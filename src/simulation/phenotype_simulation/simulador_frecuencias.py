"""
Simulador de pacientes por paquetes de enfermedad y con frecuencias de síntomas.

Diseño y decisiones: lucai/datos/ot_v3_paraguas/DISENO_simulador_frecuencias.md
No reemplaza al simulador original (StandardPhenotypeSimulator + HPODistancePhenotypeSelector), que queda igual.

Qué hace, para cada gen:
  1. Reparte los pacientes del gen (150 por defecto) entre sus enfermedades: paquetes proporcionales a la
     cantidad de síntomas de cada enfermedad, con piso de 20 (o total/k si no alcanza).
  2. Para cada paciente de un paquete:
     a. N = cantidad de síntomas, sorteada de la misma distribución empírica que el simulador original.
     b. Elige N síntomas distintos de esa enfermedad: primero los obligatorios (p = 1), después el resto
        sin repetir, con peso = probabilidad del síntoma (sin frecuencia -> 0.5; p = 0 o excluido -> nunca).
     c. A cada síntoma elegido le aplica la generalización: sortea una distancia (misma distribución que el
        original) y sube esa cantidad de pasos por la ontología (padre al azar en cada paso). Nunca llega a
        HP:0000001 ni HP:0000118, y nunca a un síntoma excluido para esa enfermedad.
     d. Si la enfermedad tiene menos síntomas que N, completa con ancestros de los ya elegidos, como hace el
        original cuando el gen no alcanza.
La etiqueta de cada paciente es el gen. La enfermedad de cada paciente se guarda aparte, solo para control.
"""
import logging
import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.ontology.hpo_graph import HPOGraphProvider
from src.simulation.phenotype_simulation.distributions import EmpiricalDistribution

logger = logging.getLogger(__name__)

RAICES = {"HP:0000001", "HP:0000118"}


def repartir_pacientes(sintomas_por_enf: Dict[str, int], total: int = 150, piso: int = 20) -> Dict[str, int]:
    """Opción A: total fijo por gen, proporcional a los síntomas, con piso por enfermedad.
    Usa el método del mayor resto para que la suma dé exactamente `total`."""
    enfs = sorted(sintomas_por_enf)
    k = len(enfs)
    piso_ef = min(piso, total / k)
    x = np.array([sintomas_por_enf[e] for e in enfs], dtype=float)
    cuota = piso_ef + (total - piso_ef * k) * x / x.sum()
    base = np.floor(cuota).astype(int)
    resto = total - base.sum()
    orden = np.argsort(-(cuota - base))
    base[orden[:resto]] += 1
    return dict(zip(enfs, base.tolist()))


class SimuladorPaquetesFrecuencia:
    def __init__(self, base_dir: str, ontology_dir: str, count_file: str, distance_file: str,
                 prob_sin_frecuencia: float = 0.5, pacientes_por_gen: int = 150, piso: int = 20,
                 usar_frecuencias: bool = True, seed: Optional[int] = None):
        self.p_default = prob_sin_frecuencia
        self.total = pacientes_por_gen
        self.piso = piso
        self.usar_frecuencias = usar_frecuencias  # False: paquetes por enfermedad pero elección pareja
        self.rng = random.Random(seed)
        np.random.seed(seed if seed is not None else None)

        # misma distribución de cantidad y distancia que el simulador original
        self.dist = EmpiricalDistribution()
        self.dist.fit({"phenotype_counts": pd.read_csv(count_file)["count"].values,
                       "distance_values": pd.read_csv(distance_file)["distance"].values})

        self.hpo = HPOGraphProvider(data_dir=ontology_dir)
        self.hpo.load()
        self._padres: Dict[str, List[str]] = {}

        d = pd.read_csv(f"{base_dir}/genes_to_phenotype.txt", sep="\t", dtype=str)
        d["p"] = pd.to_numeric(d["frequency"], errors="coerce")
        ex = pd.read_csv(f"{base_dir}/exclusiones.tsv", sep="\t", dtype=str)
        self.excluidos = defaultdict(set)
        for e, h in zip(ex.disease_id, ex.hpo_id):
            self.excluidos[e].add(h)
        # un síntoma puede venir repetido para la misma (gen, enfermedad): se queda la mayor probabilidad
        d = d.groupby(["gene_symbol", "disease_id", "hpo_id"], as_index=False)["p"].max()
        self.perfiles: Dict[str, Dict[str, Tuple[List[str], np.ndarray]]] = defaultdict(dict)
        for (g, e), grp in d.groupby(["gene_symbol", "disease_id"]):
            p = grp.p.fillna(self.p_default).to_numpy()
            ids = grp.hpo_id.tolist()
            keep = [i for i, (h, w) in enumerate(zip(ids, p)) if w > 0 and h not in self.excluidos[e]]
            if keep:
                self.perfiles[g][e] = ([ids[i] for i in keep], p[keep])
        logger.info(f"Base cargada: {len(self.perfiles)} genes, "
                    f"{sum(len(v) for v in self.perfiles.values())} pares gen-enfermedad")

    # ------------------------------------------------------------------ ontología
    def _padres_de(self, h: str) -> List[str]:
        if h not in self._padres:
            self._padres[h] = sorted(p for p in self.hpo.get_direct_parents(h) if p not in RAICES)
        return self._padres[h]

    def _generalizar(self, h: str, distancia: int, prohibidos: set) -> str:
        """Sube `distancia` pasos (padre al azar); si un paso no es posible o cae en algo prohibido, se queda."""
        actual = h
        for _ in range(distancia):
            padres = [p for p in self._padres_de(actual) if p not in prohibidos]
            if not padres:
                break
            actual = self.rng.choice(padres)
        return actual

    # ------------------------------------------------------------------ un paciente
    def _elegir(self, ids: List[str], p: np.ndarray, n: int) -> List[str]:
        n = min(n, len(ids))
        if not self.usar_frecuencias:
            return self.rng.sample(ids, n)
        oblig = [h for h, w in zip(ids, p) if w >= 0.999]
        self.rng.shuffle(oblig)
        elegidos = oblig[:n]
        faltan = n - len(elegidos)
        if faltan > 0:
            resto = [(h, w) for h, w in zip(ids, p) if w < 0.999]
            # muestreo ponderado sin reposición (Efraimidis-Spirakis): clave = u^(1/w), se toman las mayores
            claves = [(self.rng.random() ** (1.0 / w), h) for h, w in resto]
            claves.sort(reverse=True)
            elegidos += [h for _, h in claves[:faltan]]
        return elegidos

    def generar_paciente(self, gen: str, enfermedad: str) -> List[str]:
        ids, p = self.perfiles[gen][enfermedad]
        n = self.dist.sample_phenotype_count()
        excl = self.excluidos.get(enfermedad, set())
        especificos = self._elegir(ids, p, n)
        paciente: List[str] = []
        for h in especificos:
            dist = int(self.dist.sample_distances(1)[0])
            t = self._generalizar(h, dist, excl | set(paciente))
            if t in paciente:  # choque con uno ya elegido: se deja el específico
                t = h
            if t not in paciente:
                paciente.append(t)
        # completar con ancestros si la enfermedad no alcanza para N síntomas
        intentos = 0
        while len(paciente) < n and intentos < 10 * n:
            intentos += 1
            base = self.rng.choice(especificos)
            t = self._generalizar(base, self.rng.randint(1, 3), excl)
            if t != base and t not in paciente:
                paciente.append(t)
        return paciente

    # ------------------------------------------------------------------ todos
    def generar(self, genes: Optional[List[str]] = None) -> Tuple[Dict[str, List[List[str]]], Dict[str, List[str]]]:
        genes = genes or sorted(self.perfiles)
        pacientes, enfermedad_de = {}, {}
        for g in genes:
            if g not in self.perfiles:
                logger.warning(f"Gen sin perfiles utilizables: {g}")
                continue
            reparto = repartir_pacientes({e: len(v[0]) for e, v in self.perfiles[g].items()}, self.total, self.piso)
            pacientes[g], enfermedad_de[g] = [], []
            for e, k in reparto.items():
                for _ in range(k):
                    pacientes[g].append(self.generar_paciente(g, e))
                    enfermedad_de[g].append(e)
        return pacientes, enfermedad_de
