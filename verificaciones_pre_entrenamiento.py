#!/usr/bin/env python
"""
Verificaciones baratas sobre el dataset OT (dataset_A_full) antes de entrenar.

Corre tres chequeos, todos en CPU, leyendo los batch_N.pt crudos
(no pasa por el caché ni por el memmap de embeddings):

  1. Rango de x:  todo indice de nodo debe estar en [0, N_EMBEDDINGS-1].
     Un indice fuera de rango dispara device-side assert en el gather sobre
     CUDA y obliga a matar el proceso.

  2. edge_index local:  para cada grafo, edge_index debe estar en [0, n_nodos-1].
     Un indice fuera de rango conecta nodos que no existen -> grafo corrupto,
     entrena con basura sin crashear.

  3. Duplicados exactos:  hashea el contenido real de cada grafo
     (x + edge_index) junto con su gen. Dos grafos identicos con el mismo gen
     son un caso repetido: bajo el split aleatorio caen a ambos lados
     (train y val) -> leakage genuino. Se reporta la distribucion por gen y
     se cruza con el tamano del pool de fenotipos anotados de cada gen, para
     ver si los duplicados se concentran en los genes poco anotados.

Uso:
    conda activate genephenai
    python verificaciones_pre_entrenamiento.py [ruta_dataset]

Por defecto ruta_dataset = ~/lucai/datos/dataset_A_full
"""

import sys
import os
import json
import glob
import time
import hashlib
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch_geometric  # necesario para deserializar los objetos Data

# --- Config ---------------------------------------------------------------

N_EMBEDDINGS = 20126          # filas del .mmap de embeddings: indices validos 0..20111
DATASET_DIR = os.path.expanduser(
    sys.argv[1] if len(sys.argv) > 1 else "~/lucai/datos/dataset_A_full"
)
N_WORST = 25                  # cuantos genes peores mostrar en la tabla de duplicados
N_SAMPLES = 5                 # cuantos ejemplos de violacion mostrar

# --- Estado acumulado -----------------------------------------------------

# Check 1
global_min_x = None
global_max_x = None
x_violations = []             # (archivo, clave, valor_ofensor)
n_graphs_x_bad = 0

# Check 2
edge_violations = []          # (archivo, clave, edge_max, n_nodos)
n_graphs_edge_bad = 0

# Check 3
gene_n_patients = Counter()               # gen -> nro de pacientes
gene_sig_counts = defaultdict(Counter)    # gen -> Counter(digest_contenido -> nro copias)
gene_pheno_pool = defaultdict(set)        # gen -> union de HPO IDs semilla

n_graphs_total = 0


def content_digest(data):
    """Hash del contenido real del grafo: x (indices) + edge_index."""
    h = hashlib.blake2b(digest_size=16)
    h.update(data.x.contiguous().numpy().tobytes())
    if data.edge_index.numel() > 0:
        h.update(data.edge_index.contiguous().numpy().tobytes())
    return h.digest()


def pheno_list(data):
    """phenotype_ids como lista de strings, tolerante al tipo."""
    pid = getattr(data, "phenotype_ids", None)
    if pid is None:
        return []
    if torch.is_tensor(pid):
        return [str(v) for v in pid.tolist()]
    return [str(v) for v in pid]


# --- Recorrido ------------------------------------------------------------

files = sorted(
    glob.glob(os.path.join(DATASET_DIR, "batch_*.pt")),
    key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]),
)

if not files:
    print(f"ERROR: no se encontraron batch_*.pt en {DATASET_DIR}")
    sys.exit(1)

print(f"Dataset: {DATASET_DIR}")
print(f"Archivos batch_*.pt: {len(files)}")
print("Recorriendo grafos...\n")

t0 = time.time()

for fi, fpath in enumerate(files):
    batch = torch.load(fpath, weights_only=False)
    fname = os.path.basename(fpath)

    for key, data in batch.items():
        n_graphs_total += 1
        gene = data.gene

        # ---- Check 1: rango de x ----
        x = data.x
        if x.numel() > 0:
            xmax = int(x.max())
            xmin = int(x.min())
            global_max_x = xmax if global_max_x is None else max(global_max_x, xmax)
            global_min_x = xmin if global_min_x is None else min(global_min_x, xmin)
            if xmax >= N_EMBEDDINGS or xmin < 0:
                n_graphs_x_bad += 1
                if len(x_violations) < N_SAMPLES:
                    bad = [int(v) for v in x.tolist() if v >= N_EMBEDDINGS or v < 0]
                    x_violations.append((fname, key, bad[:10]))

        # ---- Check 2: edge_index local ----
        ei = data.edge_index
        n_nodos = int(x.shape[0])
        if ei.numel() > 0:
            emax = int(ei.max())
            emin = int(ei.min())
            if emax >= n_nodos or emin < 0:
                n_graphs_edge_bad += 1
                if len(edge_violations) < N_SAMPLES:
                    edge_violations.append((fname, key, emax, n_nodos))

        # ---- Check 3: duplicados ----
        gene_n_patients[gene] += 1
        gene_sig_counts[gene][content_digest(data)] += 1
        gene_pheno_pool[gene].update(pheno_list(data))

    if (fi + 1) % 50 == 0 or (fi + 1) == len(files):
        el = time.time() - t0
        print(f"  {fi+1:3d}/{len(files)} archivos | "
              f"{n_graphs_total:,} grafos | {el:5.1f}s")

el = time.time() - t0
print(f"\nRecorrido completo: {n_graphs_total:,} grafos en {el:.1f}s\n")

# --- Reporte --------------------------------------------------------------

sep = "=" * 70
n_genes = len(gene_n_patients)
patients_per_gene = [c for c in gene_n_patients.values()]

print(sep)
print("RESUMEN GENERAL")
print(sep)
print(f"Grafos totales:        {n_graphs_total:,}")
print(f"Genes distintos:       {n_genes:,}")
print(f"Pacientes/gen:         min={min(patients_per_gene)}, "
      f"max={max(patients_per_gene)}, "
      f"media={sum(patients_per_gene)/n_genes:.1f}")
balance_ok = min(patients_per_gene) == max(patients_per_gene)
print(f"Balance exacto:        {'SI' if balance_ok else 'NO (revisar)'}")

# ---- Check 1 ----
print("\n" + sep)
print("CHECK 1 — RANGO DE x  (valido: 0 .. %d)" % (N_EMBEDDINGS - 1))
print(sep)
print(f"min(x) global:         {global_min_x}")
print(f"max(x) global:         {global_max_x}")
print(f"limite:                {N_EMBEDDINGS - 1}")
if n_graphs_x_bad == 0:
    print("RESULTADO:             OK — ningun indice fuera de rango.")
else:
    print(f"RESULTADO:             *** FALLA *** {n_graphs_x_bad:,} grafos "
          f"con indices fuera de rango.")
    print("  (esto dispara device-side assert en CUDA — hay que resolverlo)")
    for fname, key, bad in x_violations:
        print(f"    {fname} | {key} | ejemplos: {bad}")

# ---- Check 2 ----
print("\n" + sep)
print("CHECK 2 — edge_index LOCAL  (valido por grafo: 0 .. n_nodos-1)")
print(sep)
if n_graphs_edge_bad == 0:
    print("RESULTADO:             OK — todos los edge_index dentro de rango.")
else:
    print(f"RESULTADO:             *** FALLA *** {n_graphs_edge_bad:,} grafos "
          f"con edge_index fuera de rango.")
    for fname, key, emax, n in edge_violations:
        print(f"    {fname} | {key} | edge_max={emax} >= n_nodos={n}")

# ---- Check 3 ----
print("\n" + sep)
print("CHECK 3 — DUPLICADOS EXACTOS  (mismo gen + mismo grafo)")
print(sep)

total_redundant = 0           # copias sobrantes: sum(count-1) sobre firmas repetidas
genes_con_dups = 0
worst = []                    # (gen, n_pac, n_distintos, n_redundantes, pool)

for gene, sig_counts in gene_sig_counts.items():
    n_pac = gene_n_patients[gene]
    n_distintos = len(sig_counts)
    redundantes = n_pac - n_distintos
    pool = len(gene_pheno_pool[gene])
    if redundantes > 0:
        genes_con_dups += 1
        total_redundant += redundantes
    worst.append((gene, n_pac, n_distintos, redundantes, pool))

pct_redundant = 100.0 * total_redundant / n_graphs_total

print(f"Grafos redundantes:    {total_redundant:,} "
      f"({pct_redundant:.2f}% del dataset)")
print(f"Genes con duplicados:  {genes_con_dups:,} / {n_genes:,} "
      f"({100.0*genes_con_dups/n_genes:.1f}%)")

if total_redundant == 0:
    print("RESULTADO:             OK — cero duplicados exactos. "
          "No hay leakage por repeticion.")
    print("  (si esperabas duplicados en genes poco anotados y da 0, "
          "sospecha del script antes de festejar)")
else:
    print("RESULTADO:             hay duplicados. Bajo split aleatorio caen "
          "a ambos lados (train/val/test).")
    print("  La decision no es 'deduplicar todo' sino que hacer con los "
          "genes de cola.")
    print("  Afecta igual a los dos brazos (OT y HPO), asi que no compromete "
          "la comparacion.\n")

    # concentracion: que fraccion de los duplicados vive en el X% de genes peores
    worst.sort(key=lambda r: r[3], reverse=True)
    top5pct = max(1, int(round(0.05 * n_genes)))
    dups_top5pct = sum(r[3] for r in worst[:top5pct])
    if total_redundant > 0:
        print(f"  Concentracion: el 5% de genes con mas duplicados "
              f"({top5pct} genes) acumula "
              f"{100.0*dups_top5pct/total_redundant:.0f}% de los duplicados.\n")

    print(f"  Peores {N_WORST} genes (mas grafos redundantes):")
    print(f"    {'gen':<15} {'pac':>4} {'distintos':>9} {'redund':>7} "
          f"{'pool_HPO':>8}")
    for gene, n_pac, n_dist, redun, pool in worst[:N_WORST]:
        print(f"    {gene:<15} {n_pac:>4} {n_dist:>9} {redun:>7} {pool:>8}")

# histograma de distintos por gen
dist_hist = Counter(len(sc) for sc in gene_sig_counts.values())
print("\n  Distribucion de grafos distintos por gen (histograma):")
for n_dist in sorted(dist_hist):
    bar = "#" * min(50, dist_hist[n_dist])
    print(f"    {n_dist:>3} distintos: {dist_hist[n_dist]:>5} genes {bar}")

# --- Volcado a JSON para el registro --------------------------------------

out = {
    "dataset": DATASET_DIR,
    "n_graphs": n_graphs_total,
    "n_genes": n_genes,
    "balance_exacto": balance_ok,
    "check1_x_range": {
        "min": global_min_x, "max": global_max_x,
        "limite": N_EMBEDDINGS - 1, "grafos_fuera_de_rango": n_graphs_x_bad,
    },
    "check2_edge_index": {"grafos_fuera_de_rango": n_graphs_edge_bad},
    "check3_duplicados": {
        "grafos_redundantes": total_redundant,
        "pct_redundante": round(pct_redundant, 4),
        "genes_con_duplicados": genes_con_dups,
    },
}
out_path = os.path.join(os.getcwd(), "verificaciones_resultado.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nResumen guardado en: {out_path}")

# --- Veredicto ------------------------------------------------------------

print("\n" + sep)
blocker = (n_graphs_x_bad > 0) or (n_graphs_edge_bad > 0)
if blocker:
    print("VEREDICTO: NO entrenar todavia. Hay violaciones de rango "
          "(check 1 y/o 2).")
else:
    print("VEREDICTO: checks 1 y 2 OK. Podes pasar al smoke test.")
    print("Los duplicados (check 3) no bloquean: son informacion para "
          "decidir que hacer con los genes de cola, no un error a corregir "
          "antes de entrenar.")
print(sep)
