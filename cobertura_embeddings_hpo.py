#!/usr/bin/env python
"""
Cobertura de embeddings para el gold standard HPO (brazo de control).

Antes de simular el brazo HPO hay que garantizar que cada HPO ID que
referencia phenotype_to_genes.txt tenga fila en el .mmap de embeddings.
Un termino sin fila -> el grafo lo referencia con un indice que no existe
o apunta a otra cosa -> grafo corrupto en silencio. Es el mismo problema
de los 578 terminos que ya se resolvio para OT, que puede reaparecer con
otros terminos en el gold standard de HPO.

Chequea tres cosas:
  1. Consistencia .index.json vs .mmap: misma cantidad de filas.
     (si al appendear los 578 se extendio el mmap pero no el indice,
      hay corrupcion por ese lado, independiente de todo lo demas)
  2. Cobertura: que HPO IDs del gold standard NO estan en el indice.
  3. Que genes quedan afectados por terminos faltantes (best-effort).

Uso:
    conda activate genephenai
    python cobertura_embeddings_hpo.py <phenotype_to_genes.txt> [index.json]

Si no pasas el index.json, lo busca en ~/lucai/datos/embeddings_ot/*.index.json
"""

import sys
import os
import re
import json
import glob
from collections import defaultdict

HP_RE = re.compile(r'HP[:_](\d{7})')
EMB_DIM = 768
BYTES_PER_FLOAT = 4  # float32


def canon(s):
    """Normaliza HP:xxxxxxx / HP_xxxxxxx a la forma con dos puntos."""
    m = HP_RE.search(s)
    return f"HP:{m.group(1)}" if m else None


# --- Resolver rutas -------------------------------------------------------

if len(sys.argv) < 2:
    print("ERROR: falta la ruta a phenotype_to_genes.txt")
    print("Uso: python cobertura_embeddings_hpo.py "
          "<phenotype_to_genes.txt> [index.json]")
    print("\nSi no sabes donde esta el archivo, busalo con:")
    print("  find ~/lucai -name phenotype_to_genes.txt 2>/dev/null")
    sys.exit(1)

gold_path = os.path.expanduser(sys.argv[1])
if not os.path.exists(gold_path):
    print(f"ERROR: no existe {gold_path}")
    sys.exit(1)

if len(sys.argv) >= 3:
    index_path = os.path.expanduser(sys.argv[2])
else:
    cand = glob.glob(os.path.expanduser(
        "~/lucai/datos/embeddings_ot/*.index.json"))
    if not cand:
        print("ERROR: no encontre ningun *.index.json en "
              "~/lucai/datos/embeddings_ot/")
        print("Pasalo como segundo argumento.")
        sys.exit(1)
    index_path = cand[0]
    if len(cand) > 1:
        print(f"AVISO: hay {len(cand)} .index.json, uso el primero:")
        print(f"       {index_path}\n")

print(f"Gold standard: {gold_path}")
print(f"Indice embeddings: {index_path}\n")

# --- Check 1: consistencia index vs mmap ----------------------------------

print("=" * 68)
print("CHECK 1 — consistencia .index.json vs .mmap")
print("=" * 68)

with open(index_path) as f:
    index = json.load(f)

n_index = len(index)
print(f"Entradas en el indice:     {n_index:,}")

# muestra de formato de claves
sample_keys = list(index.keys())[:3]
print(f"Formato de claves (muestra): {sample_keys}")

mmap_path = index_path.replace(".index.json", ".mmap")
if os.path.exists(mmap_path):
    size = os.path.getsize(mmap_path)
    row_bytes = EMB_DIM * BYTES_PER_FLOAT
    if size % row_bytes == 0:
        n_mmap = size // row_bytes
        print(f"Filas en el .mmap:         {n_mmap:,} "
              f"({size:,} bytes / {row_bytes})")
        if n_mmap == n_index:
            print("RESULTADO:                 OK — indice y mmap coinciden.")
        else:
            print(f"RESULTADO:                 *** MISMATCH *** "
                  f"indice={n_index:,} vs mmap={n_mmap:,}")
            print("  El mmap se extendio sin actualizar el indice (o al reves).")
            print("  Esto corrompe la expansion de x. Resolver ANTES de simular.")
    else:
        print(f"AVISO: el tamano del .mmap ({size:,} B) no es multiplo de "
              f"{row_bytes}. No puedo inferir filas.")
else:
    print(f"AVISO: no encontre el .mmap junto al indice ({mmap_path}).")
    print("       Salteo la verificacion de consistencia.")

# --- Leer gold standard ---------------------------------------------------

print("\n" + "=" * 68)
print("CHECK 2 — cobertura del gold standard HPO")
print("=" * 68)

hpo_in_gold = set()
hpo_to_genes = defaultdict(set)   # hpo_id -> genes que lo referencian
sample_lines = []
n_lines = 0

with open(gold_path) as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        n_lines += 1
        if len(sample_lines) < 3:
            sample_lines.append(line.rstrip("\n"))

        parts = line.rstrip("\n").split("\t")
        hpo = None
        for p in parts:
            c = canon(p)
            if c:
                hpo = c
                break
        if hpo is None:
            continue
        hpo_in_gold.add(hpo)

        # gene: best-effort. Formato estandar HPO -> columna 3 (0-based)
        # es el simbolo del gen. Se valida abajo mostrando la muestra.
        if len(parts) > 3:
            gene = parts[3].strip()
            if gene and not canon(gene):   # no es otro HP id
                hpo_to_genes[hpo].add(gene)

print(f"Lineas de datos:           {n_lines:,}")
print(f"HPO IDs unicos en el gold: {len(hpo_in_gold):,}")
print("\nMuestra de lineas crudas (verifica que la col. 4 sea el gen):")
for sl in sample_lines:
    print(f"  {sl}")

# --- Comparar -------------------------------------------------------------

emb_keys = set()
for k in index.keys():
    c = canon(k)
    emb_keys.add(c if c else k)

present = hpo_in_gold & emb_keys
missing = hpo_in_gold - emb_keys

print(f"\nPresentes en el indice:    {len(present):,}")
print(f"FALTANTES:                 {len(missing):,}")

if not missing:
    print("\nRESULTADO: OK — cobertura total. La simulacion del brazo HPO "
          "puede correr.")
else:
    pct = 100.0 * len(missing) / len(hpo_in_gold)
    print(f"\nRESULTADO: faltan {len(missing):,} terminos "
          f"({pct:.1f}% del gold).")
    print("  Hay que regenerar sus embeddings y appendearlos al .mmap "
          "(mismo procedimiento que los 578 de OT) ANTES de simular.")

    out_path = os.path.join(os.getcwd(), "hpo_ids_faltantes_gold_hpo.txt")
    with open(out_path, "w") as f:
        for h in sorted(missing):
            f.write(h + "\n")
    print(f"  Lista escrita en: {out_path}")

    # --- Check 3: genes afectados ---
    if hpo_to_genes:
        genes_afectados = set()
        for h in missing:
            genes_afectados |= hpo_to_genes.get(h, set())
        print("\n" + "=" * 68)
        print("CHECK 3 — genes afectados por terminos faltantes")
        print("=" * 68)
        print(f"Genes que referencian al menos un termino faltante: "
              f"{len(genes_afectados):,}")
        muestra = sorted(genes_afectados)[:20]
        print(f"Muestra: {muestra}")
        print("\n  (best-effort: asume que la columna 4 es el simbolo del gen.")
        print("   Verifica contra la muestra de lineas crudas de arriba.)")
    else:
        print("\n(No pude mapear genes — el formato no coincide con lo "
              "esperado. La lista de HPO IDs faltantes de arriba es lo que "
              "importa para decidir.)")

# --- Veredicto ------------------------------------------------------------

print("\n" + "=" * 68)
if not missing:
    print("VEREDICTO: cobertura OK. Segui con la simulacion del brazo HPO.")
else:
    print("VEREDICTO: NO simular todavia. Regenerar los embeddings faltantes "
          "primero, o esos genes salen con grafos corruptos en silencio.")
print("=" * 68)
