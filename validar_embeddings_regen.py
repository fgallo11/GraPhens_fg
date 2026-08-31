#!/usr/bin/env python3
"""
Valida el .pkl regenerado contra el .mmap original.

Compara TODOS los HPO IDs solapados (no una muestra) y reporta:
  - distribución de cosenos
  - normas L2 de ambos lados (detecta normalización)
  - estratificación por con/sin definición (la rama ambigua de strategies.py)
  - los peores casos, para inspección manual

Uso:
    python validar_embeddings_regen.py
"""
import json
import pickle
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------- paths
EMB_OT   = Path.home() / "lucai/datos/embeddings_ot"
BASE     = "hpo_embeddings_gsarti_biobert-nli_20250317_162424"
INDEX    = EMB_OT / f"{BASE}.index.json"
MMAP     = EMB_OT / f"{BASE}.mmap"
PKL_NEW  = Path.home() / "lucai/datos/embeddings_regen/hpo_embeddings_gsarti_biobert-nli_regen.pkl"
HP_JSON  = Path("data/ontology/hp.json")

N_ORIG   = 19534          # filas reales que tiene el .mmap hoy
DIM      = 768

# ---------------------------------------------------------------- carga
print("Cargando...")
idx = json.load(open(INDEX))
mm  = np.memmap(MMAP, dtype=np.float32, mode="r", shape=(N_ORIG, DIM))
new = pickle.load(open(PKL_NEW, "rb"))

print(f"  .index.json : {len(idx)} entradas")
print(f"  .mmap       : {N_ORIG} filas x {DIM}")
print(f"  .pkl nuevo  : {len(new)} entradas")

# HPO IDs con fila real en el .mmap (los originales)
originales = {k: v for k, v in idx.items() if v < N_ORIG}
comunes    = sorted(k for k in originales if k in new)
faltan     = sorted(k for k in originales if k not in new)

print(f"\nSolapados (original ∩ nuevo): {len(comunes)} / {len(originales)}")
if faltan:
    print(f"  ATENCION: {len(faltan)} originales NO están en el .pkl nuevo")
    print(f"  {faltan[:10]}")

# ------------------------------------------------- quién tiene definición
con_def = set()
if HP_JSON.exists():
    hp = json.load(open(HP_JSON))
    for n in hp["graphs"][0]["nodes"]:
        hid = n["id"].rsplit("/", 1)[-1].replace("_", ":")
        if ((n.get("meta") or {}).get("definition") or {}).get("val"):
            con_def.add(hid)
    print(f"Términos con definición en hp.json: {len(con_def)}")
else:
    print("hp.json no encontrado — se omite la estratificación")

# ------------------------------------------------------------- comparación
cos, n_ref, n_new = [], [], []
for h in comunes:
    a = np.asarray(new[h], dtype=np.float64).ravel()
    b = np.asarray(mm[originales[h]], dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos.append(float(a @ b / (na * nb)) if na and nb else 0.0)
    n_new.append(float(na))
    n_ref.append(float(nb))

cos   = np.array(cos)
n_ref = np.array(n_ref)
n_new = np.array(n_new)

# ------------------------------------------------------------- resultados
print("\n" + "=" * 60)
print("COSENO (nuevo vs original)")
print("=" * 60)
for q in [0, 1, 5, 25, 50, 75, 100]:
    print(f"  p{q:<3} : {np.percentile(cos, q):.6f}")
print(f"  media: {cos.mean():.6f}")
for t in [0.9999, 0.999, 0.99, 0.95, 0.90]:
    n = int((cos >= t).sum())
    print(f"  >= {t:<7}: {n:6d}  ({100*n/len(cos):5.1f}%)")

print("\n" + "=" * 60)
print("NORMAS L2  (si difieren -> normalización distinta)")
print("=" * 60)
print(f"  original : media={n_ref.mean():8.4f}  min={n_ref.min():8.4f}  max={n_ref.max():8.4f}")
print(f"  nuevo    : media={n_new.mean():8.4f}  min={n_new.min():8.4f}  max={n_new.max():8.4f}")
if abs(n_ref.mean() - 1.0) < 1e-3 and abs(n_new.mean() - 1.0) > 1e-3:
    print("  >>> los ORIGINALES están L2-normalizados y los nuevos NO")
elif abs(n_new.mean() - 1.0) < 1e-3 and abs(n_ref.mean() - 1.0) > 1e-3:
    print("  >>> los NUEVOS están L2-normalizados y los originales NO")
else:
    print("  >>> mismo régimen de normalización")

if con_def:
    print("\n" + "=" * 60)
    print("ESTRATIFICADO  (la rama ambigua: name vs description)")
    print("=" * 60)
    m = np.array([h in con_def for h in comunes])
    for etiqueta, sel in [("CON definición", m), ("SIN definición", ~m)]:
        if sel.sum():
            c = cos[sel]
            print(f"  {etiqueta:<15} n={sel.sum():6d}  media={c.mean():.6f}  "
                  f"p1={np.percentile(c,1):.6f}  >=0.999: {100*(c>=0.999).mean():5.1f}%")

print("\n" + "=" * 60)
print("PEORES 15 CASOS")
print("=" * 60)
for i in np.argsort(cos)[:15]:
    h = comunes[i]
    tag = "def" if h in con_def else "NAME"
    print(f"  {h}  cos={cos[i]:.6f}  [{tag}]")

print("\n" + "=" * 60)
umbral_ok = (cos >= 0.999).mean()
if umbral_ok > 0.95:
    print(f"VEREDICTO: PASA  ({100*umbral_ok:.1f}% por encima de 0.999)")
    print("La lógica de texto y el entorno reproducen los originales.")
    print("Siguiente: extraer los 578 del .pkl nuevo y appendear al .mmap.")
elif umbral_ok > 0.5:
    print(f"VEREDICTO: PARCIAL  ({100*umbral_ok:.1f}% por encima de 0.999)")
    print("Revisar el estratificado y los peores casos para ubicar la causa.")
else:
    print(f"VEREDICTO: NO PASA  ({100*umbral_ok:.1f}% por encima de 0.999)")
    print("Ver la tabla de patrones antes de asumir la causa.")
print("=" * 60)
