#!/usr/bin/env python3
"""
Appendea al .mmap las filas de los HPO IDs que create_hpo_dataset.py agregó
al .index.json y que todavía no tienen embedding real.

El orden lo manda el .index.json (fila i = embedding del HPO ID cuyo índice
es i), NO el orden de new_hpo_ids_sin_embedding.txt (que sale de un set()).

Verifica todo antes de escribir y no toca nada si algo no cierra.
Hace backup del .mmap antes del append.

Uso:
    python appendear_embeddings.py            # dry-run, no escribe
    python appendear_embeddings.py --write    # escribe
"""
import json
import pickle
import shutil
import sys
import numpy as np
from pathlib import Path

EMB_OT  = Path.home() / "lucai/datos/embeddings_ot"
BASE    = "hpo_embeddings_gsarti_biobert-nli_20250317_162424"
INDEX   = EMB_OT / f"{BASE}.index.json"
MMAP    = EMB_OT / f"{BASE}.mmap"
PKL_NEW = Path.home() / "lucai/datos/embeddings_regen/hpo_embeddings_gsarti_biobert-nli_regen.pkl"

N_ORIG = 19534
DIM    = 768
WRITE  = "--write" in sys.argv

print("=" * 60)
print("APPEND DE EMBEDDINGS AL .mmap" + ("" if WRITE else "   [DRY-RUN]"))
print("=" * 60)

idx = json.load(open(INDEX))
new = pickle.load(open(PKL_NEW, "rb"))

# ---- 1. estado actual del .mmap
size = MMAP.stat().st_size
filas_actuales, resto = divmod(size, DIM * 4)
print(f"\n.mmap actual: {size} bytes -> {filas_actuales} filas (resto {resto})")
assert resto == 0, "el .mmap no es múltiplo de DIM*4: hay header o corrupción"
assert filas_actuales == N_ORIG, f"esperaba {N_ORIG} filas, hay {filas_actuales} (¿ya se appendeó?)"

# ---- 2. los índices nuevos tienen que ser un rango contiguo
nuevos = {h: i for h, i in idx.items() if i >= N_ORIG}
indices = sorted(nuevos.values())
print(f".index.json: {len(idx)} entradas, {len(nuevos)} con índice >= {N_ORIG}")
print(f"rango de índices nuevos: {indices[0]}..{indices[-1]}")

esperado = list(range(N_ORIG, N_ORIG + len(nuevos)))
assert indices == esperado, "los índices nuevos NO son un rango contiguo — abortar"
assert len(set(indices)) == len(indices), "hay índices duplicados en el .index.json"
print("  OK: rango contiguo y sin duplicados")

# ---- 3. todos tienen embedding en el .pkl nuevo
faltan = [h for h in nuevos if h not in new]
if faltan:
    print(f"\nFALTAN {len(faltan)} embeddings en el .pkl:")
    print(f"  {faltan[:20]}")
    sys.exit(1)
print(f"  OK: los {len(nuevos)} tienen embedding en el .pkl")

# ---- 4. armar el bloque en el orden del índice
orden = sorted(nuevos.items(), key=lambda kv: kv[1])
bloque = np.stack([np.asarray(new[h], dtype=np.float32).ravel() for h, _ in orden])
assert bloque.shape == (len(nuevos), DIM), f"shape inesperado: {bloque.shape}"
assert bloque.dtype == np.float32
assert np.isfinite(bloque).all(), "hay NaN o inf en los embeddings nuevos"

normas = np.linalg.norm(bloque.astype(np.float64), axis=1)
print(f"\nbloque a escribir: {bloque.shape}, dtype={bloque.dtype}")
print(f"  normas L2: media={normas.mean():.4f} min={normas.min():.4f} max={normas.max():.4f}")
print(f"  (referencia de los originales: media~17.38, min~14.85, max~18.94)")
print(f"\nprimeros 3 en orden de índice:")
for h, i in orden[:3]:
    print(f"  {i}: {h}")
print(f"últimos 3:")
for h, i in orden[-3:]:
    print(f"  {i}: {h}")

final_bytes = size + bloque.nbytes
print(f"\ntamaño final esperado: {final_bytes} bytes -> {final_bytes // (DIM*4)} filas")
assert final_bytes // (DIM * 4) == len(idx), "el total no coincide con las entradas del índice"

if not WRITE:
    print("\n" + "=" * 60)
    print("DRY-RUN: todo verificado, no se escribió nada.")
    print("Para aplicar:  python appendear_embeddings.py --write")
    print("=" * 60)
    sys.exit(0)

# ---- 5. backup y append
bak = MMAP.with_suffix(".mmap.bak")
print(f"\nbackup -> {bak}")
shutil.copy2(MMAP, bak)

print("appendeando...")
with open(MMAP, "ab") as f:
    f.write(bloque.tobytes(order="C"))

# ---- 6. verificación post-escritura
size2 = MMAP.stat().st_size
filas2 = size2 // (DIM * 4)
print(f".mmap final: {size2} bytes -> {filas2} filas")
assert size2 == final_bytes, "el tamaño final no es el esperado"
assert filas2 == len(idx), "las filas no coinciden con el índice"

mm = np.memmap(MMAP, dtype=np.float32, mode="r", shape=(filas2, DIM))
errores = 0
for h, i in orden:
    if not np.allclose(mm[i], np.asarray(new[h], dtype=np.float32), rtol=0, atol=0):
        print(f"  MISMATCH en {h} (fila {i})")
        errores += 1
        if errores > 5:
            break
# control: una fila original no se movió
assert np.isfinite(mm[0]).all() and np.isfinite(mm[N_ORIG - 1]).all()

print("\n" + "=" * 60)
if errores:
    print(f"ATENCION: {errores} filas no coinciden. Restaurar desde {bak}")
else:
    print("LISTO: append verificado fila por fila.")
    print(f"Siguiente: cargar unos grafos de dataset_A_full que tengan")
    print(f"índices >= {N_ORIG} y confirmar que get() ya no tira IndexError.")
print("=" * 60)
