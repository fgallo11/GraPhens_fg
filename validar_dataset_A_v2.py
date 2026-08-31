#!/usr/bin/env python3
"""
Valida que el dataset A puede leerse ahora que el .mmap fue extendido.
Los batches son dict {clave: Data} con x = tensor 1D de índices.

Correr desde la raíz del repo.
"""
import inspect
import json
import sys
import numpy as np
import torch
from pathlib import Path

DATASET = Path.home() / "lucai/datos/dataset_A_full"
N_ORIG  = 19534

meta = json.load(open(DATASET / "metadata.json"))
MMAP = Path(meta["embedding_memmap_path"])
DIM  = meta["embedding_dim"]
NEMB = meta["num_embeddings"]

print("=" * 60)
print("VALIDACION DEL DATASET A")
print("=" * 60)
print(f"metadata: num_embeddings={NEMB} dim={DIM} samples={meta['num_samples']}")

filas = MMAP.stat().st_size // (DIM * 4)
print(f".mmap   : {filas} filas")
assert filas == NEMB, f"desajuste: mmap {filas} vs metadata {NEMB}"

mm = np.memmap(MMAP, dtype=np.float32, mode="r", shape=(NEMB, DIM))

# ---------- 1. escanear batches
print(f"\n[1] Buscando grafos con índices >= {N_ORIG} ...")
encontrados = []
max_idx = -1
total = 0

for b in [0, 1, 100, 250, 400, 532]:
    p = DATASET / f"batch_{b}.pt"
    if not p.exists():
        continue
    batch = torch.load(p, weights_only=False)
    con_nuevos = 0
    for j, (clave, g) in enumerate(batch.items()):
        xi = g.x.numpy().ravel()
        total += 1
        if xi.size:
            max_idx = max(max_idx, int(xi.max()))
        nuevos = sorted(set(int(v) for v in xi[xi >= N_ORIG]))
        if nuevos:
            con_nuevos += 1
            if len(encontrados) < 10:
                encontrados.append((b, j, clave, nuevos))
    print(f"  batch_{b}: {len(batch)} grafos, {con_nuevos} con índices nuevos")

print(f"\n  grafos revisados: {total}")
print(f"  índice máximo visto: {max_idx}  (límite del .mmap: {NEMB-1})")
assert max_idx < NEMB, "hay índices fuera de rango — el .mmap no alcanza"

if encontrados:
    print(f"  ejemplos:")
    for b, j, clave, idxs in encontrados[:5]:
        print(f"    batch_{b}[{j}] {clave}: {idxs[:6]}{' ...' if len(idxs) > 6 else ''}")
else:
    print("  ninguno en la muestra (esperable: 578 sobre 20112)")

# ---------- 2. filas nuevas
print(f"\n[2] Chequeando las {NEMB - N_ORIG} filas nuevas del .mmap...")
nuevas = np.asarray(mm[N_ORIG:NEMB], dtype=np.float64)
normas = np.linalg.norm(nuevas, axis=1)
ceros = int((normas == 0).sum())
print(f"  normas: media={normas.mean():.4f} min={normas.min():.4f} max={normas.max():.4f}")
print(f"  filas todo-cero: {ceros}")
assert ceros == 0, "hay filas nuevas vacías"
assert np.isfinite(nuevas).all(), "hay NaN/inf en las filas nuevas"
print("  OK")

# ---------- 3. get()
print("\n[3] Instanciando HPOGraphDataset...")
try:
    from src.simulation.phenotype_simulation.create_hpo_dataset import HPOGraphDataset
except Exception as e:
    print(f"  no se pudo importar: {e}")
    sys.exit(1)

sig = inspect.signature(HPOGraphDataset.__init__)
print(f"  firma: {sig}")

params = [p for p in sig.parameters if p != "self"]
kwargs = {}
for p in params:
    if sig.parameters[p].default is inspect._empty and \
       any(k in p.lower() for k in ("root", "dir", "path")):
        kwargs[p] = str(DATASET)
if not kwargs and params:
    kwargs[params[0]] = str(DATASET)

print(f"  llamando con: {kwargs}")
ds = HPOGraphDataset(**kwargs)
n = len(ds)
print(f"  len(ds) = {n}")

print("\n[4] Leyendo grafos...")
bs = meta["batch_size"]
objetivo = [b * bs + j for b, j, _, _ in encontrados[:5]]
objetivo += [0, n // 2, n - 1]

fallos = 0
for i in objetivo:
    if not (0 <= i < n):
        continue
    try:
        g = ds[i]
        x = g.x
        marca = ""
        if any(i == b * bs + j for b, j, _, _ in encontrados[:5]):
            marca = "  <-- tiene índices nuevos"
        print(f"  ds[{i}]: x={tuple(x.shape)} dtype={x.dtype} "
              f"norma_media={x.float().norm(dim=1).mean():.4f}{marca}")
        assert x.dim() == 2 and x.shape[1] == DIM, "x no quedó expandido a (N, dim)"
        assert x.dtype in (torch.float32, torch.float64), "x no son floats"
    except Exception as e:
        print(f"  ds[{i}]: FALLO -> {type(e).__name__}: {e}")
        fallos += 1

print("\n" + "=" * 60)
if fallos:
    print(f"ATENCION: {fallos} lecturas fallaron.")
else:
    print("LISTO: get() expande índices a embeddings sin IndexError.")
    print("Siguiente: convert_to_lmdb.")
print("=" * 60)
