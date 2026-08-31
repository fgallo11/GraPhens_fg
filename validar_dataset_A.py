#!/usr/bin/env python3
"""
Valida que el dataset A puede leerse ahora que el .mmap fue extendido.

1. Escanea batches buscando grafos que usen índices >= 19534 (los nuevos).
2. Verifica que esas filas existan y no sean ceros/basura en el .mmap.
3. Instancia HPOGraphDataset (introspeccionando su firma) y llama a get()
   sobre grafos que contengan índices nuevos.

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

# ---------- 1. buscar grafos que usen índices nuevos
print("\n[1] Buscando grafos con índices >= %d ..." % N_ORIG)
encontrados = []          # (batch, pos_en_batch, indices_nuevos)
max_idx_global = -1

for b in [0, 1, 100, 250, 400, 532]:
    p = DATASET / f"batch_{b}.pt"
    if not p.exists():
        continue
    batch = torch.load(p, weights_only=False)
    for j, g in enumerate(batch):
        xi = g.x.numpy().ravel()
        max_idx_global = max(max_idx_global, int(xi.max()))
        nuevos = xi[xi >= N_ORIG]
        if len(nuevos) and len(encontrados) < 8:
            encontrados.append((b, j, sorted(set(int(v) for v in nuevos))))
    print(f"  batch_{b}: revisado ({len(batch)} grafos)")

print(f"\n  índice máximo visto: {max_idx_global}  (límite del .mmap: {NEMB-1})")
assert max_idx_global < NEMB, "hay índices fuera de rango — el .mmap no alcanza"

if not encontrados:
    print("  No aparecieron índices nuevos en los batches muestreados.")
    print("  (esperable: son 578 sobre 20112, pueden ser poco frecuentes)")
else:
    print(f"  {len(encontrados)} grafos con índices nuevos:")
    for b, j, idxs in encontrados[:5]:
        print(f"    batch_{b}[{j}]: {idxs[:6]}{' ...' if len(idxs) > 6 else ''}")

# ---------- 2. las filas nuevas no son basura
print("\n[2] Chequeando las 578 filas nuevas en el .mmap...")
nuevas = np.asarray(mm[N_ORIG:NEMB], dtype=np.float64)
normas = np.linalg.norm(nuevas, axis=1)
ceros  = int((normas == 0).sum())
print(f"  filas: {nuevas.shape[0]}  normas: media={normas.mean():.4f} "
      f"min={normas.min():.4f} max={normas.max():.4f}")
print(f"  filas todo-cero: {ceros}")
assert ceros == 0, "hay filas nuevas vacías"
assert np.isfinite(nuevas).all(), "hay NaN/inf en las filas nuevas"
print("  OK")

# ---------- 3. get() real
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
    default = sig.parameters[p].default
    if any(k in p.lower() for k in ("root", "dir", "path")) and default is inspect._empty:
        kwargs[p] = str(DATASET)
if not kwargs and params:
    kwargs[params[0]] = str(DATASET)

print(f"  llamando con: {kwargs}")
ds = HPOGraphDataset(**kwargs)
print(f"  len(ds) = {len(ds)}")

print("\n[4] Leyendo grafos...")
objetivo = [(b * meta["batch_size"] + j) for b, j, _ in encontrados[:5]] or [0, 1, 2]
for i in objetivo + [0, len(ds) // 2, len(ds) - 1]:
    try:
        g = ds[i]
        x = g.x
        print(f"  ds[{i}]: x={tuple(x.shape)} dtype={x.dtype} "
              f"norma_media={x.float().norm(dim=1).mean():.4f}")
        assert x.shape[1] == DIM, "x no tiene la dimensión de embedding"
        assert x.dtype in (torch.float32, torch.float64), "x no son floats — no se expandió"
    except Exception as e:
        print(f"  ds[{i}]: FALLO -> {type(e).__name__}: {e}")
        raise

print("\n" + "=" * 60)
print("LISTO: get() expande índices a embeddings sin IndexError.")
print("Siguiente: convert_to_lmdb.")
print("=" * 60)
