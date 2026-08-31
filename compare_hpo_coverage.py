"""
Compara TODOS los HPO IDs unicos que aparecen en el dataset OT filtrado
(leaf_filter_output_A) contra las claves del .index.json de embeddings,
para saber el numero REAL de cobertura -- sin depender de cuantos casos
por gen se hayan sorteado en una simulacion puntual.

Uso:
    python compare_hpo_coverage.py \
        --leaf-filter-dir ~/lucai/datos/leaf_filter_output_A \
        --index-json data/embeddings/hpo_embeddings_gsarti_biobert-nli_20250317_162424.index.json \
        --hpo-column hpo_id
"""
import argparse
import json
from pathlib import Path


def find_data_files(leaf_filter_path: Path):
    if leaf_filter_path.is_file():
        return [leaf_filter_path]
    candidates = []
    for ext in ("*.txt", "*.csv", "*.tsv"):
        candidates.extend(leaf_filter_path.glob(ext))
    return sorted(candidates)


def extract_hpo_ids_from_file(filepath: Path, hpo_column: str) -> set:
    hpo_ids = set()
    with open(filepath, 'r') as f:
        header_line = f.readline().rstrip('\n')
        if '\t' in header_line:
            sep = '\t'
        elif ',' in header_line:
            sep = ','
        else:
            raise ValueError(
                f"No pude detectar el separador en {filepath} a partir del "
                f"header: {header_line!r}. Esperaba tab o coma."
            )
        columns = header_line.split(sep)
        if hpo_column not in columns:
            raise ValueError(
                f"La columna '{hpo_column}' no esta en el header de {filepath}. "
                f"Columnas encontradas: {columns}"
            )
        hpo_idx = columns.index(hpo_column)

        for line in f:
            parts = line.rstrip('\n').split(sep)
            if len(parts) > hpo_idx:
                hpo_id = parts[hpo_idx].strip()
                if hpo_id:
                    hpo_ids.add(hpo_id)
    return hpo_ids


def main():
    parser = argparse.ArgumentParser(description="Comparar cobertura real de embeddings HPO")
    parser.add_argument("--leaf-filter-dir", required=True)
    parser.add_argument("--index-json", required=True)
    parser.add_argument("--hpo-column", default="hpo_id")
    args = parser.parse_args()

    leaf_filter_path = Path(args.leaf_filter_dir).expanduser()
    data_files = find_data_files(leaf_filter_path)

    if not data_files:
        print(f"ERROR: no encontre archivos de datos en {leaf_filter_path}")
        return

    print(f"Archivos de datos encontrados: {[str(f) for f in data_files]}\n")

    all_hpo_ids_in_ot = set()
    for filepath in data_files:
        print(f"Leyendo {filepath}...")
        ids_here = extract_hpo_ids_from_file(filepath, args.hpo_column)
        print(f"  {len(ids_here)} HPO IDs unicos en este archivo")
        all_hpo_ids_in_ot.update(ids_here)

    print(f"\nTotal HPO IDs unicos en el dataset OT (todos los archivos): {len(all_hpo_ids_in_ot)}")

    with open(args.index_json, 'r') as f:
        embedding_index = json.load(f)
    embedding_hpo_ids = set(embedding_index.keys())
    print(f"Total HPO IDs con embedding precomputado (indice): {len(embedding_hpo_ids)}\n")

    missing = all_hpo_ids_in_ot - embedding_hpo_ids
    covered = all_hpo_ids_in_ot & embedding_hpo_ids

    print("=== RESULTADO ===")
    print(f"HPO IDs del dataset OT CON embedding:    {len(covered)} "
          f"({100 * len(covered) / len(all_hpo_ids_in_ot):.1f}%)")
    print(f"HPO IDs del dataset OT SIN embedding:     {len(missing)} "
          f"({100 * len(missing) / len(all_hpo_ids_in_ot):.1f}%)")

    if missing:
        missing_sorted = sorted(missing)
        out_path = leaf_filter_path.parent / "hpo_ids_sin_embedding_COMPLETO.txt"
        with open(out_path, 'w') as f:
            f.write('\n'.join(missing_sorted))
        print(f"\nLista completa de HPO IDs sin embedding guardada en: {out_path}")
        print(f"Primeros 20: {missing_sorted[:20]}")


if __name__ == "__main__":
    main()
