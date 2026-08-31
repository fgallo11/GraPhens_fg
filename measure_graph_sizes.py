"""
Medicion puntual y barata (~150MB, minutos) para diagnosticar el problema
de tamaño en disco del dataset de grafos GraPhens, sin correr el pipeline
completo de nuevo.

Contesta 4 preguntas:
  1. x (features) vs edge_index -- cual domina el peso de cada grafo
  2. disk / suma_teorica de un grafo guardado solo -- si hay aliasing de
     storage en torch.save (el archivo pesa mucho mas que sus tensores)
  3. nodos por grafo, comparando genes tempranos (1-102, donde la corrida 1
     midio 65 KB/grafo) vs genes tardios (~1000-2710, rango que la corrida 2
     termino promediando con 288 KB/grafo) -- confirma o descarta si el 4.4x
     es efecto de poblacion de genes, no inestabilidad por grafo individual
  4. hashes unicos vs total -- cuanto rendiria dedup de grafos identicos

Uso:
    python measure_graph_sizes.py \
        --input ~/lucai/datos/simulation_output_A/simulated_patients_100pergene.json \
        --embeddings data/embeddings/hpo_embeddings_gsarti_biobert-nli_20250317_162424.pkl \
        --n-early-genes 20 \
        --n-late-genes 20 \
        --late-start-gene-idx 1000 \
        --cases-per-gene 5

Nota: corre create_graphs_from_multiple_patients real (misma clase GraPhens
que usa create_hpo_dataset.py), asi que necesita el mismo entorno
(conda genephenai) y los mismos paths relativos (correr desde la raiz del
repo GraPhens, igual que create_hpo_dataset.py).
"""
import argparse
import hashlib
import ijson
import os
import statistics
import tempfile

import torch
from torch_geometric.data import Data

from src.graphens import GraPhens


def collect_gene_samples(input_path, n_early, n_late, late_start_idx, cases_per_gene):
    """
    Recorre el JSON en streaming (mismo mecanismo que create_hpo_dataset.py)
    y junta, sin cargar todo en memoria: los primeros n_early genes, y
    n_late genes a partir de late_start_idx. Para cada gen, se queda con
    hasta cases_per_gene casos (no todos, para mantener la muestra chica).
    """
    early = {}
    late = {}

    with open(input_path, "rb") as f:
        for i, (gene, cases) in enumerate(ijson.kvitems(f, "")):
            if i < n_early:
                early[gene] = cases[:cases_per_gene]
            elif late_start_idx <= i < late_start_idx + n_late:
                late[gene] = cases[:cases_per_gene]

            if i >= late_start_idx + n_late:
                break  # ya juntamos todo lo que necesitamos, no seguir leyendo

    return early, late


def report_one_graph(data: Data):
    """Replica el analisis que propuso el modelo consultado, por grafo."""
    parts = {}
    for key, value in data:
        if torch.is_tensor(value):
            parts[key] = value.element_size() * value.nelement()
    theoretical_total = sum(parts.values())

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        torch.save(data, tmp.name)
        disk_size = os.path.getsize(tmp.name)
        os.unlink(tmp.name)

    x_bytes = data.x.detach().cpu().numpy().tobytes()
    edge_bytes = data.edge_index.detach().cpu().numpy().tobytes()
    graph_hash = hashlib.md5(edge_bytes + x_bytes).hexdigest()

    return {
        "parts": parts,
        "theoretical_total": theoretical_total,
        "disk_size": disk_size,
        "num_nodes": int(data.x.shape[0]),
        "hash": graph_hash,
    }


def build_and_measure(graphens, gene_samples: dict, label: str):
    """Construye grafos reales para una tanda de genes y mide cada uno."""
    results = []
    for gene, cases in gene_samples.items():
        patient_data_for_gene = {
            f"{gene}_{idx}": phenotype_ids
            for idx, phenotype_ids in enumerate(cases)
            if phenotype_ids and all(isinstance(p, str) and p.startswith("HP:") for p in phenotype_ids)
        }
        if not patient_data_for_gene:
            continue

        graphs_for_gene = graphens.create_graphs_from_multiple_patients(
            patient_data_for_gene, show_progress=False
        )

        for patient_key, graph in graphs_for_gene.items():
            data = Data(
                x=torch.tensor(graph.node_features, dtype=torch.float),
                edge_index=torch.tensor(graph.edge_index, dtype=torch.long),
                phenotype_ids=patient_data_for_gene[patient_key],
                gene=gene,
            )
            measured = report_one_graph(data)
            measured["gene"] = gene
            results.append(measured)

    print(f"\n=== {label}: {len(results)} grafos medidos ===")
    if not results:
        print("  (sin grafos validos en esta tanda)")
        return results

    nodes = [r["num_nodes"] for r in results]
    disk_sizes = [r["disk_size"] for r in results]
    theo_sizes = [r["theoretical_total"] for r in results]

    print(f"  Nodos por grafo -- media: {statistics.mean(nodes):.1f}, "
          f"mediana: {statistics.median(nodes):.1f}, "
          f"min: {min(nodes)}, max: {max(nodes)}")
    print(f"  Tamaño en disco -- media: {statistics.mean(disk_sizes)/1024:.1f} KB, "
          f"mediana: {statistics.median(disk_sizes)/1024:.1f} KB")
    print(f"  Tamaño teorico (suma de tensores) -- media: {statistics.mean(theo_sizes)/1024:.1f} KB")

    ratio = statistics.mean(disk_sizes) / statistics.mean(theo_sizes) if statistics.mean(theo_sizes) > 0 else float("nan")
    print(f"  Ratio disco/teorico: {ratio:.2f}x", end="")
    if ratio > 1.3:
        print("  <-- ALTO: posible aliasing de storage en torch.save (ver nota mas abajo)")
    else:
        print("  (normal, sin indicio de aliasing)")

    # Desglose x vs edge_index sobre el primer grafo, como ejemplo representativo
    example_parts = results[0]["parts"]
    print(f"  Desglose de un grafo ejemplo: {example_parts}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Medir tamaño real de grafos GraPhens en muestra early+late")
    parser.add_argument("--input", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--n-early-genes", type=int, default=20)
    parser.add_argument("--n-late-genes", type=int, default=20)
    parser.add_argument("--late-start-gene-idx", type=int, default=1000)
    parser.add_argument("--cases-per-gene", type=int, default=5,
                         help="Casos a usar por gen en la muestra (no todos, para mantenerla chica)")
    parser.add_argument("--include-ancestors", action="store_true", default=True)
    parser.add_argument("--include-reverse-edges", action="store_true", default=True)
    args = parser.parse_args()

    print("Leyendo muestra de genes (early + late) del JSON en streaming...")
    early, late = collect_gene_samples(
        args.input, args.n_early_genes, args.n_late_genes,
        args.late_start_gene_idx, args.cases_per_gene
    )
    print(f"  Genes early recolectados: {len(early)} (indices 0-{args.n_early_genes-1})")
    print(f"  Genes late recolectados: {len(late)} (indices {args.late_start_gene_idx}-{args.late_start_gene_idx + args.n_late_genes - 1})")

    print("\nInicializando GraPhens (misma config que create_hpo_dataset.py)...")
    graphens = (GraPhens()
                .with_lookup_embeddings(args.embeddings)
                .with_augmentation(strategy=[
                    {"type": "siblings"},
                    {"type": "local", "include_ancestors": True, "include_descendants": False}
                ])
                .with_adjacency_settings(include_reverse_edges=args.include_reverse_edges))

    early_results = build_and_measure(graphens, early, "GENES TEMPRANOS (early, indices bajos)")
    late_results = build_and_measure(graphens, late, "GENES TARDIOS (late, indices altos)")

    all_results = early_results + late_results
    if all_results:
        all_hashes = [r["hash"] for r in all_results]
        unique_hashes = len(set(all_hashes))
        print(f"\n=== DEDUP: hashes unicos vs total ===")
        print(f"  {unique_hashes} unicos / {len(all_hashes)} totales "
              f"({100 * unique_hashes / len(all_hashes):.1f}% son distintos entre si)")
        if unique_hashes < len(all_hashes):
            print(f"  -> hay {len(all_hashes) - unique_hashes} grafos duplicados exactos en esta muestra")

    if early_results and late_results:
        early_nodes = statistics.mean(r["num_nodes"] for r in early_results)
        late_nodes = statistics.mean(r["num_nodes"] for r in late_results)
        print(f"\n=== COMPARACION FINAL: early vs late ===")
        print(f"  Nodos promedio -- early: {early_nodes:.1f}, late: {late_nodes:.1f}")
        print(f"  Ratio late/early: {late_nodes / early_nodes:.2f}x")
        print(f"  (si este ratio se acerca a ~4.4x, confirma que el 4.4x observado")
        print(f"   en las corridas completas es efecto de poblacion de genes,")
        print(f"   no inestabilidad por grafo individual)")


if __name__ == "__main__":
    main()
