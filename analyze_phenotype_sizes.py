"""
Analiza la distribucion de "tamaño de fenotipo" por gen en el JSON de
pacientes simulados, SIN construir grafos ni gastar disco. La idea es
confirmar (o descartar) si el tamaño real en disco que vimos en
create_hpo_dataset.py (65KB/grafo en la corrida de 102 genes vs
288KB/grafo en la de 271 batches) se explica por una distribucion no
uniforme de fenotipos-por-gen a lo largo del archivo, en vez de asumirlo.

Mide, por gen:
  - cantidad de fenotipos HPO UNICOS que aparecen en sus casos
    (proxy directo del tamaño del subgrafo que arma GraPhens)
  - promedio de longitud de cada caso individual

Y reporta esos valores tanto en orden de aparicion en el archivo
(para ver si hay tendencia/sesgo por posicion) como en general
(para tener el promedio real sobre TODOS los genes, no una muestra
parcial del principio del archivo).

Uso:
    python analyze_phenotype_sizes.py \
        --input ~/lucai/datos/simulation_output_A/simulated_patients_100pergene.json
"""
import argparse
import ijson
import statistics


def analyze(input_path: str):
    order_index = []
    unique_counts = []
    avg_case_len = []
    gene_names = []

    with open(input_path, "rb") as f:
        for i, (gene, cases) in enumerate(ijson.kvitems(f, "")):
            unique_phenotypes = set()
            case_lens = []
            for case in cases:
                unique_phenotypes.update(case)
                case_lens.append(len(case))

            order_index.append(i)
            gene_names.append(gene)
            unique_counts.append(len(unique_phenotypes))
            avg_case_len.append(statistics.mean(case_lens) if case_lens else 0)

    n = len(gene_names)
    print(f"Total genes analizados: {n}\n")

    # Promedio global (sobre TODOS los genes, no una muestra parcial)
    print("=== Promedio GLOBAL (todos los genes) ===")
    print(f"  Fenotipos unicos por gen -- media: {statistics.mean(unique_counts):.1f}, "
          f"mediana: {statistics.median(unique_counts):.1f}, "
          f"min: {min(unique_counts)}, max: {max(unique_counts)}")
    print(f"  Longitud promedio de caso -- media: {statistics.mean(avg_case_len):.1f}\n")

    # Comparacion: primeros N genes (lo que vimos en las corridas parciales)
    # vs el resto, para detectar sesgo de orden
    for n_head in (100, 271, 500):
        if n_head >= n:
            continue
        head = unique_counts[:n_head]
        tail = unique_counts[n_head:]
        print(f"=== Primeros {n_head} genes vs resto ({n - n_head} genes) ===")
        print(f"  Primeros {n_head}: media fenotipos unicos = {statistics.mean(head):.1f}")
        print(f"  Resto:            media fenotipos unicos = {statistics.mean(tail):.1f}")
        ratio = statistics.mean(head) / statistics.mean(tail) if statistics.mean(tail) > 0 else float('nan')
        print(f"  Ratio (primeros/resto): {ratio:.2f}x\n")

    # Top 10 genes mas pesados y mas livianos, para inspeccion manual
    paired = sorted(zip(gene_names, unique_counts), key=lambda x: -x[1])
    print("=== Top 10 genes con MAS fenotipos unicos (mas pesados) ===")
    for gene, count in paired[:10]:
        print(f"  {gene}: {count}")
    print("\n=== Top 10 genes con MENOS fenotipos unicos (mas livianos) ===")
    for gene, count in paired[-10:]:
        print(f"  {gene}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analizar distribucion de fenotipos por gen")
    parser.add_argument("--input", required=True, help="JSON de pacientes simulados a analizar")
    args = parser.parse_args()

    analyze(args.input)
