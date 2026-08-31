"""
Submuestrea el JSON de pacientes simulados a N casos por gen (aleatorio,
semilla fija para reproducibilidad). Usa ijson para leer en streaming
(mismo motivo que en create_hpo_dataset.py: el JSON completo no entra
cómodo en RAM) y escribe incrementalmente para no acumular el resultado
completo en memoria tampoco.

Uso:
    python subsample_patients_json.py \
        --input ~/lucai/datos/simulation_output_A/simulated_patients_all_20260720_124534.json \
        --output ~/lucai/datos/simulation_output_A/simulated_patients_100pergene.json \
        --n-per-gene 100 \
        --seed 42
"""
import argparse
import ijson
import json
import random


def subsample(input_path: str, output_path: str, n_per_gene: int, seed: int):
    rng = random.Random(seed)
    genes_processed = 0
    total_cases_out = 0

    with open(input_path, "rb") as f_in, open(output_path, "w") as f_out:
        f_out.write("{")
        first = True
        for gene, cases in ijson.kvitems(f_in, ""):
            if len(cases) <= n_per_gene:
                sampled = cases  # no hay nada que recortar
            else:
                sampled = rng.sample(cases, n_per_gene)

            if not first:
                f_out.write(",")
            first = False

            f_out.write(json.dumps(gene))
            f_out.write(":")
            f_out.write(json.dumps(sampled))

            genes_processed += 1
            total_cases_out += len(sampled)

            if genes_processed % 500 == 0:
                print(f"  procesados {genes_processed} genes, {total_cases_out} casos escritos...")

        f_out.write("}")

    print(f"Listo. {genes_processed} genes, {total_cases_out} casos totales -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submuestrear pacientes simulados por gen")
    parser.add_argument("--input", required=True, help="JSON original de pacientes simulados")
    parser.add_argument("--output", required=True, help="Ruta de salida del JSON submuestreado")
    parser.add_argument("--n-per-gene", type=int, default=100, help="Casos a conservar por gen")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para el sampleo aleatorio")
    args = parser.parse_args()

    subsample(args.input, args.output, args.n_per_gene, args.seed)
