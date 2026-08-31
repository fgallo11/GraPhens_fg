# ==========================================================================
# CELDA MARKDOWN
# ==========================================================================
"""
## Paso 2 (código) — Convertir A y B al formato que espera el leaf-filter

Ya sabemos (por lo que leímos en database.py) que GenePhenotypeDatabase
espera dos archivos con nombre fijo:

- genes_to_phenotype.txt
- phenotype_to_genes.txt

Ambos con las mismas columnas: ncbi_gene_id, gene_symbol, hpo_id, hpo_name,
frequency, disease_id.

Nuestros archivos A y B tienen columnas distintas (fuentes_ot, confidence_by_source,
fuente_fenotipo, clasificacion_dup) que no le importan al leaf-filter, y les
falta 'frequency', que dejamos como string vacío (así lo hace el propio
código de Gian en update_goldstandard_dataset.py cuando no hay ese dato).

Este script:
1. Carga A y B
2. Selecciona/renombra columnas al esquema esperado
3. Agrega columna frequency vacía
4. Guarda 4 archivos en 2 carpetas separadas (una por versión), para no
   pisar A con B
"""

# ==========================================================================
# CELDA DE CÓDIGO
# ==========================================================================
import pandas as pd
from pathlib import Path

# Carpetas de salida — una por versión, así el leaf-filter las procesa
# por separado sin mezclarlas (tal como pediste: A y B en paralelo, sin cruzar)
BASE = Path.home() / "lucai" / "datos"
OUT_A = BASE / "leaf_filter_input_A"
OUT_B = BASE / "leaf_filter_input_B"

OUT_A.mkdir(parents=True, exist_ok=True)
OUT_B.mkdir(parents=True, exist_ok=True)

archivos = {
    "A": (BASE / "ot_unified_v2_sin_clingen_puro_sin_bug.txt", OUT_A),
    "B": (BASE / "ot_unified_v2_sin_clingen_total_sin_bug.txt", OUT_B),
}

for version, (path_in, out_dir) in archivos.items():
    print(f"--- Procesando versión {version} ---")
    df = pd.read_csv(path_in, sep="\t")
    print("shape original:", df.shape)

    # Nos quedamos solo con las columnas que el pipeline de GraPhens espera,
    # en el orden que usa el propio repo (visto en update_goldstandard_dataset.py)
    df_convertido = df[["ncbi_gene_id", "gene_symbol", "hpo_id", "hpo_name", "disease_id"]].copy()
    df_convertido["frequency"] = ""  # no tenemos este dato en Open Targets, se deja vacío

    # Reordenamos columnas al orden exacto que arma el código de Gian
    df_convertido = df_convertido[
        ["ncbi_gene_id", "gene_symbol", "hpo_id", "hpo_name", "frequency", "disease_id"]
    ]

    print("shape convertido:", df_convertido.shape)
    print(df_convertido.head(3))

    # Guardamos el MISMO contenido en los dos nombres de archivo que pide
    # GenePhenotypeDatabase. Por ahora son idénticos -- el leaf-filter es
    # el que después va a editar ambos en paralelo, sacando de cada uno
    # las filas no-hoja, y ahí sí van a divergir un poco entre sí.
    genes_to_phenotype_path = out_dir / "genes_to_phenotype.txt"
    phenotype_to_genes_path = out_dir / "phenotype_to_genes.txt"

    df_convertido.to_csv(genes_to_phenotype_path, sep="\t", index=False)
    df_convertido.to_csv(phenotype_to_genes_path, sep="\t", index=False)

    print(f"Guardado en: {out_dir}")
    print()

print("Listo. Carpetas generadas:")
print(" -", OUT_A)
print(" -", OUT_B)
