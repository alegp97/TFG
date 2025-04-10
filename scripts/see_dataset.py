from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import argparse

def get_sample(df, fraction=0.1, seed=42):
    """Devuelve una muestra del DataFrame con el porcentaje especificado."""
    return df.sample(fraction=fraction, seed=seed)

def main(dataset_path):
    # Inicializar la sesión de Spark con configuración del cluster
    spark = SparkSession.builder \
        .appName("Dataset Analysis") \
        .config("spark.dynamicAllocation.enabled", "false") \
        .config("spark.shuffle.service.enabled", "false") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3") \
        .getOrCreate()

    print(f"\n🔍 Leyendo dataset desde HDFS: {dataset_path}...")

    # Leer el dataset en formato CSV
    df = spark.read.option("header", "true").csv(dataset_path)


    print("\n📜 *** Esquema del Dataset ***")
    df.printSchema()

    df.limit(5).show(truncate=False)

    spark.stop()

if __name__ == "__main__":
    # Valores por defecto si no se pasan argumentos
    dataset_path = "hdfs://atlas:9000/user/alegp97/tfg_output/processed_example_wtex_en.csv/part-00000-4a456932-6390-4378-9a86-63c71009fa38-c000.csv" # Ruta en HDFS

    main(dataset_path)
