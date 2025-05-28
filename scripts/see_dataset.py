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
        .config("spark.network.timeout", "800s") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .getOrCreate()

    print(f"\n🔍 Leyendo dataset desde: {dataset_path}...")


    ###READ DATASET
    #df = spark.read.option("multiline", "true").json(dataset_path)
    #df = spark.read.option("header", "true").option("inferSchema", "true").csv(dataset_path)
    df = spark.read.parquet(dataset_path)

    # print("\n📜 *** Esquema del Dataset ***")
    # df.printSchema()

    # df.limit(5).show(truncate=False)

    # spark.stop()

if __name__ == "__main__":
    # Valores por defecto si no se pasan argumentos
    dataset_path = "hdfs://atlas:9000/user/alegp97/tfg_output/" \
    "bert_nerdl_roberta_processed_example_wtex_en.csv/part-00000-bfad07d8-1b93-4f0d-bc4f-50b0a00b1caa-c000.snappy.parquet"
    main(dataset_path)
