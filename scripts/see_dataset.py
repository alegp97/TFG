from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import argparse

def get_sample(df, fraction=0.1, seed=42):
    """Devuelve una muestra del DataFrame con el porcentaje especificado."""
    return df.sample(fraction=fraction, seed=seed)

def main(dataset_path):
    # Inicializar la sesión de Spark con configuración del cluster
    spark = SparkSession.builder \
        .appName("Wikipedia Dataset Analysis") \
        .config("spark.driver.memory", "64g") \
        .config("spark.executor.memory", "32g") \
        .config("spark.executor.cores", "4") \
        .config("spark.dynamicAllocation.enabled", "false") \
        .config("spark.shuffle.service.enabled", "false") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3") \
        .getOrCreate()

    print(f"\n🔍 Leyendo dataset desde HDFS: {dataset_path}...")

    # Leer el dataset en formato CSV
    df = spark.read.json(dataset_path)


    print("\n📜 *** Esquema del Dataset ***")
    df.printSchema()

    print("\n📊 *** Número total de filas ***")
    print(df.count())

    print("\n🔢 *** Distribución de valores por columna ***")
    categorical_col = df.columns[0]  # Selecciona la primera columna para agrupar
    df.groupBy(categorical_col).count().orderBy(col("count").desc()).show(5, False)

    df.select(df.columns[0]).show(5, False)

    # Contar el número de IDs diferentes
    id_column = "id"  # Reemplaza con el nombre correcto de la columna en tu dataset
    unique_ids = df.select(id_column).distinct().count()

    print(f"\n🔢 Número total de IDs únicos: {unique_ids}")

    print("\n")
    df.filter(col("id").isNull()).show()

    spark.stop()

if __name__ == "__main__":
    # Valores por defecto si no se pasan argumentos
    dataset_path = "hdfs://atlas:9000/user/alegp97/tfg_input/wikipedia_train.json"  # Ruta en HDFS

    main(dataset_path)
