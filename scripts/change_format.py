from pyspark.sql import SparkSession
import glob
import os

# 1. Crear sesión de Spark
spark = SparkSession.builder \
    .appName("change_format") \
    .master("spark://atlas:7077") \
    .getOrCreate()

# Ruta del archivo JSON de entrada
base_path = os.path.expanduser("/home/alegp97/TFG/data/datasets/extracted")

# Obtener todos los archivos tipo "wiki_*" en subcarpetas
all_files = glob.glob(f"{base_path}/**/wiki_*", recursive=True)

# Filtrar solo archivos con contenido > 0 bytes
valid_files = [f for f in all_files if os.path.getsize(f) > 0]

print(f"✅ Archivos JSON válidos encontrados: {len(valid_files)}")

if len(valid_files) == 0:
    raise ValueError("❌ No hay archivos JSON válidos para leer.")

# Iniciar Spark
spark = SparkSession.builder.appName("WikipediaToParquet").getOrCreate()

# Leer como JSON
df = spark.read.json(valid_files)

# Ruta de salida para guardar el archivo Parquet
parquet_path =  os.path.expanduser("/home/alegp97/TFG/data/tfg_input/parquet2")

df_filtered = df.filter(df.text.isNotNull() & (df.text.rlike("[a-zA-Z]{30,}")))

print(f"Guardando el DataFrame como archivo Parquet...", parquet_path)
# Guardar el DataFrame como archivo Parquet
df.coalesce(1).write.mode("overwrite").parquet(parquet_path)

# Detener la sesión de Spark
spark.stop()
