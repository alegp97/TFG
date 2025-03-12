from pyspark.sql import SparkSession

# Inicializar la sesión de Spark
spark = SparkSession.builder \
    .appName("Read Wikipedia Dataset") \
    .getOrCreate()

# Ruta del dataset
DATASET_PATH = "/home/alegp97/TFG/data/input/wikipedia_train.json"

# Leer el dataset en formato JSON
df = spark.read.json(DATASET_PATH)

# Mostrar la estructura del DataFrame
df.printSchema()

# Mostrar estadísticas descriptivas de las columnas numéricas
print("\n")
print(df.count())

# Mostrar las primeras 10 filas con head()
print("Head(10):")
for row in df.head(3):
    print(row)

# Detener la sesión de Spark
spark.stop()
