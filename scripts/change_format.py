from pyspark.sql import SparkSession
import os

# 🔹 Lugar de ejecución
IP  = "atlas.ugr.es"
PORT= 4050
NAMENODE_PORT = 9000

# ⚙️ Configuración general
USE_HDFS = True  # Cambiar a False si queremos usar el sistema local en lugar de HDFS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  

# 🔹 Definir el namenode de HDFS correctamente
HDFS_NAMENODE = f"hdfs://atlas:{NAMENODE_PORT}"  # Ajustar esto según nuestra configuración

if USE_HDFS:
    HDFS_BASE_DIR = f"{HDFS_NAMENODE}/user/alegp97"
    DATA_INPUT_DIR = os.path.join(HDFS_BASE_DIR, "tfg_input")
    DATA_OUTPUT_DIR = os.path.join(HDFS_BASE_DIR, "tfg_output")
else:
    DATA_INPUT_DIR = os.path.join(BASE_DIR, "data/tfg_input")
    DATA_OUTPUT_DIR = os.path.join(BASE_DIR, "data/tfg_output")

# ⚙️ Rutas de archivos de configuración (mantienen la ruta local)
MODEL_CONFIG_PATH = os.path.join(BASE_DIR, "json/pipeline_config.json")  
SPARK_CONFIG_PATH = os.path.join(BASE_DIR, "json/spark_config.json")  
DB_DICT_CONFIG_PATH = os.path.join(BASE_DIR, "json/db_config.json")  


def iniciar_spark(app_name="TransformadorDF"):
    """Inicializa la sesión de Spark."""
    return SparkSession.builder.appName(app_name).getOrCreate()

def cargar_datos(spark, ruta, formato):
    """Carga un archivo en un formato específico y devuelve un DataFrame."""
    return spark.read.format(formato).option("header", "true").load(ruta)

def guardar_datos(df, ruta_salida, formato_salida):
    """Guarda un DataFrame en el formato especificado."""
    df.write.mode("overwrite").format(formato_salida).save(ruta_salida)

# Inicialización de Spark
spark = iniciar_spark()




# Definir la ruta del archivo de entrada y el formato
ruta_entrada = "/home/alegp97/TFG/data/tfg_input/wikipedia_train.json"  # Modificar según el caso
formato_entrada = "json"  # Puede ser "json", "parquet", etc.



# Definir la ruta de salida y el formato de salida
ruta_salida = DATA_INPUT_DIR
formato_salida = "csv"  # Puede ser "json", "csv", etc.

# Cargar y transformar datos
df = cargar_datos(spark, ruta_entrada, formato_entrada)

print("\nDatos cargados, guardando...")
guardar_datos(df.coalesce(1), ruta_salida, formato_salida)
print("\nDatos guardados")

# Cerrar sesión de Spark
spark.stop()

