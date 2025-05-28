import os

# 📌 Mostrar debugs de dentro del proyecto
DEBUG = True

# 🔹 Lugar de ejecución
IP  = "atlas.ugr.es"
PORT= 4050
NAMENODE_PORT = 9000

# ⚙️ Configuración general
USE_HDFS = True 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  

# 🔹 Definir el namenode de HDFS correctamente
HDFS_NAMENODE = f"hdfs://atlas:{NAMENODE_PORT}"  # Ajustar esto según nuestra configuración

SPARK_UI_URL = "http://atlas.ugr.es:8080"

# 📂 Directorio local para modelos NLP
MODELS_DIR = os.path.join(BASE_DIR, "models_nlp_local")  

# 📂 Rutas de datos (locales o HDFS)
if USE_HDFS:
    HDFS_BASE_DIR = f"{HDFS_NAMENODE}/user/alegp97"
    DATA_INPUT_DIR = os.path.join(HDFS_BASE_DIR, "tfg_input")
    DATA_OUTPUT_DIR = os.path.join(HDFS_BASE_DIR, "tfg_output")
else:
    DATA_INPUT_DIR = os.path.join(BASE_DIR, "data/tfg_input")
    DATA_OUTPUT_DIR = os.path.join(BASE_DIR, "data/tfg_output")

LOCAL_DATA_OUTPUT_DIR = os.path.join(BASE_DIR, "data/tfg_output")
LOCAL_DATA_INPUT_DIR = os.path.join(BASE_DIR, "data/tfg_input")

# 📂 Rutas de archivos de configuración (mantienen la ruta local)
MODEL_CONFIG_PATH = os.path.join(BASE_DIR, "json/pipeline_config.json")  
SPARK_CONFIG_PATH = os.path.join(BASE_DIR, "json/spark_config.json")  
DB_DICT_CONFIG_PATH = os.path.join(BASE_DIR, "json/db_config.json")  
os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.9" 
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3.9"
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

# ⚙️ Master configuration
SPARK_MASTER = "spark://atlas:7077"
os.environ["PYSPARK_SUBMIT_ARGS"] = f"--master {SPARK_MASTER} pyspark-shell"