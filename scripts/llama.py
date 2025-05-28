import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline

from sparknlp.base import DocumentAssembler
from sparknlp.annotator import LLAMA2Transformer

import shutil
import os
import sparknlp

# Configuración del entorno
os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.9"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3.9"
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"


from TFG.settings import *

MODELS_DIR = os.path.join(BASE_DIR, "models_nlp_local")

# model_path = os.path.join(MODELS_DIR, "llama_2_7")
model_path = f"{HDFS_NAMENODE}/user/alegp97/models_nlp/llama_2_7"

model_path = "hdfs://user/alegp97/models_nlp/llama_2_7_spark_saved"

print(f"Ruta del modelo: {model_path}")

try:
    # Iniciar la sesión de Spark
    spark = SparkSession.builder \
        .appName("tfg-llama2") \
        .master("spark://atlas:7077") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3") \
        .getOrCreate()

    spark = sparknlp.start(spark)
    print("\n✅ SparkSession iniciada.")
    


    # Datos de ejemplo
    data = [
        (1, "The future of artificial intelligence is"),
        (2, "In the heart of the jungle, a mysterious"),
        (3, "Medical research shows that")
    ]

    df = spark.createDataFrame(data, ["id", "text"]).repartition(1)

    # Ensamblador de documentos
    document_assembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("documents")

    # Cargar el modelo LLaMA 2
    # llama2 = LLAMA2Transformer.load(model_path) \
    #     .setInputCols(["documents"]) \
    #     .setOutputCol("generation") \
    #     .setMaxOutputLength(50) \
    #     .setMinOutputLength(5) \
    #     .setDoSample(True) \
    #     .setTopK(50) \
    #     .setTemperature(0.7)
    # print("✅ Modelo cargado.")
    #llama2.write().overwrite().save("hdfs://user/alegp97/models_nlp/llama_2_7_spark_saved")


    print("📦 Descargando modelo LLaMA2 preentrenado...")
    start = time.time()
    llama2 = LLAMA2Transformer.pretrained("llama_2_7b_chat_hf_int4", "en") \
        .setInputCols(["documents"]) \
        .setOutputCol("generation") \
        .setMaxOutputLength(50) \
        .setMinOutputLength(5) \
        .setDoSample(True) \
        .setTopK(50) \
        .setTemperature(0.7)
    print(f"✅ Modelo cargado en {time.time() - start:.2f}s")

    # Crear y ejecutar el pipeline
    print("🚀 Creando el pipeline...")
    pipeline = Pipeline(stages=[document_assembler, llama2])


    print("🚀 Ejecutando el pipeline...")
    model = pipeline.fit(df)
    result = model.transform(df)
    print("Resultado transformado...")

    # Mostrar resultados
    result.select("id", "text", "generation.result").show(truncate=False)

    spark.stop()
    os._exit(0)

except Exception as e:
    print(f"❌ Error en llama2.py: {e}")
    raise
