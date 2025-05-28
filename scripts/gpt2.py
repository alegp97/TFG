from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import GPT2Transformer

import openvino

import os
import sparknlp

os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.9"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3.9"
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

try:
    spark = SparkSession.builder \
        .appName("tfg-gpt2") \
        .master("spark://atlas:7077") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3") \
        .getOrCreate() 
        # .config("spark.jsl.settings.pretrained.use_openvino", "false") \
        # .config("spark.jsl.settings.prefer_openvino", "false") \



    spark = sparknlp.start(spark)
    print("\n✅ SparkSession iniciada.")

    # Datos de entrada
    data = [
        (1, "The future of artificial intelligence is"),
        (2, "In the heart of the jungle, a mysterious"),
        (3, "Medical research shows that")
    ]
    df = spark.createDataFrame(data, ["id", "text"])

    # Pipeline parcial manual (sin usar spark.ml.Pipeline)
    document_assembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")
    df = document_assembler.transform(df)

    # Cargar el modelo GPT2 de forma segura (no usar .load())
    gpt2 = GPT2Transformer.pretrained("gpt2", "en") \
        .setEngine("tensorflow") \
        .setInputCols(["document"]) \
        .setOutputCol("generated_text") \
        .setMaxOutputLength(50) \
        .setMinOutputLength(5) \
        .setDoSample(True) \
        .setTopK(50) \
        .setTemperature(0.7)

    # Aplicar transformación directa
    result = gpt2.transform(df)

    # Mostrar resultados
    result.select("id", "text", "generated_text.result").show(truncate=False)

    spark.stop()
    os._exit(0)

except Exception as e:
    print(f"❌ Error en gpt2.py: {e}")
    raise
