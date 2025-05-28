from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline

from sparknlp.base import DocumentAssembler
from sparknlp.annotator import SentenceDetectorDLModel, MarianTransformer

import shutil
import os
import sparknlp

os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.9"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3.9"
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

from TFG.settings import *

MODELS_DIR = os.path.join(BASE_DIR, "models_nlp_local")
#local_model_path = os.path.join(MODELS_DIR, "marian_finetuned_kde4_english_spanish_en")
local_model_path = os.path.join(MODELS_DIR, "opus_mt_en_fr_xx_3.1")   

try:
    spark = SparkSession.builder \
        .appName("tfg-marian") \
        .master("spark://atlas:7077") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3") \
        .getOrCreate()

    spark = sparknlp.start(spark)
    print("\n✅ SparkSession iniciada.")

    data = [
        (1, "The quick brown fox jumps over the lazy dog."),
        (2, "This paper describes a novel approach to machine translation."),
        (3, "The sun rises in the east and sets in the west.")
    ]

    df = spark.createDataFrame(data, ["id", "text"])

    document_assembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "en") \
        .setInputCols(["document"]) \
        .setOutputCol("sentence")

    translator = MarianTransformer.load(local_model_path) \
        .setInputCols(["sentence"]) \
        .setOutputCol("translation")

    pipeline = Pipeline(stages=[document_assembler, sentence_detector, translator])
    model = pipeline.fit(df)
    result = model.transform(df)

    result.select("id", "text", "translation.result").show(truncate=False)

    spark.stop()
    os._exit(0)

except Exception as e:
    print(f"❌ Error en marian.py: {e}")
    raise
