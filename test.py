from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import SentenceDetectorDLModel, Tokenizer, MarianTransformer

SPARK_NLP_JAR = "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3"

import os
import sparknlp

os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.9"  
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3.9"
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

from settings import * 

MODELS_DIR = os.path.join(BASE_DIR, "models_nlp_local")

try:
    #============================
    # Iniciar la sesión de Spark
    #============================
    spark = SparkSession.builder \
        .appName("tfgPrueba") \
        .master("spark://atlas:7077") \
        .config("spark.jars.packages", SPARK_NLP_JAR) \
        .getOrCreate()
    
    spark = sparknlp.start(spark)

    print("\nSesión de Spark iniciada con éxito")


    # translator = MarianTransformer.pretrained("opus_mt_en_es") \
    # .write() \
    # .overwrite() \
    # .save(os.path.join(MODELS_DIR, "opus_mt_en_es"))


    data = [
        (1, "The Mona Lisa is a famous painting."),
        (2, "Artificial Intelligence is transforming the world."),
        (3, "Big Data helps companies make better decisions."),
        (4, "Leonardo da Vinci was a great artist and scientist."),
        (5, "The Eiffel Tower is one of the most iconic landmarks in Paris.")
    ]

    columns = ["id", "text"]
    df = spark.createDataFrame(data, columns)

    document_assembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "en") \
        .setInputCols(["document"]) \
        .setOutputCol("sentence")

    tokenizer = Tokenizer() \
        .setInputCols(["sentence"]) \
        .setOutputCol("token")

    translator = MarianTransformer.load(os.path.join(MODELS_DIR, "opus_mnt_en_es")) \
        .setInputCols(["sentence"]) \
        .setOutputCol("translation")

    pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, translator])

    model = pipeline.fit(df)
    result = model.transform(df)

    translated_df = result.select(col("id"), col("text"), col("translation.result").alias("translated_text"))


    translated_df.show()



    if spark is not None:
        spark.stop()
        os._exit(0)


except Exception as e:
        print(f"Error main: {e}")
        raise



# import pyspark; print(pyspark.__version__)


# from pyspark.sql import SparkSession
# import os

# os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
# os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.9"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3.9"

# def init_spark_session():
#     spark = SparkSession.builder \
#         .appName("SparkNLP_App") \
#         .config("spark.driver.extraJavaOptions", "-Djava.net.preferIPv4Stack=true") \
#         .getOrCreate()
#     return spark

# spark = init_spark_session()
