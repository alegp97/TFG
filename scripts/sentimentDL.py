



from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

from sparknlp.base import DocumentAssembler
from sparknlp.annotator import UniversalSentenceEncoder, SentimentDLModel

import sparknlp
import os

# Configuración del entorno
os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.9"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3.9"
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

try:
    spark = SparkSession.builder \
        .appName("sentiment-analysis-use") \
        .master("spark://atlas:7077") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3") \
        .getOrCreate()

    spark = sparknlp.start(spark)
    print("\n✅ SparkSession iniciada.")

    # Datos de prueba
    data = [
        (1, "I absolutely loved this movie!"),
        (2, "The plot was boring and predictable."),
        (3, "I'm not sure how I feel about this."),
        (4, "An amazing experience, very emotional."),
        (5, "Terrible acting and bad direction.")
    ]
    df = spark.createDataFrame(data, ["id", "text"])

    # Pipeline con USE
    document_assembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    use_embedder = UniversalSentenceEncoder.pretrained("tfhub_use", "en") \
        .setInputCols(["document"]) \
        .setOutputCol("sentence_embeddings")

    sentiment_model = SentimentDLModel.pretrained("sentimentdl_use_imdb", "en") \
        .setInputCols(["sentence_embeddings"]) \
        .setOutputCol("sentiment")

    pipeline = Pipeline(stages=[
        document_assembler,
        use_embedder,
        sentiment_model
    ])

    model = pipeline.fit(df)
    result = model.transform(df)

    result.select("id", "text", "sentiment.result").show(truncate=False)

    spark.stop()
    os._exit(0)

except Exception as e:
    print
