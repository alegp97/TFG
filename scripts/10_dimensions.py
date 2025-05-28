# from pyspark.sql import SparkSession
# from pyspark.ml import Pipeline
# from sparknlp.base import DocumentAssembler, EmbeddingsFinisher
# from sparknlp.annotator import SentenceDetectorDLModel, AnnotatorModel
# import sparknlp
# import os

# os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.9"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3.9"
# os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

# from TFG.settings import BASE_DIR

# MODELS_DIR = os.path.join(BASE_DIR, "models_nlp_local")
# model_path = os.path.join(MODELS_DIR, "10dimensions_knowledge_en_5.5.1")

# try:
#     spark = SparkSession.builder \
#         .appName("tfg-10dimensions") \
#         .master("spark://atlas:7077") \
#         .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3") \
#         .getOrCreate()

#     spark = sparknlp.start(spark)
#     print("\n✅ SparkSession iniciada.")

#     # Datos de prueba
#     data = [
#         (1, "The Mona Lisa is a famous painting."),
#         (2, "Artificial Intelligence is transforming the world."),
#         (3, "Big Data helps companies make better decisions."),
#         (4, "Leonardo da Vinci was a great artist and scientist."),
#         (5, "The Eiffel Tower is one of the most iconic landmarks in Paris.")
#     ]
#     df = spark.createDataFrame(data, ["id", "text"])

#     # Pipeline
#     document_assembler = DocumentAssembler() \
#         .setInputCol("text") \
#         .setOutputCol("document")

#     sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "en") \
#         .setInputCols(["document"]) \
#         .setOutputCol("sentence")

#     knowledge_embedder = AnnotatorModel.load(model_path) \
#         .setInputCols(["sentence"]) \
#         .setOutputCol("embeddings")

#     embeddings_finisher = EmbeddingsFinisher() \
#         .setInputCols(["embeddings"]) \
#         .setOutputCols(["finished_embeddings"]) \
#         .setOutputAsVector(True)

#     pipeline = Pipeline(stages=[
#         document_assembler,
#         sentence_detector,
#         knowledge_embedder,
#         embeddings_finisher
#     ])

#     model = pipeline.fit(df)
#     result = model.transform(df)

#     result.select("id", "finished_embeddings").show(truncate=False)

#     spark.stop()
#     os._exit(0)

# except Exception as e:
#     print(f"❌ Error en run_10dimensions.py: {e}")
#     raise


from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Tokenizer, BertForSequenceClassification

import sparknlp
import os

os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.9"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3.9"
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"



try:
    spark = SparkSession.builder \
        .appName("bert-10dimensions") \
        .master("spark://atlas:7077") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3") \
        .getOrCreate()

    spark = sparknlp.start(spark)
    print("\n✅ SparkSession iniciada.")

    # Datos de prueba
    data = [
        (1, "I want to find meaning in life."),
        (2, "I am committed to helping others."),
        (3, "I feel unsure of who I am."),
        (4, "Success is very important to me."),
        (5, "I enjoy creating beauty.")
    ]
    df = spark.createDataFrame(data, ["id", "text"])

    # Componentes del pipeline
    documentAssembler = DocumentAssembler() \
        .setInputCol('text') \
        .setOutputCol('document')
    
    tokenizer = Tokenizer() \
        .setInputCols(['document']) \
        .setOutputCol('token')

    classifier = BertForSequenceClassification.pretrained("10dimensions_identity", "en") \
        .setInputCols(["document", "token"]) \
        .setOutputCol("class")

    pipeline = Pipeline(stages=[
        documentAssembler,
        tokenizer,
        classifier
    ])

    model = pipeline.fit(df)
    result = model.transform(df)

    result.select("id", "text", "class.result").show(truncate=False)

    spark.stop()
    os._exit(0)

except Exception as e:
    print(f"❌ Error: {e}")
    raise

