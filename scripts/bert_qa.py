from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import BertForQuestionAnswering
import os
import sparknlp

# Configuración del entorno
os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.9"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3.9"
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

try:
    spark = SparkSession.builder \
        .appName("bert-qa") \
        .master("spark://atlas:7077") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3") \
        .getOrCreate()

    data = [
        (1, "Diabetes mellitus is a chronic condition characterized by high levels of sugar in the blood.", "What is diabetes mellitus?"),
        (2, "Aspirin is used to reduce pain, fever, or inflammation.", "What is aspirin used for?"),
        (3, "The lungs are the primary organs of the respiratory system.", "Which organs are responsible for respiration?")
    ]
    df = spark.createDataFrame(data, ["id", "context", "question"])

    # DocumentAssembler para el contexto
    context_asm = DocumentAssembler() \
        .setInputCol("context") \
        .setOutputCol("document_context")

    # DocumentAssembler para la pregunta
    question_asm = DocumentAssembler() \
        .setInputCol("question") \
        .setOutputCol("document_question")

    # Modelo QA (sin tokenización intermedia)
    qa = BertForQuestionAnswering.pretrained("bert_base_cased_squad2", "en") \
        .setInputCols(["document_question", "document_context"]) \
        .setOutputCol("answer")

    pipeline = Pipeline(stages=[context_asm, question_asm, qa])
    model = pipeline.fit(df)
    result = model.transform(df)

    result.select("id", "question", "answer").show(truncate=False)

    spark.stop()
    os._exit(0)

except Exception as e:
    print(f"❌ Error en electra_qa_bert.py: {e}")
    raise
