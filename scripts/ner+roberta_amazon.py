#!/usr/bin/env python3
# ----------------------------------------------
# bert_nerdl_roberta.py
# Document → Sentence → Token →
# BertEmbeddings → NerDL → NerConverter
#           → Chunk2Doc → Token (2) → RoBERTa
# ----------------------------------------------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, arrays_zip, posexplode
from pyspark.ml import Pipeline

from sparknlp.base import DocumentAssembler
from sparknlp.annotator import (
    SentenceDetectorDLModel, Tokenizer,
    BertEmbeddings, NerDLModel, NerConverter,
    Chunk2Doc, RoBertaForTokenClassification)

import os, sparknlp
from TFG.settings import *               # BASE_DIR, etc.

# ---------- env ----------
os.environ["PYSPARK_PYTHON"]        = "/usr/bin/python3.9"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3.9"
os.environ["JAVA_HOME"]             = "/usr/lib/jvm/java-11-openjdk-amd64"

MODELS_DIR   = os.path.join(BASE_DIR, "models_nlp_local")
roberta_path = os.path.join(MODELS_DIR,
               "roberta_ner_roberta_large_ner_english_en_3.4.2")

# ---------- Spark ----------
spark = (SparkSession.builder
         .appName("bert-nerdl-roberta")
         .master("spark://atlas:7077")
         .config("spark.jars.packages",
                 "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3")
         .config("spark.serializer",
                 "org.apache.spark.serializer.KryoSerializer")
         .config("spark.kryoserializer.buffer.max","2000m")
         .getOrCreate())
spark = sparknlp.start(spark)
print("✅ SparkSession ready")



# ---------- sample data ----------

dataset_path = "/home/alegp97/TFG/data/tfg_input/amazon_review_polarity.csv"

#df = spark.read.option("multiline", "true").json(dataset_path)
df = spark.read.option("header", "true").option("inferSchema", "true").csv(dataset_path)

df = df.sample(0.01, seed=42)

df.drop("reviewText","overall", "unixReviewTime")

df.printSchema()

TARGET_COLUMN = "content"

# ---------- pipeline ----------
document = DocumentAssembler() \
    .setInputCol(TARGET_COLUMN).setOutputCol("document")

sentences = SentenceDetectorDLModel.pretrained(
    "sentence_detector_dl","en") \
    .setInputCols("document").setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols("sentence").setOutputCol("token")

embeddings = BertEmbeddings.pretrained(
    "bert_embeddings_base_cased","en") \
    .setInputCols("document","token") \
    .setOutputCol("word_embeddings") \
    .setCaseSensitive(True) \
    .setStorageRef("bert_base_cased")

ner = NerDLModel.pretrained("ner_dl_bert","en") \
    .setInputCols("document","token","word_embeddings") \
    .setOutputCol("ner")

converter = NerConverter() \
    .setInputCols("document","token","ner") \
    .setOutputCol("ner_chunk")

# ← convierte cada chunk en un nuevo Document
chunk2doc = Chunk2Doc() \
    .setInputCols("ner_chunk") \
    .setOutputCol("chunk_doc")

# tokeniza cada documento-chunk
chunk_tok = Tokenizer() \
    .setInputCols("chunk_doc") \
    .setOutputCol("chunk_token")

roberta = RoBertaForTokenClassification.load(roberta_path) \
    .setInputCols("chunk_doc","chunk_token") \
    .setOutputCol("roberta_lbl")

pipeline = Pipeline(stages=[document, sentences, tokenizer,
                            embeddings, ner, converter,
                            chunk2doc, chunk_tok, roberta])

model  = pipeline.fit(df)
result = model.transform(df)


result.select("content",
              "ner_chunk.result",     # entidades detectadas
              "roberta_lbl.result"    # nueva clasificación
             ).show(10, truncate=False)


spark.stop()
print("✅ finished")
