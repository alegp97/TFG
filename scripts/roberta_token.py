#!/usr/bin/env python3
# ----------------------------------------------
# bert_nerdl.py
# Pipeline: Document → Sentence → Token →
#           BertEmbeddings → NerDL → NerConverter
# ----------------------------------------------

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

from sparknlp.base import DocumentAssembler
from sparknlp.annotator import (
    SentenceDetectorDLModel, Tokenizer,
    BertEmbeddings, NerDLModel, NerConverter)

import os, sparknlp
from TFG.settings import *

# ---------- env ----------
os.environ["PYSPARK_PYTHON"]        = "/usr/bin/python3.9"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3.9"
os.environ["JAVA_HOME"]             = "/usr/lib/jvm/java-11-openjdk-amd64"

# ---------- Spark ----------
spark = (SparkSession.builder
         .appName("bert-nerdl")
         .master("spark://atlas:7077")
         .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3")
         .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
         .config("spark.kryoserializer.buffer.max", "2000m")
         .getOrCreate())
spark = sparknlp.start(spark)
print("✅ SparkSession ready")

# ---------- sample data ----------
data = [(1,"Barack Obama was born in Hawaii."),
        (2,"Apple Inc. is looking at buying U.K. startup for $1 billion."),
        (3,"The Eiffel Tower is located in Paris.")]
df = spark.createDataFrame(data, ["id","text"])

# ---------- pipeline ----------
document = DocumentAssembler().setInputCol("text").setOutputCol("document")

sentences = (SentenceDetectorDLModel
             .pretrained("sentence_detector_dl","en")
             .setInputCols("document")
             .setOutputCol("sentence"))

tokenizer = Tokenizer().setInputCols("sentence").setOutputCol("token")

embeddings = (BertEmbeddings
              .pretrained("bert_embeddings_base_cased","en")
              .setInputCols("document","token")
              .setOutputCol("word_embeddings")
              .setCaseSensitive(False)
              .setStorageRef("bert_base_cased"))      # must match NerDL

ner = (NerDLModel
       .pretrained("ner_dl_bert","en")
       .setInputCols("document","token","word_embeddings")
       .setOutputCol("ner"))

converter = (NerConverter()
             .setInputCols("document","token","ner")
             .setOutputCol("ner_chunk"))

pipeline = Pipeline(stages=[document, sentences, tokenizer,
                            embeddings, ner, converter])

model  = pipeline.fit(df)
result = model.transform(df)

result.select("id","text","ner_chunk.result").show(truncate=False)

spark.stop()
print("✅ finished")
