import logging
import os
import json
from pathlib import Path
import uuid
import shutil
import traceback
import gc
import time
import datetime

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import SQLTransformer
from pyspark.sql import functions as F

from sparkmeasure import StageMetrics

import sparknlp
from sparknlp.pretrained import ResourceDownloader

# Spark NLP - base & preprocesamiento
from sparknlp.base import DocumentAssembler, Finisher, TokenAssembler
from sparknlp.annotator import (
    GPT2Transformer,
    LLAMA2Transformer,
    BertEmbeddings,
    SentimentDLModel, 
    #QuestionAnswering, 
    UniversalSentenceEncoder,
    SentenceDetectorDLModel,
    WordEmbeddingsModel,
    RoBertaForTokenClassification,
    Tokenizer,
    Normalizer,
    Chunk2Doc,
    StopWordsCleaner,
    Stemmer,
    LemmatizerModel,
    LanguageDetectorDL,
    ContextSpellCheckerModel,
    DocumentNormalizer,
    MarianTransformer,
    NerDLModel,
    NerConverter
)

# Flask
from flask import current_app
from flask_socketio import SocketIO

# Configuración personalizada
from settings import HDFS_BASE_DIR, HDFS_NAMENODE, MODELS_DIR, BASE_DIR

class PipelineNLP:

    
    def __init__(self, source, input_data, spark_config, pipeline_config, debug=False):
        """
        Inicializa la sesión de Spark y carga los datos según la fuente (SQL o local).
        
        :param source: 'sql' para base de datos o 'local' para archivos locales.
        :param input_data: La consulta SQL o la ruta del archivo.
        :param spark_config: Configuración de Spark en JSON.
        :param pipeline_config: Configuración del modelo en JSON.
        :param debug: Booleano para imprimir logs.
        """
        self.source = source
        self.input_data = input_data
        self.spark_config = spark_config
        self.pipeline_config = pipeline_config
        self.debug = debug

        #self.model_manager = ModelManager(base_path="models_nlp_local")

        self.execution_metadata = {
            "start_time": None,
            "end_time": None,
            "duration": None,
            "input_type": source,
            "pipeline_config": pipeline_config,
            "spark_config": spark_config,
            "stages_executed": [],
            "models_loaded": [],
            "input_schema": None,
            "input_rows": None,
            "partitions_before": None,
            "partitions_after": None,
            "executors_memory": {},
            "read_time_sec": None,
            "write_time_sec": None,
            "stage_timings": [],
            "error": None
        }
    
    
    def init_spark_session(self):
        """Inicializa la sesión de Spark NLP con una configuración completamente dinámica desde un JSON."""
        spark_config = self.spark_config["spark"]
        socketio = current_app.extensions["socketio"]

        # Cerrar sesión anterior si existe
        try:
            existing_spark = SparkSession.getActiveSession()
            if existing_spark is not None:
                print("🔄 Cerrando sesión de Spark existente...")
                existing_spark.stop()
            if SparkContext._active_spark_context is not None:
                print("🚨 Forzando eliminación de SparkContext...")
                SparkContext._active_spark_context.stop()
                SparkContext._active_spark_context = None
            
            gc.collect()
        except Exception as e:
            print(f"⚠️ No se pudo detener la sesión previa: {e}")

        try:

            app_name = spark_config.get("app_name", "SparkNLP_App") or "SparkNLP_App"
            spark_builder = SparkSession.builder.appName(app_name)
            spark_builder = spark_builder.config(
                    "spark.jars.packages",
                    "ch.cern.sparkmeasure:spark-measure_2.13:0.25"
                )

            # Aplicar configuraciones generales (excepto configurations)
            if "master" in spark_config:
                spark_builder = spark_builder.master(spark_config["master"])

            for key, value in spark_config.items():
                if key not in ["configurations", "app_name", "master"]:
                    spark_builder = spark_builder.config(f"spark.{key}", value)

            # Extraer configuraciones avanzadas de Spark
            extra_configs = spark_config.get("configurations", {})

            # Aplicar configuraciones al builder
            for key, value in extra_configs.items():
                spark_builder = spark_builder.config(key, value)

            # Añadir Limpiar spark.local.dir
            spark_tmp_path = "/tmp/spark_temp"
            if os.path.exists(spark_tmp_path):
                shutil.rmtree(spark_tmp_path)
            spark_builder = spark_builder.config("spark.local.dir", spark_tmp_path)


            # Crear la sesión de Spark con la configuración completa
            self.spark = spark_builder.getOrCreate()

            # Iniciar Spark NLP sobre la sesión de Spark creada
            self.spark = sparknlp.start(self.spark)

             # Registrar configuración en log si está en modo debug
            if self.debug:
                # Configurar log persistente en disco
                log_file_path = os.path.join(BASE_DIR, "logs/spark_nlp.log")
                logging.basicConfig(
                    filename=log_file_path,
                    level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s",
                )

                socketio.emit("pipeline_output", {"message": "⚙️ Configuración final de Spark NLP:"})
                logs = []
                for key, value in self.spark.sparkContext.getConf().getAll():
                    log_message = f"🔹 {key} = {value}"
                    socketio.emit("pipeline_output", {"message": log_message})
                    logging.info(log_message)
                    logs.append(log_message)
                return "\n".join(logs)
            else:
                return None
            
        except Exception as e:
            # Obtener el traceback completo para más detalles
            error_trace = traceback.format_exc()

            # Enviar detalles al WebSocket para depuración en tiempo real
            error_message = f"❌ Error durante la inicialización de Spark NLP: {str(e)}"
            socketio.emit("pipeline_output", {"message": error_message})
            socketio.emit("pipeline_output", {"message": f"📜 Detalles técnicos:\n{error_trace}"})

            # Imprimir en consola para logs adicionales
            print(error_message)
            print(error_trace)

            # Lanzar la excepción con más contexto
            raise RuntimeError(f"Spark NLP Initialization Failed:\n{error_trace}")

    def get_spark_session(self):
        """Devuelve la sesión de Spark actual."""
        return self.spark

    def stop_pipeline(self):
        """Detiene la sesión de Spark."""
        if self.spark:
            self.spark.catalog.clearCache()
            self.spark.stop()
            self.spark = None
            print("🧹 SparkSession limpiada y cerrada.")
        gc.collect()
    
    def get_execution_metadata(self):
        """Devuelve los metadatos de ejecución, asegurando que todo sea JSON serializable."""
        from copy import deepcopy

        # Crear una copia para no modificar el original
        metadata = deepcopy(self.execution_metadata)

        # Formatear tiempos a ISO si existen
        for key in ["start_time", "end_time"]:
            if isinstance(metadata.get(key), datetime.datetime):
                metadata[key] = metadata[key].isoformat()

        # Formatear todos los floats a 2 decimales (si quieres un output limpio)
        for key in ["duration", "read_time_sec", "write_time_sec"]:
            if isinstance(metadata.get(key), float):
                metadata[key] = round(metadata[key], 2)

        # Formatear los tiempos por etapa
        if "stage_timings" in metadata:
            for stage in metadata["stage_timings"]:
                if isinstance(stage.get("duration_sec"), float):
                    stage["duration_sec"] = round(stage["duration_sec"], 2)

        return metadata


    def save_execution_metadata(
            self,
            model,         # PipelineModel entrenado
            df_input,      # DataFrame de entrada
            base_dir: str = None
        ) -> str:
        """
        Guarda:
        • execution_metadata.json
        • métricas de Spark stages
        • tiempos por annotator
        Todo bajo  <base_dir>/run_<timestamp>_<id>/ …
        Retorna la ruta HDFS del JSON.
        """

        # ───────────────── helpers ───────────────────────────────────
        def _time_annotators(model, df_in, spark):
            """
            Ejecuta el modelo annotator a annotator midiendo
            el tiempo de pared de cada transform().
            """
            out_df, recs = df_in, []
            for order, annot in enumerate(model.stages, start=1):
                t0 = time.perf_counter()
                out_df = annot.transform(out_df)
                recs.append((order,
                            annot.uid,
                            annot.__class__.__name__,
                            round((time.perf_counter() - t0) * 1_000, 2)))
            schema = "order INT, uid STRING, annotator STRING, elapsed_ms DOUBLE"
            return out_df, spark.createDataFrame(recs, schema)

        if base_dir is None:
            base_dir = os.path.join(HDFS_BASE_DIR, "logs_metadata")

        # ----- identificador de run -----
        ts = datetime.datetime.now().strftime("%H_%M_%S_%d_%m_%Y")
        rid = str(uuid.uuid4())[:8]
        run_tag = f"{ts}_{rid}"                       # para columnas y prints
        run_dir = f"{base_dir}/run_{run_tag}"         # dir hdfs final

        try:
            # ========== 1· Métricas de stage ========== #
            self.stg.end()   # cierra sparkmeasure

            stage_df = (
                self.stg.create_stagemetrics_DF()     
                    .withColumnRenamed("stageId", "stage_id")
                    .withColumnRenamed("jobId",   "job_id")
                    .withColumnRenamed("name",    "stage_name")
                    .withColumn("stage_sec",    F.col("stageDuration") / 1000)
                    .withColumn("executor_sec", F.col("executorRunTime") / 1000)
                    .withColumn("input_MB",     F.round(F.col("bytesRead")    / 1_048_576, 2))
                    .withColumn("output_MB",    F.round(F.col("bytesWritten") / 1_048_576, 2))
                    .withColumn("shuffle_in_MB", F.round(F.col("shuffleTotalBytesRead") / 1_048_576, 2))
                    .withColumn("shuffle_out_MB", F.round(F.col("shuffleBytesWritten")   / 1_048_576, 2))
                    .withColumn("run_tag", F.lit(run_tag))
            )

            (stage_df
                .coalesce(1)     
                .write
                .mode("error")
                .option("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
                .json(f"{run_dir}/stages"))

            # ========== 2· Métricas por annotator ===== #
            _, annot_df = _time_annotators(model, df_input, self.spark)

            (annot_df.withColumn("run_tag", F.lit(run_tag))
                    .coalesce(1)
                    .write
                    .mode("error")
                    .option("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
                    .json(f"{run_dir}/annotators"))

            # ========== 3· execution_metadata.json ==== #
            local_tmp = f"/tmp/execution_metadata_{rid}.json"
            with open(local_tmp, "w", encoding="utf-8") as f:
                json.dump(self.get_execution_metadata(), f,
                        indent=4, ensure_ascii=False)

            fs = self.spark._jvm.org.apache.hadoop.fs.FileSystem.get(
                self.spark._jvm.java.net.URI(HDFS_NAMENODE),
                self.spark._jsc.hadoopConfiguration()
            )

            fs.copyFromLocalFile(False, True,
                self.spark._jvm.org.apache.hadoop.fs.Path(local_tmp),
                self.spark._jvm.org.apache.hadoop.fs.Path(f"{run_dir}/execution_metadata.json"))

            print(f"Metadatos de ejecución guardados en: {run_dir}/")
            return f"{run_dir}/execution_metadata.json"

        except Exception as e:
            print(f"❌ Error al guardar metadatos/métricas: {e}")
            raise



    def load_from_sql(self):
        """Carga los datos desde una base de datos SQL utilizando la consulta proporcionada."""
        if self.debug:
            print("📡 Cargando datos desde SQL...")

        db_config = self.input_data["database"]
        sql_query = self.input_data["query"]["sql"]

        db_type = db_config["type"]
        host = db_config["host"]
        port = db_config["port"]
        dbname = db_config["dbname"]
        user = db_config["user"]
        password = db_config["password"]

        if db_type == "postgresql":
            url = f"jdbc:postgresql://{host}:{port}/{dbname}"
            driver = "org.postgresql.Driver"
        elif db_type == "mysql":
            url = f"jdbc:mysql://{host}:{port}/{dbname}"
            driver = "com.mysql.cj.jdbc.Driver"
        elif db_type == "sqlserver":
            url = f"jdbc:sqlserver://{host}:{port};databaseName={dbname}"
            driver = "com.microsoft.sqlserver.jdbc.SQLServerDriver"
        else:
            raise ValueError("Base de datos no soportada")

        if self.debug:
            print(f"Consulta SQL: {sql_query}")

        start_time = time.time()

        df = self.spark.read.format("jdbc").options(
            url=url,
            query=sql_query,
            user=user,
            password=password,
            driver=driver
        ).load()

        end_time = time.time()
        self.execution_metadata["read_time_sec"] = round(end_time - start_time, 3)

        if self.debug:
            print("✅ Datos cargados correctamente desde SQL")
            df.show(5)

        return df


    def load_from_local(self, file_path: str, format: str = None, delimiter: str = ",") -> DataFrame:
        """
        📥 Carga un archivo desde el sistema de archivos local en un DataFrame de Spark.

        Parámetros:
            - file_path (str): Ruta del archivo de entrada.
            - format (str, opcional): Formato del archivo. Si es None, se infiere por la extensión.
            - delimiter (str, opcional): Separador utilizado en archivos CSV y TXT. Predeterminado: ",".

        Retorna:
            - DataFrame: DataFrame de Spark con los datos cargados.
        """

        try:
            socketio = current_app.extensions["socketio"]
            socketio.emit("pipeline_output", {"message": f"📥 Cargando archivo desde {file_path}..."})

            if format is None:
                format = os.path.splitext(file_path)[-1].lower().replace(".", "")

            start_time = time.time()

            if format in ["csv", "txt"]:
                df = self.spark.read.option("header", "true").option("inferSchema", "true").option("delimiter", delimiter).csv(file_path)
            elif format == "json":
                df = self.spark.read.json(file_path)
            elif format == "parquet":
                df = self.spark.read.parquet(file_path)
            elif format == "avro":
                df = self.spark.read.format("avro").load(file_path)
            elif format == "orc":
                df = self.spark.read.orc(file_path)
            elif format in ["xls", "xlsx"]:
                df = self.spark.read.format("com.crealytics.spark.excel") \
                    .option("header", "true") \
                    .option("inferSchema", "true") \
                    .load(file_path)
            else:
                raise ValueError(f"❌ Formato '{format}' no soportado. Usa 'csv', 'txt', 'json', 'parquet', 'avro', 'orc', 'xls', 'xlsx'.")

            end_time = time.time()
            self.execution_metadata["read_time_sec"] = round(end_time - start_time, 3)

            return df

        except Exception as e:
            socketio.emit("pipeline_output", {"message": f"❌ Error al cargar el archivo: {str(e)}"})
            raise e

    def save_to_local(self, df: DataFrame, output_path: str, format: str = None, mode: str = "overwrite", delimiter: str = ","):
        """
        Guarda un DataFrame de Spark en un archivo local en distintos formatos.

        Parámetros:
            - df (DataFrame): DataFrame de Spark a guardar.
            - output_path (str): Ruta donde se guardará el archivo.
            - format (str, opcional): Formato del archivo de salida. Si es None, se infiere por la extensión.
            - mode (str, opcional): Modo de guardado ('overwrite', 'append', 'error', 'ignore'). Predeterminado: 'overwrite'.
            - delimiter (str, opcional): Separador utilizado en archivos CSV y TXT. Predeterminado: ",".

        Retorna:
            - None
        """
        if self.spark.sparkContext._jsc.sc().isStopped():
            print("❌ SparkContext está detenido. No se puede guardar el DataFrame.")
            return
        try:

            socketio = current_app.extensions["socketio"]

            if format is None:
                format = os.path.splitext(output_path)[-1].lower().replace(".", "")

            start_time = time.time()

            if format in ["csv", "txt"]:
                df.write.option("header", "true").option("delimiter", delimiter).mode(mode).csv(output_path)
            elif format == "json":
                df.write.mode(mode).json(output_path)
            elif format == "parquet":
                df.write.mode(mode).parquet(output_path)
            elif format == "avro":
                df.write.format("avro").mode(mode).save(output_path)
            elif format == "orc":
                df.write.mode(mode).orc(output_path)
            elif format in ["xls", "xlsx"]:
                df.write.format("com.crealytics.spark.excel") \
                    .option("header", "true") \
                    .mode(mode) \
                    .save(output_path)
            elif format == "delta":
                extensions = self.spark.conf.get("spark.sql.extensions", "")
                catalog = self.spark.conf.get("spark.sql.catalog.spark_catalog", "")
                if "io.delta.sql.DeltaSparkSessionExtension" not in extensions or \
                "org.apache.spark.sql.delta.catalog.DeltaCatalog" not in catalog:
                    raise EnvironmentError("Delta Lake no está habilitado en la sesión de Spark.")

                df.write.format("delta").mode(mode).save(output_path)
            else:
                raise ValueError(f"❌ Formato '{format}' no soportado. ")

            end_time = time.time()
            self.execution_metadata["write_time_sec"] = round(end_time - start_time, 3)

        except Exception as e:
            socketio.emit("pipeline_output", {"message": f"❌ Error al guardar el archivo: {str(e)}"})
            raise e
        

    
    def execute_Pipeline(self, socketio, df, document_columns, stages):
        """
        1. Arranca StageMetrics                (self.stg.begin)
        2. Construye y ejecuta el Pipeline NLP
        3. Cronometra fit + transform
        4. Limpia columnas temporales
        5. Guarda *todo* (metadatos + métricas) con save_execution_metadata()
        6. Devuelve el DataFrame transformado y el StageMetrics por si se quiere
        inspeccionar fuera.
        """
        try:
            # ───────────────────────────────────────────────────────────
            # 1) Activar StageMetrics
            # ───────────────────────────────────────────────────────────
            self.stg = StageMetrics(self.spark)
            self.stg.begin()

            # ───────────────────────────────────────────────────────────
            # 2) Validación de etapas
            # ───────────────────────────────────────────────────────────
            if not stages:
                socketio.emit("pipeline_output",
                            {"message": "❌ Error: No hay etapas válidas en el pipeline."})
                raise ValueError("No hay etapas válidas en el pipeline.")

            # ───────────────────────────────────────────────────────────
            # 3) Crear y ejecutar el Pipeline NLP
            # ───────────────────────────────────────────────────────────
            socketio.emit("pipeline_output", {"message": "🚀 Ejecutando pipeline en Spark..."})
            nlp_pipeline = Pipeline(stages=stages)

            # —— FIT
            socketio.emit("pipeline_output", {"message": "🚀 Ajustando (fit)..."})
            fit_start = time.time()
            model = nlp_pipeline.fit(df)
            df.take(1)                             # fuerza ejecución
            fit_end = time.time()

            # —— TRANSFORM
            socketio.emit("pipeline_output", {"message": "🚀 Transformando (transform)..."} )
            transform_start = time.time()
            transformed_df = model.transform(df)
            transformed_df.take(1)                 # fuerza ejecución
            transform_end = time.time()

            # ───────────────────────────────────────────────────────────
            # 4) Guardar timings en execution_metadata
            # ───────────────────────────────────────────────────────────
            self.execution_metadata["stage_timings"].extend([
                {"stage": "fit",       "duration_sec": round(fit_end - fit_start, 3)},
                {"stage": "transform", "duration_sec": round(transform_end - transform_start, 3)}
            ])
            self.execution_metadata["partitions_after"] = transformed_df.rdd.getNumPartitions()

            # ───────────────────────────────────────────────────────────
            # 5) Limpieza de columnas temporales
            # ───────────────────────────────────────────────────────────
            for col in document_columns:
                if col in transformed_df.columns:
                    transformed_df = transformed_df.drop(col)

            # ───────────────────────────────────────────────────────────
            # 6) Finalizar metadatos de ejecución
            # ───────────────────────────────────────────────────────────
            self.execution_metadata["end_time"] = datetime.datetime.now()
            self.execution_metadata["duration"] = (
                self.execution_metadata["end_time"] - self.execution_metadata["start_time"]
            ).total_seconds()
            self.execution_metadata["output_schema"] = transformed_df.schema.json()

            # ───────────────────────────────────────────────────────────
            # 7) Guardar todo (metadatos + métricas + annotators)
            # ───────────────────────────────────────────────────────────
            self.save_execution_metadata(
                model    = model,
                df_input = df,                      # se usa para cronometrar annotators
                base_dir = None                     # --> HDFS_BASE_DIR/logs_metadata
            )

            # ───────────────────────────────────────────────────────────
            return transformed_df

        except Exception as e:
            socketio.emit("pipeline_output",{"message": "❌ Error interno durante la ejecución del pipeline"})
            try:
                if getattr(self, "stg", None):
                    self.stg.end()
            except Exception:
                pass
            raise e


    def run_stages(self, df: DataFrame) -> DataFrame:
        """⚙️ Ejecuta el pipeline NLP y emite logs en tiempo real al cliente mediante WebSocket.""" 

        def _local_or_pretrained(loader, model_name, lang="en", *args, **kwargs):
            """
            Intenta cargar el modelo desde disco; si no existe, cae a .pretrained().
            """
            local_path = Path(MODELS_DIR) / model_name
            if local_path.exists():
                socketio.emit("pipeline_output",
                            {"message": f"📦 Cargando modelo local: {local_path}"})
                return loader.load(str(local_path), *args, **kwargs)
            else:
                socketio.emit("pipeline_output",
                            {"message": f"💡 Modelo local no encontrado, usando "
                                        f"pretrained('{model_name}')"})
                return loader.pretrained(model_name, lang, *args, **kwargs)
    
        try:
            # Acceder a la instancia de Flask-SocketIO para emitir logs en tiempo real
            socketio = current_app.extensions["socketio"]

            # ⏳ Iniciar la recolección de métricas
            self.execution_metadata["start_time"] = datetime.datetime.now()
            self.execution_metadata["input_rows"] = df.count()
            self.execution_metadata["input_schema"] = df.schema.json()
            self.execution_metadata["partitions_before"] = df.rdd.getNumPartitions()

            stages = []
            cached_models = {}  # Modelos cargados en memoria
            document_columns = []  # Columnas temporales generadas por el pipeline
            last_embedding_ref = None # Variable auxiliar para guardar el último storageRef usado por embeddings
            #model_manager = ModelManager(base_path="models_nlp_local")

            socketio.emit("pipeline_output", {"message": "models_nlp_local"})

            # 🔄 Construcción dinámica del pipeline con cada etapa
            # Este bucle es el que recorre todas las etapas definidas en el JSON de configuración del pipeline (`self.pipeline_config["stages"]`).
            # Cada etapa debe contener al menos una clave "name" (nombre del transformador/annotator) y opcionalmente "params" con los parámetros requeridos.
            # Según el tipo de etapa (por ejemplo, tokenizer, normalizer, ner, etc.), se inicializa el componente correspondiente de Spark NLP,
            # se configuran sus parámetros, se añade al pipeline, y se notifican los eventos al frontend mediante `socketio.emit`. 
            for stage in self.pipeline_config["stages"]:
                name = stage.get("name")
                params = stage.get("params", {})

                if not name:
                
                 
                    socketio.emit("pipeline_output", {"message": "❌ Error: Falta 'name' en una etapa."})
                    raise ValueError("Falta 'name' en una etapa.")

                socketio.emit("pipeline_output", {"message": f"📢 Agregando etapa: {name} con parámetros: {params}"})
                stage_start = time.time()


                """
                ////////////////////////////////////////////////////////////////
                        ETAPAS DE TRANSFORMACIÓN Y PREPROCESAMIENTO
                ////////////////////////////////////////////////////////////////
                """

                # 📄 Document Assembler
                if name.startswith("document_assembler"):
                    assembler = DocumentAssembler() \
                        .setInputCol(params["inputCol"]) \
                        .setOutputCol(params["outputCol"])
                    stages.append(assembler)
                    document_columns.append(params["outputCol"])

                # 📑 Sentence detector
                elif name.startswith("sentence_detector"):
                    sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx") \
                        .setInputCols(["document"]) \
                        .setOutputCol(params["outputCol"])
                    stages.append(sentence_detector)
                    document_columns.append(params["outputCol"])

                # 🎟️ Tokenizer
                elif name.startswith("tokenizer"):
                    tokenizer = Tokenizer() \
                        .setInputCols(params["inputCols"]) \
                        .setOutputCol(params["outputCol"])
                    stages.append(tokenizer)
                    document_columns.append(params["outputCol"])
                
                # 🧠 Word Embeddings
                elif name.startswith("word_embeddings"):
                    embedding_model_name = params.get("pretrained_model", "glove_100d")
                    word_embeddings = WordEmbeddingsModel.pretrained(embedding_model_name, "en") \
                        .setInputCols([params["inputCols"]]) \
                        .setOutputCol(params["outputCol"]) \
                        .setEnableInMemoryStorage(True)
                    stages.append(word_embeddings)
                    document_columns.append(params["outputCol"])

                # 🧹 Normalizer
                elif name.startswith("normalizer"):
                    normalizer = Normalizer() \
                        .setInputCols(params["inputCol"]) \
                        .setOutputCol(params["outputCol"]) \
                        .setLowercase(params.get("lowercase", True)) \
                        .setCleanupPatterns(params.get("cleanupPatterns", []))
                    stages.append(normalizer)

                # 🛑 StopWordsCleaner
                elif name.startswith("stopwords_cleaner"):
                    stopwords_cleaner = StopWordsCleaner() \
                        .setInputCols([params["inputCol"]]) \
                        .setOutputCol(params["outputCol"]) \
                        .setCaseSensitive(params.get("caseSensitive", False))
                    stages.append(stopwords_cleaner)

                # 🌱 Stemmer
                elif name.startswith("stemmer"):
                    stemmer = Stemmer() \
                        .setInputCols([params["inputCol"]]) \
                        .setOutputCol(params["outputCol"])
                    stages.append(stemmer)

                # 🍃 Lemmatizer
                elif name.startswith("lemmatizer"):
                    lemmatizer = LemmatizerModel.pretrained(params.get("model_name", "lemma_antbnc")) \
                        .setInputCols([params["inputCol"]]) \
                        .setOutputCol(params["outputCol"])
                    stages.append(lemmatizer)

                # 🗣️ Detección de idioma
                elif name.startswith("language_detector"):
                    language_detector = LanguageDetectorDL.pretrained("ld_wiki_229", "xx") \
                        .setInputCols([params["inputCol"]]) \
                        .setOutputCol(params["outputCol"])
                    stages.append(language_detector)
                
                # Chunk2Doc
                elif name.startswith("chunk2doc"):
                    chunk2doc = Chunk2Doc() \
                        .setInputCols(params["inputCols"]) \
                        .setOutputCol(params["outputCol"])
                    stages.append(chunk2doc)

                # 🔠 Spell Checker 
                elif name.startswith("spell_checker"):
                    spell_checker = ContextSpellCheckerModel.pretrained("spellcheck_dl", "en") \
                        .setInputCols([params["inputCol"]]) \
                        .setOutputCol(params["outputCol"])
                    stages.append(spell_checker)

                # 🧼 Document Normalizer (con expresiones regulares)
                elif name.startswith("document_normalizer"):
                    document_normalizer = DocumentNormalizer() \
                        .setInputCols([params["inputCol"]]) \
                        .setOutputCol(params["outputCol"]) \
                        .setAction(params.get("action", "clean")) \
                        .setPatterns(params.get("patterns", []))
                    stages.append(document_normalizer)
                
                    """
                    ////////////////////////////////////////////////////////////////
                            ETAPAS DE ESTIMACIÓN - MODELOS PREENTRENADOS
                    ////////////////////////////////////////////////////////////////
                    """
                # Bert Embeddings
                elif name.startswith("bert_embeddings"):
                    input_col  = params["inputCols"]
                    output_col = params["outputCol"]
                    model_name = params["model_name"]

                    storage_ref = params.get("storageRef", model_name)
                    last_embedding_ref = storage_ref 

                    embeddings = _local_or_pretrained(BertEmbeddings, model_name) \
                        .setInputCols(input_col) \
                        .setOutputCol(output_col) \
                        .setCaseSensitive(params.get("caseSensitive", False)) \
                        .setStorageRef(storage_ref)


                    stages.append(embeddings)
                    socketio.emit("pipeline_output",
                                {"message": f"🌍 Embeddings '{model_name}' listo → '{output_col}' ✅"})

                # 🧠 Universal Sentence Encoder
                elif name == "universal_sentence_encoder":
                    input_cols = params["inputCols"]
                    if isinstance(input_cols, str):
                        input_cols = [input_cols]

                    model_name = params["model_name"]
                    lang = params.get("lang", "en")
                    output_col = params["outputCol"]

                    use = _local_or_pretrained(UniversalSentenceEncoder, model_name, lang) \
                        .setInputCols(input_cols) \
                        .setOutputCol(output_col)

                    stages.append(use)
                    socketio.emit("pipeline_output",
                                {"message": f"🧠 USE '{model_name}' listo → '{output_col}' ✅"})


                # 🌡️ SentimentDLModel
                elif name == "sentiment_dl_model":
                    input_cols = params["inputCols"]
                    if isinstance(input_cols, str):
                        input_cols = [input_cols]

                    model_name = params["model_name"]
                    lang = params.get("lang", "en")
                    output_col = params["outputCol"]

                    sentiment = _local_or_pretrained(SentimentDLModel, model_name, lang) \
                        .setInputCols(input_cols) \
                        .setOutputCol(output_col)

                    stages.append(sentiment)
                    socketio.emit("pipeline_output",
                                {"message": f"🌡️ Modelo de sentimiento '{model_name}' listo → '{output_col}' ✅"})


                # 🌍 Traducción con Marian Transformer
                elif name.startswith("marian_transformer"):
                    input_col  = params["inputCol"]
                    output_col = params["outputCol"]
                    model_name = params["model_name"]

                    marian_transformer = _local_or_pretrained(MarianTransformer,
                                                            model_name) \
                        .setInputCols(input_col) \
                        .setOutputCol(output_col)

                    stages.append(marian_transformer)
                    socketio.emit("pipeline_output",
                                {"message": f"🌍 Marian '{model_name}' listo → '{output_col}' ✅"})


                # 🌍 Traducción con Marian Transformer
                elif name.startswith("marian_transformer"):
                    input_col  = params["inputCol"]
                    output_col = params["outputCol"]
                    model_name = params["model_name"]

                    marian_transformer = _local_or_pretrained(MarianTransformer,
                                                            model_name) \
                        .setInputCols(input_col) \
                        .setOutputCol(output_col)

                    stages.append(marian_transformer)
                    socketio.emit("pipeline_output",
                                {"message": f"🌍 Marian '{model_name}' listo → '{output_col}' ✅"})


                # 🦙 LLaMA 2 (generación)
                elif name.startswith("llama"):
                    input_col  = params["inputCol"]
                    output_col = params["outputCol"]
                    model_name = params["model_name"]                               

                    llama = _local_or_pretrained(LLAMA2Transformer, model_name) \
                        .setInputCols(input_col) \
                        .setOutputCol(output_col)

                    stages.append(llama)
                    socketio.emit("pipeline_output",
                                {"message": f"🦙 Modelo '{model_name}' listo → '{output_col}' ✅"})


                # 🤖 GPT-2 (generación creativa)
                elif name.startswith("gpt2"):
                    input_col  = params["inputCol"]
                    output_col = params["outputCol"]
                    model_name = name                                

                    gpt2 = _local_or_pretrained(GPT2Transformer, model_name) \
                        .setInputCols(input_col) \
                        .setOutputCol(output_col)

                    stages.append(gpt2)
                    socketio.emit("pipeline_output",
                                {"message": f"🤖 GPT-2 '{model_name}' listo → '{output_col}' ✅"})


                # # 📚 Electra QA (preguntas & respuestas)
                # elif name.startswith("electra_qa"):
                #     input_context = params["contextCol"]
                #     input_question = params["questionCol"]
                #     output_col     = params["outputCol"]
                #     model_name     = name                            # p.e. electra_qa_base

                #     electra = _local_or_pretrained(QuestionAnswering, model_name) \
                #         .setInputCols([input_context, input_question]) \
                #         .setOutputCol(output_col)

                #     stages.append(electra)
                #     socketio.emit("pipeline_output",
                #                 {"message": f"📚 Electra QA '{model_name}' listo → '{output_col}' ✅"})


                # 🧬 RoBERTa  
                elif name.startswith("roberta"):
                    input_col = params["inputCols"]
                    output_col = params["outputCol"]
                    model_name = name

                    roberta = _local_or_pretrained(RoBertaForTokenClassification, model_name) \
                        .setInputCols(input_col) \
                        .setOutputCol(output_col)

                    stages.append(roberta)
                    socketio.emit("pipeline_output",
                                {"message": f"🧬 RoBERTa '{model_name}' listo → '{output_col}' ✅"})


                # 🧠 10Dimensions Knowledge Model
                elif name.startswith("10dimensions_knowledge"):
                    input_col  = params["inputCol"]
                    output_col = params["outputCol"]
                    model_name = name

                    knowledge_model = _local_or_pretrained(UniversalSentenceEncoder,
                                                        model_name) \
                        .setInputCols(input_col) \
                        .setOutputCol(output_col)

                    stages.append(knowledge_model)
                    socketio.emit("pipeline_output",
                                {"message": f"🧠 10D Knowledge '{model_name}' listo → '{output_col}' ✅"})

                # 🔎 NER Converter
                elif name == "ner_converter":
                    converter = NerConverter() \
                        .setInputCols(params["inputCols"]) \
                        .setOutputCol(params["outputCol"])
                    stages.append(converter)

                # 🏷️ NER DL
                elif name == "ner_dl":
                    model_name = params.get("model_name", "ner_dl_bert")
                    lang       = params.get("lang", "en")

                    ner_dl = _local_or_pretrained(NerDLModel, model_name, lang) \
                        .setInputCols(params["inputCols"]) \
                        .setOutputCol(params["outputCol"])

                    if last_embedding_ref: # forzar a que use el storageRef correcto
                        ner_dl.setStorageRef(last_embedding_ref)

                    stages.append(ner_dl)

                # 🏁 Finisher
                elif name.startswith("finisher"):
                    input_cols = params.get("inputCols", [])
                    output_cols = params.get("outputCols", input_cols)
                    include_metadata = params.get("includeMetadata", False)
                    output_as_array = params.get("outputAsArray", False)

                    finisher = Finisher() \
                        .setInputCols(input_cols) \
                        .setOutputCols(output_cols) \
                        .setIncludeMetadata(include_metadata) \
                        .setOutputAsArray(output_as_array)
                    stages.append(finisher)

                # ⚠️ Etapa no reconocida
                else:
                    socketio.emit("pipeline_output", {"message": f"❌ Error: Tipo de etapa '{name}' no soportado."})
                    raise ValueError(f"Tipo de etapa '{name}' no soportado.")

                # ⏱️ Guardar tiempo de configuración de la etapa (NO ejecución real)
                stage_end = time.time()
                self.execution_metadata["stage_timings"].append({
                    "stage": name,
                    "duration_sec": round(stage_end - stage_start, 3)
                })
                self.execution_metadata["stages_executed"].append(name)

            
            # 🔄 Construcción dinámica del pipeline con cada etapa
            # Este bucle es el que recorre todas las etapas definidas en el JSON de configuración del pipeline (`self.pipeline_config["stages"]`).
            # Cada etapa debe contener al menos una clave "name" (nombre del transformador/annotator) y opcionalmente "params" con los parámetros requeridos.
            # Según el tipo de etapa (por ejemplo, tokenizer, normalizer, ner, etc.), se inicializa el componente correspondiente de Spark NLP,
            # se configuran sus parámetros, se añade al pipeline, y se notifican los eventos al frontend mediante `socketio.emit`. 
            transformed_df = self.execute_Pipeline(socketio, df, document_columns, stages)
            
        except Exception as e:
            # ❌ En caso de error, emitir mensaje al cliente y detener la ejecución del pipeline
            socketio.emit("pipeline_output", {"message": f"😱 ❌ Error durante la ejecución del pipeline: {str(e)} ❌😞"})
            # Obtener el traceback completo para más detalles
            error_trace = traceback.format_exc()
            # Enviar detalles al WebSocket para depuración en tiempo real
            error_message = f"❌{str(e)}"
            socketio.emit("pipeline_output", {"message": error_message})
            socketio.emit("pipeline_output", {"message": f"📜 Detalles de traza:\n{error_trace}"})
            print(error_message) # Imprimir en consola para logs adicionales
            print(error_trace)
            # Lanzar la excepción con más contexto
            raise RuntimeError(f"Spark NLP Initialization Failed:\n{error_trace}")

        return transformed_df



