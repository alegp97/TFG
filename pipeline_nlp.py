import os
import json
import shutil
import traceback
import gc
import time
from datetime import datetime

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import SQLTransformer

import sparknlp
from sparknlp.pretrained import ResourceDownloader

# Spark NLP - base & preprocesamiento
from sparknlp.base import DocumentAssembler, Finisher, TokenAssembler
from sparknlp.annotator import (
    SentenceDetector,
    SentenceDetectorDLModel,
    WordEmbeddingsModel,
    Tokenizer,
    Normalizer,
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
from settings import SAVE_ALWAYS_AS_PARQUET


class ModelManager:
    def __init__(self, base_path="models_cache"):
        self.base_path = os.path.abspath(base_path)
        os.makedirs(self.base_path, exist_ok=True)
        self.cache = {}

    def get_model_path(self, model_name: str, lang: str = "en") -> str:
        """Devuelve la ruta local donde se guardará el modelo."""
        safe_model_name = model_name.replace("/", "_")  # Evitar errores con nombres con "/"
        return os.path.join(self.base_path, f"{safe_model_name}_{lang}")

    def download_and_cache(self, model_class, model_name: str, lang: str = "en"):
        """Carga desde caché en memoria, disco o descarga si es necesario."""
        cache_key = f"{model_class.__name__}:{model_name}:{lang}"

        # ✅ En memoria
        if cache_key in self.cache:
            return self.cache[cache_key]

        model_path = self.get_model_path(model_name, lang)

        # 📂 Ya existe en disco
        if os.path.exists(model_path):
            model = model_class.load(model_path)
        else:
            # 🌐 Descargar y guardar en disco
            model = model_class.pretrained(model_name, lang)
            os.makedirs(model_path, exist_ok=True)
            model.write().save(model_path)

        self.cache[cache_key] = model
        return model




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

        self.model_manager = ModelManager(base_path="models_nlp_local")

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

            # Forzar la recolección de basura para eliminar restos de sesiones previas
            gc.collect()
        except Exception as e:
            print(f"⚠️ No se pudo detener la sesión previa: {e}")

        try:

            app_name = spark_config.get("app_name", "SparkNLP_App") or "SparkNLP_App"
            spark_builder = SparkSession.builder.appName(app_name)

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

            # Depuración: Mostrar la configuración final de Spark NLP si está en modo debug
            if self.debug:
                logs = []
                socketio.emit("pipeline_output", {"message": "⚙️ Configuración final de Spark NLP:"})
                for key, value in self.spark.sparkContext.getConf().getAll():
                    log_message = f"🔹 {key} = {value}"
                    socketio.emit("pipeline_output", {"message": log_message})
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


    # def apply_nlp_pipeline(self, df):
    #     """Aplica el modelo NLP en Spark"""
    #     if self.debug:
    #         print("⚙️ Aplicando modelo de NLP...")

    #     pipeline_model_path = self.pipeline_config["models"].get("pipeline_path")
    #     if not pipeline_model_path:
    #         raise ValueError("No se especificó un modelo en la configuración.")

    #     pipeline_model = PipelineModel.load(pipeline_model_path)
    #     transformed_df = pipeline_model.transform(df)

    #     return transformed_df
    
    def get_execution_metadata(self):
        """Devuelve los metadatos de ejecución, asegurando que todo sea JSON serializable."""
        from copy import deepcopy

        # Crear una copia para no modificar el original
        metadata = deepcopy(self.execution_metadata)

        # Formatear tiempos a ISO si existen
        for key in ["start_time", "end_time"]:
            if isinstance(metadata.get(key), datetime):
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

    
    # def calculate_mean_executor_metadata(self):
    #     exec_mem = self.spark.sparkContext._jsc.sc().getExecutorMemoryStatus()
    #     total_mem = 0
    #     free_mem = 0
    #     count = 0

    #     # ✅ Usamos .entrySet() y lo iteramos con .iterator()
    #     iterator = exec_mem.entrySet().iterator()

    #     while iterator.hasNext():
    #         entry = iterator.next()
    #         host = entry.getKey()
    #         value = entry.getValue()
    #         mem_total = value._1()
    #         mem_free = value._2()
    #         total_mem += mem_total
    #         free_mem += mem_free
    #         count += 1

    #     if count > 0:
    #         avg_total = round((total_mem / count) / (1024 ** 2), 2)
    #         avg_free = round((free_mem / count) / (1024 ** 2), 2)
    #         avg_used = round((total_mem - free_mem) / count / (1024 ** 2), 2)

    #         self.execution_metadata["executors_memory"] = {
    #             "avg_total_MB": avg_total,
    #             "avg_free_MB": avg_free,
    #             "avg_used_MB": avg_used,
    #             "num_executors": count
    #         }
    #     else:
    #         self.execution_metadata["executors_memory"] = {
    #             "error": "No executor memory data found"
    #         }


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
        try:
            if SAVE_ALWAYS_AS_PARQUET:
                format = "parquet"

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

    def run(self, df: DataFrame) -> DataFrame:
        """⚙️ Ejecuta el pipeline NLP y emite logs en tiempo real al cliente mediante WebSocket.""" 
        try:
            # 🧠 Acceder a la instancia de Flask-SocketIO para emitir logs en tiempo real
            socketio = current_app.extensions["socketio"]

            # ⏳ Iniciar la recolección de métricas
            self.execution_metadata["start_time"] = datetime.now()
            self.execution_metadata["input_rows"] = df.count()
            self.execution_metadata["input_schema"] = df.schema.json()
            self.execution_metadata["partitions_before"] = df.rdd.getNumPartitions()

            stages = []
            cached_models = {}  # Modelos cargados en memoria
            document_columns = []  # Columnas temporales generadas por el pipeline
            model_manager = ModelManager(base_path="models_nlp_local")

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
                        .setInputCols([params["inputCol"]]) \
                        .setOutputCol(params["outputCol"])
                    stages.append(tokenizer)
                    document_columns.append(params["outputCol"])
                
                # 🧠 Word Embeddings
                elif name.startswith("word_embeddings"):
                    embedding_model_name = params.get("pretrained_model", "glove_100d")
                    word_embeddings = WordEmbeddingsModel.pretrained(embedding_model_name, "en") \
                        .setInputCols(params["inputCols"]) \
                        .setOutputCol(params["outputCol"]) \
                        .setEnableInMemoryStorage(True)
                    stages.append(word_embeddings)
                    document_columns.append(params["outputCol"])

                # 🧹 Normalizer
                elif name.startswith("normalizer"):
                    normalizer = Normalizer() \
                        .setInputCols([params["inputCol"]]) \
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

                # 🔠 Spell Checker (solo si tienes instalado modelos preentrenados)
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

                # 🌍 Traducción con Marian Transformer
                elif name.startswith("marian_transformer"):
                    model_name = params["model_name"]
                    input_col = params["inputCol"]
                    output_col = params["outputCol"]

                    # Si el modelo ya está en memoria, reutilizarlo sin descargar de nuevo
                    if model_name not in cached_models:
                        socketio.emit("pipeline_output", {"message": f"🌍 Descargando modelo de traducción: {model_name}..."})
                        cached_models[model_name] = MarianTransformer.pretrained(model_name)
                        socketio.emit("pipeline_output", {"message": f"🌍 Modelo {model_name} descargado correctamente."})
                    else:
                        socketio.emit("pipeline_output", {"message": f"🌍 Modelo {model_name} ya cargado en memoria. Usando caché..."})

                    # Crear una nueva instancia del modelo sin descargar de nuevo
                    transformer = MarianTransformer.pretrained(model_name) \
                        .setInputCols([input_col]) \
                        .setOutputCol(output_col)

                    # Agregar al pipeline
                    stages.append(transformer)

                # 🔎 Named Entity Recognition (NER)
                elif name == "ner_dl":
                    model_name = params["model_name"]
                    input_cols = params.get("inputCols") or [params["inputCol"]]
                    output_col = params["outputCol"]

                    # Si el modelo ya está en memoria, reutilizarlo
                    if model_name in cached_models:
                        socketio.emit("pipeline_output", {"message": f"🔍 Modelo NER {model_name} ya cargado en memoria. Usando caché..."})
                        ner_model = cached_models[model_name]
                    else:
                        # 📥 Descargar y cargar el modelo si no está en caché
                        socketio.emit("pipeline_output", {"message": f"🔍 Descargando modelo NER: {model_name}..."})
                        ner_model = NerDLModel.pretrained(model_name, "en")
                        cached_models[model_name] = ner_model
                        socketio.emit("pipeline_output", {"message": f"🔍 Modelo NER {model_name} descargado correctamente."})

                    ner_model.setInputCols(input_cols).setOutputCol(output_col)
                    stages.append(ner_model)

                # 🔎 NER Converter
                elif name == "ner_converter":
                    converter = NerConverter() \
                        .setInputCols(params["inputCols"]) \
                        .setOutputCol(params["outputCol"])
                    stages.append(converter)

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

            # ❌ Verificar si hay etapas válidas
            if not stages:
                socketio.emit("pipeline_output", {"message": "❌ Error: No hay etapas válidas en el pipeline."})
                raise ValueError("No hay etapas válidas en el pipeline.")

            # 🚀 Ejecución del pipeline
            socketio.emit("pipeline_output", {"message": "🚀 Ejecutando pipeline en Spark..."})
            nlp_pipeline = Pipeline(stages=stages)

            # 🚀 Ajuste (fit)
            socketio.emit("pipeline_output", {"message": "🚀 Ajustando (fit)..."} )
            fit_start = time.time()
            model = nlp_pipeline.fit(df)
            df.take(1)  # ⚠️ Forzar ejecución del fit
            fit_end = time.time()

            # 🚀 Transformación
            socketio.emit("pipeline_output", {"message": "🚀 Transformando (transform)..."} )
            transform_start = time.time()
            transformed_df = model.transform(df)
            transformed_df.take(1)  # ⚠️ Forzar ejecución del transform
            transform_end = time.time()

            # ⏱️ Guardar tiempos reales
            self.execution_metadata["stage_timings"].append({
                "stage": "fit", "duration_sec": round(fit_end - fit_start, 3)
            })
            self.execution_metadata["stage_timings"].append({
                "stage": "transform", "duration_sec": round(transform_end - transform_start, 3)
            })

            # 🔢 Particiones finales
            self.execution_metadata["partitions_after"] = transformed_df.rdd.getNumPartitions()

            # # 📊 Memoria por ejecutor (media calculada desde el backend Java)
            # self.calculate_mean_executor_metadata()

            # 🧹 Eliminar columnas temporales
            for col in document_columns:
                if col in transformed_df.columns:
                    transformed_df = transformed_df.drop(col)

            # 🕒 Finalización y duración total
            self.execution_metadata["end_time"] = datetime.now()
            self.execution_metadata["duration"] = (
                self.execution_metadata["end_time"] - self.execution_metadata["start_time"]
            ).total_seconds()

        except Exception as e:
            # ❌ En caso de error, emitir mensaje al cliente y detener la ejecución del pipeline
            socketio.emit("pipeline_output", {"message": f"😱 ❌ Error durante la ejecución del pipeline: {str(e)} ❌😞"})
            # Obtener el traceback completo para más detalles
            error_trace = traceback.format_exc()

            # Enviar detalles al WebSocket para depuración en tiempo real
            error_message = f"❌{str(e)}"
            socketio.emit("pipeline_output", {"message": error_message})
            socketio.emit("pipeline_output", {"message": f"📜 Detalles de traza:\n{error_trace}"})

            # Imprimir en consola para logs adicionales
            print(error_message)
            print(error_trace)

            # Lanzar la excepción con más contexto
            raise RuntimeError(f"Spark NLP Initialization Failed:\n{error_trace}")

        return transformed_df
