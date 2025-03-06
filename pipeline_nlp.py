import os
import json

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import SQLTransformer

import sparknlp
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import MarianTransformer, NerDLModel

from flask import current_app
from flask_socketio import SocketIO




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

    
    
    def init_spark_session(self):
        """Crea la sesión de Spark con la configuración proporcionada."""
        spark_builder = SparkSession.builder.appName(self.spark_config["spark"]["app_name"])

        for key, value in self.spark_config["spark"]["configurations"].items():
            spark_builder = spark_builder.config(key, value)

        # Opción para habilitar Spark NLP si está en la configuración
        use_spark_nlp = self.spark_config["spark"].get("use_spark_nlp", False)
        socketio = current_app.extensions["socketio"]

        if use_spark_nlp:
            socketio.emit("pipeline_output", {"message": f"Usando Spark NLP por defecto..."})
            self.spark = sparknlp.start()
        else:
            print("Iniciando Spark sin Spark NLP...")
            self.spark = spark_builder.getOrCreate()

        if self.debug:
            logs = []
            print("⭐ Sesión de Spark inicializada\n ")
            print(f"⚙️ Configuraciones finales de Spark\n ")
            for key, value in self.spark.sparkContext.getConf().getAll():
                print(f"🔹 {key} = {value}")
                logs.append(f"🔹 {key} = {value}")
            return "\n".join(logs)
        else:
            return None
        
    def stop_pipeline(self):
        """Detiene la sesión de Spark."""
        if self.spark:
            self.spark.stop()
            print("Sesión de Spark detenida.")

    def apply_nlp_pipeline(self, df):
        """Aplica el modelo NLP en Spark"""
        if self.debug:
            print("⚙️ Aplicando modelo de NLP...")

        pipeline_model_path = self.pipeline_config["models"].get("pipeline_path")
        if not pipeline_model_path:
            raise ValueError("No se especificó un modelo en la configuración.")

        pipeline_model = PipelineModel.load(pipeline_model_path)
        transformed_df = pipeline_model.transform(df)

        return transformed_df
    

    def get_spark_session(self):
        """Devuelve la sesión de Spark actual."""
        return self.spark

    def load_from_sql(self):
        """Carga los datos desde una base de datos SQL utilizando la consulta proporcionada."""
        if self.debug:
            print("📡 Cargando datos desde SQL...")

        # Extraer configuración de la BD y consulta SQL
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

        # Conexión JDBC a la base de datos
        df = self.spark.read.format("jdbc").options(
            url=url,
            query=sql_query,
            user=user,
            password=password,
            driver=driver
        ).load()

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

            if format in ["csv", "txt"]:
                df = self.spark.read.option("header", "true").option("inferSchema", "true").option("delimiter", delimiter).csv(file_path)
            elif format == "json":
                df = self.spark.read.option("multiline", "true").json(file_path)
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

            socketio = current_app.extensions["socketio"]

            if format is None:
                format = os.path.splitext(output_path)[-1].lower().replace(".", "")

            if format in ["csv", "txt"]:
                df.coalesce(1).write.option("header", "true").option("delimiter", delimiter).mode(mode).csv(output_path)
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
            else:
                raise ValueError(f"❌ Formato '{format}' no soportado. Usa 'csv', 'txt', 'json', 'parquet', 'avro', 'orc', 'xls', 'xlsx'.")

        except Exception as e:
            socketio.emit("pipeline_output", {"message": f"❌ Error al guardar el archivo: {str(e)}"})
            raise e


    def run(self, df: DataFrame) -> DataFrame:
        """Ejecuta el pipeline NLP y emite logs en tiempo real al cliente mediante WebSocket.""" 
        try:
            # Acceder a la instancia de Flask-SocketIO para emitir logs en tiempo real
            socketio = current_app.extensions["socketio"]

            # Inicio del procesamiento NLP
            socketio.emit("pipeline_output", {"message": "⚙️ Iniciando procesamiento de NLP..."})

            # Mostrar la configuración recibida si está en modo debug
            if self.debug:
                log_message = f"📢 Configuración del pipeline:\n{json.dumps(self.pipeline_config, indent=2)}"
                socketio.emit("pipeline_output", {"message": log_message})

            # Verificar si la configuración tiene la clave 'stages'
            if "stages" not in self.pipeline_config:
                socketio.emit("pipeline_output", {"message": "❌ Error: No se encontró 'stages' en la configuración."})
                raise ValueError("❌ No se encontró 'stages' en la configuración.")


            stages = []
            cached_models = {}  # Modelos cargados en memoria
            document_columns = []  # Columnas temporales generadas por el pipeline

            # 🔄 Construcción dinámica del pipeline con cada etapa
            for stage in self.pipeline_config["stages"]:
                name = stage.get("name")
                params = stage.get("params", {})

                if not name:
                    socketio.emit("pipeline_output", {"message": "❌ Error: Falta 'name' en una etapa."})
                    raise ValueError("Falta 'name' en una etapa.")

                socketio.emit("pipeline_output", {"message": f"📢 Agregando etapa: {name} con parámetros: {params}"})

                # 📄 Document Assembler (Preprocesador de texto)
                if name == "document_assembler":
                    assembler = DocumentAssembler() \
                        .setInputCol(params["inputCol"]) \
                        .setOutputCol(params["outputCol"])
                    stages.append(assembler)
                    document_columns.append(params["outputCol"])

                # 🌍 Traducción con Marian Transformer
                elif name == "marian_transformer":
                    model_name = params["model_name"]
                    input_col = params["inputCol"]

                    # Si el modelo ya está en memoria, reutilizarlo
                    if model_name in cached_models:
                        socketio.emit("pipeline_output", {"message": f"🌍 Modelo {model_name} ya cargado en memoria. Usando caché..."})
                        transformer = cached_models[model_name]
                    else:
                        # 📥 Descargar y cargar el modelo si no está en caché
                        socketio.emit("pipeline_output", {"message": f"🌍 Descargando modelo de traducción: {model_name}..."})
                        transformer = MarianTransformer.pretrained(model_name)  # Descargar y carga
                        cached_models[model_name] = transformer
                        socketio.emit("pipeline_output", {"message": f"🌍 Modelo {model_name} descargado correctamente."})

                    transformer.setInputCols([input_col]).setOutputCol(params["outputCol"])
                    stages.append(transformer)

                # 🔎 Named Entity Recognition (NER)
                elif name == "ner_dl":
                    model_name = params["model_name"]
                    input_col = params["inputCol"]
                    output_col = params["outputCol"]

                    # i el modelo ya está en memoria, reutilizarlo
                    if model_name in cached_models:
                        socketio.emit("pipeline_output", {"message": f"🔍 Modelo NER {model_name} ya cargado en memoria. Usando caché..."})
                        ner_model = cached_models[model_name]
                    else:
                        # 📥 Descargar y cargar el modelo si no está en caché
                        socketio.emit("pipeline_output", {"message": f"🔍 Descargando modelo NER: {model_name}..."})
                        ner_model = NerDLModel.pretrained(model_name, "en")
                        cached_models[model_name] = ner_model
                        socketio.emit("pipeline_output", {"message": f"🔍 Modelo NER {model_name} descargado correctamente."})

                    ner_model.setInputCols([input_col]).setOutputCol(output_col)
                    stages.append(ner_model)

                # 🏁 Finisher para convertir estructuras de Spark NLP en texto plano
                elif name == "finisher":
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

                # Fin de bucle. ⚠️⚠️⚠️ Si se recibe una etapa desconocida, lanzar un error ⚠️⚠️⚠️
                else:
                    socketio.emit("pipeline_output", {"message": f"❌ Error: Tipo de etapa '{name}' no soportado."})
                    raise ValueError(f"Tipo de etapa '{name}' no soportado.")

            # ❌ Si no hay etapas válidas, lanzar error
            if not stages:
                socketio.emit("pipeline_output", {"message": "❌ Error: No hay etapas válidas en el pipeline."})
                raise ValueError("No hay etapas válidas en el pipeline.")

            # 🚀 Creación y ejecución del pipeline en Spark
            socketio.emit("pipeline_output", {"message": "🚀 Ejecutando pipeline en Spark..."})
            nlp_pipeline = Pipeline(stages=stages)

            socketio.emit("pipeline_output", {"message": "🚀 Ajustando (fit)..."} )
            model = nlp_pipeline.fit(df)

            socketio.emit("pipeline_output", {"message": "🚀 Transformando..."} )
            transformed_df = model.transform(df)

            # Eliminar columnas temporales generadas
            socketio.emit("pipeline_output", {"message": "🧹Limpiando columnas temporales..."})
            for col in document_columns:
                if col in transformed_df.columns:
                    transformed_df = transformed_df.drop(col)

            # Finalización del pipeline
            socketio.emit("pipeline_output", {"message": "✅ Transformación NLP completada 👍 "})

        except Exception as e:
            # ❌ En caso de error, emitir mensaje al cliente y detener la ejecución del pipeline
            socketio.emit("pipeline_output", {"message": f"😱 ❌ Error durante la ejecución del pipeline: {str(e)} ❌😞"})
            raise e

        return transformed_df
