import os
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

class Pipeline:
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

        # 🚀 1️⃣ Crear sesión de Spark si no existe
        if not self.spark:
            self._init_spark_session()

    

    def run(self):
        """Ejecuta el pipeline de NLP en Spark"""

        # 🚀 2️⃣ Cargar datos según la fuente (SQL o Local)
        if self.source == "sql":
            df = self.load_from_sql()
        elif self.source == "local":
            df = self.load_from_local()
        else:
            raise ValueError("Fuente de datos no válida. Use 'sql' o 'local'.")

        # # 🚀 3️⃣ Aplicar modelo NLP
        # transformed_df = self.apply_nlp_pipeline(df)

        # if self.debug:
        #     transformed_df.show(5)
        print("JAJAJA")
        df.show()
    

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
    





    def _init_spark_session(self):
        """Crea la sesión de Spark con la configuración proporcionada."""
        spark_builder = SparkSession.builder.appName(self.spark_config["spark"]["app_name"])

        for key, value in self.spark_config["spark"]["configurations"].items():
            spark_builder = spark_builder.config(key, value)

        self.spark = spark_builder.getOrCreate()

        if self.debug:
            print("✅ Sesión de Spark inicializada:")
            print(f"📊 Configuraciones finales de Spark:")
            for key, value in self.spark.sparkContext.getConf().getAll():
                print(f"🔹 {key} = {value}")

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
            print(f"📋 Consulta SQL: {sql_query}")

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


    def load_from_local(self):
        """Carga los datos desde un archivo local."""
        if self.debug:
            print(f"📂 Cargando archivo local: {self.input_data}")

        file_extension = os.path.splitext(self.input_data)[-1]

        if file_extension == ".parquet":
            df = self.spark.read.parquet(self.input_data)
        elif file_extension == ".csv":
            df = self.spark.read.option("header", "true").csv(self.input_data)
        elif file_extension == ".json":
            df = self.spark.read.json(self.input_data)
        else:
            raise ValueError("Formato de archivo no soportado")

        return df

