import os
import json
from pyspark.sql import SparkSession
import sparknlp
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

# Cargar configuración de Spark
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPARK_CONFIG_PATH = os.path.join(BASE_DIR, "spark_config.json")


def create_spark_session(use_spark_nlp=False):
    """
    Crea una sesión de Spark estándar o una optimizada para Spark NLP.
    
    :param use_spark_nlp: Si es True, inicializa Spark con Spark NLP.
    :return: SparkSession
    """
    try:
        with open(SPARK_CONFIG_PATH, "r") as f:
            spark_config = json.load(f)["spark"]

        # Si el usuario elige Spark NLP, usar la configuración predefinida de sparknlp
        if use_spark_nlp:
            spark = sparknlp.start()
        else:
            # Crear la sesión de Spark estándar con configuración personalizada
            spark_builder = SparkSession.builder.appName(spark_config["app_name"]).master(spark_config["master"])

            # Aplicar configuraciones adicionales
            for key, value in spark_config["configurations"].items():
                spark_builder = spark_builder.config(key, value)

            spark = spark_builder.getOrCreate()

        return spark

    except FileNotFoundError:
        logging.error(f"Configuration file not found: {SPARK_CONFIG_PATH}")
    except json.JSONDecodeError:
        logging.error("Error decoding the JSON configuration file.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

    return None  # Retornar None si hay un error
