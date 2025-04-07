import pandas as pd
import numpy as np
import random
from faker import Faker
import os

# 🔹 Lugar de ejecución
IP  = "atlas.ugr.es"
PORT= 4050
NAMENODE_PORT = 9000

# ⚙️ Configuración general
USE_HDFS = True  # Cambiar a False si queremos usar el sistema local en lugar de HDFS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  

# 🔹 Definir el namenode de HDFS correctamente
HDFS_NAMENODE = f"hdfs://atlas:{NAMENODE_PORT}"  # Ajustar esto según nuestra configuración

if USE_HDFS:
    HDFS_BASE_DIR = f"{HDFS_NAMENODE}/user/alegp97"
    DATA_INPUT_DIR = os.path.join(HDFS_BASE_DIR, "tfg_input")
    DATA_OUTPUT_DIR = os.path.join(HDFS_BASE_DIR, "tfg_output")
else:
    DATA_INPUT_DIR = os.path.join(BASE_DIR, "data/tfg_input")
    DATA_OUTPUT_DIR = os.path.join(BASE_DIR, "data/tfg_output")



# Número de filas a generar
NUM_ROWS = 100  #  modificarlo

# Inicializar Faker para generar texto
fake = Faker()

# Generar datos aleatorios
data = {
    "TextColumn1": [fake.sentence(nb_words=5) for _ in range(NUM_ROWS)],
    "TextColumn2": [fake.sentence(nb_words=8) for _ in range(NUM_ROWS)],
    "IntegerColumn": np.random.randint(1, 1001, NUM_ROWS),
    "FloatColumn": np.random.uniform(10.5, 99.9, NUM_ROWS).round(2),
    "Category1": np.random.choice(["A", "B", "C", "D"], NUM_ROWS),
    "Category2": np.random.choice(["X", "Y", "Z"], NUM_ROWS),
    "BinaryColumn": np.random.randint(0, 2, NUM_ROWS),
    "SmallIntColumn": np.random.randint(0, 100, NUM_ROWS),
    "ProbabilityColumn": np.random.uniform(0, 1, NUM_ROWS).round(3),
    "GaussianColumn": np.random.normal(50, 15, NUM_ROWS).round(2)
}

# Crear DataFrame en Pandas
df = pd.DataFrame(data)

# Mostrar estructura y algunas filas
print(df.info())
print(df.head())

# Ruta donde guardar el archivo
ruta_parquet  = "/home/alegp97/TFG/data/input/example_wtex_en.parquet"
ruta_csv = "/home/alegp97/TFG/data/input/example_wtex_en.csv"

# Guardar
# df.to_parquet(ruta_parquet, engine="pyarrow", index=False)
df.to_csv(ruta_csv, index=False, encoding="utf-8")

print(f"Datos guardados en:\n- {ruta_parquet} (Parquet)\n- {ruta_csv} (CSV)")
