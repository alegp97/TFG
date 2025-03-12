from datasets import load_dataset
import os

def descargar_dataset(nombre_dataset, ruta_destino, configuracion, formato="json"):
    """
    Descarga un dataset de Hugging Face y lo guarda en la ruta especificada.

    Parámetros:
    - nombre_dataset (str): Nombre del dataset en Hugging Face (ej. "wikipedia").
    - ruta_destino (str): Ruta donde se guardará el dataset.
    - formato (str): Formato de salida, puede ser "json", "csv" o "parquet".
    """
    # Cargar el dataset
    print(f"Descargando el dataset {nombre_dataset}...")

    if ( configuracion is not None):
        dataset = load_dataset(nombre_dataset, configuracion)
    else:
        dataset = load_dataset(nombre_dataset)

    # Crear directorio si no existe
    os.makedirs(ruta_destino, exist_ok=True)

    # Guardar cada split del dataset en el formato deseado
    for split in dataset.keys():
        archivo_salida = os.path.join(ruta_destino, f"{nombre_dataset}_{split}.{formato}")
        print(f"Guardando {split} en {archivo_salida}...")
        if formato == "json":
            dataset[split].to_json(archivo_salida)
        elif formato == "csv":
            dataset[split].to_csv(archivo_salida)
        elif formato == "parquet":
            dataset[split].to_parquet(archivo_salida)
        else:
            print(f"Formato {formato} no soportado.")
    
    print(f"Dataset {nombre_dataset} descargado y guardado en {ruta_destino}")




# USO
nombre_dataset = "wikipedia"
ruta_destino = "/home/alegp97/TFG/data/input/"
configuracion = "20220301.en"
descargar_dataset(nombre_dataset, ruta_destino, configuracion, formato="json")
