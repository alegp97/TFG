
# Procesamiento, Transformación y Minería de Texto en Big Data con Spark NLP (TFG)

> **Diseño e implementación de una aplicación web para ejecutar *pipelines* dinámicos de Spark NLP sobre Apache Spark.**
> Frontend simple, servidor Flask + Socket.IO, ejecución distribuida, configuración por JSON y monitorización con Spark UI.

**Stack clave:** Apache Spark (3.x) · Spark NLP (≥ 5.2.1) · Python (3.9–3.11) · Flask + Socket.IO · HDFS (opcional)
**Modos de ejecución:** local\[∗] / clúster (Standalone, YARN o Kubernetes)

---

## Índice

* [Descripción](#-descripción)
* [Arquitectura](#-arquitectura)
* [Características](#-características)
* [Requisitos](#-requisitos)
* [Instalación](#-instalación)
* [Configuración](#-configuración)
* [Ejecución](#-ejecución)

  * [Modo web (interfaz)](#modo-web-interfaz)
  * [CLI / *spark-submit*](#cli--spark-submit)
  * [En clúster](#en-clúster)
* [Ejemplos de pipelines (JSON)](#-ejemplos-de-pipelines-json)
* [Estructura del repositorio](#-estructura-del-repositorio)
* [Resultados y datasets de ejemplo](#-resultados-y-datasets-de-ejemplo)
* [Monitorización y métricas](#-monitorización-y-métricas)
* [Solución de problemas (FAQ)](#-solución-de-problemas-faq)
* [Roadmap](#-roadmap)
* [Contribución](#-contribución)
* [Licencia](#-licencia)
* [Agradecimientos](#-agradecimientos)

---

## Descripción

Este proyecto implementa una **aplicación web** que permite **definir y ejecutar *pipelines* de NLP** sobre **Apache Spark** de forma **dinámica** usando **Spark NLP**.
El usuario puede:

* Cargar datos (archivo/SQL/HDFS),
* Elegir un *pipeline* (JSON),
* Lanzar la ejecución en local o en clúster,
* Ver **logs en tiempo real** por WebSocket,
* Exportar resultados y **StageMetrics** para análisis posterior.

Está orientado a **procesamiento distribuido** de texto a gran escala con una configuración simple y reproducible.

---

## Arquitectura

* **Frontend**: HTML sencillo (index/config/results).
* **Backend**: Flask + Flask-SocketIO para eventos y logs en tiempo real.
* **Core NLP**: clase `PipelineNLP` (carga dinámica de etapas Spark NLP a partir de JSON).
* **Gestión de modelos**: `ModelManager` (descarga/caché/local vs. preentrenados).
* **Almacenamiento**: local/HDFS/SQL (entrada y salida).
* **Métricas**: integración con StageMetrics + Spark UI.

*diagrama (conceptual):*

```
Usuario ── Navegador ── Flask/Socket.IO ── PipelineNLP ── Spark Session ── (Workers) ── HDFS/FS/SQL
```

---

## Características

* **Pipelines dinámicos por JSON** (sin redeploy).
* **Soporte de múltiples tareas**: NER, clasificación, embeddings, traducción, sentimiento…
* **Entrada flexible**: archivo local/HDFS/consulta SQL.
* **Salida preparada** (CSV/Parquet) usando `Finisher` para aplanar anotaciones.
* **Logs en tiempo real** + **StageMetrics** por ejecución.
* **Ejecución local/cluster** con la misma configuración.
* **Buenas prácticas** de Spark: Kryo, particionado, evitar *shuffles* innecesarios (cuando aplica).

---

## Requisitos

* **Java** 8/11 (recomendado 11)
* **Python** 3.9–3.11
* **Spark** 3.x y (opcional) **Hadoop** si usas HDFS
* **Paquetes**:

  * `pyspark`
  * `spark-nlp` (o vía `--packages` en `spark-submit`)
  * `flask`, `flask-socketio`, `eventlet` (o `gevent`)
  * Otros utilitarios: `pandas`, etc.

> Si se ejecuta en clúster, es necesario definir variables como `SPARK_HOME`, `HADOOP_CONF_DIR`, y acceso a tu gestor (Standalone/YARN/K8s).

---

## Instalación

```bash
# 1) Crear entorno
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

# 2) Instalar dependencias
pip install -r requirements.txt



## Configuración

* **Archivo(s) de configuración**:

  * `config/` → parámetros de Spark, rutas HDFS, etc.
  * `pipelines/*.json` → definición de *pipelines* (ver ejemplos abajo).
* **Variables de entorno**:
  * `JAVA_HOME`, `SPARK_HOME`, `HADOOP_CONF_DIR`
  * `PYSPARK_PYTHON` y `PYSPARK_DRIVER_PYTHON`

---

## Ejecución

### Modo web (interfaz)

* Ejecutar app.py con python y abrir la ruta en un navegador



## Ejemplos de pipelines (JSON)

### 1) Traducción EN→ES (Marian)

```json
{
  "pipeline_name": "Marian Translation",
  "stages": [
    { "name": "document_assembler", "params": { "inputCol": "text", "outputCol": "document" } },
    { "name": "sentence_detector",   "params": { "inputCols": ["document"], "outputCol": "sentences" } },
    { "name": "marian_transformer",  "params": { "model_name": "marian_finetuned_kde4_english_spanish_en", "inputCol": "sentences", "outputCol": "translated_text_es" } },
    { "name": "finisher",            "params": { "inputCols": ["translated_text_es"], "outputCols": ["translation_es"], "cleanAnnotations": true } }
  ]
}
```

### 2) NER con BERT Embeddings + NerDL

```json
{
  "pipeline_name": "NER with BERT",
  "stages": [
    { "name": "document_assembler", "params": { "inputCol": "content", "outputCol": "document" } },
    { "name": "sentence_detector_dl", "params": { "inputCols": ["document"], "outputCol": "sentences" } },
    { "name": "tokenizer", "params": { "inputCols": ["sentences"], "outputCol": "token" } },
    { "name": "bert_embeddings", "params": { "pretrained": "small_bert_L2_128", "inputCols": ["sentences", "token"], "outputCol": "embeddings" } },
    { "name": "ner_dl_model", "params": { "pretrained": "ner_dl", "inputCols": ["sentences", "token", "embeddings"], "outputCol": "ner" } },
    { "name": "ner_converter", "params": { "inputCols": ["sentences", "token", "ner"], "outputCol": "ner_chunk" } },
    { "name": "finisher", "params": { "inputCols": ["ner_chunk"], "outputCols": ["entities"], "cleanAnnotations": true, "outputAsArray": true } }
  ]
}
```

### 3) Análisis de sentimiento con USE + SentimentDL

```json
{
  "pipeline_name": "Sentiment Analysis (USE)",
  "stages": [
    { "name": "document_assembler", "params": { "inputCol": "content", "outputCol": "document" } },
    { "name": "universal_sentence_encoder", "params": { "pretrained": "tfhub_use", "inputCols": ["document"], "outputCol": "sentence_embeddings" } },
    { "name": "sentiment_dl_model", "params": { "pretrained": "sentimentdl_use_imdb", "inputCols": ["sentence_embeddings"], "outputCol": "sentiment" } },
    { "name": "finisher", "params": { "inputCols": ["sentiment"], "outputCols": ["sentiment_label"], "cleanAnnotations": true } }
  ]
}


## Estructura del repositorio


TFG/
├─ app.py                      # Servidor Flask / entrada principal
├─ pipeline_nlp.py             # Motor de ejecución de pipelines (JSON → Spark)
├─ model_manager.py            # Gestión de modelos Spark NLP (local/pretrained)
├─ settings.py                 # Config. auxiliar de la app (rutas/constantes)
├─ templates/
│  ├─ index.html
│  ├─ config.html
│  └─ results.html
└─ README.md
```
