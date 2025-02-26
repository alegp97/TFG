from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_socketio import SocketIO
import json
import os
from werkzeug.utils import secure_filename
from pipeline import Pipeline

app = Flask(__name__)
socketio = SocketIO(app) 

# ⚙️ Rutas a los archivos de configuración
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
DATA_INPUT_DIR = os.path.join(BASE_DIR, "data/input")  
MODEL_CONFIG_PATH = os.path.join(BASE_DIR, "json/model_config.json")  
SPARK_CONFIG_PATH = os.path.join(BASE_DIR, "json/spark_config.json")  
DB_DICT_CONFIG_PATH = os.path.join(BASE_DIR, "json/db_config.json")  

# ⚙️ Asegurar que el directorio de archivos existe
os.makedirs(DATA_INPUT_DIR, exist_ok=True)

DEBUG = True

""" 
############################                         ##############################
########################### FUNCIONES DEL INDEX HTML  #############################
############################                         ##############################
"""


# 🚩 Página principal con selección de modelo y configuración de Spark
@app.route("/", methods=["GET", "POST"])
def index():
    spark_config = load_spark_config()
    
    # 🔹 Si la petición es POST, guardar configuración
    if request.method == "POST":
        return save_spark_config()

    return render_template("index.html", spark_config=json.dumps(spark_config, indent=2))


# 📝 Cargar configuración de Spark
def load_spark_config():
    if os.path.exists(SPARK_CONFIG_PATH):
        with open(SPARK_CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}

# 📝 Función para guardar configuración de Spark
@app.route("/save_spark_config", methods=["POST"])
def save_spark_config():
    try:
        # Verifica si los datos vienen en JSON o en un formulario
        if request.is_json:
            new_spark_config = request.get_json()
        else:
            new_spark_config = json.loads(request.form.get("config_json", "{}"))
        
        # Verifica si la configuración es válida antes de guardar
        if not isinstance(new_spark_config, dict):
            return jsonify({"status": "error", "message": "Formato JSON inválido"}), 400

        with open(SPARK_CONFIG_PATH, "w") as f:
            json.dump(new_spark_config, f, indent=4)

        return redirect(request.referrer) 
    except json.JSONDecodeError:
        return jsonify({"status": "error", "message": "Error en JSON de Spark"}), 400


""" 
############################                         ##############################
########################### FUNCIONES DEL CONDIG HTML #############################
############################                         ##############################
"""



# 🚩 Página de configuración del modelo y BD
@app.route("/configure/<model_key>", methods=["GET"])
def configure_model(model_key):
    model_configs = load_model_configs()
    if model_key not in model_configs:
        return "Modelo no encontrado", 404

    model_config = model_configs[model_key]
    db_config = load_db_config()
    files_in_input = os.listdir(DATA_INPUT_DIR) if os.path.exists(DATA_INPUT_DIR) else []

    return render_template(
        "config.html",
        model_key=model_key,
        config=json.dumps(model_config, indent=2),
        db_config=json.dumps(db_config, indent=2),
        input_files=files_in_input
    )

# 📝 Cargar configuraciones de modelos desde JSON
def load_model_configs():
    if os.path.exists(MODEL_CONFIG_PATH):
        with open(MODEL_CONFIG_PATH, "r") as f:
            return json.load(f).get("models", {})
    return {}

# 📝 Cargar configuración de la BD
def load_db_config():
    if os.path.exists(DB_DICT_CONFIG_PATH):
        with open(DB_DICT_CONFIG_PATH, "r") as f:
            return json.load(f)
    return {"database": {}, "query": {}}

# 📝 Guardar configuración de la BD
def save_db_config(new_config):
    with open(DB_DICT_CONFIG_PATH, "w") as f:
        json.dump(new_config, f, indent=4)

# 📝 Guardar configuración del modelo
def save_model_configs(model_configs):
    with open(MODEL_CONFIG_PATH, "w") as f:
        json.dump({"models": model_configs}, f, indent=4)

# 📝 Guardar configuración del modelo
@app.route("/save_model_config/<model_key>", methods=["POST"])
def save_model_config(model_key):
    model_configs = load_model_configs()
    if model_key not in model_configs:
        return jsonify({"status": "error", "message": "Modelo no encontrado"}), 404

    try:
        new_config = json.loads(request.form.get("config_json", "{}"))
        model_configs[model_key] = new_config
        save_model_configs(model_configs)
        return redirect(url_for("configure_model", model_key=model_key))
    except json.JSONDecodeError:
        return jsonify({"status": "error", "message": "Error en JSON del modelo"}), 400

# 📝 Guardar configuración de la base de datos
@app.route("/save_db_config", methods=["POST"])
def save_db_config_route():
    try:
        new_db_config = json.loads(request.form.get("db_json", "{}"))
        save_db_config(new_db_config)
        return redirect(request.referrer)
    except json.JSONDecodeError:
        return jsonify({"status": "error", "message": "Error en JSON de la base de datos"}), 400

# ⬆️ Manejo de carga de archivos
@app.route("/upload_file", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect(request.referrer)

    file = request.files["file"]
    if file.filename == "":
        return redirect(request.referrer)

    filename = secure_filename(file.filename)
    file.save(os.path.join(DATA_INPUT_DIR, filename))

    return redirect(request.referrer)

# 🗑️ Eliminar archivos seleccionados
@app.route("/delete_file", methods=["POST"])
def delete_file():
    data = request.get_json()
    filename = data.get("filename")

    if not filename:
        return jsonify({"status": "error", "message": "No se especificó un archivo"}), 400

    file_path = os.path.join(DATA_INPUT_DIR, filename)

    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({"status": "success", "message": "Archivo eliminado"})
    else:
        return jsonify({"status": "error", "message": "Archivo no encontrado"}), 404


# 🚀 Ejecutar Pipeline con datos de SQL o Archivo
@app.route("/run/<model_key>", methods=["POST"])
def launch_pipeline(model_key):


    input_type = request.form.get("input_type")
    selected_file = request.form.get("selected_file")

    if DEBUG:
        print(f"📢 input_type recibido: {input_type}")
        print(f"📂 selected_file recibido: {selected_file}")

    # spark_config = load_spark_config()
    # pipeline_config = model_configs[model_key]

    # # 📌 Determinar la fuente de datos
    # if input_type == "sql":
    #     input_data = load_db_config()  # ✅ Pasamos el JSON completo con la consulta SQL
    # elif input_type == "file" and selected_file:
    #     file_path = os.path.join(DATA_INPUT_DIR, selected_file)
    #     if not os.path.exists(file_path):
    #         return jsonify({"error": "Archivo no encontrado"}), 404
    #     input_data = file_path  # ✅ Se pasa solo la ruta como string
    # else:
    #     return jsonify({"error": "Entrada no válida"}), 400

    # ✅ Renderizar `results.html` primero, luego ejecutar el pipeline en WebSocket
    return render_template("results.html", model_key=model_key, input_type=input_type, selected_file=selected_file)




""" 
############################                         ##############################
########################## FUNCIONES DEL RESULTS HTML #############################
############################                         ##############################
"""
pipeline = None
# 📌 WebSocket: Ejecuta el pipeline en segundo plano y envía las salidas en tiempo real
@socketio.on("run_pipeline")
def run_pipeline(data):
    print("JAJAJA")
    model_key = data.get("model_key")
    input_type = data.get("input_type")
    selected_file = data.get("selected_file")

    spark_config = load_spark_config()

    model_configs = load_model_configs()
    if model_key not in model_configs:
        return jsonify({"error": "Modelo no encontrado"}), 404
    pipeline_config = model_configs[model_key]

    if input_type == "sql":
        input_data = load_db_config()
    elif input_type == "file" and selected_file:
        input_data = {"file_path": os.path.join("data/input", selected_file)}
    else:
        socketio.emit("pipeline_output", {"message": "❌ Entrada no válida."})
        return

    print("JAJAJA")

    # ✅ Inicializar y ejecutar el pipeline en tiempo real
    global pipeline 
    pipeline = Pipeline(source=input_type, input_data=input_data, spark_config=spark_config, pipeline_config=pipeline_config, debug=DEBUG)
    
    for output in pipeline.run(): 
        socketio.emit("pipeline_output", {"message": output})

    socketio.emit("pipeline_output", {"message": "✅ Pipeline finalizado."})


@app.route("/spark_ui")
def get_spark_ui_url():
    """Devuelve la URL de la interfaz web de Spark UI."""
    try:
        web_ui_url = pipeline.get_spark_session().sparkContext.uiWebUrl

        if not web_ui_url:
            return jsonify({"error": "⚠️ Spark UI no está disponible."}), 404

        return jsonify({"spark_ui_url": web_ui_url})

    except Exception as e:
        return jsonify({"error": f"❌ No se pudo obtener la URL de Spark UI: {str(e)}"}), 500



###################################################################################
########################### ⭐EJECUCION MAIN DE APP⭐ ############################
###################################################################################
if __name__ == "__main__":
    IP  = "atlas.ugr.es"
    PORT= 4050
    socketio.run(app, host=IP, port=PORT, debug=DEBUG)
