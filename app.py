import traceback
import subprocess 
import time
from flask import Flask, Response, render_template, request, redirect, url_for, jsonify, send_file
from flask_socketio import SocketIO
import json
import os
from werkzeug.utils import secure_filename
from pipeline_nlp import PipelineNLP
from settings import *


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*") 


#  Mostrar las rutas configuradas
print(f"📂 DATA_INPUT_DIR: {DATA_INPUT_DIR}")
print(f"📂 DATA_OUTPUT_DIR: {DATA_OUTPUT_DIR}")


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
    files_in_input = list_files(dir=DATA_INPUT_DIR)

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
    destination = request.form.get("destination", "").lower()

    if destination == "hdfs":
        # Guardar temporalmente el archivo en local antes de subirlo a HDFS
        local_temp_path = os.path.join("/tmp", filename)
        file.save(local_temp_path)

        # Subir archivo a HDFS
        hdfs_path = f"{DATA_INPUT_DIR}/{filename}"
        process = subprocess.run(["hdfs", "dfs", "-put", "-f", local_temp_path, hdfs_path],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if process.returncode != 0:
            print(f"Error al subir archivo a HDFS: {process.stderr.decode()}")
            return jsonify({"error": "No se pudo subir el archivo a HDFS"}), 500

        os.remove(local_temp_path)

    else:  # Default o "local"
        # Guardar el archivo en el sistema local
        file.save(os.path.join(LOCAL_DATA_INPUT_DIR, filename))

    return redirect(request.referrer)



# 🗑️ Eliminar archivos seleccionados
@app.route("/delete_file_input", methods=["POST"])
def delete_file_input():
    """Elimina un archivo seleccionado dentro del directorio de salida y muestra errores en pantalla."""
    data = request.get_json()
    filename = data.get("filename")
    dir = DATA_INPUT_DIR if USE_HDFS and "hdfs" in filename else LOCAL_DATA_INPUT_DIR
    return delete_file_or_dir(filename, dir)


# API flexible para listar archivos de entrada (local o HDFS)
@app.route("/list_files_by_source", methods=["GET"])
def list_files_by_source():
    source = request.args.get("source", "local")  # local o hdfs
    if source == "hdfs":
        files = list_files(dir=DATA_INPUT_DIR)
    else:
        files = list_files(dir=LOCAL_DATA_INPUT_DIR)
    return jsonify({"files": files})


# 🚀 Ejecutar Pipeline con datos de SQL o Archivo
@app.route("/run/<model_key>", methods=["POST"])
def launch_pipeline(model_key):

    input_type = request.form.get("input_type")
    selected_file = request.form.get("selected_file")
    num_partitions = request.form.get("num_partitions", 1)  

    if DEBUG:
        print(f"📢 input_type recibido: {input_type}")
        print(f"📢 num_partitions recibido: {num_partitions}")

    if request.form.get("output_save") is not None:
        output_save = True
    else:
        output_save = False

    # Determinar la fuente de datos
    if input_type == "sql":
        input_data = load_db_config()
    
    elif input_type == "file" and selected_file:
        file_source = request.form.get("file_source", "local")  # viene del <select name="file_source">

        if file_source == "hdfs":
            file_path = selected_file if selected_file.startswith(DATA_INPUT_DIR) else f"{DATA_INPUT_DIR}/{selected_file}"

            # Verificar existencia en HDFS
            check_cmd = ["hdfs", "dfs", "-test", "-e", file_path]
            process = subprocess.run(check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if process.returncode != 0:
                return jsonify({"error": "ARCHIVO SELECCIONADO NO ENCONTRADO EN HDFS"}), 404

        else:  # local
            file_path = os.path.join(LOCAL_DATA_INPUT_DIR, selected_file)

            # Verificar existencia en disco
            if not os.path.exists(file_path):
                return jsonify({"error": "ARCHIVO SELECCIONADO NO ENCONTRADO EN LOCAL"}), 404

        print(f"📢 file recibido: {file_path}")
        input_data = file_path  

    else:
        return jsonify({"error": "ENTRADA INVALIDA, SELECCIONE UN ARCHIVO DE ENTRADA"}), 40

    # Renderizar `results.html` primero, luego ejecutar el pipeline en WebSocket
    return render_template("results.html",
                           input_type=input_type, 
                           model_key=model_key, 
                           input_data=input_data,
                           output_save=output_save,
                           num_partitions=num_partitions)




""" 
############################                         ##############################
########################## FUNCIONES DEL RESULTS HTML #############################
############################                         ##############################
"""

pipeline_sessions = {}
metadata = None
# 🚩 WebSocket: Ejecuta el pipeline en segundo plano y envía las salidas en tiempo real
@socketio.on("run_pipeline")
def run_pipeline(data):
    global metadata

    print("Evento WebSocket `run_pipeline` recibido en Flask")

    input_type = data.get("input_type")
    input_data = data.get("input_data")
    model_key = data.get("model_key")
    num_partitions = int(data.get("num_partitions", 1))
    df_fraction = float(data.get("df_fraction", 100)) / 100.0
    output_save = data.get("output_save")
    output_save = (output_save.lower() == "true")

    model_configs = load_model_configs()
    if model_key not in model_configs:
        return jsonify({"error": "Modelo no encontrado"}), 404
    
    pipeline_config = model_configs[model_key]
    spark_config = load_spark_config()

    # Inicializar y ejecutar el pipeline en tiempo real
    socketio.emit("pipeline_output", {"message": "🚀 Inicializando Pipeline..."}) 
    pipeline = PipelineNLP(source=input_type, 
                        input_data=input_data, 
                        spark_config=spark_config, 
                        pipeline_config=pipeline_config, 
                        debug=DEBUG)
    pipeline_sessions[request.sid] = pipeline
    
    socketio.emit("pipeline_output", {"message": "Pipeline instanciado ⚙️ Iniciando configuración de ⭐Spark..."})
    pipeline_init_spark_msg = pipeline.init_spark_session()
    if pipeline_init_spark_msg:
        socketio.emit("pipeline_output", {"message": pipeline_init_spark_msg})
    
    socketio.emit("pipeline_output", {"message": "\n\n\n🗃️Cargando datos de entrada..."})

    if input_type == "file":    
        df = pipeline.load_from_local(file_path = input_data)
    
    if input_type == "sql":
        df = pipeline.load_from_sql() 

    socketio.emit("pipeline_output", {"message": "✅ Datos cargados y serializados correctamente."})
    if df_fraction < 1.0:
        socketio.emit("pipeline_output", {"message": f"Fraccionando el dataset al {df_fraction}"})
        df = df.sample(fraction=df_fraction)
    if num_partitions:
        socketio.emit("pipeline_output", {"message": f"Total de particiones del rdd antes de repartition: {df.rdd.getNumPartitions()}, particionando a {num_partitions}"})
        df = df.repartition(num_partitions)
    df = df.persist()
    socketio.emit("pipeline_output", {"message": "Esquema del dataframe: \n" + df.schema.json()})

    # Enviar la tabla con WebSocket, pero solo mostrarla cuando el usuario haga clic en "Ver Tabla"
    df_html = df.limit(10).toPandas().to_html()
    socketio.emit("pipeline_output", 
              {"message": "<button onclick='showTable()' "
              "class='view-table-btn'>📋 Ver Muestra de Entrada </button><div id='df_input' style='display:none;'>" + df_html + "</div>"})
    
    # 🚀 Ejecutar el pipeline 
    time_count = time.time()
    socketio.emit("pipeline_output", {"message": "⏳ Comienzo del contador de tiempo"})
    print("Ejecutando Pipeline...")
    df = pipeline.run_stages(df)
    
    # Finalización del pipeline
    socketio.emit("pipeline_output", {"message": "✅ Transformación NLP completada 👍"})

    # df_html = df.limit(5).toPandas().to_html()
    # socketio.emit("pipeline_output", 
    #         {"message": "<button onclick='showTable2()' "
    #         "class='view-table-btn'>📋 Ver Muestra Transformada </button><div id='df_input' style='display:none;'>" + df_html + "</div>"})
    
    # 📥 Guardar archivo si la fuente era local
    if input_type == "file" and output_save == True:
        output_filename = f"processed_{os.path.basename(input_data)}"
        output_file_path = os.path.join(DATA_OUTPUT_DIR, output_filename)

        socketio.emit("pipeline_output", {"message": f"💾 Guardando {output_file_path}..."})
        # if num_partitions:
        #     socketio.emit("pipeline_output", {"message": f"Reduciendo y fusionando (coalesce) las particiones a {num_partitions} "})
        #     df = df.coalesce(num_partitions)
        pipeline.save_to_local(df, output_file_path)
        socketio.emit("pipeline_output", {"message": f"✅ Guardado en {output_file_path}"})
    
    socketio.emit("pipeline_output", {"message": "⭐⭐⭐Pipeline finalizado 🚀"})
    socketio.emit("pipeline_output", {"message": f"⌛ Tiempo de ejecución final:{ time.time() - time_count} segundos "})

    metadata = pipeline.get_execution_metadata()
    
    # 📋 Emitir evento para mostrar el botón para ver el report
    socketio.emit("show_report_button", {"message": "Ver Reporte Final", "redirect_url": url_for("report")})
    stop_pipeline()  # Detener el pipeline después de la ejecución


@app.route('/report.html')
def report():
    return render_template('report.html') 


@app.route("/spark_ui")
def get_spark_ui_url():
    """Devuelve la URL de la interfaz web de Spark UI."""
    return SPARK_UI_URL


# 🚩 WebSocket: Detener la ejecución del pipeline cuando se abandona la página
@socketio.on("stop_pipeline")
def stop_pipeline():
    sid = request.sid
    pipeline = pipeline_sessions.get(sid)

    if pipeline:
        print("🛑 Deteniendo pipeline de sesión:", sid)
        try:
            pipeline.stop_pipeline()
            del pipeline_sessions[sid]
            socketio.emit("pipeline_output", {"message": "🛑 Pipeline detenido."}, to=sid)
        except Exception as e:
            socketio.emit("pipeline_output", {"message": f"❌ Error al detener el pipeline: {str(e)}"}, to=sid)
    else:
        socketio.emit("pipeline_output", {"message": "⚠️ No hay pipeline activo para esta sesión."}, to=sid)



""" 
############################                         ##############################
########################## FUNCIONES GENERALES HTML  #############################
############################                         ##############################
"""

import subprocess
from flask import send_file, abort

@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    pipeline = pipeline_sessions.pop(sid, None)
    if pipeline:
        pipeline.stop_pipeline()
        print(f"🧹 Pipeline de {sid} detenido al desconectar.")


@app.route('/download/<path:filename>')
def download_file(filename):
    """Permite la descarga de un archivo específico, ya sea en HDFS o en local."""
    if USE_HDFS and "part" in filename :
        if IP  == "atlas.ugr.es":
            return jsonify({"status": "error","message": "Descarga HDFS deshabilitada en este servidor"}), 403
        # Ruta del archivo en HDFS
        hdfs_file_path = filename if filename.startswith(DATA_OUTPUT_DIR) else os.path.join(DATA_OUTPUT_DIR, filename)
        local_temp_path = f"/tmp/{filename}"  # Archivo temporal en local
        
        # Comprobar si el archivo existe en HDFS
        check_cmd = ["hdfs", "dfs", "-test", "-e", hdfs_file_path]
        process = subprocess.run(check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.returncode != 0:
            return jsonify("Archivo no encontrado en HDFS"), 404
        
        # Descargar el archivo de HDFS a local
        get_cmd = ["hdfs", "dfs", "-get", hdfs_file_path, local_temp_path]
        process = subprocess.run(get_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.returncode != 0:
            return jsonify("Error al descargar el archivo desde HDFS"), 500
        
        # Enviar el archivo al usuario y eliminarlo después
        response = send_file(local_temp_path, as_attachment=True)
        os.remove(local_temp_path)  # Limpiar archivo temporal
        return response
    
    else:
        # Si está en local, usar la ruta de salida normal
        file_path = os.path.join(LOCAL_DATA_OUTPUT_DIR, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        return jsonify("Archivo no encontrado en local") , 404



@app.route("/get_metadata")
def get_metadata():
    return jsonify(metadata) 

# 🗑️ Eliminar archivos seleccionados
@app.route("/delete_file_output", methods=["POST"])
def delete_file_output():
    data = request.get_json()
    filename = data.get("filename")
    dir = DATA_OUTPUT_DIR if USE_HDFS and "hdfs" in filename else LOCAL_DATA_OUTPUT_DIR
    return delete_file_or_dir(filename=filename, dir=dir)

def delete_file_or_dir(filename, dir):
    try:
        print("Borrando: ", filename)

        if USE_HDFS and "user" in filename:
            if filename.startswith("hdfs://"):
                filename = filename.replace(HDFS_NAMENODE, "")
            hdfs_path = filename if filename.startswith(dir) else os.path.join(dir, filename)
            check_process = subprocess.run(["hdfs", "dfs", "-test", "-d", hdfs_path], capture_output=True)
            is_directory = check_process.returncode == 0
            delete_command = ["hdfs", "dfs", "-rm"]
            if is_directory:
                delete_command.append("-r")
            delete_command.append(hdfs_path)
            process = subprocess.run(delete_command, capture_output=True, text=True)

            print(f"🗑️ Ejecutando: {' '.join(delete_command)}")

            if process.returncode == 0:
                return jsonify({"status": "success", "message": f"{'Directorio' if is_directory else 'Archivo'} eliminado en HDFS: {filename}"}), 200
            else:
                return jsonify({"status": "error", "message": f"Error al eliminar en HDFS: {process.stderr}"}), 500
        else:
            filename = filename.lstrip("/")
            local_path = os.path.join(dir, filename)
            if os.path.exists(local_path):
                if os.path.isdir(local_path):
                    os.rmdir(local_path) 
                else:
                    os.remove(local_path)
                return jsonify({"status": "success", "message": "Archivo o directorio eliminado"}), 200
            else:
                return jsonify({"status": "error", "message": "Archivo o directorio no encontrado"}), 404
    except Exception as e:
        error_trace = traceback.format_exc()  
        print(error_trace)
        return jsonify({
            "success": False,
            "message": "❌ Error al eliminar el archivo.",
            "error": str(e),
            "traceback": error_trace 
        }), 500


def list_files(dir):
    files = []

    print(f"Listando archivos en: {dir}")

    # Comparación robusta de string para rutas locales
    if LOCAL_DATA_INPUT_DIR in dir or LOCAL_DATA_OUTPUT_DIR in dir:
        # Listar archivos en sistema local recursivamente
        for root, _, filenames in os.walk(dir):
            for filename in filenames:
                files.append(filename)  # Solo el nombre, sin la ruta completa
    else:
        
        # Listar archivos en HDFS recursivamente
        process = subprocess.Popen(["hdfs", "dfs", "-ls", "-R", dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"Error al listar archivos en HDFS: {stderr.decode()}")
            return []

        for line in stdout.decode().split("\n"):
            parts = line.split()
            if len(parts) > 7:
                file_path = parts[-1] 
                files.append(file_path)

    return files

@app.route('/list_files_local_output')
def list_files_local_output():
    return list_files(dir=LOCAL_DATA_OUTPUT_DIR)

@app.route('/list_files_output')
def list_files_output():
    return list_files(dir=DATA_OUTPUT_DIR)

@app.route("/merge_partitions", methods=["POST"])
def merge_partitions():
    try:
        data = request.get_json()
        source_path = data.get("source_path")
        output_filename = data.get("output_filename")

        if not source_path or not output_filename:
            return jsonify({"error": "source_path y output_filename son obligatorios"}), 400

        output_path = os.path.join(LOCAL_DATA_OUTPUT_DIR, output_filename)

        if USE_HDFS:
            # Descargar y mergear usando getmerge
            subprocess.run(["hdfs", "dfs", "-getmerge", source_path, output_path], check=True)
        else:
            # Fusionar archivos locales (simplificado)
            with open(output_path, "wb") as outfile:
                for root, _, files in os.walk(source_path):
                    for f in sorted(files):
                        with open(os.path.join(root, f), "rb") as infile:
                            outfile.write(infile.read())

        return jsonify({"message": f"✅ Particiones fusionadas en {output_path}"}), 200

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Error al ejecutar getmerge: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"Error al fusionar archivos: {str(e)}"}), 500
    

@app.route('/preview_file_output/<path:filename>')
def preview_file_output(filename):
    try:
        if "user" in filename:
            if not "hdfs" in filename:
                filename = os.path.join(HDFS_NAMENODE, filename)
            result = subprocess.run(["hdfs", "dfs", "-cat", filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        else:
            result = subprocess.run(["-cat", filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        return Response(result.stdout, mimetype='text/plain')
    except Exception as e:
        return f"⚠️ Error preview_file_output: {e}", 500



@app.route('/favicon.ico')
def favicon():
    return '', 204 


###################################################################################
########################### ⭐EJECUCION MAIN DE APP⭐ ############################
###################################################################################
if __name__ == "__main__":
    socketio.run(app, host=IP, port=PORT, debug=False, allow_unsafe_werkzeug=True)
