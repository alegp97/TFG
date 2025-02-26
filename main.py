
"""

FUNCTION build_pipeline(pipeline_config):
    stages_list = []

    FOR stage_def IN pipeline_config["stages"]:
        stage_name   = stage_def["name"]
        stage_params = stage_def["params"]
        
        # Verificar si stage_name existe en el registry
        IF stage_name NOT IN STAGE_REGISTRY:
            RAISE ERROR "Stage not recognized"
        
        # Obtener la función
        stage_factory_fn = STAGE_REGISTRY[stage_name]
        
        # Crear el annotator
        annotator_obj = stage_factory_fn(stage_params)
        
        # Agregar a stages_list
        ADD annotator_obj to stages_list
    
    # Crear un Pipeline de Spark con esas stages
    pipeline = Pipeline(stages=stages_list)
    RETURN pipeline


FUNCTION main():
    # 1. Definir/leer pipeline_config
    my_config = pipeline_config  # (o cargar desde JSON/YAML)
    
    # 2. Construir pipeline
    pipeline = build_pipeline(my_config)
    
    # 3. Obtener DataFrame de entrada
    df_in = leer_datos("ruta/archivo.csv")  # pseudocódigo

    # 4. Fit y transform (en caso de pretrained a veces no hay "entrenamiento", 
    #    pero spark/pipeline.fit() es necesario si existe algún Approach entrenable)
    pipeline_model = pipeline.fit(df_in)

    # 5. Transform
    df_out = pipeline_model.transform(df_in)

    # 6. Guardar o mostrar resultados
    guardar_resultados(df_out, "salida.csv")  # pseudocódigo

"""