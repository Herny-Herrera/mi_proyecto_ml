1) Requisitos Previos y Dependencias

Antes de ejecutar cualquier script, asegúrate de tener instaladas las librerías del archivo requirements.txt
michael david gualteros garcia
23:47
2) Cómo Entrenar el Modelo (train.py)

El script train.py entrena el modelo con los datos de entrenamiento y guarda los pesos entrenados.

Ejecutar el entrenamiento:

python src/train.py
Pasos en el entrenamiento:

Carga de datos: Se preprocesan y normalizan los datos (incluyendo la transformación logarítmica en la variable objetivo).
División de datos: Se dividen en X_train, y_train, X_val, y_val.
Definición del modelo: Red neuronal con varias capas ocultas y regularización.
Entrenamiento: Se entrena el modelo con fit(), usando epochs=100 y batch_size=16.
Guardado del modelo: Al final, el modelo se guarda en models/model_v2.h5
3) Cómo Evaluar el Modelo (evaluate.py)

Después del entrenamiento, se puede evaluar el modelo en datos de prueba.

python src/evaluate.py

Pasos en la evaluación:

Carga el modelo guardado (model_v2.h5).
Carga los datos de prueba (X_test, y_test).
Calcula métricas como MAE, MSE, RMSE y R².
Genera gráficos de error y comparaciones entre predicciones y valores reales.
4) Cómo Hacer Predicciones (predict.py)

Para hacer predicciones con nuevos datos, usa el script predict.py.

Ejecutar predicciones:

python src/predict.py --input "ruta_al_archivo_de_datos.csv"

Pasos en la predicción:

Carga el modelo guardado (model_v2.h5).
Preprocesa los nuevos datos de entrada.
Realiza la predicción de la renta esperada.
Guarda o muestra los resultados de la predicción
