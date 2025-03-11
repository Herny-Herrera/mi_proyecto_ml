import tensorflow as tf
import numpy as np
from data_loader import load_and_preprocess_data

# Ruta del modelo
model_path = "C:/Users/herny/Documents/2025_SEM_III/DEEP LEARNING/mi_proyecto_ml/models/model_v2.h5"

# Cargar datos
file_path = "C:/Users/herny/Documents/2025_SEM_III/DEEP LEARNING/mi_proyecto_ml/src/House_Rent_Dataset.csv"
df = load_and_preprocess_data(file_path)

# Separar características (X)
X = df.drop(columns=["Rent"]).values

# Intentar cargar el modelo con la función de pérdida correcta
try:
    model = tf.keras.models.load_model(model_path, custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
    print(f" Modelo cargado correctamente desde {model_path}")
except Exception as e:
    print(f" Error al cargar el modelo: {e}")
    exit()

# Realizar predicción con los primeros 5 datos
predictions = model.predict(X[:5])

# Convertir predicciones de la escala logarítmica a la escala original
predictions_original = np.expm1(predictions)

# Mostrar predicciones
print(" Predicciones de precios de alquiler para las primeras 5 propiedades:")
for i, pred in enumerate(predictions_original):
    print(f" Propiedad {i+1}: ${pred[0]:,.2f}")
