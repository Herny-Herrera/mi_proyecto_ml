import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Desactiva el mensaje de oneDNN

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def build_model(input_shape):
    """Construye un modelo de red neuronal."""
    model = Sequential([
        Input(shape=input_shape),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)  # Capa de salida para regresión
    ])
    
    # Compilación del modelo
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

# 📌 Prueba del modelo (solo si se ejecuta el script directamente)
if __name__ == "__main__":
    input_shape = (10,)  # Ajusta el número de características según tu dataset
    model = build_model(input_shape)
    model.summary()  # Muestra la arquitectura del modelo
