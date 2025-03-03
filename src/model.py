# Código base de model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense, Input  # ← Asegúrate de importar Input

def build_model(input_shape):
    """Construye un modelo de red neuronal."""
    model = Sequential([ 
        Input(shape=input_shape), 
        Dense(64, activation='relu'), #input_shape=input_shape),
        #Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(1)  # Salida para regresión
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
