
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_model
from data_loader import load_data

#  Ruta del archivo de datos
data_path = r"C:\\Users\\USER\\OneDrive\\Escritorio\\Maestría\\mi_proyecto_ml\\src\\House_Rent_Dataset.csv"

#  Cargar datos
X_train, X_test, y_train, y_test = load_data(data_path)

#input_shape = (X_train.shape[1],)
#input_shape

#model = build_model(input_shape)

print(f"X_train.shape: {X_train.shape}")  # Esto te dirá el número de filas y columnas
input_shape = (X_train.shape[1],)  # Asegura que tomamos el número de características

print(f"input_shape: {input_shape}")  # Verifica que sea una tupla con un solo valor

model = build_model(input_shape)




