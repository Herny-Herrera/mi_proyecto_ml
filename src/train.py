import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data_loader import load_and_preprocess_data
from model import build_model

# Cargar datos
file_path = "C:/Users/herny/Documents/2025_SEM_III/DEEP LEARNING/mi_proyecto_ml/src/House_Rent_Dataset.csv"
df = load_and_preprocess_data(file_path)

# Separar características (X) y variable objetivo (y)
X = df.drop(columns=["Rent"]).values
y = df["Rent"].values

# Dividir en entrenamiento y prueba (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear modelo
input_dim = X_train.shape[1]
model = build_model(input_dim)

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.reshape(-1, 1))

# Entrenar modelo
history = model.fit(
    X_train, y_train_scaled,
    validation_data=(X_test, y_test_scaled),
    epochs=100,
    batch_size=256,
    verbose=1,
     callbacks=[early_stopping,reduce_lr] 
)

# Guardar modelo entrenado
model.save("C:/Users/herny/Documents/2025_SEM_III/DEEP LEARNING/mi_proyecto_ml/models/model_v2.h5")

print("✅ Entrenamiento finalizado. Modelo guardado.")


# ----------------------------
# 🔹 Graficar Pérdida y MAE
# ----------------------------

# Extraer datos de entrenamiento y validación
loss = history.history['loss']
val_loss = history.history['val_loss']
mae = history.history['mae']
val_mae = history.history['val_mae']

epochs = range(1, len(loss) + 1)

# Graficar pérdida (loss)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'b-', label='Pérdida Entrenamiento')
plt.plot(epochs, val_loss, 'r-', label='Pérdida Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida (MSE)')
plt.title('Evolución de la Pérdida')
plt.legend()

# Graficar MAE
plt.subplot(1, 2, 2)
plt.plot(epochs, mae, 'b-', label='MAE Entrenamiento')
plt.plot(epochs, val_mae, 'r-', label='MAE Validación')
plt.xlabel('Épocas')
plt.ylabel('MAE')
plt.title('Evolución del MAE')
plt.legend()

# Mostrar los gráficos
plt.tight_layout()
plt.show()