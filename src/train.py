
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_model
from data_loader import load_data

#  Ruta del archivo de datos
data_path = r'C:\Users\herny\Documents\2025_SEM_III\DEEP LEARNING\Archivos_proyecto_ml_\House_Rent_Dataset.csv'

#  Cargar datos
X_train, X_test, y_train, y_test = load_data(data_path)

print(X_train.dtypes)  # Verifica los tipos de cada columna
print(X_train.head())  # Muestra algunas filas de los datos

#  Crear el modelo
input_shape = (X_train.shape[1],)  # Asegurando que sea una tupla válida
model = build_model(input_shape)



#  Definir directorio para guardar modelos
model_dir = "models/"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "best_model.h5")



# Callbacks para guardado y early stopping
checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#  Entrenar el modelo
history = model.fit(
    X_train, y_train, validation_data=(X_test, y_test),
    epochs=50, batch_size=32,
    callbacks=[checkpoint, early_stopping]
)

#  Guardar el modelo final
final_model_path = os.path.join(model_dir, "final_model.h5")
model.save(final_model_path)
print(f"Modelo guardado en {final_model_path}")




#  Función para graficar la historia del entrenamiento
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # 🔹 Pérdida (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get('loss', []), label='Loss (train)')
    plt.plot(history.history.get('val_loss', []), label='Loss (validation)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Evolución de la Pérdida')
    plt.legend()

    # 🔹 MAE (Error Absoluto Medio)
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get('mae', []), label='MAE (train)')
    plt.plot(history.history.get('val_mae', []), label='MAE (validation)')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.title('Evolución del MAE')
    plt.legend()

    plt.show()

#  Llamar a la función después de entrenar
plot_training_history(history)
