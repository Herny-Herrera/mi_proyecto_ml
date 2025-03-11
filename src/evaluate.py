import tensorflow as tf
from data_loader import load_and_preprocess_data
from sklearn.model_selection import train_test_split

# Ruta del modelo corregida
model_path = "C:/Users/herny/Documents/2025_SEM_III/DEEP LEARNING/mi_proyecto_ml/models/model_v2.h5"

# Cargar datos
file_path = "C:/Users/herny/Documents/2025_SEM_III/DEEP LEARNING/mi_proyecto_ml/src/House_Rent_Dataset.csv"
df = load_and_preprocess_data(file_path)

# Verificar que los datos se cargaron correctamente
if df is None or df.empty:
    print("❌ Error: No se pudo cargar el dataset.")
    exit()

# Separar datos en X (features) y y (target)
X = df.drop(columns=["Rent"]).values
y = df["Rent"].values

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Intentar cargar el modelo especificando 'mse' en custom_objects
try:
    model = tf.keras.models.load_model(model_path, custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
    print(f"✅ Modelo cargado correctamente desde {model_path}")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    exit()

# Evaluar el modelo
loss, mae = model.evaluate(X_test, y_test)
print(f"🔍 Evaluación del modelo - Loss: {loss:.4f}, MAE: {mae:.4f}")
