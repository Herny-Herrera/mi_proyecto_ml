import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(file_path):
    """Carga el dataset, maneja valores nulos, codifica variables categóricas y normaliza los datos."""
    
    # Cargar el dataset
    try:
        data = pd.read_csv(file_path)
        print("✅ Dataset cargado correctamente.")
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo en la ruta especificada.")
        return None
    
    # Información general del dataset
    print(data.info())
    
    # Estadísticas descriptivas
    print(data.describe())
    
    # Manejo de valores nulos
    data.dropna(inplace=True)
    
    # Eliminar la columna 'Posted On' si existe
    if 'Posted On' in data.columns:
        data.drop(columns=['Posted On'], inplace=True)
    
    # Aplicar logaritmo a Rent para normalizar su distribución
    data['Rent'] = np.log1p(data['Rent'])
    #import numpy as np

    #data['Rent'] = np.sqrt(data['Rent'])  # Aplica raíz cuadrada


    # Eliminar valores atípicos (percentil 2% y 98%)
    lower_bound = data['Rent'].quantile(0.02)
    upper_bound = data['Rent'].quantile(0.98)
    data = data[(data['Rent'] > lower_bound) & (data['Rent'] < upper_bound)]

    # Identificar columnas categóricas y aplicar Label Encoding
    categorical_cols = data.select_dtypes(include=['object']).columns
    encoder = LabelEncoder()
    for col in categorical_cols:
        data[col] = encoder.fit_transform(data[col])
    print("✅ Variables categóricas transformadas con Label Encoding.")
    
    # Normalización de los datos
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    print(f"✅ Preprocesamiento completado. Dimensiones finales: {data_scaled.shape}")
    return data_scaled

if __name__ == "__main__":
    file_path = "C:/Users/herny/Documents/2025_SEM_III/DEEP LEARNING/mi_proyecto_ml/src/House_Rent_Dataset.csv"
    df = load_and_preprocess_data(file_path)
    print(df.head())
