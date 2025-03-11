import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    """Carga el dataset, maneja valores nulos, codifica variables categóricas y normaliza los datos."""
    
    # Cargar el dataset
    try:
        data = pd.read_csv(file_path)
        print("✅ Dataset cargado correctamente.")
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo en la ruta especificada.")
        return None
    
    # Eliminar valores nulos
    data.dropna(inplace=True)
    
    # Eliminar la columna 'Posted On' si existe
    if 'Posted On' in data.columns:
        data.drop(columns=['Posted On'], inplace=True)
    
    # Aplicar logaritmo a Rent para normalizar su distribución
    data['Rent'] = np.log1p(data['Rent'])

    # Eliminar valores atípicos (percentil 2% y 98%)
    lower_bound = data['Rent'].quantile(0.02)
    upper_bound = data['Rent'].quantile(0.98)
    data = data[(data['Rent'] > lower_bound) & (data['Rent'] < upper_bound)]

    # Codificación de variables categóricas
    data = pd.get_dummies(data, drop_first=True)
    
    # Normalización de los datos
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    print(f"✅ Preprocesamiento completado. Dimensiones finales: {data_scaled.shape}")
    return data_scaled

if __name__ == "__main__":
    file_path = "C:/Users/herny/Documents/2025_SEM_III/DEEP LEARNING/mi_proyecto_ml/src/House_Rent_Dataset.csv"
    df = load_and_preprocess_data(file_path)
    print(df.head())
