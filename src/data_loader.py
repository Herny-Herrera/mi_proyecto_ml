# Código base de data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath, test_size=0.2, random_state=42):
    """Carga el dataset desde un archivo CSV y lo divide en conjuntos de entrenamiento y prueba."""
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)  # Eliminar filas con valores nulos
    
    X = df.drop(columns=['Rent'])  # Variables predictoras
    y = df['Rent']  # Variable objetivo
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test