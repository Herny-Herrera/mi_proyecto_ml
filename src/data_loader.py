import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath, test_size=0.2, random_state=42):
    """Carga el dataset, preprocesa variables categóricas y lo divide en entrenamiento y prueba."""
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)  # Eliminar filas con valores nulos

    # 🔹 Convertir fechas a formato numérico (timestamp UNIX en segundos)
    df['Posted On'] = pd.to_datetime(df['Posted On']).view('int64') // 10**9

    # 🔹 Eliminar columnas irrelevantes
    df.drop(columns=['Point of Contact', 'Area Locality'], inplace=True)

    # 🔹 Convertir variables categóricas en variables numéricas
    df = pd.get_dummies(df, columns=['Floor', 'Area Type', 'City', 'Furnishing Status', 'Tenant Preferred'], drop_first=True)

    # 🔹 Separar variables predictoras (X) y objetivo (y)
    X = df.drop(columns=['Rent'])
    y = df['Rent']

    # 🔹 Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

# 📌 Prueba rápida
if __name__ == "__main__":
    filepath = r"C:\Users\herny\Documents\2025_SEM_III\DEEP LEARNING\Archivos_proyecto_ml_\House_Rent_Dataset.csv"
    X_train, X_test, y_train, y_test = load_data(filepath)

    print("✅ Datos
