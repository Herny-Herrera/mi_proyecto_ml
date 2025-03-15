# Predicción del Precio de Alquiler de Viviendas con una Red Neuronal

## Integrantes
>> Sebastián Heredia
>> Alejandra Bolívar 
>> Michael Gualteros
>> Herny Herrera
>> Juan Rodriguez

## Descripción del Proyecto
Este proyecto implementa una red neuronal utilizando Backpropagation para predecir el precio de alquiler de viviendas en una ciudad, basado en un conjunto de datos con información sobre propiedades residenciales.

## Estructura del Proyecto
```
mi_proyecto_ml/
│── notebooks/                  # Jupyter Notebooks para exploración y análisis
│   ├── 01_exploracion.ipynb    # Análisis exploratorio de datos
│   ├── 02_entrenamiento.ipynb  # Entrenamiento del modelo
│   ├── 03_evaluacion.ipynb     # Evaluación del modelo
│
│── src/                        # Código fuente del proyecto
│   ├── data_loader.py          # Carga y preprocesamiento de datos
│   ├── model.py                # Definición del modelo de red neuronal
│   ├── train.py                # Entrenamiento del modelo (falta implementar)
│   ├── evaluate.py             # Evaluación del modelo entrenado
│   ├── predict.py              # Predicción con el modelo entrenado
│   ├── utils.py                # Funciones auxiliares (opcional)
│
│── models/                     # Modelos entrenados y checkpoints
│   ├── model_v1.pth            # Modelo en PyTorch (si se usa)
│   ├── model_v2.h5             # Modelo en TensorFlow/Keras
│
│── requirements.txt            # Librerías necesarias
│── .gitignore                  # Archivos a ignorar en Git
│── README.md                   # Documentación del proyecto
```

## Instalación y Configuración
1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/mi_proyecto_ml.git
   cd mi_proyecto_ml
   ```
2. Crea un entorno virtual e instala dependencias:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Uso del Proyecto
### 1. Preprocesamiento de Datos
Ejecutar el script para cargar y limpiar los datos:
```bash
python src/data_loader.py
```

### 2. Entrenamiento del Modelo
(Este script no está incluido en los archivos subidos, pero debe ser implementado en `train.py`)
```bash
python src/train.py
```

### 3. Evaluación del Modelo
Para evaluar el rendimiento de la red neuronal:
```bash
python src/evaluate.py
```

### 4. Realizar Predicciones
Ejecuta el siguiente comando para predecir precios de alquiler:
```bash
python src/predict.py
```

## Métricas de Evaluación
- **MSE (0.26)**:
- **MAE (0.32)**:
- **R² (0.73)**:

(*Agregar resultados una vez entrenado el modelo*)

## Requisitos
- Python 3.8+
- TensorFlow/Keras
- Pandas, NumPy, Scikit-learn
- Matplotlib (opcional para visualizaciones)

## Contacto
Si tienes preguntas, contacta a michael david gualteros garcia  a través de <mgualterosg@ucentral.edu.co>.


