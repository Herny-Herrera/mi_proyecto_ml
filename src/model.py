
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2



def build_model(input_shape):
    """Define una red neuronal para la predicción del precio de alquiler."""
    
 
    model = Sequential([
    Dense(512, activation='relu', kernel_regularizer=l2(0.01), input_shape=(10,)),
    Dropout(0.3),
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1, activation='linear')
])

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])        
    return model

if __name__ == "__main__":
    input_dim = 20  # Ajustar según el número de características
    model = build_model(input_dim)
    model.summary()
