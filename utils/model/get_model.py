import os as _os
_os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras as _keras

def model_v1(input_shape: tuple):
    model = _keras.Sequential()
    model.add(_keras.layers.LSTM(64, input_shape=input_shape, activation='relu', return_sequences=True))
    model.add(_keras.layers.Dropout(0.2))
    model.add(_keras.layers.LSTM(32, activation='relu', return_sequences=False))
    model.add(_keras.layers.Dropout(0.2))
    model.add(_keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['acc'])
    return model
