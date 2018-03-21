from keras import Sequential
from keras.layers import Dense


def create_model(input_shape):
    model = Sequential()
    model.add(Dense(units=8, input_shape=input_shape, activation='relu'))
    model.add(Dense(units=14, activation='relu'))
    model.add(Dense(units=3, activation='linear'))
    model.compile(loss='mse',optimizer='adam')

    return model