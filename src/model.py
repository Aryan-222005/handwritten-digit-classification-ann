from keras.models import Sequential
from keras.layers import Dense, Dropout

def build_model():
    model = Sequential()

    model.add(Dense(128, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10, activation='softmax'))

    return model