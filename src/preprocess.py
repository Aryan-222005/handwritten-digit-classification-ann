import tensorflow as tf
from keras.utils import to_categorical
mnist = tf.keras.datasets.mnist

def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return X_train, y_train, X_test, y_test


def preprocess_data(X_train, X_test):
    # Normalize
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Flatten
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)

    return X_train, X_test


def encode_labels(y_train, y_test):
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return y_train, y_test