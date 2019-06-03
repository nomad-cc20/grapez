from numpy.core.multiarray import ndarray
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Lambda
import numpy as np


class FFNN:
    """
    An implementation of feed-forward net.
    """

    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int, n_layers: int):
        """
        FFNN constructor.
        :param n_inputs: number of neurons in the input layer
        :param n_hidden: number of neurons in a hidden layer
        :param n_outputs: number of neurons in the output layer
        :param n_layers: number of hidden layers
        """
        self.net = Sequential()
        self.net.add(Dense(n_hidden, input_dim=n_inputs, activation='tanh'))
        for i in range(n_layers - 1):
            self.net.add(Dense(n_hidden, input_dim=n_hidden, activation='tanh'))
        #self.net.add(Flatten())
        self.net.add(Dense(n_outputs))
        self.net.add(Activation('softmax'))

        self.net.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def fit(self, patterns: ndarray, expected: ndarray, n_epochs: int):
        """
        Pattern training.
        :param patterns: patterns in an array
        :param expected: array of expected values
        :param n_epochs: number of epochs
        """
        self.net.fit(patterns, expected, epochs=n_epochs, verbose=2)

    def predict(self, patterns: ndarray) -> ndarray:
        """
        Pattern prediction.
        :param patterns: patterns in an array
        :return: array of predictions
        """
        softmaxes = self.net.predict_on_batch(patterns)
        predictions = np.zeros(600)
        for i in range(600):
            predictions[i] = np.argmax(softmaxes[i])
        return predictions
