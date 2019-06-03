import numpy as np
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten, Activation
from keras.models import Sequential
from numpy.core.multiarray import ndarray
from keras_preprocessing import image


class CN:
    """
    An implementation of convolution net.
    """

    def __init__(self, n_filters: int, kernel_size: (int, int), input_shape: (int, int, int),
                 strides: (int, int), n_layers: int, n_clusters):
        """
        CN constructor.
        :param n_filters: number of filters
        :param kernel_size: the size of a kernel
        :param input_shape: the shape of input images
        :param strides: the step offset
        :param n_layers: number of layers
        :param n_clusters: number of clusters
        """
        self.net = Sequential()
        self.net.add(Convolution2D(
            filters=n_filters,
            kernel_size=kernel_size,
            padding='same',
            input_shape=input_shape,
            strides=strides
        ))
        self.net.add(Activation('relu'))
        self.net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.net.add(Flatten())
        for i in range(n_layers):
            self.net.add(Dense(n_clusters))
        self.net.add(Activation('softmax'))
        self.net.compile(
            loss = 'categorical_crossentropy',
            optimizer='Adam',
            metrics=['accuracy'])

    def fit(self, patterns: ndarray, batch_size: int, n_epochs: int):
        """
        Pattern training.
        :param patterns:
        :param batch_size:
        :param n_epochs:
        """
        self.net.fit_generator(
            patterns,
            steps_per_epoch = patterns.samples / batch_size,
            epochs=n_epochs,
            verbose=1)

    def predict(self, path: str, bits) -> ndarray:
        """
        Pattern prediction.
        :param path: path to image
        :param bits: bit depth
        :return:
        """
        img = image.load_img(path)
        img = np.array(img)
        img = img / (2 ** bits - 1)
        img = np.expand_dims(img, axis=0)

        return self.net.predict(img)
