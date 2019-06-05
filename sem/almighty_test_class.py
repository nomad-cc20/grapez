import os
import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from sem.image_preprocessor import ImagePreProcessor
from sem.net_Kohonen_map import Kohonen
from sem.net_convolution import CNN
from sem.net_feedforward_Keras import FFNN


class Almighty:
    """
    A testing class for nets.
    """
    def __init__(self, train_path: str, test_path: str, folders):
        """
        Constructor.
        :param train_path: Path to training set.
        :param test_path: Path to testing set.
        :param folders: Folders of set. One per cluster.
        """
        self.folders = folders
        self.train_path = train_path
        self.test_path = test_path
        self.n_clusters = len(folders)

    def test_kohonen(self, cell_size: int, block_size: int, alpha: float, corr: float, n_epochs: int) -> (float, float):
        """
        A test of the Kohonen map.
        :param cell_size: Image preprocessing cell size.
        :param block_size: Image preprocessing block size (cells).
        :param alpha: Kohonen map alpha value.
        :param corr:  Kohonen map correction value.
        :param n_epochs: Kohonen map learning epochs count.
        :return: Resulting evaluation score (0-1). The first is for train set, the other is for test set.
        """
        ipp = ImagePreProcessor(cell_size, block_size)

        clusters_train = np.zeros(3)
        clusters_val = np.zeros(3)
        pattern_length = 0

        i = 0
        for folderName in os.listdir(self.train_path):
            for fileName in os.listdir(self.train_path + '/' + folderName):
                img = ipp.serve(self.train_path + '/' + folderName + '/' + fileName)
                if len(img) > pattern_length:
                    pattern_length = len(img)
                clusters_train[i] = clusters_train[i] + 1
            i = i + 1

        i = 0
        for folderName in os.listdir(self.test_path):
            for fileName in os.listdir(self.test_path + '/' + folderName):
                img = ipp.serve(self.test_path + '/' + folderName + '/' + fileName)
                if len(img) > pattern_length:
                    pattern_length = len(img)
                clusters_val[i] = clusters_val[i] + 1
            i = i + 1

        patterns_train = np.zeros((int(clusters_train[0] + clusters_train[1] + clusters_train[2]), pattern_length))
        i = 0
        for folderName in os.listdir(self.train_path):
            for fileName in os.listdir(self.train_path + '/' + folderName):
                img = ipp.serve(self.train_path + '/' + folderName + '/' + fileName)
                patterns_train[i] = img
                i = i + 1
        patterns_train = np.transpose(patterns_train)

        patterns_val = np.zeros((int(clusters_val[0] + clusters_val[1] + clusters_val[2]), pattern_length))
        i = 0
        for folderName in os.listdir(self.test_path):
            for fileName in os.listdir(self.test_path + '/' + folderName):
                img = ipp.serve(self.test_path + '/' + folderName + '/' + fileName)
                patterns_val[i] = img
                i = i + 1
        patterns_val = np.transpose(patterns_val)

        expected_train = np.zeros(int(clusters_train[0] + clusters_train[1] + clusters_train[2]))
        for i in range(int(clusters_train[0]), int(clusters_train[0] + clusters_train[1])):
            expected_train[i] = 1
        for i in range(int(clusters_train[0] + clusters_train[1]), int(clusters_train[0] + clusters_train[1] + clusters_train[2])):
            expected_train[i] = 2

        expected_val = np.zeros(int(clusters_val[0] + clusters_val[1] + clusters_val[2]))
        for i in range(int(clusters_val[0]), int(clusters_val[1])):
            expected_val[i] = 1
        for i in range(int(clusters_val[1]), int(clusters_val[2])):
            expected_val[i] = 2

        net_km = Kohonen(self.n_clusters, pattern_length)
        net_km.train(patterns_train, alpha, corr, n_epochs, 1)

        return (net_km.evaluate(patterns_train, expected_train, int(clusters_train[0] + clusters_train[1] + clusters_train[2])),
                net_km.evaluate(patterns_val, expected_val, int(clusters_val[0] + clusters_val[1] + clusters_val[2])))

    def test_feedforward(self, cell_size: int, block_size: int, n_hidden: int, n_layers: int, n_epochs) -> (float, float):
        """
        A test of the feed-forward multilayer neural network.
        :param cell_size: Image preprocessing cell size.
        :param block_size: Image preprocessing block size (cells).
        :param n_hidden: A hidden layer neurons count.
        :param n_layers: Count of hidden layers.
        :param n_epochs: Network learning epochs count.
        :return: Resulting evaluation score (0-1). The first is for train set, the other is for test set.
        """
        ipp = ImagePreProcessor(cell_size, block_size)

        clusters_train = np.zeros(3)
        clusters_val = np.zeros(3)
        pattern_length = 0

        i = 0
        for folderName in os.listdir(self.train_path):
            for fileName in os.listdir(self.train_path + '/' + folderName):
                img = ipp.serve(self.train_path + '/' + folderName + '/' + fileName)
                if len(img) > pattern_length:
                    pattern_length = len(img)
                clusters_train[i] = clusters_train[i] + 1
            i = i + 1

        i = 0
        for folderName in os.listdir(self.test_path):
            for fileName in os.listdir(self.test_path + '/' + folderName):
                img = ipp.serve(self.test_path + '/' + folderName + '/' + fileName)
                if len(img) > pattern_length:
                    pattern_length = len(img)
                clusters_val[i] = clusters_val[i] + 1
            i = i + 1

        patterns_train = np.zeros((int(clusters_train[0] + clusters_train[1] + clusters_train[2]), pattern_length))
        i = 0
        for folderName in os.listdir(self.train_path):
            for fileName in os.listdir(self.train_path + '/' + folderName):
                img = ipp.serve(self.train_path + '/' + folderName + '/' + fileName)
                patterns_train[i] = img
                i = i + 1

        patterns_val = np.zeros((int(clusters_val[0] + clusters_val[1] + clusters_val[2]), pattern_length))
        i = 0
        for folderName in os.listdir(self.test_path):
            for fileName in os.listdir(self.test_path + '/' + folderName):
                img = ipp.serve(self.test_path + '/' + folderName + '/' + fileName)
                patterns_val[i] = img
                i = i + 1

        expected_train = np.zeros(int(clusters_train[0] + clusters_train[1] + clusters_train[2]))
        for i in range(int(clusters_train[0]), int(clusters_train[0] + clusters_train[1])):
            expected_train[i] = 1
        for i in range(int(clusters_train[0] + clusters_train[1]),
                       int(clusters_train[0] + clusters_train[1] + clusters_train[2])):
            expected_train[i] = 2

        expected_val = np.zeros(int(clusters_val[0] + clusters_val[1] + clusters_val[2]))
        for i in range(int(clusters_val[0]), int(clusters_val[1])):
            expected_val[i] = 1
        for i in range(int(clusters_val[1]), int(clusters_val[2])):
            expected_val[i] = 2

        net_ffn = FFNN(pattern_length, n_hidden, self.n_clusters, n_layers)
        net_ffn.fit(patterns_train, expected_train, n_epochs)

        return (net_ffn.evaluate(patterns_train, expected_train,
                                 int(clusters_train[0] + clusters_train[1] + clusters_train[2])),
                net_ffn.evaluate(patterns_val, expected_val, int(clusters_val[0] + clusters_val[1] + clusters_val[2])))

    def test_convolution(self, n_filters: int, kernel_size: (int, int), input_shape: (int, int, int),
                         strides: (int, int), n_layers: int, n_epochs: int) -> (float, float):
        """
        A test of the convolution neural network.
        :param n_filters: Number of filters.
        :param kernel_size: Size of a kernel.
        :param input_shape: The input shape.
        :param strides: Stride offset.
        :param n_layers: Number of hidden layers.
        :param n_epochs : Network learning epochs count.
        :return: Resulting evaluation score (0-1). The first is for train set, the other is for test set.
        """
        bits = 8
        batch_size = 32

        train_datagen = ImageDataGenerator(
            rotation_range=0,
            width_shift_range=0,
            height_shift_range=0,
            rescale=1.0 / (2 ** bits - 1),
            zoom_range=0,
            fill_mode='nearest'
        )
        train_generator = train_datagen.flow_from_directory(
            self.train_path,
            target_size=(input_shape[0], input_shape[1]),
            batch_size=batch_size,
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(
            rotation_range=0,
            width_shift_range=0,
            height_shift_range=0,
            rescale = 1.0 / (2 ** bits - 1),
            zoom_range=0,
            fill_mode='nearest'
        )
        test_generator = test_datagen.flow_from_directory(
            self.test_path,
            target_size=(input_shape[0], input_shape[1]),
            batch_size=batch_size,
            class_mode='categorical'
        )

        net_c = CNN(n_filters, kernel_size, input_shape, strides, n_layers, self.n_clusters)
        net_c.fit(train_generator, batch_size, n_epochs)

        return net_c.evaluate(train_generator, 100), net_c.evaluate(test_generator, 100)
