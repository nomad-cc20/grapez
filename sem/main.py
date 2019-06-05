from sem.almighty_test_class import Almighty


def _test_kohonen(test_class: Almighty, cell_size: int, block_size: int, alpha: float, corr: float, n_epochs: int):
    """
        A test of a Kohonen map.
        :param cell_size: Image preprocessing cell size.
        :param block_size: Image preprocessing block size (cells).
        :param alpha: Kohonen map alpha value.
        :param corr:  Kohonen map correction value.
        :param n_epochs: Kohonen map learning epochs count.
        :return: Resulting evaluation score (0-1). The first is for train set, the other is for test set.
    """
    result = test_class.test_kohonen(cell_size, block_size, alpha, corr, n_epochs)
    print("Tested Kohonen:\ncell size: %d,\nblock size: %d,\nalpha: %f,\ncorrection: %f,\nepochs: %d\n"
          "with a result of" % (cell_size, block_size, alpha, corr, n_epochs))
    print(result)


def _test_feedforward(test_class: Almighty, cell_size: int, block_size: int, n_hidden: int, n_layers: int, n_epochs):
    """
        A test of feed-forward multilayer neural network.
        :param cell_size: Image preprocessing cell size.
        :param block_size: Image preprocessing block size (cells).
        :param n_hidden: A hidden layer neurons count.
        :param n_layers: Count of hidden layers.
        :param n_epochs: Network learning epochs count.
        :return: Resulting evaluation score (0-1). The first is for train set, the other is for test set.
    """
    result = test_class.test_feedforward(cell_size, block_size, n_hidden, n_layers, n_epochs)
    print("Tested feed-forward net:")
    print("cell size: " + str(cell_size))
    print("block size: " + str(block_size))
    print("hidden layer neurons: " + str(n_hidden))
    print("hidden layers: " + str(n_layers))
    print("epochs" + str(n_epochs))
    print("with a result of")
    print(result)


def _test_convolution(test_class: Almighty, n_filters: int, kernel_size: (int, int), input_shape: (int, int, int),
                      strides: (int, int), n_layers: int, n_epochs: int):
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
    result = test_class.test_convolution(n_filters, kernel_size, input_shape, strides, n_layers, n_epochs)
    print("Tested convolution net:")
    print("filters: " + str(n_filters))
    print("kernel size: " + str(kernel_size))
    print("input shape: " + str(input_shape))
    print("strides: " + str(strides))
    print("hidden layers: " + str(n_layers))
    print("epochs: " + str(n_epochs))
    print("with a result of")
    print(result)


tc = Almighty('img/train', 'img/test', {'pos', 'neg', 'env'})

# print(_test_kohonen(tc, 8, 2, 0.75, 0.995, 1000))
# print(_test_kohonen(tc, 4, 2, 0.75, 0.995, 1000))
# print(_test_kohonen(tc, 8, 3, 0.75, 0.995, 1000))

print(_test_feedforward(tc, 8, 2, 6, 1, 1000))
print(_test_feedforward(tc, 4, 2, 6, 1, 1000))
print(_test_feedforward(tc, 8, 3, 6, 1, 1000))
print(_test_feedforward(tc, 8, 2, 10, 1, 1000))
print(_test_feedforward(tc, 8, 2, 6, 3, 1000))

print(_test_convolution(tc, 32, (3, 3), (40, 40, 3), (1, 1), 1, 1000))
print(_test_convolution(tc, 64, (3, 3), (40, 40, 3), (1, 1), 1, 1000))
print(_test_convolution(tc, 32, (4, 4), (40, 40, 3), (1, 1), 1, 1000))
print(_test_convolution(tc, 32, (2, 2), (40, 40, 3), (1, 1), 2, 1000))
print(_test_convolution(tc, 32, (3, 3), (40, 40, 3), (1, 1), 3, 1000))
