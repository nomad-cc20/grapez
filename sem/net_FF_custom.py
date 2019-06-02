import numpy as np
from numpy.core.multiarray import ndarray


def _init_weights(n1: int, n2: int):
    """
    A layer creation - weights initialization.
    :param n1: number of inputs
    :param n2: number of outputs
    :return: the layer
    """
    return np.random.randn(n1 + 1, n2)


def _d_activate_hidden(z: ndarray) -> ndarray:
    """
    Derivation of the activation function of the hidden layer.
    :param z: actual values of the hidden layer
    :return: derived function
    """
    return 1 - z * z    # tanh(z)'=1-z^2


class FFNN:
    """
    An implementation of feed-forward net.
    """
    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int):
        """
        FFNN constructor.
        :param n_inputs: number of neurons in the input layer
        :param n_hidden: number of neurons in the hidden layer
        :param n_outputs: number of neurons in the output layer
        """
        self.v = _init_weights(n_inputs, n_hidden)
        self.w = _init_weights(n_hidden, n_outputs)
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

    def _aggregate_hidden(self, x: ndarray) -> ndarray:
        """
        The hidden layer aggregate function.
        :param x: values from the previous layer
        :return: aggregated values
        """
        x = np.append(1, x)
        z_in = np.zeros((self.n_hidden, 1))
        for j in range(self.n_hidden):
            for i in range(self.n_inputs + 1):
                z_in[j] = z_in[j] + self.v[i, j] * x[i]
        return z_in

    def _activate_hidden(self, z_in: ndarray) -> ndarray:
        """
        The hidden layer activate function.
        :param z_in: aggregated input
        :return: activated values
        """
        z = np.zeros((self.n_hidden, 1))
        for j in range(self.n_hidden):
            z[j] = np.tanh(z_in[j])
        return z

    def _aggregate_output(self, z: ndarray) -> ndarray:
        """
        The output layer aggregate function.
        :param z: values from the hidden layers
        :return: aggregated values
        """
        z = np.append(1, z)
        y_in = np.zeros((self.n_outputs, 1))
        for k in range(self.n_outputs):
            for j in range(self.n_hidden + 1):
                y_in[k] = y_in[k] + self.w[j, k] * z[j]
        return y_in

    def _activate_output(self, y_in: ndarray) -> ndarray:
        """
        The output layer activate function.
        :param y_in: aggregated inputs
        :return: activated values
        """
        y = np.zeros((self.n_outputs, 1))
        for k in range(self.n_outputs):
            y[k] = y_in[k]
        return y

    def response(self, x: ndarray) -> (ndarray, ndarray):
        """
        Response of the net.
        :param x: input values
        :return: net output, hidden layer output
        """
        z_in = self._aggregate_hidden(x)
        z = self._activate_hidden(z_in)
        y_in = self._aggregate_output(z)
        y = self._activate_output(y_in)
        return y, z

    def _compute_delta_k(self, t: ndarray, y: ndarray) -> ndarray:
        """
        Delta of the output layer.
        :param t: expected values
        :param y: actual values
        :return: output layer delta
        """
        delta_k = np.zeros((self.n_outputs, 1))
        for k in range(self.n_outputs):
            delta_k[k] = (t[k] - y[k]) * 2
        return delta_k

    def _compute_delta_j(self, delta_k: ndarray, z: ndarray) -> ndarray:
        """
        Delta of the hidden layer.
        :param delta_k: output layer delta
        :param z: actual values of the hidden layer
        :return: hidden layer delta
        """
        delta_j = np.zeros((self.n_hidden, 1))
        for j in range(self.n_hidden):
            for k in range(self.n_outputs):
                delta_j[j] = delta_j[j] + delta_k[k] * self.w[j, k]
                delta_j[j] = delta_j[j] * _d_activate_hidden(z[j])
        return delta_j

    def _actualize_v(self, alpha: float, delta_j: ndarray, x: ndarray):
        """
        Actualization of hidden layer weights.
        :param alpha: learning speed
        :param delta_j: hidden layer delta
        :param x: actual values
        """
        x = np.append(1, x)
        for i in range(self.n_inputs + 1):
            for j in range(self.n_hidden):
                self.v[i, j] = self.v[i, j] + alpha * delta_j[j] * x[i]

    def _actualize_w(self, alpha: float, delta_k: ndarray, z: ndarray):
        """
        Actualization of output layer weights.
        :param alpha: learning speed
        :param delta_k: output layer delta
        :param z: actual values
        """
        z = np.append(1, z)
        for j in range(self.n_hidden + 1):
            for k in range(self.n_outputs):
                self.w[j, k] = self.w[j, k] + alpha * delta_k[k] * z[j]

    def _bpg_iter(self, x: ndarray, t: ndarray, alpha: float):
        """
        Back propagation of error - one iteration.
        :param x: input values
        :param t: expected outputs
        :param alpha: learning speed
        """
        y, z = self.response(x)
        delta_k = self._compute_delta_k(t, y)
        delta_j = self._compute_delta_j(delta_k, z)
        self._actualize_w(alpha, delta_k, z)
        self._actualize_v(alpha, delta_j, x)

    def bpg_epoch(self, inputs: ndarray, targets: ndarray, alpha: float):
        """
        Back propagation of error.
        :param inputs: input values
        :param targets: expected outputs
        :param alpha: learning speed
        :return:
        """
        dim = np.size(inputs, 1)
        for index in range(dim):
            self._bpg_iter(inputs[:, index], targets[:, index], alpha)

    def compute_error(self, inputs: ndarray, targets: ndarray) -> float:
        """
        Computation of mean quadratic error of the net.
        :param inputs: input values
        :param targets: expected values
        :return: error value
        """
        error = 0
        for index in range(np.size(inputs, 1)):
            y, z = self.response(inputs[:, index])
            e = 0
            for k in range(len(y)):
                e = e + (targets[k, index] - y[k]) ** 2
            error = error + e
        error = error / np.size(inputs, 1)
        return error
