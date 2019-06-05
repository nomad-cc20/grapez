import numpy as np
from numpy.core.multiarray import ndarray
from tqdm import tqdm


class Kohonen:
    """
    An implementation of Kohonen map.
    """

    def __init__(self, m: int, n: int):
        """
        Kohonen onstructor.
        :param m: number of clusters
        :param n: size of patterns (row count)
        """
        self.m = m
        self.n = n
        self.w = np.random.randn(n, m)

    def train(self, x: ndarray, alpha: float, corr: float, epochs: int, r0: int):
        """
        Training of patterns.
        :param x: pattern matrix
        :param alpha: learning speed
        :param corr: learning speed multiplier
        :param epochs: number of epochs
        :param r0: the initial affected neighbourhood offset
        """
        r = r0
        for i in tqdm(range(epochs)):
            for j in range(len(x[0, :])):
                self._train_iter(x[:, j], alpha, r)

            alpha = corr * alpha
            r = int(np.round(r0 * (epochs - i) / epochs))

    def _train_iter(self, x_j: ndarray, alpha: float, r: int):
        """
        A single pattern training iteration.
        :param x_j: a pattern
        :param alpha: learning speed
        :param r: affected neighbourhood offset
        """
        d = np.zeros((self.m, 1))
        for i in range(self.m):
            for j in range(self.n):
                d[i] = d[i] + (self.w[j, i] - x_j[j]) ** 2
        ind = np.argmin(d)
        for i in range(max(0, ind - r), min(ind + r + 1, self.m)):
            for j in range(self.n):
                self.w[j, i] = self.w[j, i] + alpha * (x_j[j] - self.w[j, i])

    def equip(self, x_j: ndarray) -> (int, ndarray):
        """
        Equiping of a pattern.
        :param x_j: the pattern
        :return: cluster index, distance to cluster
        """
        d = np.zeros((self.m, 1))
        for i in range(self.m):
            for j in range(self.n):
                d[i] = d[i] + (self.w[j, i] - x_j[j]) ** 2
        ind = np.argmin(d)
        return ind, d[ind]

    def evaluate(self, patterns: ndarray, expected: ndarray, count: int) -> float:
        """
        Evaluates the net.
        :param patterns: validation data
        :param expected: expected output per pattern
        :param count: count of patterns
        :return: accuracy
        """
        right = 0
        for i in range(count):
            prediction = self.equip(patterns[:, i])[0]
            if prediction == expected[i]:
                right = right + 1

        return right / count
