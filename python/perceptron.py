"""Perceptron Learning Algorithm"""

import numpy as np
from collections import namedtuple


__description__ = 'Perceptron Learning Algorithm'
__author__ = 'Zarak (@z0k)'


SUP = 1
INF = -1
DATADIR = '.'


Point = namedtuple('Point', ['x1', 'x2'])


class Data(object):
    def __init__(self, dim=2, num_points=100):
        self.__p1 = Point(np.random.uniform(INF, SUP),
                          np.random.uniform(INF, SUP))
        self.__p2 = Point(np.random.uniform(INF, SUP),
                          np.random.uniform(INF, SUP))
        self.__dim = dim
        self.__num_points = num_points

    @property
    def p1(self):
        return self.__p1

    @property
    def p2(self):
        return self.__p2

    @property
    def X(self):
        """Random points"""
        dim = self.__dim
        num_points = self.__num_points
        X_without_dummies = np.random.uniform(INF, SUP, (dim, num_points))
        return np.vstack([np.ones((1, num_points)), X_without_dummies])

    @property
    def line(self):
        """Random line"""
        return self._gram_schmidt()

    def _line(self):
        """Creates a line based on p1 and p2."""
        p1 = np.array(self.__p1)
        p2 = np.array(self.__p2)
        return p1 - p2

    def _gram_schmidt(self):
        """Finds a solution using the Gram-Schmidt process."""
        u = np.random.uniform(-1, 1, (3, 1))
        x = np.array([1, *self._line()]).reshape(3, -1)
        w = u - np.dot(x.T, u) / np.dot(x.T, x) * x
        return w

    def positive_points(self):
        return self.line.T.dot(self.X) > 0


    def plot(self):
        pass
