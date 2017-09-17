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
    def __init__(self):
        self.__p1 = Point(np.random.uniform(INF, SUP),
                          np.random.uniform(INF, SUP))
        self.__p2 = Point(np.random.uniform(INF, SUP),
                          np.random.uniform(INF, SUP))

    @property
    def X(self):
        """Random points"""
        pass

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
        x = np.array([1, *self._line()]).reshape(3, -1)
        u = np.random.uniform(-1, 1, (3, 1))
        w = u - np.dot(x.T, u) / np.dot(x.T, x) * x
        return w


    def plot(self):
        pass
