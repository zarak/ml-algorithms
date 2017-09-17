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
    def line(self):
        """Creates a line based on p1 and p2."""
        p1 = np.array(self.__p1)
        p2 = np.array(self.__p2)
        return p1 - p2

    @property
    def X(self):
        """Random points"""
        pass

    def _gram_schmidt(self):
        """Finds a solution using the Gram-Schmidt process."""
        line = np.array([1, *self.line()])

        u = np.
        

    def plot(self):
        pass
