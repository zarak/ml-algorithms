"""Perceptron Learning Algorithm"""

import numpy as np
from collections import namedtuple


__description__ = 'Perceptron Learning Algorithm'
__author__ = 'Zarak (@z0k)'


SUPREMUM = 1
INFIMUM = -1
DATADIR = '.'


Point = namedtuple('Point', ['x1', 'x2'])


class Data(object):
    def __init__(self):
        self.__p1 = Point(np.random.rand(), np.random.rand())
        self.__p2 = Point(np.random.rand(), np.random.rand())

    @property
    def line(self):
        """Creates a line based on p1 and p2."""
        p1 = self.__p1
        p2 = self.__p2
        return p1 - p2

    @property
    def data(self):
        """Random points"""
        pass

    def _gram_schmidt(self):
        """Finds a solution using the Gram-Schmidt process."""
        line = np.array([1, *self.line()])

        u = np.
        

