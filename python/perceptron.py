"""Perceptron Learning Algorithm"""

import numpy as np


__description__ = 'Perceptron Learning Algorithm'
__author__ = 'Zarak (@z0k)'


SUPREMUM = 1
INFIMUM = 0
DATADIR = '.'


class Data(object):
    def __init__(self):
        self.__x0 = np.random.rand()
        self.__x1 = np.random.rand()

    @property
    def f(self):
        """Target function"""
        pass

    @property
    def data(self):
        """Random points"""
        pass
        

