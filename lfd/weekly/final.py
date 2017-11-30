import numpy as np
from pathlib import Path


DATA_PATH = Path('/home/z/Documents/ml-algorithms/lfd/data')


def load_data():
    data_train = np.loadtxt(DATA_PATH/'features.train')
    X_train = data_train[:, 1:]
    y_train = data_train[:, 0]
    data_test = np.loadtxt(DATA_PATH/'features.test')
    X_test = data_test[:, 1:]
    y_test = data_test[:, 0]
    return X_train, X_test, y_train, y_test


def digit_versus_all(digit, y):
    """Returns the boolean array of when a label is a particular digit"""
    return y == digit


def generate_digit2labels(y):
    """Creates the dictionary to access the boolean target array by digit"""
    return {digit: digit_versus_all(digit, y) for digit in range(10)}
