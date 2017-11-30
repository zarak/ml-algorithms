import numpy as np
from pathlib import Path
from lfd.models.linear import LinearRegression


DATA_PATH = Path('../data')


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
    labels = np.ones(y.shape)
    labels[y == digit] = 1
    labels[y != digit] = -1
    return labels


def generate_digit2labels(y):
    """Creates the dictionary to access the boolean target array by digit"""
    return {digit: digit_versus_all(digit, y) for digit in range(10)}


def question7():
    X_train, X_test, y_train, y_test = load_data()
    digit2labels_train = generate_digit2labels(y_train)
    for digit in range(5, 10):
        lm = LinearRegression(l2=1)
        y_train = digit2labels_train[digit]
        lm.fit(X_train.T, y_train.T)
        preds = lm.predict(X_train.T)
        E_in = np.mean(preds != y_train)
        print(f"E_in for {digit} versus all is", E_in)


if __name__ == "__main__":
    question7()
