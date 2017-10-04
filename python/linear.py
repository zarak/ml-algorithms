import numpy as np

SHAPE = (3, 1)

class LinearRegression(object):
    def __init__(self):
        self._w = np.zeros(SHAPE)
        self._iterations = 0

    @property
    def w(self):
        return self._w

    def _activation(self, z):
        return np.sign(z)

    def fit(self, train, labels):
        X = train
        y = labels
        self._w = y.dot(X.T).dot(np.linalg.inv(X.dot(X.T)))

    def predict(self, test):
        return self._activation(self._w.dot(test))
