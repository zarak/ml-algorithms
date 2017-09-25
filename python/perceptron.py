import random
import numpy as np

SHAPE = (3, 1)

class Perceptron(object):
    def __init__(self):
        self._w = np.zeros(SHAPE)
        self._iterations = 0

    @property
    def w(self):
        return self._w

    @property
    def iterations(self):
        return self._iterations

    def fit(self, train, labels):
        """Train on the synthetic training data

        Args:
        """
        X = train
        y = labels
        z = self.w.T.dot(X)
        h = self._activation(z)
        data = np.vstack((X, y))
        idx = (h != y).reshape(-1)
        while idx.any():
            misclassified_points = data[:, idx] 
            print(misclassified_points.shape)
            random_point = random.choice(misclassified_points.T)
            
            self._w = self._w + (random_point[:-1]
                    * random_point[-1]).reshape(3, 1)
            self._iterations += 1

            z = self.w.T.dot(X)
            print(self.w)
            h = self._activation(z)
            idx = (h != y).reshape(-1)
            print(idx)

    def _activation(self, z):
        return np.sign(z)
