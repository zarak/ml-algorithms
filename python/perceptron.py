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

    def fit(self, data):
        """Train on the synthetic training data

        Args:
            data (Data): Synthetic Data object
        """
        # logits
        z = self.w.T.dot(data.X)
        h = self._activation(z)
        idx = (h <= 0).reshape(-1)
        misclassified_points = d.X[:, idx]
        if not misclassified_points:
            random_point = random.choice(misclassified_points.T)

    def _activation(self, z):
        return np.sign(z)
