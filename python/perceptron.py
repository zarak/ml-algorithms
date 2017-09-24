import numpy as np

SHAPE = (3, 1)

class Perceptron(object):
    def __init__(self):
        self._w = np.zeros(SHAPE)

    @property
    def w(self):
        return self._w

    def fit(self, data):
        """Train on the synthetic training data

        Args:
            data (Data): Synthetic Data object
        """
        # misclassified_points = self.w.T.dot(data.X)

