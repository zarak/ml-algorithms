import random
import numpy as np
import synthetic

SHAPE = (3, 1)

class Perceptron(object):
    def __init__(self):
        self._w = np.zeros(SHAPE)
        self._iterations = 0

    @property
    def w(self):
        """The weight vector (initialized with all zeros)"""
        return self._w

    @property
    def iterations(self):
        """A count of the number of iterations until convergence of the PLA."""
        return self._iterations

    @property
    def g(self):
        """Final out of sample estimate g."""
        # TODO: need to handle case when self.w[-1]
        return self.w / self.w[-1]

    def fit(self, train, labels):
        """Train on the synthetic training data

        Args:
            train (array): Data to train on, array with shape (dims, num_samples)
            labels (array): Labels for the training data with shape (1,
            num_samples)
        """
        X = train
        y = labels
        z = self.w.T.dot(X)
        h = self._activation(z)
        data = np.vstack((X, y))
        idx = (h != y).reshape(-1)
        while idx.any():
            misclassified_points = data[:, idx] 
            random_point = random.choice(misclassified_points.T)
            self._w = self._w + (random_point[:-1]
                    * random_point[-1]).reshape(3, 1)
            self._iterations += 1
            z = self.w.T.dot(X)
            h = self._activation(z)
            idx = (h != y).reshape(-1)

    def predict(self, test):
        """Make predictions on test data."""
        return self._activation(self.w.T.dot(test))

    def _activation(self, z):
        """The activation function returns the sign of the input. An input of
        0 returns 0"""
        return np.sign(z)


def main(points):
    num_iterations = []
    probability_not_equal = []
    for i in range(1000):
        d = synthetic.Data(num_points=points)
        p = Perceptron()
        p.fit(d.X, d.labels(np.sign))
        X_test = synthetic.Data(num_points=10000).X
        y_test = np.sign(d.line.T.dot(X_test))
        preds = p.predict(X_test)
        num_iterations.append(p.iterations)
        probability_not_equal.append(np.mean(preds != y_test))
    avg_iterations = np.mean(num_iterations)
    avg_probability_not_equal = np.mean(probability_not_equal)
    print(f"Average number of iterations over 1000 runs for N={points}: {avg_iterations}")
    print(f"Average probability that g != f: over 1000 runs for N={points}: {avg_probability_not_equal}") 


if __name__ == "__main__":
    main(10)
    main(100)
