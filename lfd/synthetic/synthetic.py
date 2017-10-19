"""Synthetic Data"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.artist import setp
from collections import namedtuple
plt.style.use('dark_background')


__description__ = 'Randomly generated data'
__author__ = 'Zarak (@z0k)'


MAX = 1
MIN = -1
DATADIR = '.'


Point = namedtuple('Point', ['x1', 'x2'])
Line = namedtuple('Line', ['w0', 'w1', 'w2'])


def plot_target(line, ax):
    """Draw a line representing the target function."""
    intercept, slope, _ = -1 * line
    x1 = -1
    x2 = 1
    y1 = slope * x1 + intercept
    y2 = slope * x2 + intercept
    ax.plot((x1, x2), (y1, y2), color='green', label='f')
    return ax


def plot_hypothesis(line, ax):
    """Draw a line representing the hypothesis function."""
    intercept, slope, _ = -1 * line
    x1 = -1
    x2 = 1
    y1 = slope * x1 + intercept
    y2 = slope * x2 + intercept
    ax.plot((x1, x2), (y1, y2), color='orange', label='h')
    return ax


class Data(object):
    def __init__(self, num_train_points=100, num_test_points=1000, dim=2):
        self.__p1 = Point(np.random.uniform(MIN, MAX),
                          np.random.uniform(MIN, MAX))
        self.__p2 = Point(np.random.uniform(MIN, MAX),
                          np.random.uniform(MIN, MAX))
        self._dim = dim
        self._X_train = self._initialize_points(num_train_points)
        self._X_test = self._initialize_points(num_test_points)
        self._line = self._generate_line()
        self._y_train = self.labels(self.X_train, np.sign)
        self._y_test = self.labels(self.X_test, np.sign)

    @property
    def _p1(self):
        return self.__p1

    @property
    def _p2(self):
        return self.__p2

    @property
    def X_train(self):
        return self._X_train

    @property
    def X_test(self):
        return self._X_test

    @property
    def y_train(self):
        """Use np.sign as the default activation function for the train labels"""
        return self._y_train

    @property
    def y_test(self):
        """Use np.sign as the default activation function for the test labels"""
        return self._y_test

    @property
    def line(self):
        """Randomly generated line represented as (w0, w1, w2)."""
        return np.array(self._line).reshape(3, -1)

    @property
    def positive_points(self):
        """All points on the positive side of the line."""
        is_positive = (self.line.T.dot(self.X_train) > 0).reshape(-1)
        return self.X_train[:,is_positive]

    @property
    def negative_points(self):
        """All points on the negative side of the line."""
        is_negative = (self.line.T.dot(self.X_train) < 0).reshape(-1)
        return self.X_train[:, is_negative]

    def labels(self, data, activation):
        """Gets the labels of the synthetic data set.

        Args:
            activation (function): Use a custom activation function

        Returns:
            Array of shape (num_points, 1)
        """
        return activation(self.line.T.dot(data))

    def plot(self):
        """Plots all the points and the line."""
        positive_x = self.positive_points[1, :]
        positive_y = self.positive_points[2, :]
        negative_x = self.negative_points[1, :]
        negative_y = self.negative_points[2, :]
        plt.scatter(positive_x, positive_y, marker='o')
        plt.scatter(negative_x, negative_y, marker='x')

        # Plot the two random points to generate the line
        xs = [p[0] for p in [self._p1, self._p2]]
        ys = [p[1] for p in [self._p1, self._p2]]
        plt.scatter(xs, ys, color='green', marker='D')

        ax = plt.gca()
        plot_target(self.line, ax)

        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.xlabel('$x_1$', fontsize=12)
        plt.ylabel('$x_2$', fontsize=12)
        # plt.show()
        return ax

    def _initialize_points(self, num_points):
        """Random points"""
        dim = self._dim
        X_without_dummies = np.random.uniform(MIN, MAX, (dim, num_points))
        return np.vstack([np.ones((1, num_points)), X_without_dummies])

    def _generate_line(self):
        """Creates a line based on p1 and p2."""
        p1 = self._p1
        p2 = self._p2
        slope = (p2.x2 - p1.x2) / (p2.x1 - p1.x1)
        intercept = p1.x2 - slope * p1.x1
        return Line(-intercept, -slope, 1)

    # TODO
    # def __repr__(self):
        # pass

    # def __iter__(self):
        # pass


class NoisyData(Data):
    def __init__(self, num_train_points=1000, num_test_points=1000):
        super().__init__(num_train_points, num_test_points)
        self._y_train = self._generate_targets(self.X_train)
        self._y_test = self._generate_targets(self.X_test)
        self._add_noise()

    def _generate_targets(self, X):
        return np.sign(X[1]**2 + X[2]**2 - 0.6).reshape(1, -1)

    def _add_noise(self):
        """Randomly flips the sign on 10% subset of training date by mutating
        the target vectors.""" 
        self._y_train = self._y_train * np.random.choice([1, -1],
                self._y_train.shape, p=[0.9, 0.1])
        self._y_test = self._y_test * np.random.choice([1, -1],
                self._y_test.shape, p=[0.9, 0.1])

    def add_features(self):
        """Creates three additional nonlinear features and returns as
        a tuple of training and test sets."""
        new_X_train = np.vstack([self.X_train, self.X_train[1] * self.X_train[2],
            self.X_train[1]**2, self.X_train[2]**2])
        new_X_test = np.vstack([self.X_test, self.X_test[1] * self.X_test[2],
            self.X_test[1]**2, self.X_test[2]**2])
        return new_X_train, new_X_test


if __name__ == "__main__":
    pass
