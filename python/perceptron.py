"""Perceptron Learning Algorithm"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.artist import setp
from collections import namedtuple
plt.style.use('dark_background')


__description__ = 'Perceptron Learning Algorithm'
__author__ = 'Zarak (@z0k)'


SUP = 1
INF = -1
DATADIR = '.'


Point = namedtuple('Point', ['x1', 'x2'])
Line = namedtuple('Line', ['w0', 'w1', 'w2'])


def plot_target(line, ax):
    """Given a line object, plot a line across the entire figure."""
    intercept, slope, _ = -1 * line
    x1 = -1
    x2 = 1
    y1 = slope * x1 + intercept
    y2 = slope * x2 + intercept
    ax.plot((x1, x2), (y1, y2), color='green', label='f')


def plot_hypothesis(line, ax):
    """Given a line object, plot a line across the entire figure."""
    intercept, slope, _ = -1 * line
    x1 = -1
    x2 = 1
    y1 = slope * x1 + intercept
    y2 = slope * x2 + intercept
    ax.plot((x1, x2), (y1, y2), color='orange', label='h')


class Data(object):
    def __init__(self, dim=2, num_points=100):
        self.__p1 = Point(np.random.uniform(INF, SUP),
                          np.random.uniform(INF, SUP))
        self.__p2 = Point(np.random.uniform(INF, SUP),
                          np.random.uniform(INF, SUP))
        self.__dim = dim
        self.__num_points = num_points
        self.__X = self._initialize_points()
        self.__line = self._generate_line()

    @property
    def p1(self):
        return self.__p1

    @property
    def p2(self):
        return self.__p2

    def _generate_line(self):
        """Creates a line based on p1 and p2."""
        p1 = self.p1
        p2 = self.p2
        slope = (p2.x2 - p1.x2) / (p2.x1 - p1.x1)
        intercept = p1.x2 - slope * p1.x1
        return Line(-intercept, -slope, 1)

    @property
    def X(self):
        return self.__X

    @property
    def line(self):
        """Randomly generated line represented as (w0, w1, w2)."""
        return np.array(self.__line).reshape(3, -1)

    @property
    def positive_points(self):
        """All points on the positive side of the line."""
        is_positive = (self.line.T.dot(self.X) > 0).reshape(-1)
        return self.X[:,is_positive]

    @property
    def negative_points(self):
        """All points on the negative side of the line."""
        is_negative = (self.line.T.dot(self.X) < 0).reshape(-1)
        return self.X[:, is_negative]

    def plot(self):
        """Plots all the points and the line."""
        positive_x = self.positive_points[1, :]
        positive_y = self.positive_points[2, :]
        negative_x = self.negative_points[1, :]
        negative_y = self.negative_points[2, :]
        plt.scatter(positive_x, positive_y, marker='o')
        plt.scatter(negative_x, negative_y, marker='x')

        # Plot the two random points to generate the line
        xs = [p[0] for p in [self.p1, self.p2]]
        ys = [p[1] for p in [self.p1, self.p2]]
        plt.scatter(xs, ys, color='green', marker='D')

        ax = plt.gca()
        plot_target(self.line, ax)

        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.legend(fancybox=True)
        plt.xlabel('$x_1$', fontsize=12)
        plt.ylabel('$x_2$', fontsize=12)
        plt.show()


    # def _gram_schmidt(self):
        # """Finds a solution using the Gram-Schmidt process."""
        # u = np.random.uniform(-1, 1, (3, 1))
        # x = np.array([1, *self._line()]).reshape(3, -1)
        # w = u - np.dot(x.T, u) / np.dot(x.T, x) * x
        # return w

    def _initialize_points(self):
        """Random points"""
        dim = self.__dim
        num_points = self.__num_points
        X_without_dummies = np.random.uniform(INF, SUP, (dim, num_points))
        return np.vstack([np.ones((1, num_points)), X_without_dummies])
