"""Synthetic Data"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.artist import setp
from collections import namedtuple
plt.style.use('dark_background')


__description__ = 'Randomly generated data'
__author__ = 'Zarak (@z0k)'


SUP = 1
INF = -1
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


def plot_hypothesis(line, ax):
    """Draw a line representing the hypothesis function."""
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
        self._dim = dim
        self._num_points = num_points
        self._X = self._initialize_points()
        self._line = self._generate_line()

    @property
    def _p1(self):
        return self.__p1

    @property
    def _p2(self):
        return self.__p2

    @property
    def X(self):
        return self._X

    @property
    def line(self):
        """Randomly generated line represented as (w0, w1, w2)."""
        return np.array(self._line).reshape(3, -1)

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
        xs = [p[0] for p in [self._p1, self.p2]]
        ys = [p[1] for p in [self._p1, self.p2]]
        plt.scatter(xs, ys, color='green', marker='D')

        ax = plt.gca()
        plot_target(self.line, ax)

        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.legend(fancybox=True)
        plt.xlabel('$x_1$', fontsize=12)
        plt.ylabel('$x_2$', fontsize=12)
        plt.show()

    def _initialize_points(self):
        """Random points"""
        dim = self._dim
        num_points = self._num_points
        X_without_dummies = np.random.uniform(INF, SUP, (dim, num_points))
        return np.vstack([np.ones((1, num_points)), X_without_dummies])

    def _generate_line(self):
        """Creates a line based on p1 and p2."""
        p1 = self._p1
        p2 = self._p2
        slope = (p2.x2 - p1.x2) / (p2.x1 - p1.x1)
        intercept = p1.x2 - slope * p1.x1
        return Line(-intercept, -slope, 1)

    def __repr__(self):
        pass

    def __iter__(self):
        pass
