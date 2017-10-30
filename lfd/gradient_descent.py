import numpy as np


class GradientDescent():
    """
    Gradient descent optimizer on the error function
    E(u, v) = (u * exp(v) - 2 * v * exp(-u))**2
    """
    def __init__(self, w, learning_rate=0.1):
        self.lr = learning_rate
        self._w = w.astype(np.float64)

    @property
    def w(self):
        return self._w

    def E_in(self):
        u = self._w[0]
        v = self._w[1]
        return (u * np.exp(v) - 2 * v * np.exp(-u))**2

    def du(self):
        u = self._w[0]
        v = self._w[1]
        return 2 * (np.exp(v) + 2 * v * np.exp(-u)) * (u * np.exp(v)
                - 2 * v * np.exp(-u))

    def dv(self):
        u = self._w[0]
        v = self._w[1]
        return 2 * (u * np.exp(v) - 2 * np.exp(-u)) * (u * np.exp(v)
                - 2 * v * np.exp(-u))

    def gradient(self):
        return np.array([self.du(), self.dv()]).reshape(2, 1)

    def optimize(self, timesteps):
        for i in range(timesteps):
            dE = self.gradient()
            v = -dE
            self._w += self.lr * v
        return self.w
