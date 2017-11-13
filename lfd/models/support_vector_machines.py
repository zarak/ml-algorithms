from lfd import synthetic
import numpy as np
from quadprog import solve_qp


class SVM:
    def __init__(self):
        self._w = None
        self._alphas = None

    def _set_dimensions(self, X):
        # X should be a matrix of shape (N, 3)
        assert X.shape[1] == 3
        self._N = X.shape[0]
        self._d = X.shape[1] - 1

    def _construct_matrices(self, X, y):
        self._set_dimensions(X)
        self._p = np.zeros((self._d + 1, 1))
        self._c = np.ones((self._N, 1))

        # Components of Q
        zero = np.zeros((1, 1))
        zero_d = np.zeros((self._d, 1))
        eye_d = np.eye(self._d)
        
        # Create Q, the quadratic term
        Q_0 = np.hstack([zero, zero_d.T])
        Q_1 = np.hstack([zero_d, eye_d])
        self._Q = np.vstack([Q_0, Q_1])

        # Construct A
        Y = np.diag(np.squeeze(y))
        self._A = Y @ X

    def _format_for_solver(self):
        # Make Q positive definite
        Q = self._Q + np.eye(3) * 1e-5
        # Squeeze and transpose to put in appropriate format for solver
        A = self._A.T
        p = np.squeeze(self._p)
        c = np.squeeze(self._c)
        return Q, p, A, c

    @property
    def w(self):
        return self._w

    @property
    def num_support_vectors(self):
        if self._alphas is not None:
            return np.sum(self._alphas != 0)
        else:
            return None

    def fit(self, X, y):
        # Ensure that there is not just one label for all points
        self._set_dimensions(X)
        self._construct_matrices(X, y)
        Q, p, A, c = self._format_for_solver()
        # Minimize     1/2 u^T Q u - p^T u
        # Subject to   A^T u >= c
        # result = solve_qp(Q, p, A, c)[0]
        xf, _, _, _, lagr, _ = solve_qp(Q, p, A, c)
        self._w = xf.reshape(-1, 1)
        self._alphas = lagr

    def predict(self, X):
        # X should be a matrix of shape (N, 3)
        assert X.shape[1] == 3
        return np.sign(X @ self.w)
