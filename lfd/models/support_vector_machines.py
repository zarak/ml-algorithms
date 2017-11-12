from lfd import synthetic
import numpy as np
from quadprog import solve_qp


class SVM:
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

    @property
    def w(self):
        return self._w

    @property
    def alphas(self):
        return self._alphas

    def fit(self, X, y):
        # Ensure that there is not just one label for all points
        self._set_dimensions(X)
        self._construct_matrices(X, y)
        # Squeeze and transpose to put in appropriate format for solver
        Q = self._Q + np.eye(3) * 1e-5 # Make Q positive definite
        A = self._A.T
        p = np.squeeze(self._p)
        c = np.squeeze(self._c)
        print(Q.shape)
        print(A.shape)
        print(p.shape)
        print(c.shape)
        # Minimize     1/2 u^T Q u - p^T u
        # Subject to   A u >= c
        # result = solve_qp(Q, p, A, c)[0]
        xf, _, _, _, lagr, _ = solve_qp(Q, p, A, c)
        self._w = xf
        self._alphas = lagr

    def predict(self, X):
        # X should be a matrix of shape (N, 3)
        assert X.shape[1] == 3
        return np.sign(X @ self.w)
