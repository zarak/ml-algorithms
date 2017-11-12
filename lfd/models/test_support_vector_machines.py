import numpy as np
from quadprog import solve_qp

def test_solve_qp():
    Q = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]) + np.eye(3) * 1e-5
    p = np.zeros(3)
    c = np.ones(4)
    A = np.array([[-1, 0, 0], [-1, -2, -2], [1, 2, 0], [1, 3,
        0]]).astype(np.float64).T
    result = solve_qp(Q, p, A, c)
    assert np.isclose(np.array(result[0]), np.array([-1, 1, -1])).all()

