import perceptron
import pytest
import numpy as np

p = perceptron.Perceptron()


def test_initialize_weight_vector_with_zeros():
    assert all(p._w == np.zeros((3, 1)))

def test_sign_of_zero():
    """sign is the function which classifies points"""
    assert p.sign(0) == 0

@pytest.mark.skip()
def test_misclassified_points():
    """The sign of misclassified points should be negative"""
    assert all(np.array(map(p.sign, p.misclassified)) < 0)

@pytest.mark.skip()
def test_misclassified_points_empty():
    """The list of mis"""
