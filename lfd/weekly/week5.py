

def expected_in_sample_error_lin(N):
    """The expected in sample error with respect to a dataset D using linear
    regression."""
    sigma = 0.1
    d = 8
    return sigma**2 * (1 - (d + 1) / N)



def question1():
    results = [f"The value for N={N} is {expected_in_sample_error_lin(N):.5f}" for N in [10, 25, 100, 500, 1000]]
    return '\n'.join(results)

