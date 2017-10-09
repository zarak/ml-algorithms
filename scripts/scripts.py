import linear
import perceptron
import synthetic
import numpy as np
import matplotlib.pyplot as plt


# Week 1
def question7():
    iterations = []    
    for _ in range(1000):
         lm = linear.LinearRegression()
         d = synthetic.Data(10)
         lm.fit(d.X_train, d.y_train)
         p = perceptron.Perceptron(lm.w.T)
         p.fit(d.X_train, d.y_train)
         iterations.append(p.iterations)
    print(np.mean(iterations))


def question9():
    model_weights = []
    for _ in range(100):
        nd = synthetic.NoisyData()
        Z_train, Z_test = nd.add_features()
        lm = linear.LinearRegression()
        lm.fit(Z_train, nd.y_train)
        model_weights.append(lm.w)
    print(np.mean(model_weights, axis=0))


# Week 2
def question1():
    """Returns distributions of the fraction of heads obtained in ten coin
    flips among 1000 fair coins. Experiment is repeated 100000 times."""
    v1s = [] # Fraction of heads of first coin flipped
    vrands = [] # Fraction of heads of a randomly chosen coin
    vmins = [] # Fraction of heads of the coin with min frequency of heads
    for _ in range(100000):
        flips = np.random.randint(0, 2, (1000, 10))
        c1 = flips[0, :]
        crand = flips[np.random.randint(0, len(flips)), :]
        cmin = flips[np.argmin(np.sum(flips, axis=1)), :]
        v1s.append(np.mean(c1))
        vrands.append(np.mean(crand))
        vmins.append(np.mean(cmin))
    averages = np.mean((v1s, vrands, vmins), axis=1)
    print("v1: {}, vrand: {}, vmin: {}".format(*averages))
    return v1s, vrands, vmins
    # return averages


def norm(u, v):
    return np.abs(u - v)


def hoeffding_RHS(epsilon, N):
    return 2 * np.exp(-2 * epsilon**2 * N)


def question2():
    v1, vrand, vmin = question1()
    u = 0.5
    epsilon = np.arange(0, 0.6, 0.1)
    plt.plot(epsilon, hoeffding_RHS(epsilon, 10), label='RHS of Hoeffding')
    plt.plot(epsilon, np.repeat(norm(v1, u), 6), label='v1')
    plt.plot(epsilon, np.repeat(norm(vrand, u), 6), label='vrand')
    plt.plot(epsilon, np.repeat(norm(vmin, u), 6), label='vmin')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # question7()
    # question9()
    question2()
