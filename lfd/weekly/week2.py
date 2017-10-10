from collections import Counter
from models import linear
from models import perceptron
import synthetic
import numpy as np
import matplotlib.pyplot as plt


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

def hoeffding_LHS(dist):
    counter = Counter(dist)
    freqs = [counter[epsilon] / 100000 for epsilon in np.array(range(11)) / 10]
    probs = [1 - freqs[5] if i == 0 else 1 - np.sum(freqs[5-i:5+i]) for i in range(6)]
    return probs

def hoeffding_RHS(epsilon, N, M=1):
    return 2 * M * np.exp(-2 * epsilon**2 * N)

def question2():
    v1, vrand, vmin = question1()
    u = 0.5
    epsilon = np.arange(0, 0.6, 0.1)
    plt.plot(epsilon, hoeffding_RHS(epsilon, 10), label='RHS of Hoeffding')
    plt.plot(epsilon, hoeffding_LHS(v1), label='v1', linestyle='--',
            marker='s', color='green')
    plt.plot(epsilon, hoeffding_LHS(vrand), label='vrand', linestyle='--',
            marker='x', color='blue', alpha=0.5)
    plt.plot(epsilon, hoeffding_LHS(vmin), label='vmin', linestyle='--', marker='o')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # question7()
    # question9()
    pass
