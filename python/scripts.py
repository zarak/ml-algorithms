import linear
import perceptron
import synthetic
import numpy as np

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
    v1s = []
    vrands = []
    vmins = []
    for _ in range(10000):
        flips = np.random.randint(0, 2, (1000, 10))
        c1 = flips[0, :]
        crand = flips[np.random.randint(0, len(flips)), :]
        cmin = flips[np.argmin(np.sum(flips, axis=1)), :]
        v1s.append(np.mean(c1))
        vrands.append(np.mean(crand))
        vmins.append(np.mean(cmin))
    averages = np.mean((v1s, vrands, vmins), axis=1)
    print("v1: {}, vrand: {}, vmin: {}".format(*averages))


if __name__ == "__main__":
    # question7()
    # question9()
    question1()
