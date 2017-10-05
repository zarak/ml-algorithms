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


if __name__ == "__main__":
    question7()
    question9()
