import linear
import perceptron
import synthetic
import numpy as np

iterations = []    
for _ in range(1000):
     lm = linear.LinearRegression()
     d = synthetic.Data(10)
     lm.fit(d.X_train, d.y_train)
     p = perceptron.Perceptron(lm.w.T)
     p.fit(d.X_train, d.y_train)
     iterations.append(p.iterations)

print(np.mean(iterations))
