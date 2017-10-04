import synthetic
import numpy as np


class LinearRegression(object):
    @property
    def w(self):
        return self._w

    def _activation(self, z):
        return np.sign(z)

    def fit(self, train, labels):
        X = train
        y = labels
        self._w = (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(y)

    def predict(self, test):
        return self._activation(test.dot(self.w))


def out_of_sample_error(model, data):
    X_test = data.X_test.T
    y_test = data.y_test.T

    predictions = model.predict(X_test)
    mismatch_probability = np.mean(predictions != y_test)
    return mismatch_probability


def in_sample_error(model, data):
    X_train = data.X_train.T
    y_train = data.y_train.T
    
    model.fit(X_train, y_train)

    predictions = model.predict(X_train)

    mismatch_probability = np.mean(predictions != y_train)
    return mismatch_probability, model, data


if __name__ == '__main__':
    num_train_points = 30
    num_test_points = 300
    in_sample_scores = []
    out_of_sample_scores = []
    for _ in range(1000):
        model = LinearRegression()
        data = synthetic.Data(num_train_points, num_test_points)
    
        prob, g, data = in_sample_error(model, data)
        in_sample_scores.append(prob)
        out_of_sample_scores.append(out_of_sample_error(g, data))

    print("In sample error: ", np.mean(in_sample_scores))
    print("Out of sample error: ", np.mean(out_of_sample_scores))
