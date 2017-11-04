import synthetic
import numpy as np


class LogisticRegression(object):
    def __init__(self, learning_rate=0.01):
        self._w = np.zeros([3, 1])
        self.lr = learning_rate
        self._line = None
        self._epochs = 0

    @property
    def w(self):
        return self._w

    @property
    def line(self):
        return self._line

    @property
    def epochs(self):
        return self._epochs

    def fit(self, train, labels):
        assert train.shape[0] == 3
        X = train
        y = labels
        initial_weights = np.copy(self.w)
        while True:
            random_idx = np.random.permutation(range(100))
            for idx in random_idx:
                random_point = X[:, idx]
                label = y[:, idx]
                g = self.gradient(label, random_point)
                self._w -= self.lr * g
            self._epochs += 1
            if self.termination_condition(initial_weights):
                self._line = self.w / self.w[-1]
                break
            initial_weights = np.copy(self.w)

    def termination_condition(self, initial_weights):
        return np.linalg.norm(self.w - initial_weights) < 0.01

    def predict(self, X):
        return self.w.T.dot(X)

    def sigmoid(self, logits):
        return 1 / (1 + np.exp(-logits))

    def gradient(self, label, random_point):
        return (-(label * random_point) / (1 + np.exp(label * random_point.dot(self.w)))).reshape(3, 1)

    def cross_entropy(self, x, y):
        return np.log(1 + np.exp(-(y * self.w.T.dot(x))))


def out_of_sample_error(model, data):
    X_train = data.X_train
    y_train = data.y_train
    X_test = data.X_test
    y_test = data.y_test
    num_test_points = y_test.shape[1]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    for i in range(num_test_points):
        cross_entropy_error = model.cross_entropy(X_test[:, i], predictions[:, i])
    epochs = model.epochs
    return cross_entropy_error, epochs


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=5)
    num_train_points = 100
    num_test_points = 1000
    out_of_sample_scores = []
    for _ in range(100):
        model = LogisticRegression()
        data = synthetic.Data(num_train_points, num_test_points)
        error, epochs = out_of_sample_error(model, data)
        out_of_sample_scores.append((error, epochs))
    print("Out of sample error and average steps: ",
            np.mean(out_of_sample_scores, axis=0))
