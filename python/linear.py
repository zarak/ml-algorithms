import synthetic
import numpy as np

SHAPE = (3, 1)

class LinearRegression(object):
    def __init__(self):
        self._w = np.zeros(SHAPE)
        self._iterations = 0

    @property
    def w(self):
        return self._w

    def _activation(self, z):
        return np.sign(z)

    def fit(self, train, labels):
        X = train
        y = labels
        self._w = y.dot(X.T).dot(np.linalg.inv(X.dot(X.T)))

    def predict(self, test):
        return self._activation(self._w.dot(test))


# TODO: Need to improve API for Data class. Use a method which splits data into
# training and test sets.
def out_of_sample_error(model, training_data, test_data):
    # X_train = test_data.X
    # y_train = training_data.labels(np.sign)
    
    # model.fit(X_train, y_train)
    
    # X_test = synthetic.Data(num_points=1000).X
    # y_test = np.sign(training_data.line.T.dot(X_test))

    # predictions = model.predict(X_test)

    # mismatch_probability = np.mean(predictions != y_test)
    # return mismatch_probability
    pass


def in_sample_error(model, training_data):
    X_train = training_data.X
    y_train = training_data.labels(np.sign)
    
    model.fit(X_train, y_train)

    predictions = model.predict(X_train)

    mismatch_probability = np.mean(predictions != y_train)
    return mismatch_probability, model


if __name__ == '__main__':
    num_points = 100
    in_sample_scores = []
    final_models = []
    for _ in range(1000):
        model = LinearRegression()
        training_data = synthetic.Data(num_points=num_points)
    
        prob, g = in_sample_error(model, training_data)
        in_sample_scores.append(prob)
        final_models.append(g)

    # Have to specify axis for np.mean
    print("In sample error: ", np.mean(in_sample_scores, axis=0))
