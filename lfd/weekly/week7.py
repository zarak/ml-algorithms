from lfd.models.linear import LinearRegression
import pandas as pd
import numpy as np
import week6


TARGET_VARIABLE_COLUMN = 2
NUM_TRAIN = 25
NUM_VAL = 10


def train_test_split(X, y, reverse):
    if reverse:
        X_train = X[-10:]
        y_train = y[-10:]
        X_test = X[:-10]
        y_test = y[:-10]
    else:
        X_train = X[:25]
        y_train = y[:25]
        X_test = X[25:]
        y_test = y[25:]
    return X_train, X_test, y_train, y_test


def question2(reverse=False):
    data_in, data_out = week6.load_data()

    X = data_in[:, :TARGET_VARIABLE_COLUMN]
    y = data_in[:, TARGET_VARIABLE_COLUMN]

    X_test = data_out[:, :TARGET_VARIABLE_COLUMN]
    y_test = data_out[:, TARGET_VARIABLE_COLUMN]
    # Add the bias component of the data
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

    X_train, X_val, y_train, y_val = train_test_split(X, y, reverse=reverse)
    X_train = np.hstack((np.ones((10, 1)), X_train))
    X_val = np.hstack((np.ones((25, 1)), X_val))

    Z_train = week6.non_linear_transformation(X_train)
    Z_val = week6.non_linear_transformation(X_val)
    Z_test = week6.non_linear_transformation(X_test)

    # Reshape y to 2-dimensional array
    y_train = y_train.reshape(1, -1)
    y_val = y_val.reshape(1, -1)
    y_test = y_test.reshape(1, -1)

    validation_error_scores = {}
    test_error_scores = {}
    for k in range(3, 8):
        Z_train_subset = np.copy(Z_train[:, :k + 1])
        Z_val_subset = np.copy(Z_val[:, :k + 1])
        Z_test_subset = np.copy(Z_test[:, :k + 1])

        # Take the transpose to put in format expected by LinearRegression
        lm = LinearRegression(l2=0)
        lm.fit(Z_train_subset.T, y_train)

        val_preds = lm.predict(Z_val_subset.T)
        val_error = week6.error_rate(y_val, val_preds)
        validation_error_scores[k] = val_error

        test_preds = lm.predict(Z_test_subset.T)
        test_error = week6.error_rate(y_test, test_preds)
        test_error_scores[k] = test_error

        error_scores = validation_error_scores, test_error_scores

    print(error_scores)
    min_val_error = min(validation_error_scores, key=validation_error_scores.get)
    min_test_error = min(test_error_scores, key=test_error_scores.get)
    return min_val_error, min_test_error


if __name__ == "__main__":
    print(question2(True))
