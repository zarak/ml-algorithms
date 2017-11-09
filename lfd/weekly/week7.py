from lfd.models.linear import LinearRegression
import pandas as pd
import numpy as np
import week6


TARGET_VARIABLE_COLUMN = 2
NUM_TRAIN = 25
NUM_VAL = 10


def train_test_split(X, y, train_size):
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    return X_train, X_test, y_train, y_test


def question1():
    data_in, data_out = week6.load_data()

    X = data_in[:, :TARGET_VARIABLE_COLUMN]
    y = data_in[:, TARGET_VARIABLE_COLUMN]

    X_train, X_val, y_train, y_val = train_test_split(X, y, NUM_TRAIN)
    X_train = np.hstack((np.ones((NUM_TRAIN, 1)), X_train))
    X_val = np.hstack((np.ones((NUM_VAL, 1)), X_val))

    Z_train = week6.non_linear_transformation(X_train)
    Z_val = week6.non_linear_transformation(X_val)

    # Reshape y to 2-dimensional array
    y_train = y_train.reshape(1, -1)
    y_val = y_val.reshape(1, -1)

    error_scores = {}
    for k in range(3, 8):
        Z_train_subset = np.copy(Z_train[:, :k + 1])
        Z_val_subset = np.copy(Z_val[:, :k + 1])

        # Take the transpose to put in format expected by LinearRegression
        lm = LinearRegression(l2=0)
        lm.fit(Z_train_subset.T, y_train)

        val_preds = lm.predict(Z_val_subset.T)
        error = week6.error_rate(y_val, val_preds)
        error_scores[k] = error

    print(error_scores)
    return min(error_scores, key=error_scores.get)


if __name__ == "__main__":
    print(question1())
