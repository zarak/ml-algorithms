from lfd.models.linear import LinearRegression
import pandas as pd
import numpy as np


def non_linear_transformation(X):
    # num_examples = X.shape[0]
    feature_1 = np.expand_dims(X[:, 1]**2, axis=1)
    feature_2 = np.expand_dims(X[:, 2]**2, axis=1)
    feature_3 = np.expand_dims(X[:, 1] * X[:, 2], axis=1)
    feature_4 = np.expand_dims(np.abs(X[:, 1] - X[:, 2]), axis=1)
    feature_5 = np.expand_dims(np.abs(X[:, 1] + X[:, 2]), axis=1)
    return np.hstack((X, feature_1, feature_2, feature_3, feature_4, feature_5))


def accuracy(labels, preds):
    return np.mean(labels == preds)


def error_rate(labels, preds):
    return np.mean(labels != preds)


def question2(reg):
    lm = LinearRegression(reg)
    data_in = np.fromfile('../data/week6/in.dta', sep=' ').reshape(-1, 3)
    data_out = np.fromfile('../data/week6/out.dta', sep=' ').reshape(-1, 3)
    num_in = data_in.shape[0]
    num_out = data_out.shape[0]
    X_train = np.hstack((np.ones((num_in, 1)), data_in[:, :2]))
    # print(X_train.shape)
    y_train = data_in[:, 2].reshape(1, -1)
    # print(y_train.shape)
    X_test = np.hstack((np.ones((num_out, 1)), data_out[:, :2]))
    # print(X_test.shape)
    y_test = data_out[:, 2].reshape(1, -1)
    # print(y_test.shape)
    Z_train = non_linear_transformation(X_train)
    # print(Z_train.shape)
    Z_test = non_linear_transformation(X_test)
    # print( Z_test.shape)
    lm.fit(Z_train.T, y_train)
    train_preds = lm.predict(Z_train.T)
    test_preds = lm.predict(Z_test.T)
    print(error_rate(y_train, train_preds), error_rate(y_test, test_preds))
    return error_rate(y_train, train_preds), error_rate(y_test, test_preds)


def question3():
    k = -3
    l2 = 10**k
    question2(l2)


def question4():
    k = 3
    l2 = 10**k
    question2(l2)


def question5():
    values_of_k = range(-2, 3)
    E_out = {}
    for k in values_of_k:
        l2 = 10**k
        _, E_out[k] = question2(l2)
    best_k = min(E_out, key=E_out.get)
    print("The best value of k is", best_k)
    print("The corresponding out of sample error is", E_out[best_k])


if __name__ == "__main__":
    # question2()
    # question3()
    # question4()
    question5()
