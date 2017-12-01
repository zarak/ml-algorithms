import numpy as np
from pathlib import Path
from lfd.models.linear import LinearRegression


DATA_PATH = Path('../data')


# def one_versus_five(X, y):
    # """Returns data for just the digits 1 and 5, with 1 classified as +1 and
    # 5 as -1"""
    # # def wrapper():
        # # print("Loading 1 versus 5 data...")
        # # X_train, X_test, y_train, y_test = load_data()
        # # train_mask = (y_train == 0) | (y_train == 8)
        # # print(np.sum(train_mask))
        # # test_mask = (y_test == 0) | (y_test == 8)
        # # print(np.sum(test_mask))
        # # y_train_binary = digit_versus_all(0, y_train)
        # # y_test_binary = digit_versus_all(0, y_test)
        # # print(y_train_binary[:20])
        # # return X_train[train_mask, :], X_test[test_mask, :], \
                # # y_train_binary[train_mask, :], y_test_binary[test_mask, :]
    # # return wrapper
    # mask = (y == 0) | (y == 8)
    # y_binary= digit_versus_all(0, y)
    # print(y_binary[mask, :][:20])
    # return X[mask, :], y_binary[mask, :]


def one_versus_five():
    X_train, X_test, y_train, y_test = load_data()
    train_idx = np.where((y_train == 5) | ( y_train == 1 ))
    test_idx = np.where((y_test == 5) | ( y_test == 1 ))
    # True when the label is 1
    y_train_1_vs_5 = np.copy(y_train[train_idx])
    y_test_1_vs_5 = np.copy(y_test[test_idx])
    y_train_1_vs_5[y_train_1_vs_5 == 1] = 1
    y_train_1_vs_5[y_train_1_vs_5 == 5] = -1
    return X_train[train_idx], X_test[test_idx], y_train_1_vs_5, y_test_1_vs_5


# @one_versus_five
def load_data():
    data_train = np.loadtxt(DATA_PATH/'features.train')
    X_train = data_train[:, 1:]
    y_train = data_train[:, 0]
    data_test = np.loadtxt(DATA_PATH/'features.test')
    X_test = data_test[:, 1:]
    y_test = data_test[:, 0]
    return X_train, X_test, y_train, y_test


def digit_versus_all(digit, y):
    """Returns the boolean array of when a label is a particular digit"""
    labels = np.zeros(y.shape)
    labels[y == digit] = 1
    labels[y != digit] = -1
    return labels.reshape(-1, 1)


def generate_digit2labels(y):
    """Creates the dictionary to access the boolean target array by digit"""
    return {digit: digit_versus_all(digit, y) for digit in range(10)}


def non_linear_transformation(X):
    # num_examples = X.shape[0]
    feature_1 = np.expand_dims(X[:, 1] * X[:, 2], axis=1)
    feature_2 = np.expand_dims(X[:, 1]**2, axis=1)
    feature_3 = np.expand_dims(X[:, 2]**2, axis=1)
    return np.hstack((X, feature_1, feature_2, feature_3))


def add_bias(X):
    num_examples = X.shape[0]
    bias = np.ones((num_examples, 1))
    return np.hstack([bias, X])


def question7():
    X_train, X_test, y_train, y_test = load_data()
    X_train = add_bias(X_train)
    X_test = add_bias(X_test)
    digit2labels_train = generate_digit2labels(y_train)

    for digit in range(5, 10):
        lm = LinearRegression(l2=1)
        y_train = digit2labels_train[digit]
        lm.fit(X_train.T, y_train.T)
        preds = lm.predict(X_train.T)
        E_in = np.mean(preds != y_train)
        print(f"E_in for {digit} versus all is", E_in)


def question8():
    X_train, X_test, y_train, y_test = load_data()
    X_train = add_bias(X_train)
    X_test = add_bias(X_test)
    Z_train = non_linear_transformation(X_train)
    Z_test = non_linear_transformation(X_test)
    digit2labels_train = generate_digit2labels(y_train)
    digit2labels_test = generate_digit2labels(y_test)

    for digit in range(0, 5):
        lm = LinearRegression(l2=1)
        y_train = digit2labels_train[digit]
        y_test = digit2labels_test[digit]
        lm.fit(Z_train.T, y_train.T)
        preds = lm.predict(Z_test.T)
        E_out = np.mean(preds != y_test)
        print(f"E_in for {digit} versus all is", E_out)


def question9():
    X_train, X_test, y_train, y_test = load_data()
    X_train = add_bias(X_train)
    X_test = add_bias(X_test)
    Z_train = non_linear_transformation(X_train)
    Z_test = non_linear_transformation(X_test)
    digit2labels_train = generate_digit2labels(y_train)
    digit2labels_test = generate_digit2labels(y_test)

    for digit in range(0, 10):
        X_lm = LinearRegression(l2=1)
        Z_lm = LinearRegression(l2=1)

        y_train = digit2labels_train[digit]
        y_test = digit2labels_test[digit]

        X_lm.fit(X_train.T, y_train.T)
        X_train_preds = X_lm.predict(X_train.T)
        X_test_preds = X_lm.predict(X_test.T)

        Z_lm.fit(Z_train.T, y_train.T)
        Z_train_preds = Z_lm.predict(Z_train.T)
        Z_test_preds = Z_lm.predict(Z_test.T)

        X_E_in = np.mean(X_train_preds != y_train)
        Z_E_in = np.mean(Z_train_preds != y_train)
        X_E_out = np.mean(X_test_preds != y_test)
        Z_E_out = np.mean(Z_test_preds != y_test)
        print(f"E_in for {digit} versus all is", X_E_in)
        print(f"E_out for {digit} versus all is", X_E_out)
        print(f"E_in for {digit} versus all with feature transformation is", Z_E_in)
        print(f"E_out for {digit} versus all with feature transformation is", Z_E_out)


def question10():
    # X_train, X_test, y_train, y_test = load_data()
    X_train, X_test, y_train, y_test = one_versus_five()
    # Make sure that the data has been subset for 1 versus 5
    assert X_train.shape[0] == 1561
    assert X_test.shape[0] == 424
    X_train_with_bias = add_bias(X_train)
    X_test_with_bias = add_bias(X_test)
    Z_train = non_linear_transformation(X_train_with_bias)
    Z_test = non_linear_transformation(X_test_with_bias)
    for l2 in [0.01, 1]:
        Z_lm = LinearRegression(l2=l2)
        Z_lm.fit(Z_train.T, y_train.T)
        Z_train_preds = Z_lm.predict(Z_train.T)
        Z_test_preds = Z_lm.predict(Z_test.T)
        Z_E_in = np.mean(Z_train_preds != y_train)
        Z_E_out = np.mean(Z_test_preds != y_test)
        print(f"E_in for {l2} is", Z_E_in)
        print(f"E_out for {l2} is", Z_E_out)


if __name__ == "__main__":
    # question9()
    question10()
