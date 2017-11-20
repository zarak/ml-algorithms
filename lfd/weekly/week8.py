import numpy as np
from sklearn.svm import SVC
from pathlib import Path


DATA_PATH = Path('/home/z/Documents/ml-algorithms/lfd/data')


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
    return y == digit


def generate_digit2labels(y):
    """Creates the dictionary to access the boolean target array by digit"""
    return {digit: digit_versus_all(digit, y) for digit in range(10)}
    

def default_params():
    """Returns the parameter settings for questions 2 through 4"""
    kernel = 'poly'
    Q = 2
    C = 0.01
    return kernel, Q, C
    

def question2():
    kernel, Q, C = default_params()
    X_train, X_test, y_train, y_test = load_data()
    digit2labels_train = generate_digit2labels(y_train)
    for digit in range(0, 10, 2):
        svm = SVC(kernel=kernel, degree=Q, C=C)
        y_train = digit2labels_train[digit]
        svm.fit(X_train, y_train)
        preds = svm.predict(X_train)
        E_in = np.mean(preds != y_train)
        print(f"E_in for {digit} versus all is", E_in)


def question3():
    kernel, Q, C = default_params()
    X_train, X_test, y_train, y_test = load_data()
    digit2labels_train = generate_digit2labels(y_train)
    for digit in range(1, 10, 2):
        svm = SVC(kernel=kernel, degree=Q, C=C)
        y_train = digit2labels_train[digit]
        svm.fit(X_train, y_train)
        preds = svm.predict(X_train)
        E_in = np.mean(preds != y_train)
        print(f"E_in for {digit} versus all is", E_in)


def question4():
    kernel, Q, C = default_params()
    X_train, X_test, y_train, y_test = load_data()
    digit2labels_train = generate_digit2labels(y_train)

    svm_0 = SVC(kernel=kernel, degree=Q, C=C)
    y_train_0 = digit2labels_train[0]
    svm_0.fit(X_train, y_train_0)

    svm_1 = SVC(kernel=kernel, degree=Q, C=C)
    y_train_1 = digit2labels_train[1]
    svm_1.fit(X_train, y_train_1)

    return np.abs(np.sum(svm_0.n_support_) - np.sum(svm_1.n_support_))


def one_versus_five():
    X_train, X_test, y_train, y_test = load_data()
    train_idx = np.where((y_train == 5) | ( y_train == 1 ))
    test_idx = np.where((y_test == 5) | ( y_test == 1 ))
    # True when the label is 1
    y_train_1_vs_5 = y_train[(y_train == 5) | ( y_train == 1 )] == 1
    y_test_1_vs_5 = y_test[(y_test == 5) | ( y_test == 1 )] == 1
    return X_train[train_idx], X_test[test_idx], y_train_1_vs_5, y_test_1_vs_5


def question5():
    kernel = 'poly'
    X_train, X_test, y_train, y_test = one_versus_five()

    Q = 2
    C_values = [0.001, 0.01, 0.1, 1.0]
    for C in C_values:
        svm = SVC(kernel=kernel, coef0=1, gamma=1, degree=Q, C=C)
        svm.fit(X_train, y_train)
        train_preds = svm.predict(X_train)
        test_preds = svm.predict(X_test)
        E_in = np.mean(y_train != train_preds)
        E_out = np.mean(y_test != test_preds)
        print(f"Number of support vectors for C={C}", np.sum(svm.n_support_))
        print(f"E_in for C={C}", E_in)
        print(f"E_out for C={C}", E_out)
        

def question6():
    kernel = 'poly'
    X_train, X_test, y_train, y_test = one_versus_five()
    
    Q_values = [2, 5]
    C_values = [0.0001, 0.001, 0.01, 1.0]
    for C in C_values:
        for Q in Q_values:
            svm = SVC(kernel=kernel, coef0=1, gamma=1, degree=Q, C=C)
            svm.fit(X_train, y_train)
            train_preds = svm.predict(X_train)
            test_preds = svm.predict(X_test)
            E_in = np.mean(y_train != train_preds)
            E_out = np.mean(y_test != test_preds)
            print(f"E_in for Q={Q} and C={C}", E_in)
            print(f"E_out for Q={Q} and C={C}", E_out)
            print(f"Number of support vectors for Q={Q} and C={C}",
                    np.sum(svm.n_support_))


