import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from pathlib import Path
from collections import Counter


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
    params = dict(
        kernel='poly',
        degree=2,
        C=0.01,
        coef0=1,
        gamma=1
    )
    return params
    

def question2():
    params = default_params()
    X_train, X_test, y_train, y_test = load_data()
    digit2labels_train = generate_digit2labels(y_train)
    for digit in range(0, 10, 2):
        svm = SVC(**params)
        y_train = digit2labels_train[digit]
        svm.fit(X_train, y_train)
        preds = svm.predict(X_train)
        E_in = np.mean(preds != y_train)
        print(f"E_in for {digit} versus all is", E_in)


def question3():
    params = default_params()
    X_train, X_test, y_train, y_test = load_data()
    digit2labels_train = generate_digit2labels(y_train)
    for digit in range(1, 10, 2):
        svm = SVC(**params)
        y_train = digit2labels_train[digit]
        svm.fit(X_train, y_train)
        preds = svm.predict(X_train)
        E_in = np.mean(preds != y_train)
        print(f"E_in for {digit} versus all is", E_in)


def question4():
    params = default_params()
    X_train, X_test, y_train, y_test = load_data()
    digit2labels_train = generate_digit2labels(y_train)

    svm_0 = SVC(**params)
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


def CV10fold(X):
    """Returns the indices for a training and validation set split
    for 10-fold cross-validation"""
    # X should have shape (num observations, num features)
    assert X.shape[1] == 2
    num_examples = X.shape[0]
    fold_size = num_examples // 10
    remainder = num_examples % 10
    shuffled_indices = np.random.permutation(range(num_examples))
    leftover_indices = list(shuffled_indices[-remainder:])
    for fold in range(10):
        val_idx = shuffled_indices[fold*fold_size:(fold+1)*fold_size]
        # train_idx = np.setxord1d(shuffled_indices, val_idx)
        train_idx = list(set(shuffled_indices) - set(val_idx))
        if leftover_indices:
            # print(leftover_indices)
            train_idx.append(leftover_indices.pop())
        train_idx = np.array(train_idx)
        yield train_idx, val_idx


def question7_8():
    params = default_params()
    kernel = params['kernel']
    Q = params['degree']
    coef0 = params['coef0']
    gamma  = params['gamma']

    X, X_test, y, y_test = one_versus_five()
    C_values = [0.0001, 0.001, 0.01, 0.1, 1.0]

    cnt = Counter()
    min_errors = []
    for _ in range(100):
        error_scores = {}
        for C in C_values:
            fold_scores = []
            for i, (train_idx, val_idx) in enumerate(CV10fold(X)):
                # print(f"Processing fold {i}...")
                svm = SVC(kernel=kernel, coef0=coef0, gamma=gamma, degree=Q,
                        C=C)
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_val = X[val_idx]
                y_val = y[val_idx]
                svm.fit(X_train, y_train)
                val_preds = svm.predict(X_val)
                val_error = np.mean(val_preds != y_val)
                fold_scores.append(val_error)
            average_10fold_score = np.mean(fold_scores)
            error_scores[C] = average_10fold_score
        min_errors.append(min(error_scores.values()))
        C_with_min_error = min(error_scores, key=error_scores.get)
        cnt[C_with_min_error] += 1
    print(cnt)
    # For question 8
    print(f"The minimum error over 100 runs is {np.mean(min_errors)}")
