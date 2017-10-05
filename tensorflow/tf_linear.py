import numpy as np
import synthetic
import tensorflow as tf


def initialize_placeholders(num_features=3):
    """Initializes placeholders for data."""
    X_train = tf.placeholder(shape=(None, num_features), dtype=tf.float32, name="X_train")
    y_train = tf.placeholder(shape=(None, 1), dtype=tf.float32, name="y_train")
    X_test = tf.placeholder(shape=(None, num_features), dtype=tf.float32, name="X_test")
    y_test = tf.placeholder(shape=(None, 1), dtype=tf.float32, name="y_test")
    return X_train, y_train, X_test, y_test


def fit(X_train, y_train):
    """Uses the analytic solution to linear regression to find parameter
    theta."""
    XT = tf.transpose(X_train)
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X_train)), XT),
            y_train)
    return theta


def predict(X_test, theta):
    """Use theta to generate a hypothesis for the target function."""
    return tf.sign(tf.matmul(X_test, theta))


def question6():
    """Estimate the out of sample error Eout by training on 100 points and then
    using the learned model to classify 1000 fresh points."""
    X_train, y_train, X_test, y_test = initialize_placeholders()
    
    theta = fit(X_train, y_train)

    preds = predict(X_test, theta)
    prob = tf.not_equal(preds, y_test)

    probs = []
    with tf.Session() as sess:
        for _ in range(1000):
            d = synthetic.Data(num_train_points=100, num_test_points=1000)
            prob_val = sess.run(prob, feed_dict={X_train: d.X_train.T,
                y_train: d.y_train.T,
                X_test: d.X_test.T,
                y_test: d.y_test.T})
            probs.append(prob_val)
        print(np.mean(probs))


def question8():
    """The target function here is f(x_1, x_2) = sign(x_1**2 + x_2**2 - 0.6)"""
    X_train, y_train, X_test, y_test = initialize_placeholders()
    
    theta = fit(X_train, y_train)

    preds = predict(X_train, theta)
    prob = tf.not_equal(preds, y_train)

    probs = []
    with tf.Session() as sess:
        for _ in range(1000):
            nd = synthetic.NoisyData()
            prob_val = sess.run(prob, feed_dict={X_train: nd.X_train.T,
                y_train: nd.y_train.T,
            })
            probs.append(prob_val)
        print("In sample error with noisy data: ", np.mean(probs))


def question10():
    """The target function here is f(x_1, x_2) = sign(x_1**2 + x_2**2 - 0.6)"""
    X_train, y_train, X_test, y_test = initialize_placeholders(num_features=6)
    
    theta = fit(X_train, y_train)

    preds = predict(X_test, theta)
    prob = tf.not_equal(preds, y_test)

    probs = []
    with tf.Session() as sess:
        for _ in range(1000):
            nd = synthetic.NoisyData()
            Z_train, Z_test = nd.add_features()
            prob_val = sess.run(prob, feed_dict= {X_train: Z_train.T,
                y_train: nd.y_train.T,
                X_test: Z_test.T,
                y_test: nd.y_test.T
            })
            probs.append(prob_val)
        print("Out of sample error with noisy data using transformed features: ", np.mean(probs))


if __name__ == "__main__":
    question8()
    question10()
