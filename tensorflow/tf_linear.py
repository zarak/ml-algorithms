import numpy as np
import synthetic
import tensorflow as tf



X_train = tf.placeholder(shape=(None, 3), dtype=tf.float32, name="X_train")
y_train = tf.placeholder(shape=(None, 1), dtype=tf.float32, name="y_train")
X_test = tf.placeholder(shape=(None, 3), dtype=tf.float32, name="X_test")
y_test = tf.placeholder(shape=(None, 1), dtype=tf.float32, name="y_test")

XT = tf.transpose(X_train)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X_train)), XT),
        y_train)

preds = tf.sign(tf.matmul(X_test, theta))
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
