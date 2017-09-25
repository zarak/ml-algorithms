import numpy as np
import tensorflow as tf
import synthetic

SHAPE = (3, 1)


def create_placeholders():
    X = tf.placeholder(dtype=tf.float32, shape=(3, None), name="X")
    y = tf.placeholder(dtype=tf.float32, shape=(1, None), name="y")
    return X, y


def initialize_weights():
    w = tf.Variable(tf.zeros(SHAPE), dtype=tf.float32, name="w")
    return w


def linear(w, X):
    return tf.matmul(tf.transpose(w), X)


def activation(z):
    return tf.sign(z)


def update_weights(w, random_point):
    random_point = tf.reshape(random_point, (4, 1))
    w = w + random_point[:3] * random_point[3]
    return w


def fit():
    pass

if __name__ == "__main__":
    d = synthetic.Data()
    X = d.X
    y = d.labels(np.sign)

    X_train, y_train = create_placeholders() 
    w = initialize_weights()

    z = linear(w, X_train)
    h = activation(z)

    mask = tf.not_equal(h, y_train)
    data = tf.concat([X_train, y_train], axis=0) 
    # misclassified points
    mis = tf.reshape(tf.boolean_mask(data, tf.tile(mask, [4, 1])), (4, -1))
    num_misclassified = tf.cast(tf.size(mis) / 4, dtype=tf.int32)
    random_index = tf.random_uniform((), dtype=tf.int32, minval=0, maxval=num_misclassified)

    random_point = tf.transpose(mis)[random_index]
    w = update_weights(w, random_point)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        weights, misclassified = sess.run([w, num_misclassified], feed_dict={X_train: X, y_train: y})
        while misclassified != 0:
            print(misclassified)
            sess.run([w, num_misclassified], feed_dict={X_train: X, y_train: y})
