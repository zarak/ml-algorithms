import numpy as np
import tensorflow as tf
import synthetic

SHAPE = (3, 1)
LOGDIR = "tf-logs/"

def create_placeholders():
    X = tf.placeholder(dtype=tf.float32, shape=(3, None), name="X")
    y = tf.placeholder(dtype=tf.float32, shape=(1, None), name="y")
    return X, y


def initialize_weights():
    w = tf.Variable(tf.zeros(SHAPE), dtype=tf.float32, name="w")
    tf.summary.histogram("weights", w)
    return w


def linear(w, X):
    return tf.matmul(tf.transpose(w), X)


def activation(z):
    return tf.sign(z)


def build_graph():
    X_train, y_train = create_placeholders() 
    w = initialize_weights()

    z = linear(w, X_train)
    h = activation(z)

    mask = tf.not_equal(h, y_train)
    data = tf.concat([X_train, y_train], axis=0) 
    # misclassified points
    mis = tf.reshape(tf.boolean_mask(data, tf.tile(mask, [4, 1])), (4, -1))
    num_misclassified = tf.cast(tf.size(mis) / 4, dtype=tf.int32)
    tf.summary.scalar("misclassified", num_misclassified)
    # random_index = tf.random_uniform((), dtype=tf.int32, minval=0,
            # maxval=num_misclassified)

    # random_point = tf.transpose(mis)[random_index]
    random_point = tf.random_shuffle(tf.transpose(mis))[0]

    random_point = tf.reshape(random_point, (4, 1))
    training_op = tf.assign(w, w + random_point[:3] * random_point[3])

    return X_train, y_train, training_op, num_misclassified


if __name__ == "__main__":
    d = synthetic.Data()
    X = d.X
    y = d.labels(np.sign)

    X_train, y_train, training_op, num_misclassified = build_graph()
    init = tf.global_variables_initializer()
    summ = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGDIR)
    with tf.Session() as sess:
        sess.run(init)

        misclassified = sess.run(num_misclassified, feed_dict={X_train: X, y_train: y})
        iterations = 0
        while misclassified >  0:
            weights, s = sess.run(
                    [training_op, summ], feed_dict={X_train: X, y_train: y})
            writer.add_summary(s)
            print(misclassified)
            iterations += 1
            misclassified = sess.run(num_misclassified, feed_dict={X_train: X, y_train: y})
            if misclassified == 0:
                break

        print("Final weights:\n", weights)
        print("Number of iterations:", iterations)
