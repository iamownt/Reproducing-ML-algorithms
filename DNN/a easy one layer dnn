#just one layer DNN
import numpy as np

X1 = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
y1 = np.array([[0, 1, 1, 0]]).T
np.random.seed(1)
weights = 2 * np.random.random((3, 1)) - 1
for i in range(5000):
    output = 1/(1 + np.exp(-np.dot(X1, weights)))
    weights -= 1*np.dot(X1.T ,(output - y1)*output*(1-output))    
print(1/(1 + np.exp(-np.dot(np.array([0, 0, 1]), weights))))
    
import tensorflow as tf

X = tf.placeholder(tf.float32, [None, 3], name='X')
y = tf.placeholder(tf.float32, [None, 1], name='y')
hidden1 = tf.layers.dense(X, 1, activation=tf.nn.sigmoid)
loss = tf.reduce_mean(tf.square(y-hidden1))
training_op = tf.train.AdamOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    for i in range(5000):
        sess.run(training_op, feed_dict={X:X1, y:y1})
    print(sess.run(hidden1, feed_dict={X:X1}))
