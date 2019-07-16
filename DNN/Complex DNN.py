import numpy as np

class NeuralNetwork:
    """expect the input params neurons a list"""
    def __init__(self, neurons=[1], random_state=42):
        """expect X a numpy array"""
        self.neurons = neurons
        self.random_state = random_state
    
    def layers(self, input1):
        np.random.seed(self.random_state)
        self.params = {}
        
        for ind, n_neurons in enumerate(self.neurons):
            if ind == 0:
                input_size = input1.shape[1]
                output_size = n_neurons
            else:
                input_size = output_size
                output_size = n_neurons
            self.params['W'+str(ind+1)] = np.random.randn(input_size, output_size)
            self.params['b'+str(ind+1)] = np.random.randn(1, output_size)
        
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_backward(self, dA, Z):
        dZ = np.array(dA)
        dZ[Z<=0] = 0
        return dZ
    
    def forward_propagation(self, A, W, b):
        Z = np.dot(A, W) + b
        return self.relu(Z), Z
    
    def full_foward_propagation(self, input1):
        self.memory = {'A0':input1}
        A_curr = input1
        for ind, n_neurons in enumerate(self.neurons):
            W_curr = self.params['W'+str(ind+1)]
            b_curr = self.params['b'+str(ind+1)]
            A_curr, Z_curr = self.forward_propagation(A_curr, W_curr, b_curr)
            self.memory['A'+str(ind+1)] = A_curr
            self.memory['Z'+str(ind+1)] = Z_curr
        return A_curr, Z_curr
    
    def backward_propagation(self, A_prev, dA_curr, W_curr, b_curr, Z_curr):
        
        
        dZ_curr = self.relu_backward(dA_curr, Z_curr)
        dW_curr = 1/self.m*np.dot(A_prev.T, dZ_curr)
        db_curr = 1/self.m*np.sum(dZ_curr, axis=0, keepdims=True)
        dA_prev = np.dot(dZ_curr, W_curr.T)
        
        return dA_prev, dW_curr, db_curr
    
    def full_backward_propagation(self, Y, Y_hat):
        self.grads_values = {}
        dA_prev = 2*(Y - Y_hat)
        for layer_prev_ind, n_neurons in reversed(list(enumerate(self.neurons))):
            layer_curr_ind = layer_prev_ind + 1
            dA_curr = dA_prev
            A_prev = self.memory["A"+str(layer_prev_ind)]
            Z_curr = self.memory['Z'+str(layer_curr_ind)]
            W_curr = self.params['W'+str(layer_curr_ind)]
            b_curr = self.params['b'+str(layer_curr_ind)]
            dA_prev, dW_curr, db_curr = self.backward_propagation(A_prev, dA_curr, W_curr, b_curr, Z_curr)
            self.grads_values['dW'+str(layer_curr_ind)] = dW_curr
            self.grads_values['db'+str(layer_curr_ind)] = db_curr
            
    def update(self):
        for ind, neurons in enumerate(self.neurons):
            self.params['W'+str(ind+1)] -= 0.01*self.grads_values['dW'+str(ind+1)]
            self.params['b'+str(ind+1)] -= 0.01*self.grads_values['db'+str(ind+1)]
        
    def train(self, X, y):
        self.X = X
        self.m = X.shape[0]
        self.n = X.shape[1]
        self.y = y
        self.layers(self.X)
        for i in range(5000):
            Y_hat, _ = self.full_foward_propagation(self.X)
            self.full_backward_propagation(Y_hat, self.y)
            self.params_values = self.update()
    
    def predict(self, X_test):
        Y_hat, _ = self.full_foward_propagation(X_test)
        return Y_hat
        
        
        
import tensorflow as tf

X = tf.placeholder(tf.float32, [None, 8], name='X')
y = tf.placeholder(tf.float32, [None, 1], name='y')
hidden1 = tf.layers.dense(X, 6, activation=tf.nn.relu)
hidden2 = tf.layers.dense(hidden1, 1, activation=tf.nn.relu)
#hidden3 = tf.layers.dense(hidden2, 1, activation=tf.nn.relu)
loss = tf.reduce_mean(tf.square(y-hidden2))
training_op = tf.train.AdamOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    for i in range(5000):
        sess.run(training_op, feed_dict={X:X_train, y:y_train})
    a = sess.run(hidden2, feed_dict={X:X_train})
    b = sess.run(hidden2, feed_dict={X:X_test})
    print(mean_squared_error(a, y_train))
    print(mean_squared_error(b, y_test))
