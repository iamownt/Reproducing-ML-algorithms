import numpy as np

class LR():
    "create a LogisticRegression classifier!"
    def __init__(self, iterations = 5000, learning_rate=0.01, fit_process=False, C=0.01):
        self.iterations = 5000
        self.fit_process = fit_process
        self.eta = learning_rate
        self.epsilon = 1e-7
        self.C = C
    
    def sigmoid(self, w, X):
        return 1/(1 + np.exp(-X.dot(w)))
    
    def I_func(self, arr):
        for i in range(len(arr)):
            if arr[i] >= 0.5:
                arr[i] = 1
            else:
                arr[i] = 0
        return arr
    
    def fit(self, X, y):
        """expect X a numpy array with shape(m, n), y a numpy array with shape(m,)"""
        m = X.shape[0]
        n = X.shape[1]
        w = np.r_[np.random.randn(n, 1), np.array([[1]])]
        X_train = np.concatenate((X, np.ones((m, 1))), axis=1)
        y_train = y.reshape(-1, 1)
        best_loss = np.infty
        
        for i in range(self.iterations):
            y_prob = self.sigmoid(w, X_train)
            #y_pred = self.I_func(y_prob)
            error = y_prob - y_train
            l2 = 0.5 * np.sum(np.square(w[:-1]))
            gradients = 1/m * X_train.T.dot(error) + np.r_[self.C*w[:-1],np.zeros([1, 1])]
            #gradients = 1/m * X_train.T.dot(error)
            
            loss = -np.mean(y_train*np.log(y_prob+self.epsilon) + (1-y_train)*np.log(1-y_prob+self.epsilon)) + self.C*l2
            w = w - self.eta*gradients
            if self.fit_process:
                if i % 500 == 0:
                    print('iterations:', i, ' Loss:', loss)
            if loss < best_loss:
                best_loss = loss
            else:
                print('Early Stopping!')
                break
                
        self.w = w
        
    def predict(self, X):
        m = X.shape[0]
        X_test = np.concatenate((X, np.ones((m, 1))), axis=1)
        y_test_prob = self.sigmoid(self.w, X_test)
        y_pred = self.I_func(y_test_prob)
        
        return y_pred
    
    def predict_prob(self, X):
        m = X.shape[0]
        X_test = np.concatenate((X, np.ones((m, 1))), axis=1)
        y_test_prob = self.sigmoid(self.w, X_test)
        
        return y_test_prob
        
 from sklearn.iris import load_iris
 from sklearn.model_selection import train_test_spit
 from sklearn.linear_model import LogisticRegression
 # just try the dataset you like best.
 #but the LR what I create just support 2 class classifier...
 
 lr = LR()
 lr_sk = LogisticRegression()
 lr.fit(X, y)
 lr_sk.fit(X, y)
 accuracy_score(lr.predict(X), y)
 accuracy_score(lr.predict(X), y)
..............
 
    
