import numpy as np

class Perceptron():
    def __init__(self, learning_rate = 1):
        self.learning_rate = learning_rate
        
    def sign(self, w, x, b):
        return w.dot(x) + b
    
    def fit(self, X, y):
        
        m = X.shape[0]
        n = X.shape[1]
        self.w = np.ones(n)
        self.b = 1
        any_wrong = True
        
        while any_wrong == True:
            count = 0
            for i in range(m):
                X_1 = X[i]
                y_1 = y[i]
                if y_1 * self.sign(self.w, X_1, self.b) <= 0:
                    count +=1
                    self.w = self.w + self.learning_rate*y_1*X_1
                    self.b = self.b + self.learning_rate*y_1
            if count == 0:
                any_wrong = False
                
        return None
    
    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for i in range(len(X)):
            if self.sign(self.w, X[i], self.b) > 0:
                predictions[i] = 1
            else:
                predictions[i] = -1
        return predictions
            
