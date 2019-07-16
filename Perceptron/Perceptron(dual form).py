class dual_Perceptron():
    
    def __init__(self, learning_rate=1):
        self.learning_rate = learning_rate
        
    def fit(self, X, y):
        m = X.shape[0]
        #n = X.shape[1]
        self.alpha = np.zeros(m)
        self.b = 0
        self.yx = np.array([y[i]*X[i] for i in range(m)])
        any_wrong = True
        count=0
        while any_wrong == True:
            for i in range(m):
                X_1 = X[i]
                y_1 = y[i]
                if y_1*(self.alpha.dot(self.yx).dot(X_1)+self.b) <= 0:
                    self.alpha[i] += self.learning_rate
                    self.b += self.learning_rate*y_1
                    count+=1
            if count == 0:
                any_wrong = False
                
            return None
    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for i in range(len(X)):
            if self.alpha.dot(self.yx).dot(X[i])+self.b <= 0:
                predictions[i] = -1
            else:
                predictions[i] = 1
        return predictions
