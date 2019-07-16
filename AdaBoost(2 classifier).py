import numpy as np

X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1]).reshape(-1, 1)
    
class AdaBoost_2():
    """create a AdaBoost classifier for 2 classes"""
    def __init__(self,n_estimators=3):
        self.epsilon = 1e-7
        self.n_estimators = n_estimators
    
    def Gx(self, X, y, val):
        flag1 = X <= val
        flag2 = X > val
        y1 = np.zeros(len(X)).reshape(-1, 1)
        tup = np.unique(y[flag1], return_counts=True)
        y1[flag1] = tup[0][np.argmax(tup[1])]
        y1[flag2] = -tup[0][np.argmax(tup[1])]
        return y1
        
    def error_rate(self, w, X, y):
        best_loss = np.infty
        values = np.unique(X.ravel())
        for value in values:
            y_pred = self.Gx(X, y, value)
            loss = np.sum(w*((y_pred!=y).astype(np.int)))
            if loss < best_loss:
                best_loss = loss
                best_value = value
        alpha = 0.5*np.log((1-best_loss)/(best_loss+self.epsilon))
        
        return alpha, best_value
    
    def update_w(self, w, y, alpha, val):
        w_new = w*np.exp(-alpha*y*self.Gx(X, y, val))/np.sum(w*np.exp(-alpha*y*self.Gx(X, y,val)))
        return w_new        
    
    def fit(self, X, y):
        w = np.array([1/len(X) for i in range(len(X))]).reshape(-1, 1)
        self.alpha_lis = []
        self.best_value_lis = []
        for i in range(self.n_estimators):
            alp, bes_val = self.error_rate(w, X, y)
            self.alpha_lis.append(alp)
            self.best_value_lis.append(bes_val)
            w = self.update_w(w, y, alp, bes_val)
        
            
    def predict(self, X):
        y_pred = np.zeros(len(X)).reshape(-1, 1)
        for i in range(self.n_estimators):
            y_pred += self.alpha_lis[i] * self.Gx(X, y, self.best_value_lis[i])
        y_pred = np.sign(y_pred)
        return y_pred
