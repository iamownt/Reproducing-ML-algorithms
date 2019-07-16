import numpy as np
import pandas as pd

class ID3(object):
    """Create ID3 Decision Tree."""
    def __init__(self, epsilon=0):
        self.epsilon = epsilon
    
    def entropy(self, target_col):
        elements, counts = np.unique(target_col, return_counts=True)
        entro = 0
        for i in range(len(counts)):
            prob = counts[i]/np.sum(counts)
            entro += -prob * np.log2(prob)
        return entro
        
    def InfoGain(self, X, y, split_attri):
        vals, counts = np.unique(X[split_attri], return_counts=True)
        cond_entropy = 0
        for i in range(len(vals)):
            prob = counts[i]/np.sum(counts)
            ind = X[split_attri]==vals[i]
            cond_entropy += prob*self.entropy(y[ind])
        return self.entropy(y) - cond_entropy

    def fit(self, X, y):
        """
        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        try:
            X = pd.DataFrame(X, columns=['feature_%d'% i for i in range(X.shape[1])])
        except:
            raise ValueError("Please input a numpy array, I do not want to write more codes")
        self.features = ['feature_%d'% i for i in range(X.shape[1])]
        y_ori = y
        def tree_re(X, y_ori, features, y, parent_node=None):
            if len(np.unique(y)) <= 1:
                return np.unique(y)[0]
            elif len(X) == 0:
                return np.argmax(np.bincount(y_ori))
            elif len(features) == 0:
                return parent_node
            else:
                parent_node = np.argmax(np.bincount(y))
                item_values = [self.InfoGain(X, y, feature) for feature in features]
                if max(item_values) < self.epsilon:
                    return(tree)
                else:
                    best_feature = features[np.argmax(item_values)]
                    tree = {best_feature:{}}
                    features = [i for i in features if i != best_feature]
                    for value in np.unique(X[best_feature]):
                        sub_data = X.where(X[best_feature] == value).dropna()
                        sub_y = y[X[best_feature] == value]
                        subtree = tree_re(sub_data, y_ori, features, sub_y, parent_node)
                        tree[best_feature][value] = subtree
                
            return(tree)
        self.tree = tree_re(X, y_ori, self.features, y)
        
    def pred(self, query, default=1):
        for key in list(query.keys()):
            if key in list(self.tree.keys()):
                try:
                    result = self.tree[key][query[key]]
                except:
                    return default
                result =self.tree[key][query[key]]
                if isinstance(result, dict):
                    return self.pred(query, result)
                else:
                    return result 
    
    def predict(self, X_test):
        try:
            X_test = pd.DataFrame(X_test, columns=['feature_%d'% i for i in range(X_test.shape[1])])
        except:
            raise ValueError("Please input a numpy array, I do not want to write more codes")
        
        X_test = X_test.to_dict(orient = 'records')
        predictions = np.zeros(len(X_test))
        
        for i in range(len(X_test)):
            predictions[i] = self.pred(X_test[i])
        return predictions
