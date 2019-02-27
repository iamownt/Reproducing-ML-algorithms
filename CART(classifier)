import numpy as np

class Tree:
    def __init__(self, col=-1, value=None, trueBranch=None, falseBranch=None,\
                 results=None, data=None):
        self.col = col
        self.value = value
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.results = results
        self.data = data

class CART(object):
    """create the CART tree with pruning"""
    def __init__(self, epsilon=0):
        self.epsilon = epsilon
    
    def gini(self, y):
        elements, counts = np.unique(y, return_counts=True)
        imp = np.sum([np.square(counts[i]/np.sum(counts)) for i in range(len(counts))])
        return 1 - imp
    
    def splitdata(self, X, y, col, value):
        """expect X a numpy array"""
        if (isinstance(value, int) or isinstance(value, float)):
            tuple1 = (X[X[:, col] >= value], y[X[:, col] >= value])
            tuple2 = (X[X[:, col] < value], y[X[:, col] < value])
        else:
            tuple1 = (X[X[:, col] == value], y[X[:, col] == value])
            tuple2 = (X[X[:, col] != value], y[X[:, col] != value])
        return tuple1, tuple2
    
    def buildtree(self, X, y):
        """停止条件结点中样本个数小于预定阀值， gini指数小于预定阀值， 没有更多特征"""
        col_len = X.shape[1]
        best_gini = 0
        best_value = None
        best_set = None
        curr_gini = self.gini(y)
        for col in range(col_len):
            col_values = np.unique(X[:, col])
            for value in col_values:
                tuple1, tuple2 = self.splitdata(X, y, col, value)
                prob = tuple1[0].shape[0]/X.shape[0]
                gini_gain = curr_gini - prob*self.gini(tuple1[1]) - (1 - prob)*self.gini(tuple2[1])
                if gini_gain > best_gini:
                    best_gini = gini_gain
                    best_value = (col, value)
                    best_set = (tuple1, tuple2)
        if gini_gain > self.epsilon:
            trueBranch = self.buildtree(best_set[0][0], best_set[0][1])
            falseBranch = self.buildtree(best_set[1][0], best_set[1][1])
            return Tree(col=best_value[0], value=best_value[1], trueBranch=trueBranch, \
                        falseBranch=falseBranch)
        else:
            return Tree(results=np.unique(y, return_counts=True), \
                        data=(X, y))
    def prune(self, tree, mini_gini=0):
        """普通剪枝叶已无效，采用较为简单的剪枝方法"""
        if tree.trueBranch.results == None:self.prune(tree.trueBranch)
        if tree.falseBranch.results == None:self.prune(tree.falseBranch)
        
        if tree.trueBranch.results != None and tree.falseBranch.results !=None:
            len1 = len(tree.trueBranch.data[0])
            len2 = len(tree.falseBranch.data[0])
            len3 = len1 + len2
            now_gini = (len1/len3)*self.gini(tree.trueBranch.data[1])+(len2/len3)*self.gini(tree.falseBranch.data[1])
            #add_gini = self.gini(np.concatenate((self.gini(tree.trueBranch.data[1], \
             #                                              gini(tree.falseBranch.data[1])))))
            if now_gini < mini_gini:
                r1 = np.concatenate((tree.trueBranch.data[0], tree.falseBranch.data[0]))
                r2 = np.concatenate((tree.trueBranch.data[1], tree.falseBranch.data[1]))
                tree.data = (r1, r2)
                tree.results = np.unique(r2, return_counts=True)
                tree.trueBranch = None
                tree.falseBranch = None
        return tree
    
    def fit(self, X,  y, mini_gini = 0):
        """expect X a numpy array"""
        self.mini_gini = mini_gini
        decision_tree = self.buildtree(X, y)
        prune_tree = self.prune(decision_tree, self.mini_gini)
        self.prune_tree = prune_tree
    
    def classify(self, X_test, tree):
        if tree.results != None:
            return tree.results
        else:
            branch = None
            v = X_test[tree.col]
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.trueBranch
                else:
                    branch = tree.falseBranch
            else:
                if v == tree.value:
                    branch = tree.trueBranch
                else:
                    branch = tree.falseBranch
            return self.classify(X_test, branch)
    
    def predict(self, X_test):
        predictions = np.zeros(len(X_test))
        for i in range(len(X_test)):
            elements, counts = self.classify(X_test[i], self.prune_tree)
            predictions[i] = elements[np.argmax(counts)]
        return predictions
        
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
indx = np.random.permutation(150)
X = X[indx]
y = y[indx]
X_train = X[:140]
y_train = y[:140]
X_test = X[140:]
y_test = y[140:]
tree_sklearn = DecisionTreeClassifier()
tree_own = CART()
tree_own.fit(X_train, y_train)
tree_sklearn.fit(X_train, y_train)
np.mean(tree_own.predict(X_test)==y_test)
np.mean(tree_sklearn.predict(X_test)==y_test)
#output:0.9
#output:0.9
