# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 23:48:24 2019

@author: ownt
"""

import numpy as np
from scipy.stats import mode
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit

class RF():
    "Create the RandomForestClassifier and ExtraTreesClassifier"
    def __init__(self, n_estimators=800, max_leaf_nodes=16, *args):
        self.n_estimators = 500
        self.max_leaf_nodes = max_leaf_nodes
        
    def fit(self, X, y):
        tree = DecisionTreeClassifier(max_leaf_nodes=self.max_leaf_nodes)
        self.forest = [clone(tree) for _ in range(self.n_estimators)]
        rs = ShuffleSplit(n_splits=self.n_estimators, test_size=0.8, random_state=42)
        mini_sets = []
        for mini_train_index, mini_test_index in rs.split(X):
            X_train_mini = X[mini_train_index]
            y_train_mini = y[mini_train_index]
            mini_sets.append((X_train_mini, y_train_mini))
        for i in range(len(mini_sets)):
            self.forest[i].fit(mini_sets[i][0], mini_sets[i][1])
    
    def predict(self,X):
        y_pred = np.empty((self.n_estimators, len(X)))
        for index, tree in enumerate(self.forest):
            y_pred[index] = tree.predict(X)
            
        y_pred, n_votes = mode(y_pred, axis=0)
        
        return y_pred
