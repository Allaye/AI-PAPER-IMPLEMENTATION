#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


class LinearRegression:
    
    def __init__(self, lr=0.005, iters=1000):
        self.lr = lr,
        self.iters = iters
        self.weights = None
        self.bias = None
        
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in len(iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        y_predicted =  np.dot(X, self.weights) + self.bias
        return y_predicted
    
    def mse(self, y_true, y_predicted):
        return np.mean((y_true - y_predicted) **2 )



