#!/usr/bin/env python
# coding: utf-8
import torch
# In[ ]:

from torch import nn
import numpy as np


class LinearRegression:
    """
        a class to perform linear regression calculations,  modelling the relationship between a scalar response and one
        or more explanatory variables (also known as dependent and independent variables).
        using the least squares method to minimize the sum of the squares of the differences between the target and the
        predicted values.
        formula: y = mx + b
        m = slope
        b = y-intercept
        x = independent variable
        y = dependent variable
        for more info: https://en.wikipedia.org/wiki/Linear_regression
    """

    def __init__(self, lr=0.005, iters=1000):
        self.lr = lr,
        self.iters = iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias = self.lr * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

    def mse(self, y_true, y_predicted):
        return np.mean((y_true - y_predicted) ** 2)


class LinearRegression2(nn.Module):

    def __init__(self, lr: float = 0.005, iters=1000):
        super().__init__()
        self.lr = lr,
        self.iters = iters
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x: torch.Tensor):
        return self.weights * x + self.bias

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD([self.weights, self.bias], lr=self.lr)
        for epoch in range(self.iters):
            y_predicted = self.forward(X)
            loss = criterion(y_predicted, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


model = LinearRegression2()
print('ss', model.parameters().__next__())
print('params', model.state_dict())
