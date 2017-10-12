#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 04:07:40 2017

Boosting from scratch

@author: abhijit
"""

import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline

#' ## Gradient descent
#'
#' ### Using linear regression models
rng = np.random.RandomState(43)
ns = 100
x = np.linspace(0, 10, ns)
y = 3  + 8*x + + 5*x**2 - 2 * x**3+np.random.normal(0, 1, ns)

a,b = 0,0
learning_rate = 0.0001

def loss(a,b):
    e = np.sum((y - a - b*x)**2)
    return(e)

a,b=0,0
for i in range(5000):
    print(i)
    l1 = loss(a,b)
    dLda = -np.sum(y - a - b*x)
    dLdb = -np.sum(x*(y - a - b*x))
    a1 = a - learning_rate * dLda
    b1 = b - learning_rate * dLdb
    l2 = loss(a1, b1)
    a = a1
    b = b1
    if np.abs(l1-l2) < 0.001:
        break

#' ### Using decision trees
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(max_depth=2)
x = x[:, np.newaxis]
dt.fit(x,y)
p = dt.predict(x)

plt.scatter(x,y)
plt.scatter(x,p, c='red')

res1 = y-p
tr2 = dt.fit(x, res1)
p1 = p + dt.predict(x)
plt.scatter(x,y)
plt.scatter(x,p, c='red')

res2 = y - p1
tr3 = dt.fit(x, res2)
p2 = p+tr3.predict(x)
plt.scatter(x,y)
plt.scatter(x,p, c='red')

res3 = y-p2
tr4 = dt.fit(x,res3)
p3 = p2 + tr4.predict(x)
plt.scatter(x,y)
plt.scatter(x,p, c='red')

res4 = y-p3
tr5 = dt.fit(x, res4)
p4 = p3+tr5.predict(x)
plt.scatter(x,y)
plt.scatter(x,p, c='red')


from sklearn.metrics import mean_squared_error
mean_squared_error(y, p1)
mean_squared_error(y, p2)
mean_squared_error(y, p3)

p = np.zeros_like(y)
for i in range(10):
    res = y - p
    tr = dt.fit(x, res)
    p = p + tr.predict(x)
    print(mean_squared_error(y,p))



#==============================================================================
# Binary data
#==============================================================================

from sklearn.datasets import load_breast_cancer
breast  = load_breast_cancer()
X = breast['data']
y = breast['target']

#' The logistic loss function is defined to be
#' $$ L(y, p) = \ln (1 + e^{-yp})$$
#' It's gradient function is therefore
#' $$ \grad L(y,p) = -y + e^{-yp}/(1+e^{-yp})$$
def loss_logistic(y, p):
    return(np.log(1 + np.exp(-y * p)))
def grad(y,p):
    return(-y * np.exp(-y*p)/(1+np.exp(-y*p)))

dt1 = DecisionTreeRegressor(min_samples_leaf=5)

p = np.ones(len(y))*0.5
learning_rate = 0.0001
res1 = grad(y, p)
mod1 = dt1.fit(X,res1)
p1 = dt1.predict(X)
pd.Series(p+p1).describe()
p = p+learning_rate*p1
res1 = grad(y, p)

mod2=  dt1.fit(X,res1)
p2 = mod2.predict(X)
pd.Series(p2).describe()
p = p+learning_rate*p2
pd.Series(p).describe()

from sklearn.metrics import accuracy_score

accuracy_score(y, p1 > 0.5)
accuracy_score(y, p1+p2 > 0.5)
