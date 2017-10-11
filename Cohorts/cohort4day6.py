#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 13:13:28 2017

@author: abhijit
"""

import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
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

y1 = np.sin(x) + np.random.normal(0,0.4, len(x))
p = np.zeros(len(x))
X = x[:,np.newaxis]
dt = DecisionTreeRegressor(max_depth=2)

for i in range(100): # interpolation
    res = y1-p
    tr = dt.fit(X,res)
    p = p + tr.predict(X)
plt.plot(x,y1,'b.',x,p,'r.')



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
def grad(x,y):
    return(-y * np.exp(-y*x)/(1+np.exp(-y*x)))

dt1 = DecisionTreeRegressor(min_samples_leaf=5)

mod1 = dt1.fit(X,y)
p1 = dt1.predict(X)

res1 = grad(y, p1)

#==============================================================================
# Using sklearn
#==============================================================================

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

rng = np.random.RandomState(43)
ns = 500
x = np.linspace(0, 10, ns)
y = np.sin(x) + rng.normal(0,0.3, len(x))
X = x[:,np.newaxis]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3)

gbr = GradientBoostingRegressor(n_estimators = 100,max_depth=2)
gbr.fit(X,y)
p = gbr.predict(X)
plt.plot(x,y,'b.',x,p,'r.')

rng = np.random.RandomState(30)
mse = []
r2 = []
for n in range(100):
    gbr.set_params(n_estimators=n+1)
    if n>0:
        gbr.set_params(warm_start=True)
    else:
        gbr.set_params(warm_start=False)
    gbr.fit(X_train,y_train)
    p = gbr.predict(X_test)
    mse.append(mean_squared_error(y_test, p))
    r2.append(r2_score(y_test, p))

plt.plot(range(100),mse)
gbr.set_params(n_estimators=50, warm_start=False)
gbr.fit(X_train, y_train)
plt.plot(X_test, y_test, 'b.', X_test, gbr.predict(X_test),'r.')
rng = np.random.RandomState(30)


mse_train = []
r2_train = []
for n in range(100):
    gbr.set_params(n_estimators=n+1)
    if n>0:
        gbr.set_params(warm_start=True)
    else:
        gbr.set_params(warm_start=False)
    gbr.fit(X_train,y_train)
    p = gbr.predict(X_train)
    mse_train.append(mean_squared_error(y_train, p))
    r2_train.append(r2_score(y_train, p))

rng = np.random.RandomState(30)
mse_2 = []
r2_2 = []
for n in range(100):
    gbr.set_params(n_estimators=n+1, subsample=0.8)
    if n>0:
        gbr.set_params(warm_start=True)
    else:
        gbr.set_params(warm_start=False)
    gbr.fit(X_train,y_train)
    p = gbr.predict(X_test)
    mse_2.append(mean_squared_error(y_test, p))
    r2_2.append(r2_score(y_test, p))

plt.plot(range(100), mse, 'g', range(100), mse_2, 'r')

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
breast=load_breast_cancer()
X, y = breast['data'], breast['target']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3)

from sklearn.metrics import accuracy_score

gbc = GradientBoostingClassifier(n_estimators=100, max_depth=3)
gbc.fit(X_train, y_train)
p = gbc.predict(X_test)
accuracy_score(y_test, p)

rng = np.random.RandomState(30)
acc = []
for n in range(100):
    gbc.set_params(n_estimators=n+1)
    if n>0:
        gbc.set_params(warm_start=True)
    else:
        gbc.set_params(warm_start=False)
    gbc.fit(X_train,y_train)
    p = gbc.predict(X_test)
    acc.append(accuracy_score(y_test, p))

plt.plot(range(100), acc)
gbc.set_params(n_estimators = 40, warm_start=False)
gbc.fit(X_train,y_train)
p_gbc = gbc.predict(X_test)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=500, max_depth=3)
rfc.fit(X_train,y_train)
p_rf = rfc.predict(X_test)
