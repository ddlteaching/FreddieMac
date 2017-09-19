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
%matplotlib inline

rng = np.random.RandomState(43)
x = rng.rand(400)
y = 3*x - 2*x**2 + rng.normal(0,0.2, 400)
plt.plot(x,y,'.')


x = x[:,np.newaxis]
from sklearn.linear_model import LinearRegression

lrmodel = LinearRegression(fit_intercept=True)

m1 = lrmodel.fit(x,y)

p1 = lrmodel.predict(x)

p = p1

res = y - p1

m2 = lrmodel.fit(x, res, sample_weight = np.abs(res))

p2 = lrmodel.predict(x)

p = p + p2

res2 = y - p

m3 = lrmodel.fit(x, res2, sample_weight = np.abs(res2))
p3 = lrmodel.predict(x)



plt.plot(x,y,'.')
plt.plot(x, p1, '.')
plt.plot(x, p1+p2,'.')
plt.plot(x, p1+p2+p3,'.')


from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(max_depth=3)

indx = np.argsort(x, 0).ravel()
x = x[indx]
y = y[indx]

tr1 = dt.fit(x,y)

p1 = dt.predict(x)

res1 = y - p1

tr2 = dt.fit(x, res1)

p2 = tr2.predict(x)

res2 = y - (p1+p2)

tr3 = dt.fit(x, res2)

p3 = tr3.predict(x)


plt.plot(x,y,'.')
plt.plot(x, p1, lw=2)
plt.plot(x, p1+p2, lw=2)
plt.plot(x, p1+p2+p3, lw=2)
ra
from sklearn.metrics import mean_squared_error
mean_squared_error(y, p1)
mean_squared_error(y, p1+p2)
mean_squared_error(y, p1+p2+p3)

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

def grad(x,y):
    return(-y * np.exp(-y*x)/(1+np.exp(-y*x)))

dt1 = DecisionTreeRegressor(min_samples_leaf=5)

mod1 = dt1.fit(X,y)
p1 = dt1.predict(X)

res1 = grad(y, p1)

mod2=  dt1.fit(X,res1)
p2 = mod2.predict(X)

p = p1+p2


from sklearn.metrics import accuracy_score

accuracy_score(y, p1 > 0.5)
accuracy_score(y, p1+p2 > 0.5)
