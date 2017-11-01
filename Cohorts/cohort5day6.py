#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 08:13:23 2017

@author: abhijit
"""

import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline

def f(x):
    return(np.array(x)**2)

def grad(x):
    return(2*x)

learning_rate = 0.01

x = [4]
for i in range(150):
    cur = x[-1]-learning_rate*grad(x[-1])
    x.append(cur)

y = np.linspace(-4,4,100)
plt.plot(y, f(y))
plt.scatter(np.array(x), f(np.array(x)), s=50, color='red')
plt.plot(np.array(x), f(np.array(x)), color='green')

def f(x):
    return(x**4 - 3 * x**3 + 2)

def grad(x):
    return(4 * x**3 - 9 * x**2)

y = np.linspace(-3,4,201)
plt.plot(y, f(y))

learning_rate=.005
x = [4]
for i in range(10):
    cur = x[-1] - learning_rate * grad(x[-1])
    x.append(cur)

plt.plot(y, f(y))
plt.scatter(np.array(x), f(np.array(x)), s=50, color='red')
plt.plot(np.array(x), f(np.array(x)), color='green')

#' ### Using linear regression models

rng = np.random.RandomState(24)
x = rng.standard_normal(100)
y = 3 - 2*x + rng.normal(0,0.4,100)
plt.scatter(x,y)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x[:, np.newaxis], y)

#' We can look at the squared error loss as the objective function we want to minimize
def f(a,b):
    return(np.sum((y - a - b*x)**2))

#' The gradient will be defined by the two partial derivatives
def grad_a(a,b):
    return(2*np.sum((y-a-b*x)*(-1)))
def grad_b(a,b):
    return(2*np.sum((y - a - b*x)*(-x)))

a, b = [0], [0]
learning_rate = 0.003
for i in range(50):
    a1 = a[-1] - learning_rate*grad_a(a[-1],b[-1])
    b1 = b[-1] - learning_rate*grad_b(a[-1],b[-1])
    a.append(a1)
    b.append(b1)

fig,ax = plt.subplots(1,1)
ax.scatter(x,y)
for i in range(50):
    ax.plot(x, a[i] + b[i]*x)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth=2)

rng = np.random.RandomState(24)
x = rng.standard_normal(100)
y = 3 - 2*x + x**2 + rng.normal(0,0.4,100)
plt.scatter(x,y)

dt.fit(x[:, np.newaxis],y)
p = dt.predict(x[:, np.newaxis])
plt.scatter(x,y)
plt.scatter(x,p, c='red')

def squared_loss(y, p):
    return(np.sum((y - p)**2))
def grad(y, p): # with respect to p
    return(-2*(y - p))

learning_rate = 1
res1 = - learning_rate * grad(y, p)
tr2=dt.fit(x[:, np.newaxis],res1)
p1 = p +tr2.predict(x[:, np.newaxis])
plt.scatter(x,y)
plt.scatter(x,p1, c='red')

res2 = - learning_rate * grad(y, p1)
tr2=dt.fit(x[:, np.newaxis],res2)
p2 = p1 +tr2.predict(x[:, np.newaxis])
plt.scatter(x,y)
plt.scatter(x, p, c='green')
plt.scatter(x,p2, c='red')

from sklearn.metrics import mean_squared_error
p = [np.zeros_like(y)+ np.mean(y)]
for i in range(10):
    res = -0.3*grad(y,p[-1])
    tr = dt.fit(x[:,np.newaxis], res)
    p.append(p[-1]+tr.predict(x[:,np.newaxis]))
    print(mean_squared_error(y,p[-1]))

plt.scatter(x,y)
plt.scatter(x, p[0], c='green')
plt.scatter(x, p[2], c='red')

from sklearn.ensemble import RandomForestRegressor

p_rf = [np.zeros_like(y)+np.mean(y)]
np.random.RandomState(43)
for i in range(50,501,50):
    mod_rf = RandomForestRegressor(n_estimators = i, max_depth=2)
    mod_rf.fit(x[:, np.newaxis], y)
    p_rf.append(mod_rf.predict(x[:, np.newaxis]))

plt.scatter(x, y)
plt.scatter(x, p_rf[0], c='green')
plt.scatter(x, p_rf[10], c='red')
plt.scatter(x, p[9], c='black')