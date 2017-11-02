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
#' First let's look at the function $y = x^2$
x=[6]
cur=6

def f(x):
    return(np.array(x)**2)

def grad(x):
    return(2*x)

learning_rate = .9

for i in range(50):
    cur = x[-1]-learning_rate*grad(x[-1])
    x.append(cur)
x
%matplotlib inline
import matplotlib.pyplot as plt

y = np.linspace(-4,4,100)

x = [4]
learning_rate = .8
for i in range(10):
    cur = x[-1] - learning_rate * grad(x[-1])
    x.append(cur)

plt.scatter(x,f(x), s=30, color='red')
plt.plot(x,f(x))
plt.plot(y, f(y),'k')

#' Another more illustrative example on the role of the learning rate ($\gamma$)
#' is using the function $f(x) = x^4 - 3x^3 + 2$, which has two local minima. Starting
#' at 4, we'll reach different minima depending on the learning rate.

def f(x):
    return(x**4 - 3 * x**3 + 2)

def grad(x):
    return(4 * x**3 - 9 * x**2)

y = np.linspace(-3,4,201)
plt.plot(y, f(y))

learning_rate = 0.01
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


#' We can look at the squared error loss as the objective function we want to minimize
def f(a,b):
    return(np.sum((y - a - b*x)**2))

#' The gradient will be defined by the two partial derivatives
def grad_a(a,b):
    return(2*np.sum((y-a-b*x)*(-1)))
def grad_b(a,b):
    return(2*np.sum((y - a - b*x)*(-x)))

#' Now we start with gradient descent on the __paramter space__ (a,b).
a, b = [0], [0]
learning_rate = 0.1
for i in range(10):
    a1 = a[-1] - learning_rate*grad_a(a[-1],b[-1])
    b1 = b[-1] - learning_rate*grad_b(a[-1],b[-1])
    a.append(a1)
    b.append(b1)

fig,ax = plt.subplots(1,1)
ax.scatter(x,y)
for i in range(10):
    ax.plot(x, a[i] + b[i]*x)


#' ### Using decision trees on the linear regression problem
from sklearn.tree import DecisionTreeRegressor

rng = np.random.RandomState(24)
x = rng.standard_normal(100)
y = 3 - 2*x + rng.normal(0,0.4,100)
plt.scatter(x,y)

dt = DecisionTreeRegressor(max_depth=2)
x = x[:, np.newaxis]
dt.fit(x,y)
p = dt.predict(x)

plt.scatter(x,y)
plt.scatter(x,p, c='red')

#' Here too, we are minimizing the squared error loss, so :
def squared_loss(y, p):
    return(np.sum((y - p)**2))
def grad(y, p): # with respect to p
    return(np.sum(-2*(y - p)))

learning_rate = .1
res1 = - learning_rate * grad(y, p)
tr2 = dt.fit(x, res1) # train to learn about the step
p1 = p + dt.predict(x) # Go to the new location
plt.scatter(x,y)
plt.scatter(x,p, c='red')

res2 = - learning_rate * grad(y, p1)
tr3 = dt.fit(x, res2)
p2 = p+tr3.predict(x)
plt.scatter(x,y)
plt.scatter(x,p, c='red')

res3 =  -learning_rate * grad(y,p2)
tr4 = dt.fit(x,res3)
p3 = p2 + tr4.predict(x)
plt.scatter(x,y)
plt.scatter(x,p, c='red')

res4 = - learning_rate * grad(y, p3)
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
from sklearn.tree import DecisionTreeRegressor
breast  = load_breast_cancer()
X = breast['data']
y = breast['target']

#' The logistic loss function is defined to be
#' $$ L(y, p) = p \log(1+e^{-y}) + (1-p)*\log(1+e^y)$$
#' It's gradient function is therefore
#' $$ \grad L(y,p) = p - \frac{1}{1+e^{-y}}$$
def loss_logistic(y, p):
    return(np.sum(p * np.log(1 + np.exp(-y))+ (1 - p) * np.log(1 + np.exp(y))))
def grad(y,p):
    return(


dt1 = DecisionTreeRegressor(min_samples_leaf=5)

p = np.ones(len(y))*0.5 # Starting value
learning_rate = 0.1
res1 =  - learning_rate * grad(y, p)
mod1 = dt1.fit(X,res1)
p1 = dt1.predict(X)
pd.Series(p+p1).describe()
p = p + p1
res1 =  - learning_rate * grad(y, p)

mod2=  dt1.fit(X,res1)
p2 = mod2.predict(X)
pd.Series(p2).describe()
p = p+p2
pd.Series(p).describe()

from sklearn.metrics import accuracy_score

loss_logistic(y, 0.5+p1)
loss_logistic(y, 0.5 + p1 + p2)
accuracy_score(y, np.zeros_like(y))
accuracy_score(y, 0.5+p1 > 0.5)
accuracy_score(y, 0.5+p1+p2 > 0.5)
