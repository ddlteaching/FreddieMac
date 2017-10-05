#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 04:22:00 2017

Random forest introduction

@author: Abhijit Dasgupta
Copyright (c) Abhijit Dasgupta, 2017. All rights reserved
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, KFold


########################################
#  Bias-variance tradeoff
########################################


rng = np.random.RandomState(35)
sns.set_style('dark')

x = np.sort(rng.uniform(0,1,size=100))
truth = x - 0.7*x**2 
pred_knn = np.ndarray((100, 1000))
pred_lr = np.ndarray((100,1000))

X = x[:,np.newaxis]
model_knn = KNeighborsRegressor(n_neighbors=3)
model_lr = LinearRegression(fit_intercept=True)

for i in range(1000):
    
    y = truth + rng.uniform(-.1, .1, size=100)
    
    model_knn.fit(X, y)
    model_lr.fit(X,y)
    
    pred_knn[:,i] = cross_val_predict(model_knn, X, y, cv=KFold(3, shuffle=True))
    pred_lr[:,i] = cross_val_predict(model_lr, X, y, cv=KFold(3, shuffle=True))


plt.scatter(x,y, s=20)
plt.xlabel('x'); plt.ylabel('y')
plt.savefig('bvexample.png')



avgpred_knn = pred_knn.mean(axis=1)
avgpred_lr = pred_lr.mean(axis=1)

indx_sort= np.argsort(x)
plt.plot(x[indx_sort], truth[indx_sort], 'k-',linewidth=3 )
plt.scatter(x, avgpred_knn, label = 'kNN regression', alpha=0.5)
plt.scatter(x, avgpred_lr, color='red', label='Linear regression', alpha=0.5)
plt.legend(loc='best')
plt.savefig('avgpred.png')


bias_knn = (avgpred_knn - truth)**2
bias_lr = (avgpred_lr - truth)**2
plt.scatter(x, bias_knn, color='blue', label = 'kNN regression')
plt.scatter(x, bias_lr, color='red', label='Linear regression')
plt.legend(loc='best')
plt.xlabel('x'); plt.ylabel('Squared prediction bias')
plt.savefig('bias.png')


sd_knn = np.sqrt(pred_knn.var(axis=1))
sd_lr = np.sqrt(pred_lr.var(axis=1))

plt.scatter(x, sd_knn,color='blue', s=20, label='kNN regression')
plt.scatter(x, sd_lr, color='red', s=20, label='Linear regression')
plt.legend(loc='best')
plt.xlabel('x'); plt.ylabel('Estimated prediction variance')
plt.savefig('variance.png')

########################################
# The advantage of averaging
########################################

rng = np.random.RandomState(5)

single_pred = np.ndarray((100,100))
ten_pred = np.ndarray((100,100))

for i in range(100):
    y = truth+ rng.uniform(-.4, .4, 100)
    model_lr.fit(X,y)
    single_pred[:,i] = model_lr.predict(X)
    
    tmp = np.ndarray((100,10))
    for j in range(10):
        y = truth + rng.uniform(-.1, .1, 100)
        model_lr.fit(X,y)
        tmp[:,j] = model_lr.predict(X)
    ten_pred[:,i] = tmp.mean(axis=1).ravel()

f,ax = plt.subplots(1,2)
ax[0].plot(x,single_pred,'b')
ax[0].plot(x, single_pred.mean(1),'k',
  x,truth, 'r',linewidth=2)
ax[0].set_title('Single')
ax[1].plot(x,ten_pred,'b')
ax[1].plot(x, ten_pred.mean(1),'k',
  x,truth,'r',linewidth=2)
ax[1].set_title('Avg of 10')
plt.savefig('reduce.png')


########################################
#  Bootstrap
########################################

rng = np.random.RandomState(8)

sampling = np.ndarray((5000,))
boots = np.ndarray((5000,))

first_sample = rng.normal(0,1,50)

for i in range(5000):
    sampling[i] = rng.normal(0,1,50).mean()
    boots[i] = rng.choice(first_sample, size=first_sample.shape, replace=True).mean()

f, ax = plt.subplots(2,sharex=True)
sns.distplot(sampling, ax=ax[0])
ax[0].set_title('Sampling')
sns.distplot(boots, ax=ax[1])
ax[1].set_title('Bootstrapping')

## proprtion of values in a bootstrap sample

prop_unique = np.zeros(10000)
x = np.arange(50)
for i in range(10000):
    prop_unique[i] = len(np.unique(rng.choice(x, size=50)))/50

pd.Series(prop_unique).describe()
sns.distplot(prop_unique)

###########################################
# Bagging a overfit model
###########################################

from sklearn.preprocessing import PolynomialFeatures

rng = np.random.RandomState(37)
y = truth + rng.normal(0, 0.2,100)
poly = PolynomialFeatures(degree=20)
model_over = LinearRegression(fit_intercept=True)
X_poly = poly.fit_transform(X)
model_over.fit(X_poly, y)

f, ax = plt.subplots(1,2,sharey=True)
ax[0].scatter(x, y, s=20)
ax[0].plot(x, model_over.predict(X_poly), 'r-')
ax[0].set_title('Overfit model')


bagged_pred = np.ndarray((100,5000))
indx = np.arange(len(y))
for i in range(5000):
    indx1 = rng.choice(indx, size=100, replace=True)
    X1 = X[np.sort(indx1),:]
    y1 = y[np.sort(indx1)]
    X2 = poly.fit_transform(X1)
    model_over.fit(X2, y1)
    bagged_pred[:,i] = model_over.predict( X2)

bagged_prediction = bagged_pred.mean(1)
ax[1].scatter(x,y,s=20)
ax[1].plot(x, bagged_prediction,'r')
ax[1].set_title('After bagging')
plt.savefig('afterbagging.png')


################################################
# Role of max_features
################################################

from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


boston = load_boston()
X, y = boston['data'], boston['target']

features_at_split = np.arange(2,12,2)
cv_scores = np.zeros(len(features_at_split))
for i, param in enumerate(features_at_split):
    cv_scores[i] = cross_val_score(DecisionTreeRegressor(max_features=param, random_state=2, criterion='mse'), X, y, cv=5).mean()
cv_scores



