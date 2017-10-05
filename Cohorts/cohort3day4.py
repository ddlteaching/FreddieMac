#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:50:18 2017

@author: abhijit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.tree import DecisionTreeRegressor


rng = np.random.RandomState(35)
sns.set_style('dark')

x = np.sort(rng.uniform(0,1,size=100))
truth = x - 0.7*x**2 
y = truth + rng.normal(0, 0.2,100)
X = x[:, np.newaxis]

mod = DecisionTreeRegressor(max_depth=20)
mod.fit(X,y)
plt.scatter(y, mod.predict(X))
plt.scatter(x,y)
plt.plot(x, mod.predict(X))

bagged_pred = pd.DataFrame(np.zeros((100,5000)))
indx = np.arange(len(y))
for i in range(5000):
    indx1 = rng.choice(indx, size=100, replace=True)
    X1 = X[np.sort(indx1),:]
    y1 = y[np.sort(indx1)]
    mod.fit(X1, y1)
    indx2 = list(set(indx).difference(set(indx1)))
    X2 = X[np.sort(indx2),:]
    bagged_pred.loc[indx2,i] = mod.predict(X2)

bagged_pred[bagged_pred==0.0] = np.nan
np.mean(pd.isnull(bagged_pred), axis=1)
np.mean(bagged_pred, axis=1)

plt.scatter(x, y, c='red')
plt.plot(x, mod.predict(X), c='green')
plt.plot(x, np.mean(bagged_pred, axis=1), c='black')

from sklearn.ensemble import BaggedClassifier
