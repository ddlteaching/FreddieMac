#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 08:30:57 2017

Class code, Cohort 1, Day 6

@author: abhijit
"""

import numpy as np
import matplotlib.pyplot as plt
x = np.random.uniform(-1,1,100)
y = 3+4*x + np.random.normal(0,0.2,100)

p1 = np.mean(y)*np.ones(len(y))

res1 = y-p1

x = x[:,np.newaxis]
from sklearn.linear_model import LinearRegression
lrmodel = LinearRegression(fit_intercept=True)
mod1 = lrmodel.fit(x,res1)
p2 = lrmodel.predict(x)

plt.plot(x,y,'.')
plt.plot(x, p1)
plt.plot(x, p1+p2)


y2 = x[:,0]**2 + np.random.normal(0,.2,100)
plt.plot(x,y2,'.')

indx = np.argsort(x, axis=0).ravel()
x = x[indx,:]
y2 = y2[indx]
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth = 5)

mod1 = dt.fit(x,y2)
p1 = mod1.predict(x)
res1 = y2 - p1

mod2 = dt.fit(x, res1)
p2 = mod2.predict(x)
res2 = y2 - p2

mod3 = dt.fit(x, res2)
p3 = mod3.predict(x)

plt.plot(x,y2,'.')
plt.plot(x, p1 , lw=2)
plt.plot(x, p1+p2, lw=2)
plt.plot(x, p1+p2+p3, lw=2)


#==============================================================================
# Adaboost idea
#==============================================================================

# start at mod2

mod2a = dt.fit(x, y2, sample_weight=np.abs(res1))
p2a = mod2a.predict(x)

plt.plot(x,y2,'.')
plt.plot(x, p1)
plt.plot(x, p2a)

#==============================================================================
# Bagging vs Boosting
#==============================================================================

dt = DecisionTreeRegressor(max_depth =4)
dat
## Bootstrap sample
 
dat = pd.DataFrame({'x': x.ravel(), 'y':y})
dat1 = dat.iloc[np.random.choice(range(len(y)),len(y)),:].copy()

mod1bag = dt.fit(dat1['x'][:,np.newaxis], dat1['y'])

dat2 = dat.iloc[np.random.choice(range(len(y)),len(y)),:].copy()
mod2bag = dt.fit(dat2['x'][:,np.newaxis],dat2['y'])

#==============================================================================
# Gradient Boosting
#==============================================================================

from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
boston = load_boston()
X = boston['data']
y = boston['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rfr = RandomForestRegressor(n_estimators=100,max_depth = 5, min_samples_leaf = 10)
gbr = GradientBoostingRegressor(n_estimators=100,max_depth=5, min_samples_leaf=10, criterion='mse')

rfr = rfr.fit(X_train, y_train)
gbr = gbr.fit(X_train, y_train)

predict_rfr = rfr.predict(X_test)
predict_gbr = gbr.predict(X_test)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, predict_rfr)
mean_squared_error(y_test, predict_gbr)

plt.plot(y_test, predict_rfr,'.',label='RF')
plt.plot(y_test, predict_gbr, '.',label='Boost')
plt.plot([10,10],[50,50], 'k:')
plt.legend(loc='best')

plt.plot(rfr.feature_importances_, gbr.feature_importances_,'.')

gbr.oob_improvement_
