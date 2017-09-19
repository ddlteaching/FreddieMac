#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:04:20 2017

@author: abhijit
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.random.uniform(0, 10, 200)
y = 4 + np.sin(x/3) + np.random.normal(0,.2,200)
plt.plot(x,y, '.')

p = x

res = y - p

x = x[:,np.newaxis]
from sklearn.linear_model import LinearRegression
lrmodel = LinearRegression(fit_intercept=True)
lrmodel.fit(x, res)

p2 = lrmodel.predict(x)


plt.plot(x,y, '.')
plt.plot(x,p,'.')
plt.plot(x, p+p2, '.')

#==============================================================================
# 
#==============================================================================
indx=np.argsort(x, axis=0).ravel()
x = x[indx,:]
y = y[indx]
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth=8, min_samples_leaf = 5)

x1=np.append(x, np.array([4,6]))[:,np.newaxis]
y1= np.append(y, np.array([3.6,3.9]))
indx1=np.argsort(x1, axis=0).ravel()
x1 = x1[indx1,:]
y1 = y1[indx1]

mod1 = dt.fit(x1, y1)
p1 = mod1.predict(x1)
res1 = y1 - p1

mod2 = dt.fit(x1, y1, sample_weight=np.abs(res1))
p2 = mod2.predict(x1)
res2 = y1 - p2

mod3 = dt.fit(x1, y1, sample_weight=np.abs(res2))
p3 = mod3.predict(x1)



plt.plot(x1,y1,'.')
plt.plot(x1, p1, lw=3 )
plt.plot(x1, p2, lw=3 )
plt.plot(x1, p3, lw=3 )


mod2a = dt.fit(x1, res1)
p2a = mod2a.predict(x1)
res2a = y1 - (p1+p2a)
mod3a = dt.fit(x1, res2a)
p3a = mod3a.predict(x1)

plt.plot(x1, y1, '.')
plt.plot(x1, p1)
plt.plot(x1, p1+p2a)
plt.plot(x1, p1+p2a+p3a)

from sklearn.ensemble import RandomForestRegressor
rfm = RandomForestRegressor(n_estimators = 100, max_depth=5, min_samples_leaf=5)
rfm.fit(x1, y1)
p_rfm = rfm.predict(x1)

plt.plot(x1, y1, '.')
plt.plot(x1, p_rfm, lw=3)
plt.plot(x1, p1+p2a, lw=3)
#==============================================================================
# Random Forest vs GBM Deathmatch
#==============================================================================

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

boston = load_boston()
X, y = boston['data'], boston['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

rfm = RandomForestRegressor(n_estimators=100, max_depth = 5, min_samples_leaf = 5)
gbm = GradientBoostingRegressor(n_estimators=100, criterion='mse', max_depth = 5, min_samples_leaf=5)

rfm.fit(X_train, y_train)
gbm.fit(X_train, y_train)

p_rfm = rfm.predict(X_test)
p_gbm = gbm.predict(X_test)

plt.plot(y_test, p_rfm, '.', label='Random Forest')
plt.plot(y_test, p_gbm, '.', label = 'Boosting')
plt.legend(loc='best')

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, p_rfm)
mean_squared_error(y_test, p_gbm)

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.datasets import load_breast_cancer
breast = load_breast_cancer()
X, y = breast['data'], breast['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

rfc = RandomForestClassifier(n_estimators = 100, max_depth=5, min_samples_leaf = 5)
gbc = GradientBoostingClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, criterion='mse')

rfc.fit(X_train, y_train)
gbc.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, roc_auc_score
accuracy_score(y_test, rfc.predict(X_test))
accuracy_score(y_test, gbc.predict(X_test))
roc_auc_score(y_test, rfc.predict(X_test))
roc_auc_score(y_test, gbc.predict(X_test))

from sklearn.linear_model import LogisticRegression
logc = LogisticRegression(fit_intercept=True)
logc.fit(X_train, y_train)
accuracy_score(y_test, logc.predict(X_test))
roc_auc_score(y_test, logc.predict(X_test))

from sklearn.model_selection import GridSearchCV

param_grid={'n_estimators': range(20,101,20),
            'max_depth': [2,4,6,8,10],
            'min_samples_leaf': [1, 5, 10, 20]}

selector_gb = GridSearchCV(gbc, param_grid, cv=5)
selector_gb.fit(X, y)
selector_gb.best_params_

selector_rf = GridSearchCV(rfc, param_grid, cv=5)
selector_rf.fit(X,y)
selector_rf.best_params_

gbm_opt = selector_gb.best_estimator_
rfm_opt = selector_rf.best_estimator_

gbm_opt.fit(X_train, y_train)
rfm_opt.fit(X_train, y_train)

accuracy_score(y_test, gbm_opt.predict(X_test))
accuracy_score(y_test, rfm_opt.predict(X_test))
