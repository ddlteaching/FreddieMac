#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 04:54:52 2017

Stacking2

@author: abhijit
"""

## IPython magics
%matplotlib inline
%cd ~/ARAASTAT/Teaching/FreddieMacFinal/data

## Preamble
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

#==============================================================================
# King County Housing Data
#==============================================================================

kcdata = pd.read_csv('kc_house_data.csv')
kcdata.head()
kcdata.columns

kcdata = kcdata.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13,14,19,20]]
kcdata = pd.get_dummies(kcdata, columns = ['bedrooms','bathrooms','floors','waterfront',
                                           'condition','grade'])


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_predict
# from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

models = {'rf': RandomForestRegressor(max_depth=4, min_samples_leaf = 10, n_estimators=100),
          'gbt': GradientBoostingRegressor(max_depth=4, min_samples_leaf = 10, n_estimators=100),
          'lr': LinearRegression(),
          'knn': KNeighborsRegressor(n_neighbors=10)}

X_train, X_test, y_train, y_test = train_test_split(kcdata.drop(['price'], axis=1), kcdata['price'],
                                                                test_size = 0.3)
y_train, y_test = np.log(y_train), np.log(y_test)

# Training

preds_train = {}
preds_test = {}
errs = {}
for k in models.keys():
    preds_train[k] = cross_val_predict(models[k], X_train, y_train, cv=5)
    trained = models[k].fit(X_train,y_train)
    preds_test[k] = trained.predict(X_test)
    errs[k] = np.sqrt(mean_squared_error(y_test, preds_test[k]))

metadf_train = pd.DataFrame(preds_train)
metadf_test = pd.DataFrame(preds_test)

## Stacking
### Version 1

# stacked_mod = XGBRegressor(n_estimators=150, max_depth=4)
# stacked1 = stacked_mod.fit(metadf_train, y_train)
# preds1_a = stacked1.predict(metadf_test)
#
# ### Version 2
#
# X_train.index = pd.Index(range(X_train.shape[0]))
# X_test.index = pd.Index(range(X_test.shape[0]))
#
# metadf_train2 = pd.concat([X_train, metadf_train], axis=1)
# metadf_test2 = pd.concat([X_test, metadf_test], axis=1)
#
# stacked2 = stacked_mod.fit(metadf_train2, y_train)
# preds2_a = stacked2.predict(metadf_test2)
#
# np.sqrt(mean_squared_error(y_test, preds1_a))
# np.sqrt(mean_squared_error(y_test, preds2_a))

#==============================================================================

## Stacking
### Version 1

stacked_mod = RandomForestRegressor(n_estimators = 500, max_depth = 4, min_samples_leaf=20)
stacked1 = stacked_mod.fit(metadf_train, y_train)
preds1_b = stacked1.predict(metadf_test)

### Version 2

X_train.index = pd.Index(range(X_train.shape[0]))
X_test.index = pd.Index(range(X_test.shape[0]))

metadf_train2 = pd.concat([X_train, metadf_train], axis=1)
metadf_test2 = pd.concat([X_test, metadf_test], axis=1)

stacked2 = stacked_mod.fit(metadf_train2, y_train)
preds2_b = stacked2.predict(metadf_test2)

np.sqrt(mean_squared_error(y_test, preds1_b))
np.sqrt(mean_squared_error(y_test, preds2_b))

#==============================================================================

## Ensemble average
preds1_c = metadf_test.mean(axis=1)
np.sqrt(mean_squared_error(y_test, preds1_c))
