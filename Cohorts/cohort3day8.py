#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 08:22:36 2017

@author: abhijit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import xgboost as xgb
from scipy import sparse
import pickle
import os
os.chdir('/Users/abhijit/ARAASTAT/Teaching/FreddieMacFinal/data')
from sklearn.metrics import accuracy_score


## Early stopping
from sklearn.preprocessing import LabelEncoder
def preprocess(D, target_name):
    X, y = D.drop(target_name, axis=1), D[target_name]
    X = pd.get_dummies(X)
    if(dat.dtypes[target_name]=='object'):
        le = LabelEncoder()
        y = le.fit_transform(y)
    dummy_names = ['f' + str(x) for x in range(X.shape[1])]
    feature_map = dict(zip(dummy_names, list(X.columns)))
    data = xgb.DMatrix(X, feature_names=dummy_names, label = y)
    return([data, feature_map])

dat = pd.read_csv('adult.data', header=None,
                  names = ['age','workclass', 'fnlwgt','education','edyr', 'marital',
                  'occupation', 'relationship','race','sex','capitalgain',
                  'capitalloss','hrsweek','country','income_class'])
testdat = pd.read_csv('adult.test', header=None, skiprows=1,
                      names = ['age','workclass', 'fnlwgt','education', 'edyr','marital',
                  'occupation', 'relationship','race','sex','capitalgain',
                  'capitalloss','hrsweek','country','income_class'])
testdat['income_class'] = pd.Series([x.replace('.','') for x in testdat['income_class']])


dat = dat.drop(['country'], axis=1)
testdat = testdat.drop(['country'], axis=1)

mydtrain, fmap = preprocess(dat, 'income_class')
mydtest, fmap = preprocess(testdat,'income_class')

param = {'max_depth':6,
         'min_samples_leaf': 10,
'eta':1, # Learning rate
'silent':1,
'objective':'binary:logistic' }
param['eval_metric'] = 'error' # misclassification error
watchlist = [ (mydtrain, 'train'), (mydtest,'eval')]

bst = xgb.train(param, mydtrain, num_boost_round=20, evals = watchlist,
    early_stopping_rounds=3)

param['eval_metric'] = 'error@0.7' # Change cutoff for +ve to 0.7
bst = xgb.train(param, mydtrain, num_boost_round=20, evals = watchlist,
    early_stopping_rounds=3)

param['eval_metric'] = 'auc' # Use AUC as metric for evaluation
bst = xgb.train(param, mydtrain, num_boost_round=20, evals = watchlist,
    early_stopping_rounds=3)

from sklearn.metrics import f1_score
def eval_f1(preds, dtrain):
    N = len(preds)
    actual = dtrain.get_label()
    predict = preds > 0.5
    f1 = -f1_score(actual, predict)
    return('f1_metric',f1)

del(param['eval_metric'])
bst = xgb.train(param, mydtrain, num_boost_round=20, evals = watchlist,early_stopping_rounds=3, feval = eval_f1)

#==============================================================================
# Stacking
#==============================================================================
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier

breast = load_breast_cancer()
X, y = breast['data'], breast['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

models = [RandomForestClassifier(n_estimators=200, max_features=n) for n in range(2,16,2)]

from sklearn.model_selection import cross_val_predict

predictions = defaultdict(list)
predictions_test = defaultdict(list)

for m in models:
    m.fit(X_train, y_train)
    p = cross_val_predict(m, X_train, y_train, cv=5, method='predict_proba')[:,1]
    n = m.get_params()['max_features']
    predictions[n] = p
    predictions_test[n] = m.predict_proba(X_test)[:,1]
    #preds_class[n] = m.predict(X_test)

from sklearn.metrics import brier_score_loss
for k in predictions_test:
    print(k, "{0:.4f}".format(brier_score_loss(y_test, predictions_test[k])))
    
X_train_new = pd.DataFrame(predictions).values
X_test_new = pd.DataFrame(predictions_test).values

clf2 = RandomForestClassifier(n_estimators=200, random_state=5)
clf2.fit(X_train_new, y_train)
p = clf2.predict(X_test_new)
"{0:.4f}".format(brier_score_loss(y_test, p))

#==============================================================================
# Example
#==============================================================================
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import pandas as pd
import numpy as np

lupus = pd.read_csv('lupus.csv')
lupus.head()

dat = lupus.loc[:,['age','male','dead','lupus','ventilator']]
dat_vent= dat[dat['ventilator']==1]
dat_novent = dat[dat['ventilator']==0]

rf = RandomForestRegressor(n_estimators=500)
rf_vent = rf.fit(dat_vent.drop('dead', axis=1),dat_vent['dead'])
rf_novent = rf.fit(dat_novent.drop('dead', axis=1), dat_novent['dead'])

xb = xgb.XGBRegressor(n_estimators = 20)
xb_vent = xb.fit(dat_vent.drop('dead', axis=1), dat_vent['dead'])
xb_novent = xb.fit(dat_novent.drop('dead', axis=1), dat_novent['dead'])

from sklearn.model_selection import cross_val_predict
rng = np.random.RandomState(35)
p_vent_dvent = cross_val_predict(xb_vent, dat_vent.drop('dead', axis=1),
    dat_vent['dead'], cv=3)
p_vent_dnovent= xb_vent.predict(dat_novent.drop('dead',axis=1))
p_novent_dnovent = cross_val_predict(xb_novent, dat_novent.drop('dead',axis=1),
    dat_novent['dead'], cv=3)
p_novent_dvent = xb_novent.predict(dat_vent.drop('dead', axis=1))

p_vent = np.concatenate([p_vent_dvent,p_vent_dnovent])
p_novent= np.concatenate([p_novent_dvent, p_novent_dnovent])

eff_vent = p_vent - p_novent
pd.Series(eff_vent).describe()

dat1 = dat_vent.append(dat_novent)
plt.hist(eff_vent)
plt.scatter(dat1['age'], eff_vent)
