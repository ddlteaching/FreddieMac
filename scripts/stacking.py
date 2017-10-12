#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 05:17:38 2017

Stacking

@author: abhijit
"""

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
%matplotlib inline

sns.set_style('darkgrid')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from collections import defaultdict

breast = load_breast_cancer()
X, y = breast['data'], breast['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

models = [RandomForestClassifier(n_estimators=200, max_features=n) for n in range(2,16,2)]
models[0].get_params()['max_features']
from sklearn.model_selection import cross_val_predict

predictions = defaultdict(list)
predictions_test = defaultdict(list)
preds_class = defaultdict(list)

for m in models:
    m.fit(X_train, y_train)
    p = cross_val_predict(m, X_train, y_train, cv=5, method='predict_proba')[:,1]
    n = m.get_params()['max_features']
    predictions[n] = p
    predictions_test[n] = m.predict_proba(X_test)[:,1]
    preds_class[n] = m.predict(X_test)

X_train_new = pd.DataFrame(predictions).values
X_test_new = pd.DataFrame(predictions_test).values

clf2 = RandomForestClassifier(n_estimators=200, random_state=5)
clf2.fit(X_train_new, y_train)
p = clf2.predict_proba(X_test_new)[:,1]
from sklearn.metrics import brier_score_loss

for k in predictions_test:
    print(k, "{0:.4f}".format(brier_score_loss(y_test, predictions_test[k])))

"{0:.4f}".format(brier_score_loss(y_test, p))
