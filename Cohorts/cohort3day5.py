#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 09:38:17 2017

@author: abhijit
"""

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
%matplotlib inline

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
breast = load_breast_cancer()

X, y = breast['data'], breast['target']

from collections import defaultdict

clf = RandomForestClassifier(random_state=50, n_estimators=500)
scores =  defaultdict(list)
names = breast['feature_names']

for i in range(50):
    print(i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    r = clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, r.predict(X_test))
    for j in range(X.shape[1]):
        X_t = X_test.copy()
        np.random.shuffle(X_t[:,j])
        shuff_acc = accuracy_score(y_test, r.predict(X_t))
        scores[names[j]].append((acc-shuff_acc)/acc)

print(sorted([(np.round(np.mean(score), decimals=4), feat) for feat, score in scores.items()], reverse=True))

imps = np.array([np.mean(score) for score in scores.values()]).ravel()
imp_sd = np.array([np.std(score) for score in scores.values()]).ravel()
indx = np.argsort(imps)
names = np.array(list(scores.keys()))

plt.barh(range(X.shape[1]), imps[indx], color='r',
         xerr = imp_sd/np.sqrt(50))
plt.yticks(range(len(indx)), np.array(names)[indx])

#==============================================================================
# Variable selection
#==============================================================================

from sklearn.feature_selection import RFECV
estimator = RandomForestClassifier(n_estimators=200, random_state=5)
selector = RFECV(estimator, cv=5, scoring='roc_auc')
selector.fit(X,y)

#==============================================================================
# Example
#==============================================================================
import os
os.chdir('/Users/abhijit/ARAASTAT/Teaching/FreddieMacFinal/data')


dat = pd.read_csv('adult.data', header=None, index_col=False, names = ['age','workclass', 'fnlwgt','education', 'edyrs','marital', 
                  'occupation', 'relationship','race','sex','capitalgain',
                  'capitalloss','hrsweek','country','income_class'])
testdat = pd.read_csv('adult.test', header=None, skiprows=1,
                      names = ['age','workclass', 'fnlwgt','education', 'marital', 
                  'occupation', 'relationship','race','sex','capitalgain',
                  'capitalloss','hrsweek','country','income_class'])

X = pd.get_dummies(dat.iloc[:,:-1])
y = dat['income_class']


from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
y = le.fit_transform(y)



