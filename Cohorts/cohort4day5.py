#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:14:20 2017

@author: abhijit
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pandas as pd

def find_oob(x, n_obs):
    """Description
    x = indices of a bootstrap sample
    n_obs = number of observations in my sample
    
    Returns:
        Indices of OOB sample
    """
    oob = list(set(range(n_obs)).difference(set(x)))
    return(oob)

def get_bootstrap(n_obs, n_boot, rng = np.random.RandomState(20)):
    """
    n_obs = number of observations in sample
    n_boot = number of bootstrap samples to take
    rng = seeding the random number generator
    """
    indx = np.arange(n_obs)
    boot_indx = rng.choice(indx, size = (n_obs, n_boot), replace=True)
    return(boot_indx)

def get_oob(boots, n_obs):
    return([find_oob(x, n_obs) for x in boots.T])

def myRFRegressor(dat, target_var, n_boot = 250, max_feature = 5, rng = np.random.RandomState(35)):
    """
    dat is a pd.DataFrame object
    """
    feature_names = list(dat.columns)
    feature_names.remove(target_var)
    X, y = dat[feature_names], dat[target_var]
    boot_indx = get_bootstrap(X.shape[0], n_boot, rng=rng)
    oob_indx = get_oob(boot_indx, X.shape[0])
    oob_preds = np.zeros_like(boot_indx) - 1
    baseLearner = DecisionTreeRegressor(max_features = max_feature)
    engines = []
    for i in range(n_boot):
        X_boot = X.iloc[boot_indx[:,i],:]
        y_boot = y[boot_indx[:,i]]
        X_oob = X.iloc[np.array(oob_indx[i]),:]
        baseLearner.fit(X_boot, y_boot)
        engines.append(baseLearner)
        oob_preds[np.array(oob_indx[i]),i] = baseLearner.predict(X_oob)
    oob_preds = pd.DataFrame(oob_preds)
    oob_preds[oob_preds==-1] = np.nan
    predictions = oob_preds.agg(np.nanmean, axis=1)
    return({'engines': engines, 'predictions': predictions})


def predictRF(engines, test_dat):
    prediction = np.zeros(test_dat.shape[0])
    for i in range(test_dat.shape[0]):
        x = test_dat.iloc[i,:]
        preds = [machine.predict(x[ np.newaxis,:]) for machine in engines]
        prediction[i] = np.mean(np.array(preds))
    return(prediction)



from sklearn.datasets import load_iris

iris = load_iris()
iris_df = pd.DataFrame(iris['data'], columns = iris['feature_names'])
iris_df['Species'] = iris['target']
train_RF = myRFRegressor(iris_df, 'Species', n_boot=100, max_feature = 3)



#==============================================================================
# Feature importances
#==============================================================================

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
breast = load_breast_cancer()
X, y = breast['data'], breast['target']
clf = RandomForestClassifier(random_state=50, n_estimators = 2000, max_depth = 3, min_samples_leaf= 0.1)
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.3)
clf.fit(X_train, y_train)

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
indices = np.argsort(importances) # Reverse order

import matplotlib.pyplot as plt
plt.figure()
plt.title("Feature importances")
plt.barh(range(X.shape[1]), importances[indices],
       color="r", xerr=std[indices]/np.sqrt(2000), align="center")
plt.yticks(range(X.shape[1]), breast['feature_names'][indices])
plt.ylim([-1, X.shape[1]])

#==============================================================================
# Permutation-based importance
#==============================================================================

from sklearn.metrics import roc_auc_score, accuracy_score
from collections import defaultdict

clf = RandomForestClassifier(random_state = 50, n_estimators = 500)
scores = defaultdict(list)
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
# Data analysis, and categorical variables
#==============================================================================

import os
os.chdir('/Users/abhijit/ARAASTAT/Teaching/FreddieMacFinal/data')


dat = pd.read_csv('adult.data', header=None, 
                  names = ['age','workclass', 'fnlwgt','education', 'edyr','marital', 
                  'occupation', 'relationship','race','sex','capitalgain',
                  'capitalloss','hrsweek','country','income_class'])
testdat = pd.read_csv('adult.test', header=None, skiprows=1,
                      names = ['age','workclass', 'fnlwgt','education','edyr', 'marital', 
                  'occupation', 'relationship','race','sex','capitalgain',
                  'capitalloss','hrsweek','country','income_class'])


X = pd.get_dummies(dat.iloc[:,:-1])
y = dat['income_class']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

clf = RandomForestClassifier(n_estimators = 250)
clf.fit(X, y)

X_test = pd.get_dummies(testdat.iloc[:,:-1])
y_test = le.fit_transform(testdat['income_class'])


#==============================================================================
# There is a problem with not having same number of levels of a predictor. We'll
# concatenate first, then process, then separate the training and test data
#==============================================================================


#==============================================================================
# The coding of income_class is different in test and training
#==============================================================================

testdat['income_class'] = pd.Series([x.replace('.','') for x in testdat['income_class']])

dat_full = dat.append(testdat)
X = pd.get_dummies(dat_full.iloc[:,:-1])
y = le.fit_transform(dat_full['income_class'])

X_train,y_train = X.iloc[:dat.shape[0],:], y[:dat.shape[0]]
X_test, y_test = X.iloc[dat.shape[0]:,:], y[dat.shape[0]:]

#==============================================================================
#  Computing metrics for binary outcomes
#==============================================================================
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, brier_score_loss
from sklearn.metrics import f1_score

rf_class = RandomForestClassifier(warm_start=True, random_state=50, oob_score=True)
rf_reg = RandomForestRegressor(warm_start=True, random_state=50, oob_score=True)

nboot = range(50,500,50)

scores_class = {'oob':[], 'acc':[], 'auc':[], 'precision':[], 'recall':[], 'brier':[], 'f1':[]}
scores_reg = {'oob':[], 'acc':[], 'auc':[], 'precision':[], 'recall':[], 'brier':[], 'f1':[]}

for n_est in nboot:
    print(n_est)
    rf_class.set_params(n_estimators = n_est)
    rf_reg.set_params(n_estimators = n_est)
    
    rf_class.fit(X,y)
    rf_reg.fit(X, y)
    
    p_class = rf_class.predict(X_test)
    p_class_prob = rf_class.predict_proba(X_test)[:,1]
    p_reg = rf_reg.predict(X_test)
    
    scores_class['oob'].append(rf_class.oob_score_)
    scores_class['acc'].append(accuracy_score(y_test, p_class))
    scores_class['auc'].append(roc_auc_score(y_test, p_class_prob))
    scores_class['precision'].append(precision_score(y_test, p_class))
    scores_class['recall'].append(recall_score(y_test, p_class))
    scores_class['brier'].append(brier_score_loss(y_test, p_class_prob))
    scores_class['f1'].append(f1_score(y_test, p_class))
    
    p_reg_class = p_reg > 0.5
    
    scores_reg['oob'].append(rf_reg.oob_score_)
    scores_reg['acc'].append(accuracy_score(y_test, p_reg_class))
    scores_reg['auc'].append(roc_auc_score(y_test, p_reg))
    scores_reg['precision'].append(precision_score(y_test, p_reg_class))
    scores_reg['recall'].append(recall_score(y_test, p_reg_class))
    scores_reg['brier'].append(brier_score_loss(y_test, p_reg))
    scores_reg['f1'].append(f1_score(y_test, p_reg_class))


#==============================================================================
# Calibration
#==============================================================================

from sklearn.calibration import calibration_curve

c_class = calibration_curve(y_test, p_class_prob, n_bins=10)
c_reg = calibration_curve(y_test, p_reg, n_bins=10)

plt.plot([0,1],[0,1], 'k:', label='Perfect calibration')
plt.plot(c_class[1], c_class[0], label ='Classification')
plt.plot(c_reg[1], c_reg[0], label='Regression')
