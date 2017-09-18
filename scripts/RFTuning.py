#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 00:39:29 2017

Random forest parameters and tuning

@author: abhijit
"""

import os
os.chdir('/Users/abhijit/ARAASTAT/Teaching/FreddieMacFinal/data')


import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
%matplotlib inline

sns.set_style('darkgrid')
#==============================================================================
# The random forest has 4 main tuning parameters: 
# min_samples_leaf (or min_samples_split),
# max_depth
# max_features (Fraction of features to select for splits)
# n_estimators (No. of bootstraps)
#==============================================================================

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
breast = load_breast_cancer()

X, y = breast['data'], breast['target']


clf = RandomForestClassifier(oob_score=True, n_estimators = 500, random_state=5)
oobs = []
min_leafs = np.linspace(0.01, 0.2, num=21)
for ml in min_leafs:
    clf.set_params(min_samples_leaf = ml)
    clf.fit(X, y)
    oobs.append(clf.oob_score_)

plt.plot(min_leafs, oobs)
plt.xlabel('Percentage data in a leaf')
plt.ylabel('OOB accuracy')
plt.title('Min sample in leaf')
plt.savefig('../present/RFminsample.png')

mtrys = [None,'sqrt','log2']
nboots = range(50,501,50)
clf = RandomForestClassifier(oob_score=True, n_estimators = 500, random_state=5)
oobs = []
for m in mtrys:
    for n in nboots:
        clf.set_params(max_features = m, n_estimators = n)
        clf.fit(X, y)
        oobs.append(clf.oob_score_)
oobs = np.array(oobs).reshape((len(nboots), len(mtrys)), order='F') # by col
oobs = pd.DataFrame(oobs, columns = ['None','sqrt','log2'])

plt.plot(nboots, oobs['None'], label = 'Bagged forest')
plt.plot(nboots, oobs['sqrt'], label = 'Square root')
plt.plot(nboots, oobs['log2'], label = 'Log2')
plt.xlabel('Number of estimators')
plt.ylabel('OOB accuracy')
plt.legend(loc='best')
plt.title('Number of predictors in split')
plt.savefig('../present/mtry.png')
    

depths = [1, 2, 3,5, 10, 20]
nboots = range(50,501, 50)
clf = RandomForestClassifier(oob_score=True, random_state=5)
oobs = []

for d in depths:
    for n in nboots:
        clf.set_params(max_depth = d, n_estimators = n)
        clf.fit(X, y)
        oobs.append(clf.oob_score_)

oobs = np.array(oobs).reshape((len(nboots), len(depths)), order='F') # by col
oobs = pd.DataFrame(oobs, columns = [str(d) for d in depths])

for d in oobs.columns:
    plt.plot(nboots, oobs[d], label=d)
plt.legend(loc='best')
plt.xlabel('Number of estimators')
plt.ylabel('OOB accuracy')
plt.title('Max depth')
plt.savefig('../present/depth.png')


#==============================================================================
# How many estimators?
#==============================================================================

nboots = range(50, 5000, 50)
clf = RandomForestClassifier(random_state=50, warm_start=True, oob_score=True)
oobs = []
for n in nboots:
    clf.set_params(n_estimators = n)
    clf.fit(X,y)
    oobs.append(clf.oob_score_)

plt.plot(nboots, oobs)
plt.xlabel('Number of estimators')
plt.ylabel('OOB accuracy')
plt.savefig('../present/no_est.png')

#==============================================================================
# Accessing individual predictors
#==============================================================================

clf = RandomForestClassifier(random_state=50, n_estimators = 2000, max_depth = 3, min_samples_leaf= 0.1)
clf.fit(X,y)
predictions=np.array([tr.predict(X) for tr in clf]).T
prob_predictions = np.array([tr.predict_proba(X)[:,1] for tr in clf]).T

probs = pd.DataFrame(prob_predictions).apply(np.mean, axis = 1)
probs_se = pd.DataFrame(prob_predictions).apply(np.std, axis=1)

#==============================================================================
# Feature importance. Default version shows mean decrease in Gini after each split
#==============================================================================
X, y = breast['data'], breast['target']
clf = RandomForestClassifier(random_state=50, n_estimators = 2000, max_depth = 3, min_samples_leaf= 0.1)
clf.fit(X,y)
clf.feature_importances_

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances) # Reverse order

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.barh(range(X.shape[1]), importances[indices],
       color="r", xerr=std[indices]/np.sqrt(2000), align="center")
plt.yticks(range(X.shape[1]), breast['feature_names'][indices])
plt.ylim([-1, X.shape[1]])

#==============================================================================
# Permutation-based importance
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from collections import defaultdict
from sklearn.datasets import load_iris
iris = load_iris()
sns.set_style('darkgrid')

clf = RandomForestClassifier(random_state = 50, n_estimators = 500)
scores = defaultdict(list)

X = iris['data']
y = iris['target']


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


print(sorted([(np.round(np.mean(score),4), feat) for
              feat, score in scores.items()], reverse=True))

imps = np.array([np.mean(score) for score in scores.values()]).ravel()
imp_sd = np.array([np.std(score) for score in scores.values()]).ravel()
indx = np.argsort(imps)
names = np.array(list(scores.keys()))

plt.barh(range(X.shape[1]), imps[indx], color='r',
         xerr = imp_sd/np.sqrt(50))
plt.yticks(range(len(indx)), np.array(names)[indx])

