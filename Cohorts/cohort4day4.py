#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 13:00:57 2017

@author: abhijit
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()
y = digits['target']==9
X_train, X_test, y_train, y_test = train_test_split(digits['data'], y, test_size=0.25,random_state=0)

from sklearn.dummy import DummyClassifier
dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train,y_train)
pred_most_frequent = dummy_majority.predict(X_test)
dummy_majority.score(X_test, y_test)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
pred_tree  = tree.predict(X_test)
tree.score(X_test,y_test)

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, roc_auc_score

confusion_matrix(y_test, pred_tree)
confusion_matrix(y_test, pred_most_frequent)

tree.predict_proba(X_test)
fpr,tpr, thresholds = roc_curve(y_test, tree.predict_proba(X_test)[:,1])
roc_auc_score(y_test, tree.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'k--')
plt.xlabel('Sensitivity')
plt.ylabel('1-Specificity')

fpr,tpr, thresholds = roc_curve(y_test, dummy_majority.predict_proba(X_test)[:,1])

#==============================================================================
# Bootstrap
#==============================================================================

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

sns.distplot(sampling)
sns.distplot(boots)


## proprtion of values in a bootstrap sample

prop_unique = np.zeros(10000)
x = np.arange(50)
for i in range(10000):
    prop_unique[i] = len(np.unique(rng.choice(x, size=50, replace=True)))/50

###########################################
# Bagging a overfit model
###########################################

from sklearn.preprocessing import PolynomialFeatures

rng = np.random.RandomState(37)
x = np.sort(rng.uniform(0,1,size=100))
truth = x - 0.7*x**2 
y = truth + rng.normal(0, 0.2,100)
poly = PolynomialFeatures(degree=20)
model_over = LinearRegression(fit_intercept=True)
X_poly = poly.fit_transform(X)
model_over.fit(X_poly, y)

f, ax = plt.subplots(1,2,sharey=True)
ax[0].scatter(x, y, s=20)
ax[0].plot(x, model_over.predict(X_poly), 'r-')
ax[0].set_title('Overfit model')

plt.scatter(x,y)
plt.plot(x, model_over.predict(X_poly),'r-')

bagged_pred = np.zeros((100,5000))-1
indx = np.arange(len(y))
for i in range(5000):
    indx1 = rng.choice(indx, size=100, replace=True)
    X1 = X[np.sort(indx1),:]
    y1 = y[np.sort(indx1)]
    X2 = poly.fit_transform(X1)
    model_over.fit(X2, y1)
    indx2 = list(set(indx).difference(set(indx1)))
    X_test = X[np.sort(indx2),:]
    X2_test = poly.fit_transform(X_test)
    bagged_pred[indx2,i] = model_over.predict( X2_test)

bagged_pred[bagged_pred== -1] = np.nan
bagged_prediction = np.nanmean(bagged_pred, axis=1)

f, ax = plt.subplots(1,2,sharey=True)
ax[0].scatter(x, y, s=20)
ax[0].plot(x, model_over.predict(X_poly), 'r-')
ax[0].set_ylim((-1,1))
ax[0].set_title('Overfit model')

ax[1].scatter(x,y,s=20)
ax[1].plot(x, bagged_prediction,'r')
ax[1].set_ylim((-1,1))
ax[1].set_title('After bagging')


from  sklearn.tree import DecisionTreeRegressor
model_over = DecisionTreeRegressor(max_depth=20)
bagged_pred = np.zeros((100,5000))-1
indx = np.arange(len(y))
for i in range(5000):
    indx1 = rng.choice(indx, size=100, replace=True)
    X1 = X[np.sort(indx1),:]
    y1 = y[np.sort(indx1)]
    #X2 = poly.fit_transform(X1)
    model_over.fit(X1, y1)
    indx2 = list(set(indx).difference(set(indx1)))
    X_test = X[np.sort(indx2),:]
#    X2_test = poly.fit_transform(X_test)
    bagged_pred[indx2,i] = model_over.predict( X_test)

bagged_pred[bagged_pred==-1] = np.nan
#==============================================================================
# Role of max_features
#==============================================================================
plt.scatter(x,y,s=20)
plt.plot(x, bagged_prediction,'r')
plt.ylim((-.5,.7))

plt.scatter(x,y, s=20)
plt.plot(x, model_over.predict(X))


#==============================================================================
# Role of max_features
#==============================================================================

from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


boston = load_boston()
X, y = boston['data'], boston['target']

features_at_split = np.arange(2,12,2)
cv_scores = np.zeros(len(features_at_split))
for i, param in enumerate(features_at_split):
    cv_scores[i] = cross_val_score(DecisionTreeRegressor(max_features=param, random_state=2, criterion='mse'), X, y, cv=5).mean()
cv_scores

breast = load_breast_cancer()
X, y = breast['data'], breast['target']

features_at_split = np.arange(2,16,2)
cv_scores = np.zeros(len(features_at_split))
for i, param in enumerate(features_at_split):
    cv_scores[i] = cross_val_score(DecisionTreeRegressor(max_features=param, random_state=2, criterion='mse'), X, y, cv=5).mean()
cv_scores




