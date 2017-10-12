#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 08:41:54 2017

@author: abhijit
"""

import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline
from sklearn.tree import DecisionTreeRegressor

#==============================================================================
# Binary data
#==============================================================================

from sklearn.datasets import load_breast_cancer
breast  = load_breast_cancer()
X = breast['data']
y = breast['target']

def loss_logistic(y, p):
    return(np.sum(np.log(1 + np.exp(-y * p))))
def grad(y,p):
    return(-y * np.exp(-y*p)/(1+np.exp(-y*p)))

dt = DecisionTreeRegressor(min_samples_leaf=5)
learning_rate = 0.05
p = np.ones(len(y))*0.5
loss_logistic(y, p)
for i in range(50):
    res = -grad(y, p)
    mod = dt.fit(X, res)
    p1 = dt.predict(X)
    p = p+learning_rate*p1
loss_logistic(y, p)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error

rng = np.random.RandomState(43)
ns = 500
x = np.linspace(0, 10, ns)
y = np.sin(x) + rng.normal(0,0.3, len(x))
X = x[:,np.newaxis]

p = np.zeros_like(y)
for i in range(100):
    res = y - p
    tr = dt.fit(X, res)
    p = p + tr.predict(X)
    print(mean_squared_error(y,p))

plt.plot(x, y, 'b.', x, p, 'r')

from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=10)
gbr.fit(X,y)
p = gbr.predict(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
mse=[]
for n in range(100):
    gbr.set_params(n_estimators=n+1)
    if n>0:
        gbr.set_params(warm_start=True)
    else:
        gbr.set_params(warm_start=False)
    gbr.fit(X_train,y_train)
    p = gbr.predict(X_test)
    mse.append(mean_squared_error(y_test, p))

gbr.set_params(n_estimators=40, warm_start=False)
gbr.fit(X_train, y_train)
indx = np.argsort(X_test.ravel())
plt.plot(X_test[indx,:], y_test[indx], 'b.', X_test[indx,:], gbr.predict(X_test[indx,:]),'r')

mean_squared_error(y_test, gbr.predict(X_test))

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 500, max_depth=3)
rfr.fit(X_train, y_train)
mean_squared_error(y_test, rfr.predict(X_test))
plt.plot(X_test[indx,:], y_test[indx], 'b.', X_test[indx,:], rfr.predict(X_test[indx,:]),'r')

#==============================================================================
# XGBoost
#==============================================================================

import xgboost as xgb
from xgboost import XGBRegressor
%cd ~/ARAASTAT/Teaching/FreddieMacFinal/data/

dtrain = xgb.DMatrix('agaricus.txt.train')
dtest = xgb.DMatrix('agaricus.txt.test')
param = {'max_depth':2,
'eta':1, # Learning rate
'silent':1,
'objective':'binary:logistic' }
num_round = 2
watchlist = [(dtest,'eval'), (dtrain,'train')]
bst = xgb.train(param, dtrain, num_round, evals=watchlist)
preds = bst.predict(dtest)
labels = dtest.get_label()
print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))

bst.save_model('0001.model')
# dump model
bst.dump_model('dump.raw.txt')
# dump model with feature map
#bst.dump_model('dump.nice.txt','featmap.txt')


# save dmatrix into binary buffer
dtest.save_binary('dtest.buffer')
# save model
bst.save_model('xgb.model')
# load model and data in
bst2 = xgb.Booster(model_file='xgb.model')
dtest2 = xgb.DMatrix('dtest.buffer')
preds2 = bst2.predict(dtest2)

#==============================================================================
#  Create your own data
#==============================================================================
from sklearn.preprocessing import LabelEncoder

def preprocess(D, target_name):
    """
    D is a pandas DataFrame, where the target variable is D[target_name]
    """
    X, y = D.drop(target_name), D[target_name]
    X = pd.get_dummies(X)
    if(dat.dtypes[target_name]=='object'):
        le = LabelEncoder()
        y = le.fit_transform(y)
    dummy_names = ['f' + str(x) for x in range(X.shape[1])]
    feature_map = dict(zip(dummy_names, list(X.columns)))
    data = xgb.DMatrix(X, feature_names=dummy_names, label = y)
    return([data, feature_map])

## Example for zip

a = ['a','b','c','d','e']
b = [1,2,3,4,5]
dict(zip(a,b))

# alternatively
d = {}
for i in range(len(a)):
    d[a[i]] = b[i]

#==============================================================================
# Define your own objective function
#==============================================================================

# user define objective function, given prediction, return gradient and second order gradient
# this is log likelihood loss
def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels # Gradient
    hess = preds * (1.0-preds) # Hessian
    return grad, hess

bst1 = xgb.train(param, dtrain, num_round, evals=watchlist, obj = logregobj)


param = {'max_depth':2,
'eta':1, # Learning rate
'silent':1, 'early_stopping_rounds':2,
'objective':'binary:logistic' }
num_round = 25
watchlist = [(dtest,'eval'), (dtrain,'train')]
bst = xgb.train(param, dtrain, num_round, evals=watchlist)
preds = bst.predict(dtest)

#==============================================================================
# sklearn syntax
#==============================================================================

import xgboost as xgb

import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.datasets import load_iris, load_digits, load_boston
rng = np.random.RandomState(31337)

dat = pd.read_csv('../data/adult.data', header=None,
                  names = ['age','workclass', 'fnlwgt','education', 'edyr','marital',
                  'occupation', 'relationship','race','sex','capitalgain',
                  'capitalloss','hrsweek','country','income_class'])
testdat = pd.read_csv('../data/adult.test', header=None, skiprows=1,
                      names = ['age','workclass', 'fnlwgt','education', 'edyr','marital',
                  'occupation', 'relationship','race','sex','capitalgain',
                  'capitalloss','hrsweek','country','income_class'])
testdat['income_class'] = pd.Series([x.replace('.','') for x in testdat['income_class']])

dat = dat.drop(['country'], axis=1)
testdat = testdat.drop(['country'], axis=1)

iris = load_iris()
y = iris['target']
X = iris['data']
kf = KFold(n_splits=2, shuffle=True, random_state=rng)
for train_index, test_index in kf.split(X):
    xgb_model = xgb.XGBClassifier().fit(X[train_index],y[train_index])
    predictions = xgb_model.predict(X[test_index])
    actuals = y[test_index]
    print(confusion_matrix(actuals, predictions))

print("Boston Housing: regression")
boston = load_boston()
y = boston['target']
X = boston['data']
kf = KFold(n_splits=2, shuffle=True, random_state=rng)
for train_index, test_index in kf.split(X):
    xgb_model = xgb.XGBRegressor().fit(X[train_index],y[train_index])
    predictions = xgb_model.predict(X[test_index])
    actuals = y[test_index]
    print(mean_squared_error(actuals, predictions))

print("Parameter optimization")
y = boston['target']
X = boston['data']
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=.3)
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train, early_stopping_rounds=2, eval_set = [(X_test, y_test)])
clf = GridSearchCV(xgb_model,
                   {'max_depth': [2,4,6],
                    'n_estimators': [50,100,200]}, verbose=1)
clf.fit(X,y)

### Adult data
#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#adult_mod = xgb.XGBRegressor()
#adult_mod.fit(dat.drop('income_class')) # To be ctd

             
iris = load_iris()
y = iris['target']
X = iris['data']
rng = np.random.RandomState(35)
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=.3)
param = {}
param['objective'] = 'multi:softmax'
param['eta'] = 0.1 # learning rate
param['max_depth'] = 6
param['silent'] = 1
#param['nthread'] = 4
param['num_class'] = 3

clf = xgb.XGBClassifier(max_depth=3, objective = 'multi:softprob')
clf.fit(X_train, y_train, early_stopping_rounds = 3, eval_set = [(X_test, y_test)])
