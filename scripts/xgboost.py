#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 18:28:27 2017

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


%cd ~/ARAASTAT/Teaching/FreddieMacFinal/scripts/

dtrain = xgb.DMatrix('agaricus.txt.train') # target is embedded. Use dtrain.get_label() to extract
dtest = xgb.DMatrix('agaricus.txt.test')
param = {'max_depth':2, 
'eta':1, # Learning rate
'silent':1, 
'objective':'binary:logistic' }
num_round = 2
watchlist = [(dtest,'eval'), (dtrain,'train')]
bst = xgb.train(param, dtrain, num_round, evals=watchlist)
# make prediction
preds = bst.predict(dtest)
labels = dtest.get_label()
print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))
# This is accuracy error
# 1 - sklearn.metrics.accuracy_score(labels, preds > 0.5)

bst.save_model('0001.model')
# dump model
bst.dump_model('dump.raw.txt')
# dump model with feature map
bst.dump_model('dump.nice.txt','featmap.txt')


# save dmatrix into binary buffer
dtest.save_binary('dtest.buffer')
# save model
bst.save_model('xgb.model')
# load model and data in
bst2 = xgb.Booster(model_file='xgb.model')
dtest2 = xgb.DMatrix('dtest.buffer')
preds2 = bst2.predict(dtest2)

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

#==============================================================================
#  Create your own data
#==============================================================================
from sklearn.preprocessing import LabelEncoder

def preprocess(D, target_name):
    X, y = D.drop(target_name), D[target_name]
    X = pd.get_dummies(X)
    if(dat.dtypes[target_name]=='object'):
        le = LabelEncoder()
        y = le.fit_transform(y)
    dummy_names = ['f' + str(x) for x in range(X.shape[1])]
    feature_map = dict(zip(dummy_names, list(X.columns)))
    data = xgb.DMatrix(X, feature_names=dummy_names, label = y)
    return([data, feature_map])


    
dat = pd.read_csv('../data/adult.data', header=None, 
                  names = ['age','workclass', 'fnlwgt','education', 'marital', 
                  'occupation', 'relationship','race','sex','capitalgain',
                  'capitalloss','hrsweek','country','income_class'])
testdat = pd.read_csv('../data/adult.test', header=None, skiprows=1,
                      names = ['age','workclass', 'fnlwgt','education', 'marital', 
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
'objective':'binary:logistic' ,
'num_round': 20}

watchlist = [(mydtrain, 'train'), (mydtest,'eval')]

bst = xgb.train(param, mydtrain, evals = watchlist)

preds = bst.predict(mydtest)

y_obs = mydtest.get_label()

for i in range(len(20)):
    print(accuracy_score(y_obs, bst.predict(mydtest, ntree_limit = i+1)>0.5))

blah = [preds[y_obs==0], preds[y_obs==1]]
plt.boxplot(blah)

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston, load_breast_cancer

breast = load_breast_cancer()
X, y = breast['data'], breast['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

breast_train, breast_test = xgb.DMatrix(X_train, label=y_train), xgb.DMatrix(X_test,label=y_test)

param = {'max_depth':6, 
         'min_samples_leaf': 10,
'eta':1, # Learning rate
'silent':1, 
'objective':'binary:logistic' ,
'num_round': 10}

breast_bst =xgb.train(param, breast_train)
preds = breast_bst.predict(breast_test)

breast_bst.get_fscore()
for i in range(10):
    print(accuracy_score(y_test, breast_bst.predict(breast_test, ntree_limit=i+1)>0.5))

#==============================================================================
# Multiclass classification 
#==============================================================================

# Download dermatology data from web

dermdata = pd.read_csv('../data/dermatology.data')

X,y = dermdata.iloc[:,:-1], dermdata.iloc[:,-1]-1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

xgtrain = xgb.DMatrix(X_train, label=y_train)
xgtest = xgb.DMatrix(X_test, label=y_test)

param = {}
param['objective'] = 'multi:softmax'
param['eta'] = 0.1 # learning rate
param['max_depth'] = 6
param['silent'] = 1
#param['nthread'] = 4
param['num_class'] = 6
watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
num_round = 5
bst = xgb.train(param, xgtrain, num_round, watchlist );
pred = bst.predict( xgtest );

param['objective'] = 'multi:softprob'
bst = xgb.train(param, xgtrain, num_round, watchlist );
# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
yprob = bst.predict( xgtest ).reshape( y_test.shape[0], 6 )
ylabel = np.argmax(yprob, axis=1)

#==============================================================================
# sklearn syntax
#==============================================================================

import xgboost as xgb

import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.datasets import load_iris, load_digits, load_boston
rng = np.random.RandomState(31337)


print("Iris: multiclass classification")
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
xgb_model = xgb.XGBRegressor()
clf = GridSearchCV(xgb_model,
                   {'max_depth': [2,4,6],
                    'n_estimators': [50,100,200]}, verbose=1)
clf.fit(X,y)
print(clf.best_score_)
print(clf.best_params_)

print("Pickling sklearn API models")
# must open in binary format to pickle
pickle.dump(clf, open("best_boston.pkl", "wb"))
clf2 = pickle.load(open("best_boston.pkl", "rb"))
print(np.allclose(clf.predict(X), clf2.predict(X)))

X = digits['data']
y = digits['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = xgb.XGBClassifier()
clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc",
        eval_set=[(X_test, y_test)])

