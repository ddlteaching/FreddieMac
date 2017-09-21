#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:13:11 2017

@author: abhijit
"""


%cd ~/ARAASTAT/Teaching/FreddieMacFinal/scripts/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import xgboost as xgb
from scipy import sparse
import pickle

dtrain  = xgb.DMatrix('../data/agaricus.txt.train')
dtest = xgb.DMatrix('../data/agaricus.txt.test')

params = {'max_depth': 2, 
          'eta' : 0.05,
          'silent' : 1, 
          'min_samples_leaf': 0.1,
          'objective': 'binary:logistic'}
watchlist = [(dtrain,'train'), (dtest, 'eval')]
bst = xgb.train(params, dtrain, num_boost_round=200,
                evals = watchlist)
preds = bst.predict(dtest)

from sklearn.metrics import accuracy_score, roc_auc_score
accuracy_score( dtest.get_label(), preds>0.5)
roc_auc_score( dtest.get_label(), preds>0.5)

bst.save_model('0001.model')
bst.dump_model('dump.raw.txt')

bst2 = xgb.Booster(model_file = '0001.model')
bst2.predict(dtest)

#==============================================================================
# objective function
#==============================================================================

def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1 / (1 + np.exp(-preds))
    grad = preds - labels
    hess = preds*(1-preds)
    return([grad, hess])

bst = xgb.train(params, dtrain, num_boost_round=50,
                evals = watchlist, obj=logregobj)

#==============================================================================
# Regression
#==============================================================================

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
boston = load_boston()
X, y = boston['data'], boston['target']
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size= 0.2)

params = {'max_depth': 2, 
          'eta' : 0.0.5,
          'silent' : 1, 
          'min_samples_leaf': 0.1}

bos_train = xgb.DMatrix(X_train, label=y_train)
bos_test = xgb.DMatrix(X_test)
bos_model = xgb.train(params, bos_train, num_boost_round=100)

err = []
for i in range(100):
    pred = bos_model.predict(bos_test, ntree_limit = i+1)
    err.append(mean_squared_error(y_test, pred))

#==============================================================================
# scikit-learn interface
#==============================================================================

from xgboost import XGBRegressor

xgbmodel = XGBRegressor()
xgbmodel.fit(X_train, y_train)
preds = xgbmodel.predict(X_test)


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

X, y = make_regression(n_samples = 1000, 
                       n_features = 1500, 
                       n_informative=100)


models = [LinearRegression(fit_intercept = True),
          RandomForestRegressor(max_depth=4, n_estimators=100),
          GradientBoostingRegressor(max_depth=4, n_estimators = 100),
          XGBRegressor(max_depth=4, n_estimators=100, subsample = 0.6, 
                       colsample_bytree = 0.3)
    ]

err = []
for m in models:
    m.fit(X, y)
    pred = m.predict(X)
    err.append(mean_squared_error(y, pred))

preds = []
for m in models:
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    preds.append(pred)

preds1 = np.array(preds, order="F").T
newpred = np.mean(preds1, axis=1)

param_grid = {"learning_rate": [0.05],
              'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
clf = GridSearchCV(XGBRegressor(max_depth=4, n_estimators=100),
                   param_grid)
clf.fit(X,y)
results = pd.DataFrame(clf.cv_results_)
plt.plot(results['param_colsample_bytree'], 
         results['mean_test_score'])



from sklearn.datasets import make_classification, make_regression
X,y = make_classification(n_samples=5000, n_features = 100, n_informative = 40, n_redundant=10)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models = [RandomForestClassifier(max_depth=4, n_estimators = 200),
          XGBClassifier(max_depth=4, n_estimators=200)]

err = []
for m in models:
    m.fit(X,y)
    p = m.predict(X)
    err.append(accuracy_score(y,p>0.5))#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 03:25:38 2017

xgboost testing

@author: abhijit
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print( "Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

train = pd.read_csv('train_modified.csv')
target = 'Disbursed'
IDcol = 'ID'

predictors = [x for x in train.columns if x not in [target, IDcol]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

modelfit(xgb1, train, predictors)
