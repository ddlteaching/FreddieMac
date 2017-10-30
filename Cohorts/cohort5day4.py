#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 08:25:18 2017

@author: abhijit
"""

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.2)

iris_model = DecisionTreeClassifier()
iris_model. fit(X_train, y_train)
from sklearn.model_selection import validation_curve

max_depth = np.arange(1, 20)
train_score, val_score = validation_curve(DecisionTreeClassifier(),X_train, y_train,'max_depth',max_depth,cv=5)
plt.plot(max_depth, np.median(train_score, axis=1), color='blue',
         label = 'training score')
plt.plot(max_depth, np.median(val_score, axis=1), color = 'red',
         label = 'validation score')
plt.legend(loc='best')
plt.ylim(0,1)
plt.xticks(max_depth)
plt.xlabel('max_depth')
plt.ylabel('score')


boston = sklearn.datasets.load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston['data'], boston['target'], test_size=0.2)
boston_model1 = DecisionTreeRegressor(max_depth=2)
boston_model2 = DecisionTreeRegressor(max_depth = 5)
boston_model3 = DecisionTreeRegressor(max_depth = 10)

boston_model1.fit(X_train,y_train)
boston_model2.fit(X_train,y_train)
boston_model3.fit(X_train,y_train)

fig, ax = plt.subplots(1,3, sharey=True)
ax[0].scatter(y_test, boston_model1.predict(X_test))
ax[1].scatter(y_test, boston_model2.predict(X_test))
ax[2].scatter(y_test, boston_model3.predict(X_test))

from sklearn.model_selection import GridSearchCV
boston_model = DecisionTreeRegressor()
param_grid = {'max_depth': [1,2,5,10,15,20],
              'min_samples_leaf':[0.2, 0.1, 0.05, 1]}
gs_boston = GridSearchCV(boston_model, param_grid, cv=5)
gs_boston.fit(X_train, y_train)
boston_model_best = gs_boston.best_estimator_
gs_boston.best_params_
gs_boston.best_score_
d = pd.DataFrame(gs_boston.cv_results_)
d


#==============================================================================
# 
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

## proprtion of values in a bootstrap sample

prop_unique = np.zeros(10000)
x = np.arange(50)
for i in range(10000):
    prop_unique[i] = len(np.unique(rng.choice(x, size=50)))/50

sns.distplot(prop_unique)
pd.Series(prop_unique).describe()

#==============================================================================
# 
#==============================================================================
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

def get_bootstrap(n_obs, n_boot, rng = np.random.RandomState(20)):
    """Summary
    *n_obs*: number of observations in data set
    *n_boot*: number of bootstrap samples to generate
    *rng*: (Optional) seed of random number generator

    RESULTS:
    An array that gives the indices of each bootstrap sample as a column in a
    2-d numpy array.
    """
    indx = np.arange(n_obs)
    boot_indx = rng.choice(indx, size = (n_obs, n_boot), replace=True)
    return(boot_indx)

overall = [0,1,2,3,4]
sample = [1,1,2,3,3]
oob=set(overall).difference(set(sample))


#'
#' ### Determining the OOB samples for each bootstrap sample
def find_oob(x, n_obs):
    """
    *x*: index of rows that are in a bootstrap sample
    *n_obs*: Number of observations in the original data

    RESULTS:
    A list with the indices of the OOB sample.
    """
    oob = list(set(range(n_obs)).difference(set(x)))
    return(oob)

def get_oob(boots, n_obs):
    """
    *boots*: A 2-d array that is the output of get_bootstrap
    *n_obs*: Number of observations in the data

    RESULTS:
    A list of OOB indices, one for each bootstrap sample. Note that this is essentially
    a ragged array
    """
    return([find_oob(x, n_obs) for x in boots.T])

def myRFRegressor(dat, target_var, n_boot = 250,
                    max_features = 5,
                    rng = np.random.RandomState(35)):
    """
    Summary

    dat: A pandas DataFrame object
    target_var: A string denoting the column name of the target variable in dat
    n_boot: number of bootstrap samples to take
    max_features: Maximum number of features to consider at each split
    rng: (Optional) random number seed.

    RESULTS:
    A dictionary containing the trained decision trees as well as the OOB predictions from the training set
    """
    feature_names = list(dat.columns)
    feature_names.remove(target_var) # Removes the name of the target variable from the list of names
    X, y = dat[feature_names], dat[target_var]
    boot_indx = get_bootstrap(X.shape[0], n_boot, rng=rng) # Generate bootstrap samples
    oob_indx = get_oob(boot_indx, X.shape[0]) # Get OOB samples
    oob_preds = np.zeros_like(boot_indx) - 1 # Storage for OOB predictions. -1 is meant to be outside data range, will convert to missing
    baseLearner = DecisionTreeRegressor()
    engines = [] # Storage for fitted decision tree objects
    for i in range(n_boot):
        X_boot = X.iloc[boot_indx[:,i],:]
        y_boot = y[boot_indx[:,i]]
        X_oob = X.iloc[np.array(oob_indx[i]),:]
        baseLearner.fit(X_boot, y_boot)
        engines.append(baseLearner)
        oob_preds[np.array(oob_indx[i]),i] = baseLearner.predict(X_oob)
    oob_preds = pd.DataFrame(oob_preds)
    oob_preds[oob_preds==-1] = np.nan
    predictions = oob_preds.apply(np.nanmean, axis=1) #backward compatible to pandas 0.19
    return({'engines': engines, 'predictions': predictions})

def predictRF(engines, test_dat):
    """Summary

    engines: A list of fitted decision trees that will be used as scorers for new data
    test_dat: A test data set
    RESULTS:
    Predictions for each row of test_dat
    """
    prediction = np.zeros(test_dat.shape[0])
    for i in range(test_dat.shape[0]):
        x = test_dat.iloc[i,:]
        preds = [machine.predict(x.values.reshape(1,-1)) for machine in engines]
        prediction[i] = np.mean(np.array(preds))
    return(prediction)

