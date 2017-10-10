#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 08:19:37 2017

@author: abhijit
"""
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
    baseLearner = DecisionTreeRegressor(max_features=max_features)
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
    predictions = oob_preds.np.nanmean, axis=1)
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



import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
boston = load_boston()
X, y = boston['data'], boston['target']
dat = pd.DataFrame(X, columns = boston['feature_names'])
dat['target'] = y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
myRF = myRFRegressor(dat, 'target')
p = predictRF(myRF['engines'], pd.DataFrame(X_test))

#==============================================================================
#  A data analysis
#==============================================================================
import os
os.chdir('/Users/abhijit/ARAASTAT/Teaching/FreddieMacFinal/data')
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#' Read in the data. In this case the dataset only consists of data without a first
#' row that contains the column names (or headers). This necessitates the
#' `header=None` option. The test data file also contains some annotation in the
#' first row that we don't want to import. This necessitates the `skiprows=1` option.

dat = pd.read_csv('adult.data', header=None,
                  names = ['age','workclass', 'fnlwgt','education', 'edyrs','marital',
                  'occupation', 'relationship','race','sex','capitalgain',
                  'capitalloss','hrsweek','country','income_class'])
testdat = pd.read_csv('adult.test', header=None, skiprows=1,
                      names = ['age','workclass', 'fnlwgt','education', 'edyrs','marital',
                  'occupation', 'relationship','race','sex','capitalgain',
                  'capitalloss','hrsweek','country','income_class'])

X = pd.get_dummies(dat.iloc[:,:-1])
y = dat['income_class']

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
y = le.fit_transform(y)

X_test = pd.get_dummies(testdat.iloc[:,:-1])
y_test = le.fit_transform(testdat['income_class'])

from sklearn.ensemble import RandomForestClassifier
rf_class = RandomForestClassifier(random_state=50, n_estimators=200)
rf_class.fit(X, y)
p = rf_class.predict(X_test)

#==============================================================================
# There is a problem with not having same number of levels of a predictor. We'll
# concatenate first, then process, then separate the training and test data
#==============================================================================

dat_full = dat.append(testdat)
X = pd.get_dummies(dat_full.iloc[:,:-1])
y = le.fit_transform(dat_full['income_class'])
X_train,y_train = X.iloc[:dat.shape[0],:], y[:dat.shape[0]]
X_test, y_test = X.iloc[dat.shape[0]:,:], y[dat.shape[0]:]

rf_class.fit(X_train, y_train)
p = rf_class.predict(X_test)

#==============================================================================
# The coding of income_class is different in test and training
#==============================================================================

testdat['income_class'] = pd.Series([x.replace('.','') for x in testdat['income_class']])
dat_full = dat.append(testdat)
X = pd.get_dummies(dat_full.iloc[:,:-1])
y = le.fit_transform(dat_full['income_class'])
X_train,y_train = X.iloc[:dat.shape[0],:], y[:dat.shape[0]]
X_test, y_test = X.iloc[dat.shape[0]:,:], y[dat.shape[0]:]
rf_class.fit(X_train, y_train)
p = rf_class.predict(X_test)

#==============================================================================
# Calibration
#==============================================================================


from sklearn.calibration import calibration_curve

p_class_prob = rf_class.predict_proba(X_test)[:,1]
c_class = calibration_curve(y_test, p_class_prob, n_bins=10)
plt.plot([0,1],[0,1], 'k:', label='Perfect calibration')
plt.plot(c_class[1], c_class[0], label ='Classification')


#==============================================================================
# Boosting
#==============================================================================
ns = 100
x = np.linspace(0, 10, ns)
y = 3  + 8*x + + 5*x**2 - 7 * x**3+np.random.normal(0, 1, ns)

a,b = 0,0
learning_rate = 0.0001

def loss(a,b):
    e = np.sum((y - a - b*x)**2)
    return(e)

a,b=0,0
for i in range(5000):
    print(i)
    l1 = loss(a,b)
    dLda = -np.sum(y - a - b*x)
    dLdb = -np.sum(x*(y - a - b*x))
    a1 = a - learning_rate * dLda
    b1 = b - learning_rate * dLdb
    l2 = loss(a1, b1)
    a = a1
    b = b1
    if np.abs(l1-l2) < 0.001:
        break
    
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(max_depth=2)
x = x[:, np.newaxis]
dt.fit(x,y)
p = dt.predict(x)

plt.scatter(x,y)
plt.scatter(x,p, c='red')

res1 = y-p
tr2 = dt.fit(x, res1)
p = p + dt.predict(x)
plt.scatter(x,y)
plt.scatter(x,p, c='red')

res2 = y - p
tr3 = dt.fit(x, res2)
p = p+tr3.predict(x)
plt.scatter(x,y)
plt.scatter(x,p, c='red')

res3 = y-p
tr4 = dt.fit(x,res3)
p = p + tr4.predict(x)
plt.scatter(x,y)
plt.scatter(x,p, c='red')

res4 = y-p
tr5 = dt.fit(x, res4)
p = p+tr5.predict(x)
plt.scatter(x,y)
plt.scatter(x,p, c='red')

