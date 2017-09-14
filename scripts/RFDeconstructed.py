#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 04:22:00 2017

Random forest deconstructed

@author: Abhijit Dasgupta
Copyright (c) Abhijit Dasgupta, 2017. All rights reserved
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pandas as pd

def find_oob(x, n_obs):
    oob = list(set(range(n_obs)).difference(set(x)))
    return(oob)

def get_bootstrap(n_obs, n_boot, rng = np.random.RandomState(20)):
    indx = np.arange(n_obs)
    boot_indx = rng.choice(indx, size = (n_obs, n_boot), replace=True)
    return(boot_indx)

def get_oob(boots, n_obs):
    return([find_oob(x, n_obs) for x in boots.T])

def myRFRegressor(dat, target_var, n_boot = 250, rng = np.random.RandomState(35)):
    """
    dat is a pd.DataFrame object
    """
    feature_names = list(dat.columns)
    feature_names.remove(target_var)
    X, y = dat[feature_names], dat[target_var]
    boot_indx = get_bootstrap(X.shape[0], n_boot, rng=rng)
    oob_indx = get_oob(boot_indx, X.shape[0])
    oob_preds = np.zeros_like(boot_indx) - 1
    baseLearner = DecisionTreeRegressor()
    for i in range(n_boot):
        X_boot = X.iloc[boot_indx[:,i],:]
        y_boot = y[boot_indx[:,i]]
        X_oob = X.iloc[np.array(oob_indx[i]),:]
        baseLearner.fit(X_boot, y_boot)
        oob_preds[np.array(oob_indx[i]),i] = baseLearner.predict(X_oob)
    oob_preds = pd.DataFrame(oob_preds)
    oob_preds[oob_preds==-1] = np.nan
    predictions = oob_preds.agg(np.nanmean, axis=1)
    return(predictions)


        