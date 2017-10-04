#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 08:07:17 2017

@author: abhijit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

iris = pd.read_csv('iris.csv')

#==============================================================================
# Visual prediction rule
#==============================================================================
iris['Prediction'] = ''
iris.loc[iris['Petal.Length'] < 2.5, 'Prediction'] = 'setosa'
iris.loc[(iris['Petal.Length'] > 2.5) & (iris['Petal.Width'] < 1.7), 'Prediction'] = 'versicolor'
iris.loc[(iris['Petal.Length'] > 2.5) & (iris['Petal.Width'] > 1.7),'Prediction'] = 'virginica'

pd.crosstab(iris['Species'], iris['Prediction'])


#==============================================================================
# Decision Tree
#==============================================================================

# Gini Index

def gini_index(groups_indx, data, target_name):
        """
    group_indx is a dict with keys 'left' and 'right' where each component is a list of indices for each group
    data is the full data set from which indices are computed
    """
    n_instances = len(groups_indx['left']) + len(groups_indx['right'])
     # frequency-weighted Gini
    GI = 0
    for key in groups_indx:
        size = len(groups_indx[key])
        if size == 0:
            continue
        x = data[target_name].loc[groups_indx[key]]
        p = x.value_counts()/len(x)
        score = 1 - np.sum(p**2)
        GI += score * (size/n_instances)
    return(GI)


def test_split(variable, value, data):
    out = {'left': data.index[data[variable] < value],
           'right': data.index[data[variable] >= value]}
    return(out)
#

def get_split(dat, target_var, min_size):
    gini = 1.0
    out = {}
    for var in dat.columns:
        for x in np.sort(dat[var].unique())[1:-1]:
            grp = test_split(var, x, dat)
            if len(grp['left']) < min_size or len(grp['right']) < min_size:
                continue
            g = gini_index(grp, dat, target_var)
            if g >= gini:
                continue
            gini = g
            out['var'] = var
            out['value'] = x
            out['groups'] = grp
            out['gini'] = gini
    return(out)

#














