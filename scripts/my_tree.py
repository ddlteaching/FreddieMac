#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 10:15:49 2017

@author: abhijit
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 08:17:50 2017

@author: abhijit
"""
import numpy as np
import pandas as pd

def gi_group(indx, dat, target_var):
    """
    dat = DataFrame object containing the full dataset
    indx = index of rows that are included in the group
    target_var = string giving the column name of the target variable in the dataset
    
    """
    x = dat[target_var].loc[indx]
    p = x.value_counts()/len(x)
    score = 1 - np.sum(p**2)
    return(score)

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
        score = gi_group(groups_indx[key], data, target_name)
        GI += score * (size/n_instances)
    return(GI)

def test_split(variable, value, data):
    out = {'left': list(data.index[data[variable] < value]),
           'right': list(data.index[data[variable] >= value])}
    return(out)

def get_split(dat, target_var, min_size):
    gini = 1.0
    out = {}
    for var in dat.columns:
        if var == target_var:
            continue
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

def to_terminal(indx, dat, target_var):
    return(dat.loc[indx,target_var].value_counts().argmax())


def split(node, max_depth, min_size, depth, dat, target_var):
    """
    node = the output of get_split
    """
    if depth > max_depth:
        node['left'],node['right'] = to_terminal(node['groups']['left'], dat, target_var), to_terminal(node['groups']['right'], dat, target_var)
        del(node['groups'])
        return(node)
    for key in node['groups']:
        print(key)
        indx = node['groups'][key]
        if gi_group(indx, dat,target_var)==0:
            node[key] = to_terminal(indx, dat, target_var)
            continue
        if len(indx) <= 2*min_size:
            node[key] = to_terminal(indx, dat, target_var)
            continue
        node[key] = get_split(dat.loc[indx], target_var, min_size)
        split(node[key], max_depth, min_size, depth+1, dat, target_var)
    del(node['groups'])

def build_tree(train, max_depth, min_size, target_var):
    root = get_split(train, target_var, min_size)
    split(root, max_depth, min_size, 1, train, target_var)
    return(root)
#
def predict_obs(tree, x):
    """
    Find the prediction for a single observation
    """
    if x[tree['var']]< tree['value']:
        if isinstance(tree['left'], dict):
            p = predict_obs(tree['left'], x)
        else:
            p = tree['left']
    else:
        if isinstance(tree['right'],dict):
            p = predict_obs(tree['right'], x)
        else:
            p = tree['right']
    return(p)

def predict(tree, test_dat):
    preds = pd.Series(np.zeros( (test_dat.shape[0],) ))
    preds.index=test_dat.index
    for indx in test_dat.index:
        preds.loc[indx] = predict_obs(tree, test_dat.loc[indx])
    return(preds)


if __name__ == '__main__':
    import os
    os.chdir('/Users/abhijit/ARAASTAT/Teaching/FreddieMacFinal/data')
    iris = pd.read_csv('iris.csv')
    tree = build_tree(iris, 5, 10, 'Species')
    predict(tree, iris)
