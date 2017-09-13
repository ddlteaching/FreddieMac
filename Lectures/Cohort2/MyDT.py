#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 05:30:33 2017
My decision tree using pandas
@author: abhijit
"""
import numpy as np
import pandas as pd

#%% Gini index
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
#%%

def test_split(variable, value, data):
    out = {'left': data.index[data[variable] < value],
           'right': data.index[data[variable] >= value]}
    return(out)
#%%

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

#%%

def to_terminal(indx, dat, target_var):
    return(dat.loc[indx,target_var].value_counts().argmax())

#%%
def split(node, max_depth, min_size, depth, dat, target_var):
    left, right = node['groups']['left'], node['groups']['right']
    del(node['groups'])
 #   if not left or not right:
 #       node['left'] = node['right'] = to_terminal(left+right, dat, target_var)
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left, dat, target_var), to_terminal(right, dat, target_var)
        return
    if (len(left)) <= min_size:
        node['left'] = to_terminal(left, dat, target_var)
    else:
        node['left'] = get_split(dat.loc[left], target_var, min_size)
        split(node['left'], max_depth, min_size, depth+1, dat, target_var)
    if len(right) <= min_size:
        node['right'] = to_terminal(right, dat, target_var)
    else:
        node['right'] = get_split(dat.loc['right'], target_var, min_size)
        split(node['right'], max_depth, min_size, depth+1, dat, target_var)

#%%
def build_tree(train, max_depth, min_size, target_var):
    root = get_split(train, target_var, min_size)
    split(root, max_depth, min_size, 1, train, target_var)
    return(root)
#%%
def predict(node, test_dat):
    if test_dat[node['var']]< node['value']:
        if isinstance(node['left'], dict):
            return(predict(node['left'], test_dat))
        else:
            return(node['left'])
    else:
        if isinstance(node['right'],dict):
            return(predict(node['right'], test_dat))
        else:
            return(node['right'])
