#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 13:00:03 2017

@author: abhijit
"""

import os
os.chdir('/Users/abhijit/ARAASTAT/Teaching/FreddieMacFinal/data')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

iris = pd.read_csv('iris.csv')

iris['Prediction'] = ''
iris.loc[iris['Petal.Length'] < 2.5, 'Prediction'] = 'setosa'
iris.loc[(iris['Petal.Length'] >= 2.5) & (iris['Petal.Width'] < 1.7), 'Prediction'] = 'versicolor'
iris.loc[(iris['Petal.Length'] >= 2.5) & (iris['Petal.Width'] > 1.7),'Prediction'] = 'virginica'
pd.crosstab(iris['Species'], iris['Prediction'])


usecolors = dict(zip(iris['species'].unique(),
                ['red','blue','green']))
#

fig, ax = plt.subplots()
type1 = ax.scatter(iris.loc[iris.species=='setosa','petal_length'],
                   iris.loc[iris.species=='setosa','petal_width'],
                   s = 50,
                   c = 'red')
type2 = ax.scatter(iris.loc[iris.species=='versicolor','petal_length'],
                   iris.loc[iris.species=='versicolor', 'petal_width'], 
                   s = 50, 
                   c = 'green')
type3 = ax.scatter(iris.loc[iris.species == 'virginica','petal_length'],
                   iris.loc[iris.species=='virginica', 'petal_width'],
                   s = 50, 
                   c = 'blue')
plt.legend([type1,type2, type3], ['setosa','versicolor','virginica'])
plt.xlabel('Petal Length')
plt.ylabel('Petal Width');

#==============================================================================
#           Decision Trees
#==============================================================================

# Gini index for a single group
def gi_group(indx, dat, target_var):
    x = dat[target_var].loc[indx]
    p = x.value_counts()/len(x)
    score = 1 - np.sum(p**2)
    return(score)

# Gini index for a split

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
#

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

#==============================================================================
# Scikit-learn API
#==============================================================================

import sklearn
from sklearn.tree import DecisionTreeClassifier

X = iris.iloc[:,:4]
y = iris.iloc[:,-1]
model = DecisionTreeClassifier(max_depth=2, min_samples_leaf=10)
model.fit(X, y)
model.predict(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)

model.fit(X_train, y_train)
pred = model.predict(X_test)

model.score(X_train, y_train)
model.score(X_test, y_test)


from sklearn.model_selection import KFold
   
from sklearn.model_selection import cross_val_score
cross_val_score(model, X_train, y_train, cv=5)    

model = DecisionTreeClassifier(max_depth=1, min_samples_leaf=2)
cross_val_score(model, X_train, y_train, cv=5)    



from sklearn.datasets import load_breast_cancer
breast = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(breast['data'], 
                                                    breast['target'], 
                                                    test_size=.2, 
                                                    random_state=3)
breast_model.fit(X_train, y_train)

breast_model = DecisionTreeClassifier(max_depth=2, min_samples_leaf = 20)
cross_val_score(breast_model, X_train, y_train, cv=5)

from sklearn.model_selection import GridSearchCV
parameter_grid = {'max_depth': [1, 2, 3, 5, 10, 20],
                  'min_samples_leaf': [1,5,10,20,50]}
grid_search = GridSearchCV(breast_model, parameter_grid, cv=5)
pd.DataFrame(grid_search.cv_results_).loc[:, ['params','mean_test_score','mean_test_score']]

grid_search.best_estimator_.predict(X_test)
grid_search.best_params_
grid_search.best_score_


from sklearn.model_selection import validation_curve
max_depth = np.arange(1,20)
train_score, val_score = validation_curve(DecisionTreeClassifier(min_samples_leaf=20), X_train, y_train, 'max_depth', max_depth, cv=5)

plt.plot(max_depth, np.mean(train_score, axis=1), color='blue', label='training score')
plt.plot(max_depth, np.mean(val_score, axis=1), color='red', label='validation score')
plt.legend()


#==============================================================================
# Regression
#==============================================================================

boston= sklearn.datasets.load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston['data'], boston['target'], test_size=0.2)
from sklearn.tree import DecisionTreeRegressor
boston_model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=20)
boston_model.fit(X_train, y_train)
preds = boston_model.predict(X_test)
plt.scatter(y_test, preds)

max_depth = np.arange(1,40)
train_score, val_score = validation_curve(DecisionTreeRegressor(min_samples_leaf=20), X_train, y_train, 'max_depth', max_depth, cv=5)

plt.plot(max_depth, np.mean(train_score, axis=1), color='blue', label='training score')
plt.plot(max_depth, np.mean(val_score, axis=1), color='red', label='validation score')
plt.legend()

parameter_grid = {'max_depth': [1, 2, 3, 5, 10,15,  20,],
                  'min_samples_leaf': [1,5,10,20,50]}
grid_search = GridSearchCV(boston_model, parameter_grid, cv=5)
grid_search.fit(X_train, y_train)
pd.DataFrame(grid_search.cv_results_).loc[:, ['params','mean_test_score','mean_test_score']]

grid_search.best_estimator_.fit(X_train, y_train)
grid_search.best_estimator_.feature_importances_

def importance_plot(vals, names):
    fig, ax = plt.subplots()
    ax.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), names)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    plt.show()

importance_plot(grid_search.best_estimator_.feature_importances_, boston['feature_names'])


from sklearn.datasets import make_moons
moons_data, moons_label = make_moons(200, noise = 0.1, random_state=4)

fig, ax = plt.subplots()
ax.scatter(moons_data[moons_label==0,0],moons_data[moons_label==0,1], c='blue')
ax.scatter(moons_data[moons_label==1,0],moons_data[moons_label==1,1], c='red')

x_min, x_max = np.min(moons_data[:,0]), np.max(moons_data[:,0])
y_min, y_max = np.min(moons_data[:,1]), np.max(moons_data[:,1])


xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
X_test = np.c_[xx.ravel(), yy.ravel()]

mod = DecisionTreeClassifier()
mod.fit(moons_data, moons_label)
p=mod.predict(X_test)

fig, ax = plt.subplots()
ax.scatter(moons_data[moons_label==0,0],moons_data[moons_label==0,1], c='blue')
ax.scatter(moons_data[moons_label==1,0],moons_data[moons_label==1,1], c='red')

fig, ax = plt.subplots()
ax.scatter(X_test[p==0,0], X_test[p==0,1], color='blue')
ax.scatter(X_test[p==1,0], X_test[p==1,1], color='red')

