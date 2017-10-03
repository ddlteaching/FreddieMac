#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 21:23:29 2017

@author: abhijit
"""



#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#%%

iris = sns.load_dataset('iris')
usecolors = dict(zip(iris['species'].unique(),
                ['red','blue','green']))
#%% Plot iris data
fig,ax = plt.subplots(figsize=(4,2.5), dpi=200)
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
#%% Plot with partition lines
fig,ax = plt.subplots(figsize=(4,2.5), dpi=200)
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
plt.ylabel('Petal Width')
ax.axvline(x = 2.5, color='black', ls = '--')
ax.axhline(y = 1.7, xmin = .3, xmax = 1, color='black', ls= '--')
#%% Predictions
iris['Prediction'] = ''
iris.loc[iris['petal_length'] < 2.5, 'Prediction'] = 'setosa'
iris.loc[(iris['petal_length'] > 2.5) & (iris['petal_width'] < 1.7), 'Prediction'] = 'versicolor'
iris.loc[(iris['petal_length'] > 2.5) & (iris['petal_width'] > 1.7),'Prediction'] = 'virginica'
pd.crosstab(iris['species'], iris['Prediction'])

from sklearn.metrics import accuracy_score
accuracy_score(iris['species'],iris['Prediction'])
