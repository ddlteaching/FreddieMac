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

























