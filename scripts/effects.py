#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 06:03:38 2017

Interpretation

@author: abhijit
"""

%matplotlib inline
%cd ~/ARAASTAT/Teaching/FreddieMacFinal/data

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

boston = load_boston()
X, y = boston['data'], boston['target']
X_train, X_test, y_train, y_test = train_test_split(boston['data'],boston['target'], test_size=0.3)

boston_df = pd.DataFrame(X, columns = boston['feature_names'])
boston_df['price'] = y

mod = RandomForestRegressor(n_estimators = 300, max_depth=20, min_samples_leaf=10)

preds = cross_val_predict(RandomForestRegressor(n_estimators=300, max_depth=20, 
                                                min_samples_leaf=10 ),
    X, y, cv=5)

plt.plot(boston_df['CRIM'], preds,'.')
sns.regplot(boston_df['CRIM'], preds, lowess=True)
sns.regplot(boston_df['NOX'], preds, lowess=True)
sns.regplot(boston_df['RM'], preds, lowess=True)

x = pd.cut(boston_df['RM'],np.arange(3.5,9.5,1))

bl = pd.Series([str(u) for u in x])
levels = list(np.sort(bl.unique()))[:-1]

effs = []
for i in range(len(levels)-1):
    train,target = np.delete(X[bl==levels[i],:],5,1), y[bl==levels[i]]
    mod.fit(train, target)
    p1 = mod.predict(np.delete(X[bl==levels[i+1],:],5,1))
    mod.fit(np.delete(X[bl==levels[i+1],:],5,1), y[bl==levels[i+1]])
    p2 = mod.predict(np.delete(X[bl==levels[i+1],:],5,1))
    effs.append(p1-p2)
    
    

    
    
