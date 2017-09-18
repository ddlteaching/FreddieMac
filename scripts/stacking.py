#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 05:17:38 2017

Stacking

@author: abhijit
"""

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
%matplotlib inline

sns.set_style('darkgrid')

from sklearn.datasets import load_breast_cancer

breast = load_breast_cancer()
X, y = breast['data'], breast['target']