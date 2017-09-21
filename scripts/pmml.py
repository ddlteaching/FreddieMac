#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 23:44:14 2017

Exploring how to export scikit-learn models to PMML

Information about which models can be converted is at
https://github.com/jpmml/jpmml-sklearn


@author: abhijit
"""
%matplotlib inline
%cd ~/ARAASTAT/Teaching/FreddieMacFinal/scripts

## Simple version

import pandas as pd

iris_df = pd.read_csv('../data/iris.csv')

from sklearn2pmml import PMMLPipeline
from sklearn.tree import DecisionTreeClassifier

iris_pipeline = PMMLPipeline([
        ('classifier', DecisionTreeClassifier())
        ])
iris_pipeline.fit(iris_df[iris_df.columns.difference(['Species'])], iris_df["Species"])

from sklearn2pmml import sklearn2pmml

sklearn2pmml(iris_pipeline, 'DecisionTreeIris.pmml', with_repr=True)

## More complex

from sklearn2pmml import PMMLPipeline
from sklearn2pmml.decoration import ContinuousDomain
from sklearn_pandas import DataFrameMapper
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression

iris_pipeline = PMMLPipeline([
	("mapper", DataFrameMapper([
		(["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"],
   [ContinuousDomain(), Imputer()])
	])),
	("pca", PCA(n_components = 3)),
	("selector", SelectKBest(k = 2)),
	("classifier", LogisticRegression())
])
iris_pipeline.fit(iris_df, iris_df["Species"])

from sklearn2pmml import sklearn2pmml

sklearn2pmml.sklearn2pmml(iris_pipeline, "LogisticRegressionIris.pmml", with_repr = True)
