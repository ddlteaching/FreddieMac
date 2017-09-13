#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decision tree implementation, evaluation

@author: abhijit
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
%matplotlib inline

def importance_plot(vals, names):
    fig, ax = plt.subplots()
    ax.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), names)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    plt.show()



from sklearn.datasets import load_iris
iris1 = load_iris()
# iris2 = sns.load_dataset('iris')

boston = sklearn.datasets.load_boston()
boston2 = pd.DataFrame(boston['data'], columns = boston['feature_names'])
boston2['MedianIncome'] = boston['target']
boston2.to_csv('boston.csv', index=False)

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


## Classification

X, y = iris1['data'], iris1['target']
iris_model = DecisionTreeClassifier()
iris_model.fit(X,y)

y_pred = iris_model.predict(X)
pd.crosstab(y, y_pred)
scores = cross_val_score(iris_model, X, y, cv=10)

importance = iris_model.feature_importances_
y_pred = iris_model.predict(X)

## Decision surfaces

X1 = X[:,2:]
x_min, x_max = X1[:,0].min()-0.1, X1[:,0].max()+0.1
y_min, y_max = X1[:,1].min()-0.1, X1[:,1].max()+0.1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

from matplotlib.colors import ListedColormap

iris_model2 = DecisionTreeClassifier()
iris_model2.fit(X1, y)
Z = iris_model2.predict(np.c_[xx.ravel(), yy.ravel()])
zz = Z.reshape(xx.shape)

cs = plt.contourf(xx, yy, zz, cmap = plt.cm.RdYlBu)
plt.scatter(X1[:,0], X1[:,1], c = y, 
            cmap = ListedColormap(['r','y','b']),
            edgecolor='k', s = 20)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

## Model validation

scores = cross_val_score(iris_model, X, y, cv=10)

### Grid search
iris_model.get_params()
from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth': [1, 2, 3, 5, 10],
              'min_samples_leaf': [0.05, 0.10, .20, 1],
              'splitter': ['best','random']}

grid_search = GridSearchCV(iris_model, param_grid, cv = 5)
grid_search.fit(X,y)

grid_search.best_estimator_
grid_search.best_params_

### Validation curves

from sklearn.model_selection import validation_curve
max_depth = np.arange(1, 20)
train_score, val_score = validation_curve(DecisionTreeClassifier(),
                                          X, y,
                                          'max_depth',
                                          max_depth,
                                          cv=5)

plt.plot(max_depth, np.median(train_score, axis=1), color='blue',
         label = 'training score')
plt.plot(max_depth, np.median(val_score, axis=1), color = 'red',
         label = 'validation score')
plt.legend(loc='best')
plt.ylim(0,1)
plt.xticks(max_depth)
plt.xlabel('max_depth')
plt.ylabel('score')

### Learning curve

from sklearn.model_selection import learning_curve

indx = np.arange(X.shape[0])
np.random.shuffle(indx)

fig, ax = plt.subplots(1,3, sharey=True)

for i, depth in enumerate([2, 5, 15]):
    N, train_lc, val_lc = learning_curve(DecisionTreeClassifier(max_depth = depth),
                                         X[indx,:], y[indx], cv = 5,
                                         train_sizes = np.linspace(0.1,1, 25))
    ax[i].plot(N, np.mean(train_lc,1), color='blue', label='training score')
    ax[i].plot(N, np.mean(val_lc, 1), color = 'red', label ='validation score')
    ax[i].hlines(np.mean([train_lc[-1], val_lc[-1]]), N[0], N[-1], color='gray', linestyle='dashed')
    ax[i].set_ylim(0,1)
    ax[i].set_xlim(N[0],N[-1])
    ax[i].set_xlabel('training size')
    ax[i].set_ylabel('score')
    ax[i].legend(loc='best')
    ax[i].set_title('max_depth = {0}'.format(depth), size=14)


## Regression
boston = sklearn.datasets.load_boston()
boston2 = pd.DataFrame(boston['data'], columns = boston['feature_names'])
boston2['MedianIncome'] = boston['target']
boston2.to_csv('boston.csv', index=False)


boston_model1 = DecisionTreeRegressor(max_depth=2)
boston_model2 = DecisionTreeRegressor(max_depth = 5)
boston_model3 = DecisionTreeRegressor(max_depth = 10)


X,y = boston['data'], boston['target']
boston_model1.fit(X,y)
boston_model2.fit(X,y)
boston_model3.fit(X,y)

fig, ax = plt.subplots(1,3, sharey=True)
ax[0].scatter(y, boston_model1.predict(X))
ax[1].scatter(y, boston_model2.predict(X))
ax[2].scatter(y, boston_model3.predict(X))

boston_model = DecisionTreeRegressor()
boston_model.get_params()

param_grid = {'max_depth': [1,2,5,10,20],
              'min_samples_leaf':[0.2, 0.1, 0.05, 1]}
gs_boston = GridSearchCV(boston_model, param_grid, cv=5)
gs_boston.fit(X,y)

boston_model_best = gs_boston.best_estimator_
gs_boston.best_score_
d = pd.DataFrame(gs_boston.cv_results_)
results = d[['param_max_depth','param_min_samples_leaf', 'mean_test_score']].pivot_table(values = 'mean_test_score',
 index='param_max_depth', columns = 'param_min_samples_leaf')


boston_model_best.fit(X,y)
boston_model_best.feature_importances_
importance_plot(boston_model_best.feature_importances_, boston['feature_names'])

### Validation curves

from sklearn.model_selection import validation_curve
max_depth = np.arange(1, 20)
train_score, val_score = validation_curve(DecisionTreeRegressor(),
                                          X, y,
                                          'max_depth',
                                          max_depth,
                                          cv=5)

plt.plot(max_depth, np.median(train_score, axis=1), color='blue',
         label = 'training score')
plt.plot(max_depth, np.median(val_score, axis=1), color = 'red',
         label = 'validation score')
plt.legend(loc='best')
plt.ylim(0,1)
plt.xticks(max_depth)
plt.xlabel('max_depth')
plt.ylabel('score')

### Marginal effects

from sklearn.model_selection import cross_val_predict
D = pd.DataFrame(boston['data'], columns = boston['feature_names'])
D['target'] = boston['target']
D['predicted'] = cross_val_predict(boston_model, X, y, cv = 5)

bl = pd.DataFrame(D.mean()).T
bl = bl.loc[np.repeat(bl.index, D.shape[0])]
bl.index = D.index
bl['RM'] = D['RM']
bl = bl.iloc[:,:-2]

marginal_predict = boston_model_best.predict(bl)

plt.plot(bl['RM'], marginal_predict,'.')

##########################################################
# Vertebrate data
##########################################################

vertebrates = pd.read_csv('vertebrate.csv')

vertebrates.head()
vertebrates['Class Label'].unique()

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer

cols_to_retain = ['Body Temperature', 'Skin Cover', 'Gives Birth', 'Aquatic Creature', 'Aerial Creature', 'Has Legs', 'Hibernates']

X_feature = vertebrates[cols_to_retain]
X_dict = X_feature.T.to_dict().values()


# turn list of dicts into a numpy array
vect = DictVectorizer(sparse=False)
X_vector = vect.fit_transform(X_dict)
vect.get_feature_names()

X_df = pd.DataFrame(X_vector, columns = vect.get_feature_names())

# Used to vectorize the class label
le = LabelEncoder()
y= le.fit_transform(vertebrates['Class Label'][:-1])

vert_model = DecisionTreeClassifier(criterion='gini')
vert_model.fit(X_vector[:-1],y)

##########################################################
# Breast cancer data
##########################################################

brain = sklearn.datasets.load_breast_cancer()