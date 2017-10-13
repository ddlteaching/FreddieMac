#' ## Data analysis and modeling with random forests

#' We will go through the steps of a data analysis using a random forest classifier
#' to illustrate both the way to go from dataset to predictions, as well as to
#' teach some Python concepts.

#' First change the directory to move Python to the data
import os
os.chdir('/Users/abhijit/ARAASTAT/Teaching/FreddieMacFinal/data')

#' Then import the usual stack of packages
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#' Read in the data. In this case the dataset only consists of data without a first
#' row that contains the column names (or headers). This necessitates the
#' `header=None` option. The test data file also contains some annotation in the
#' first row that we don't want to import. This necessitates the `skiprows=1` option.

dat = pd.read_csv('adult.data', header=None,
                  names = ['age','workclass', 'fnlwgt','education', 'marital',
                  'occupation', 'relationship','race','sex','capitalgain',
                  'capitalloss','hrsweek','country','income_class'])
testdat = pd.read_csv('adult.test', header=None, skiprows=1,
                      names = ['age','workclass', 'fnlwgt','education', 'marital',
                  'occupation', 'relationship','race','sex','capitalgain',
                  'capitalloss','hrsweek','country','income_class'])

#' Let's take a quick peek at this dataset.
dat.head()

#' Uh oh! It looks like the age data got pushed to the index of the DataFrame.
#' Further inspection reveals that in fact, there is one column header missing,
#' for the fifth column. That is easily fixed as follows:
dat = pd.read_csv('adult.data', header=None,
                  names = ['age','workclass', 'fnlwgt','education', 'edyrs',
                  'marital',
                  'occupation', 'relationship','race','sex','capitalgain',
                  'capitalloss','hrsweek','country','income_class'])
testdat = pd.read_csv('adult.test', header=None, skiprows=1,
                      names = ['age','workclass', 'fnlwgt','education', 'edyrs',
                      'marital',
                      'occupation', 'relationship','race','sex','capitalgain',
                  '   capitalloss','hrsweek','country','income_class'])
pd.Series(dat['capitalloss']).describe()


X = pd.get_dummies(dat.iloc[:,:-1])
y = dat['income_class']

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
y = le.fit_transform(y)

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                    train_size = .3, random_state=32)
# np.sum(y==1)
# np.sum(y==0)
X_test = pd.get_dummies(testdat.iloc[:,:-1])
y_test = le.fit_transform(testdat['income_class'])

#==============================================================================
# The coding of income_class is different in test and training
#==============================================================================

testdat['income_class'] = pd.Series([x.replace('.','') for x in testdat['income_class']])

#==============================================================================
# There is a problem with not having same number of levels of a predictor. We'll
# concatenate first, then process, then separate the training and test data
#==============================================================================

dat_full = dat.append(testdat)
X = pd.get_dummies(dat_full.iloc[:,:-1])
y = le.fit_transform(dat_full['income_class'])

X_train,y_train = X.iloc[:dat.shape[0],:], y[:dat.shape[0]]
X_test, y_test = X.iloc[dat.shape[0]:,:], y[dat.shape[0]:]




from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, brier_score_loss
from sklearn.metrics import f1_score

rf_class = RandomForestClassifier(warm_start=True, random_state=50, oob_score=True)
rf_reg = RandomForestRegressor(warm_start=True, random_state=50, oob_score=True)
pd.DataFrame(X_train).describe()
nboot = range(50,500,50)

scores_class = {'oob':[], 'acc':[], 'auc':[], 'precision':[], 'recall':[], 'brier':[], 'f1':[]}
scores_reg = {'oob':[], 'acc':[], 'auc':[], 'precision':[], 'recall':[], 'brier':[], 'f1':[]}

X

for n_est in nboot:
    print(n_est)
    rf_class.set_params(n_estimators = n_est)
    rf_reg.set_params(n_estimators = n_est)

    rf_class.fit(X,y)
    rf_reg.fit(X, y)

    p_class = rf_class.predict(X_test)
    p_class_prob = rf_class.predict_proba(X_test)[:,1]
    p_reg = rf_reg.predict(X_test)

    scores_class['oob'].append(rf_class.oob_score_)
    scores_class['acc'].append(accuracy_score(y_test, p_class))
    scores_class['auc'].append(roc_auc_score(y_test, p_class_prob))
    scores_class['precision'].append(precision_score(y_test, p_class))
    scores_class['recall'].append(recall_score(y_test, p_class))
    scores_class['brier'].append(brier_score_loss(y_test, p_class_prob))
    scores_class['f1'].append(f1_score(y_test, p_class))

    p_reg_class = p_reg > 0.5

    scores_reg['oob'].append(rf_reg.oob_score_)
    scores_reg['acc'].append(accuracy_score(y_test, p_reg_class))
    scores_reg['auc'].append(roc_auc_score(y_test, p_reg))
    scores_reg['precision'].append(precision_score(y_test, p_reg_class))
    scores_reg['recall'].append(recall_score(y_test, p_reg_class))
    scores_reg['brier'].append(brier_score_loss(y_test, p_reg))
    scores_reg['f1'].append(f1_score(y_test, p_reg_class))

sns.set_style('darkgrid')

f, ax = plt.subplots(2,3, sharex=True, squeeze=False, figsize=(8,4))
for i, loss in enumerate(['acc','auc','precision','recall','brier','f1']):
    row = i // 3
    col = i % 3
    ax[row,col].plot(nboot, scores_class[loss], label='Classification')
    ax[row,col].plot(nboot, scores_reg[loss], label='Regression')
    if i > 2:
        ax[row,col].set_xlabel('Number of estimators')
    ax[row,col].set_title(loss)
plt.savefig('../present/LossFns.png', dpi=150)

#==============================================================================
# Calibration
#==============================================================================

from sklearn.calibration import calibration_curve

p_class_prob = rf_class.predict_proba(X_test)
c_class = calibration_curve(y_test, p_class_prob, 10)
c_reg = calibration_curve(y_test, p_reg, 10)

plt.plot([0,1],[0,1], 'k:', label='Perfect calibration')
plt.plot(c_class[1], c_class[0], label ='Classification')
plt.plot(c_reg[1], c_reg[0], label='Regression')
plt.xlabel('Mean predicted value')
plt.ylabel('Fraction positive')
plt.legend(loc='best')
plt.title('Calibration curves')
plt.savefig('../present/calibration.png')

plt.hist(p_class_prob, label='Classification', histtype='step')
plt.hist(p_reg, label='Regression', histtype='step')
