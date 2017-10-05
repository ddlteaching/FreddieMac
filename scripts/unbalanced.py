#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 05:42:03 2017

@author: abhijit
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()
y = digits['target']==9
X_train, X_test, y_train, y_test = train_test_split(digits['data'], y, test_size=0.25,random_state=0)

from sklearn.dummy import DummyClassifier
dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train,y_train)
pred_most_frequent = dummy_majority.predict(X_test)
dummy_majority.score(X_test, y_test)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
pred_tree  = tree.predict(X_test)
tree.score(X_test,y_test)

dummy = DummyClassifier(). fit( X_train, y_train) 
pred_dummy = dummy.predict( X_test)
dummy.score(X_test,y_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred_tree)
confusion_matrix(y_test, pred_most_frequent)

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
print(classification_report(y_test, pred_most_frequent))
print(classification_report(y_test, pred_tree))

from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score

fpr,tpr, thresholds = roc_curve(y_test, dummy_majority.predict_proba(X_test)[:,1])

precision,recall,thresholds = precision_recall_curve(y_test, dummy_majority.predict_proba(X_test)[:,1])

plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'k--')
plt.xlabel('Sensitivity')
plt.ylabel('1-Specificity')

plt.plot(precision, recall)
plt.xlabel('Precision')
plt.ylabel('Recall')


