---
  title: Machine Learning with Python
  subtitle: A District Data Labs Training Program
  author: Abhijit Dasgupta, PhD
  date: \textcopyright 2017 Abhijit Dasgupta. All rights reserved.
  toc-title: Contents
  titlepage: True
  titlepage-color: 06386e
  titlepage-text-color: ffffff
  titlepage-rule-color: ffffff
  titlepage-rule-height: 1
  papersize: letter
  section-title: True
  logo: DDLLogo.png
---


# Introduction

Python is a popular general-purpose computing language. It is an open-source
language released under a liberal [license](https://docs.python.org/3/license.html)
that is compatible with the [GPL](https://www.gnu.org/licenses/gpl-3.0.en.html).

In recent times, Python has become one of the preferred open-source languages for doing
data science (along with [R](http://www.r-project.org)). This has been driven
by the development of the [`numpy`](http://www.numpy.org),
[`scipy`](http://www.scipy.org) and [`matplotlib`](http://matplotlib.org) packages in the 90s
to mimic Matlab, and then development of [`pandas`](http://pandas.pydata.org),
 [`statsmodels`](http://www.statsmodels.org) and [`sckit-learn`](http://scikit-learn.org)
in the 2000s to add statistical and machine learning functionality akin to R.
This has come to be known, along with some other packages, as the [PyData Stack](https://pydata.org/downloads.html).

## Installing Python

The easiest way to install Python for data science is using the Anaconda Python Distribution,
provided by [Anaconda, Inc.](https://www.anaconda.com/). This distribution bundles together
over 400 packages (depending on your operating system) useful for data science applications.
To install Python:

1. Download the Anaconda Installer from [Anaconda](https://www.anaconda.com/download)
based on your operating system. Currently Python version 3.6 is preferred since the support for
Python version 2.7 will cease soon.
2. Open the installer and install Anaconda

> Note for Mac users: The Mac OS comes with a default Python installation that is part of the
> operating system. Anaconda is installed at a different location and doesn't overwrite the
> system Python. The installation changes the default Python to Anaconda, so when you run Python from
> the terminal by typing `python`, the Anaconda version will be used. Optionally you can keep your default
> system version of Python as the default and create an alias in your .bashrc file to access the
> Anaconda version of Python.

## Training

This training will consist of four modules:

1. [Introduction to Python](#IntroToPython)
2. [Decision Trees](#DecisionTrees)
3. [Bagging and Random Forests](#RandomForests)
4. [Boosting and XGBoost](#XGBoost)

We will start with an introduction to Python programming for new users of Python, to
get users up to speed with basic Python syntax for data science. This will lead up to
using basic `pandas` for data manipulation. Additional Python packages will be introduced in
later sections. We will introduce selected intermediate Python programming concepts with an
explanatory note, as needed.

Next, we will introduce [decision trees](https://en.wikipedia.org/wiki/Decision_tree_learning), specifically
the Classification and Regression trees, or CART. We will discuss the conceptual basis of decision trees with
binary splits. We will then formulate the algorithm and derive a Python program to implement decision trees. Next,
we will introduce the `scikit-learn` package and its implementation of decision trees. We will learn how
to train decision trees, score new data to make predictions, and tune decision trees for optimal
performance.

Ensemble learning is a general method for using multiple learning methods to derive a meta-machine
that can perform better than the original machines. We develop two ensemble machines using
decision trees as _base learners_, namely, Random Forests and Gradient Boosted Machines. The former is
based on the general method of _bagging_ or _bootstrap aggregating_ a number of base learners to create
an improved predictive engine. The latter uses an optimization principle called _boosting_ which
recursively fits base learners to data to optimize some loss function or fit criterion. The particular
implementation of boosting that we will explore is [_XGBoost_](http://xgboost.readthedocs.io/en/latest/), a
scalable, fast and efficient implementation of gradient boosted trees.

As we proceed with these four modules, we will also introduce various important statistical and computational
concepts that are relevant to machine learning, for example, the bias-variance tradeoff, prediction error and its
assessment in ensemble models, and others.



\newpage
