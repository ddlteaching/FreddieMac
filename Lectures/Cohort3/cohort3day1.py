#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 08:37:26 2017

@author: abhijit
"""

1+1
4*6

a = '123'
b = 123
c = 123.0
d = 123.00000000001

type(a)
type(b)
type(c)

a == b
b == c

# Lists, Tuples and Dictionaries
# [], () , { }

L = [1,2,3,4,5]
L[0]
L[:2]
L[2:]
L[-2:]


L1 = [1, 2, 'a', True, 3.456]

T = (1,2,3,4,5)
T[:2]

L1[2] = 'b'
T[2] = 5

D  = {'first': ['John','Paul','Ringo'],
      'last': ['Lennon','McCartney','Starr'],
      'age': ['dead', 75, 71],
      'married': [True, False]}
 
 
 D.keys()
 D.values()
 
 'age' in D
 'gender' in D
 'John' in D['first']
 
 L1 = [1,2,2,2,2,3,5,6,6,6]
set(L1)
 L2 = L1
 L1[0] = 1000
 L2 = L1.copy()
 L1[1] = 24
 
 # Loops
 
for x in L:
     y = x+1
     print(y)

s = 0
for x in L:
    s = s + x
    print(s)

for i in range(10):
    for j in range(4):
        print(i+j)

for x in L:
    print(x ** 2)

# List comprehension

squares = [x ** 2 for x in L]

evens = [x for x in squares if (x % 2 == 0) & (x > 10)]

# Strings

1 + 1
'1' + '1'
'a' + 'b' + 'c'

4 * 'a'

list('abcd')
z = 'abcde'
z[0]
z[-1]
z[1:3]

y = 'abracadabra'
y[:4] + y[-4:]

filename = 'clinical_data.txt'
filename[:-4] + '.csv'

filename.replace('txt','csv')

names = ['Avril','taYlor','KATE','CHRIStina']

names = [x.capitalize() for x in names]

data = [24, 24,50, 28, 58, 29]

csvdata = ','.join([str(x) for x in data])

[float(x) for x in csvdata.split(',')]

d = 'Jack, Ryan, 45, Male, Intelligence'

[x.strip() for x in d.split(',')]

# Functions

s = 0
for x in L:
    s = s+x
print(s)

def mysum(L):
    """
    Function to sum numbers in a list
    """
    s = 0
    for x in L:
        s = s+x
    return(s)

L2 = [5,6,7,8]
mysum(L2)

L3 = [2, 5, 'a', 9]
def mysum(L):
    """
    Function to sum numbers in a list
    """
    s = 0
    for x in L:
        if type(x) == str:
            continue
        s = s+x
    return(s)


import numpy as np

L
A = np.array(L)
X = np.arange(100000)
np.sum(X)
np.median(X)

len(X)
X1 = list(X[:100])

%timeit mysum(X1)
%timeit np.sum(X[:100])

import matplotlib.pyplot as plt

x = np.linspace(0,10, num=51)
y = np.sin(x)

plt.plot(x, y)
plt.plot(x, np.sin(x), 'k-', x, np.cos(x), 'r--')

plt.plot(x, np.sin(x), label = 'Sine')
plt.plot(x, np.cos(x), label = 'Cosine')
plt.legend()
plt.xlabel('x')
plt.title('Trig functions');
#==============================================================================
# 
#==============================================================================

## Matrices

B = np.array([[1,2,3], [4,5,6]]) # row-wise
B1 = np.array([[1,2,3], [4,5,6]], order='F')
         
Z = np.zeros((5,10))
np.ones((2,2))
np.eye(3)
        
np.zeros_like(B) 
         
rng = np.random.RandomState(48)
rng.rand(10)

B
B[0,0]
B[0,:]
B[:,1]
B[:2,:2]
B[:, [0,2]]

B[0,0] = 'a'

A = rng.normal(0,1, (100,2))
A1 = A[A>0]
np.sort(A[(A > -1) & (A < 1)])

#==============================================================================
# Pandas
#==============================================================================

import pandas as pd

iris = pd.read_csv('iris.csv')

iris.columns
iris.index
iris.shape
iris.dtypes
iris['Species']
iris.describe(include='all')
iris['Sepal.Length'].mean()
iris['Species'].value_counts()
iris['Sepal.Length'].plot(kind = 'hist')

iris.groupby('Species').mean()
iris.groupby('Species').agg([np.mean, np.std])

A = pd.Series([1, np.nan, 2, None])
A.isnull()
A[~A.isnull()]
A.dropna()
A.fillna(0)
A.fillna(method='ffill')
A.fillna(A.mean())

D = pd.DataFrame({'X1': [1,2, None, 4], 'X2': [3, None, 5, 6]})
D
D.isnull()
D.mean()
D.fillna(D.mean())
D.fillna({'X1': 0, 'X2': 100})

D[0,0] # Doesn't work

 D.loc[0,'X1']
D.iloc[0,0]
D.index = pd.Index(['a','b','c','d'])
D.loc['c','X2']
D.loc['c']
