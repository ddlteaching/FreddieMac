#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 12:32:30 2017

@author: abhijit
"""

1 + 1
6/4

a = '123'
b = 123
c = 123.0

type(a)
type(b)
type(c)

a == b
b == c
d = 123.00000000000000001
c == d

True == 1

# Lists, Tuples and Dictionaries
## THree kinds of brackets: [ ], ( ), { }

L = [1,2,3,4,5]
len(L)
L[0]
L[4]
L[-1]

L[:2]
L[2:]
L[1:3]


L1 = [1,2,'a', True, 59.3]

L2 = [L, L1]

T = (1,2,3,4,5)
T[0]
T[:3]
len(T)

L[3] = 59
T[3]  = 59


 D = {'first': ["John",'Paul','Ringo'],
      'last' : ['Lennon','McCartney', 'Starr'],
      'age' : ['dead', 75, 71],
       'married': [True, False]}

D['instrument'] = ['guitar','guitar','drums']
 
 D['first'].append('George')
D[0]
D['age'] 

'Age' in D
 
L3 = [1,2,2,2,3,4,4,5,6,6,6,6,]
 
for x in L:
    y = x+ 1
    print(y)

 for i in range(10):
     for j in range(4):
         print(i*j)

s = 0
for x in L:
    s = s + x
print(s)

L = [1,2,3,4,5]

squares = [element ** 2 for element in L]

[element ** 2 for element in L if element < 3]
[element ** 2 for element in L if element % 2 == 0]
[element ** 2 for element in L if (element < 3 ) and (element % 2 == 0)]


#==============================================================================
# Strings
#==============================================================================

A = 'abcd'
list(A)
A[0]
A[-1] 

'a' + 'b'
10 * 'b' 

filename = 'Misc.txt' 
filename[:-4] + '.csv' 
filename.replace('txt','csv') 
 

names = ['Avril','taYlor','KATE','CHRIStina']
[name.capitalize()    for name in names]

dat = [24, 56, 7,5,39]
csvdat = ','.join([str(x) for x in dat])
[float(x) for x in csvdat.split(',')]

'301-345-4953'.split('-')

d = 'Jack, Ryan, 45, Male, Intelligence'
d.split(',')

[x.strip() for x in d.split(',')]

#==============================================================================
# functions
#==============================================================================

s = 0
for x in L:
    s = s + x
s

def mysum(L):
    """
    A function to sum a list of numbers
    USAGE:
        L = [1,2,3,4,5]
        mysum(L)
    """
    s=0
    for x in L:
        if type(x) == str:
            continue
        s = s + x
    return(s)


import numpy as np
import matplotlib.pyplot as plt

L_array = np.array(L)

np.sum(L_array)

%timeit mysum(L)
%timeit np.sum(L_array)

X = np.arange(1000000)

x = np.linspace(0,10,num=51)
y = np.sin(x)
z = np.cos(x)
plt.plot(x, y)
plt.plot(x, y, 'k-', x, z, 'r--')

plt.plot(x, y, label = 'Sine')
plt.plot(x, z, label = 'Cosine')
plt.legend()
plt.xlabel('x')
plt.title('Trig curves')

a, b = 1, 4

fig, ax = plt.subplots(nrows=1, ncols =2)
ax[0].plot(x,y)
ax[1].plot(x,z)

#==============================================================================
# Matrices
#==============================================================================


M = np.array([[1,2,3],[4,5,6]])
M.shape
M[:,0]
M[1, :]
M[0,0]
M[0,[0,2]]
M.T

M * 2
B = np.array([9,2,4])
M * B
M.dot(B)

np.zeros((3,2))
np.ones((5,2))
np.eye(4)
np.zeros_like(M)

B = np.zeros_like(M)
B1 = B
B1[0,0] = 100
B1
B2 = B.copy()
B2[0,1] = 50
B2

B2[B2 > 0]
#==============================================================================
# Random numbers
#==============================================================================

rng = np.random.RandomState(46)
rng.rand(10)

#==============================================================================
#  Pandas
#==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris = pd.read_csv('iris.csv')
iris['Sepal.Length']
iris['Sepal.Length'].mean()
iris['Species'].value_counts()

iris['Sepal.Length'].plot(kind = 'hist')

iris.groupby('Species').mean()
iris.groupby('Species').agg([np.mean, np.std])

A = pd.Series([1, np.nan, 2, None])
A.mean()

A[~A.isnull()]
A.dropna()
A.fillna(0)
A.fillna(method='ffill')
A.fillna(A.mean())

D = pd.DataFrame({'X1': [1,2, None, 4], 'X2': [3, None, 5, 6]})
D
D.mean()
D.fillna(D.mean())
D.fillna({'X1': 0, 'X2': 100})

D[0,0]
D.loc[0,'X1']
D.iloc[0, 0]

D.index = pd.Index(['a','b','c','d'])
D.loc['a','X1']
D.loc['a',:]
D.loc[:,'X1']










import os
os.chdir('/Users/abhijit/Downloads')


















