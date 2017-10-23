#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 08:00:15 2017

@author: abhijit
"""

1+1

a = '123'
b = 123
c = 123.0

a ==b
b==c
True == 1

>
>=
<=
b !=c

# Lists, Tuples and Dictionaries
# Correspond to [ ], ( ), { }

L = [1,2,3,4,5]

L[0]
L[4]
L[-1]
L[1:]
L[:3]
L[1:3]


L1 = [1,2, 'a',True, 3.56]
L1[3]=4
L1.append(5)

# Tuples

T = (1,2,3,4,5)
T[0]
T[2:]
T[2] = 10
T2 = (1,2, 'a', 56, 'abraca')


# Dictionaries

D1 = {'first': ['John','Paul','Ringo'], 
      'last': ['Lennon','McCartney','Starr'], 
      'age': [67, 69, 61] }

D1['first']

D2 = {'first': ['John','Paul','Ringo'], 
      'last': ['Lennon','McCartney','Starr'], 
      'age': [67, 69, 61] ,
      2.3: 'Beatles'}

# Keys can be strings, numbers or tuples
# Values can be anything

D1.keys()
D1.values()

'age' in D1
'gender' in D1

D1['gender'] = 'all males'
D1


# Sets

L3 = [1,2,3,3,2,4,5,6,6,8,7,4,5,9]
set(L3)
list(set(L3))


# Loops

L
for x in L:
    y = x + 1
    print(y)

for key in D1:
    for x in D1[key]:
        print(x)

s = 0
for x in L:
    s = s + x
print(s)

for indx in range(len(L)):
    print(L[indx])
S
for element in L:
    print(element ** 2)

Sq=[element ** 2 for element in L]

L4 = [x for x in range(10)]

# make me a new list which has only the even numbers in L4

[x for x in L4 if x % 2 == 0]
[x for x in L4 if (x % 2 == 0) or (x > 5)]

# Strings

'a' + 'b' 
list('abcde')

5*'a'

s = 'abcdefg'
s[0]
s[-1]

filename = 'Abra.txt'
filename[:4]+'.csv'
filename.replace('txt','csv')
filename[-3:] = 'csv'

names = ['Avril','taYlor','KATE','CHRIStina']
names = [x.capitalize() for x in names]

dat = [29, 34, 52, 39, 34, 53]
out=','.join( str(x) for x in dat)
[float(x) for x in out.split(',')]

d = 'Jack, Ryan, 45, Male, Intelligence'
d.split(', ')

# Functions

s = 0
for x in L:
    s = s + x
s

def mysum(lst):
    s = 0
    for x in lst:
        s = s + x
    return(s)

mysum(range(1000))
mysum([1,2,'a',4,5])

def mysum(lst):
    """Description:
    
    Summing a list or tuple
    
    USAGE: mysum(x)
    """
    s = 0
    for x in lst:
        if type(x) == str:
            continue
        s = s + x
    return(s)

# numpy and matplotlib

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 
# The above line only works in IPython

A = np.array(L)
X  = np.arange(100000)
np.sum(X)
np.mean(X)

%timeit np.sum(X)
%timeit mysum(X)

x=np.linspace(0, 10, num=51)
plt.plot(x, np.sin(x))
plt.plot(x, np.sin(x),'k-', x, np.cos(x),'r--');

plt.plot(x, np.sin(x), label = 'Sine')
plt.plot(x, np.cos(x), label = 'Cosine')
plt.legend()
plt.xlabel('x')
plt.title('Trig functions')

 B=np.array([[1,2,3],[4,5,6]])
B.shape
B.T

B[0,0]
B[:, 0]
B[0,:]
B[0, :2]
B[:,[0,2]]

np.zeros((5,10))
np.ones((3,2))
np.eye(5)
np.zeros_like(B)
A = np.ones_like(B)
A+B
A*B
A.dot(B.T)

B1 = B.copy()
B1[0,0 ] = 700
