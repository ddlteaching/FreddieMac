# Calculator

1 + 1
4 * 6
6/4

# Python data types
# float, int, bool, str

a  = '123'
b = 123
c = 123.0
d = 123.0000000000001

type(a)
type(b)

a == b # Testing equality, gives bool

True == 1

# Lists, Tuples and Dictionaries

# Three kinds of brackets: [ ] , () , { }

## Lists

L = [1,2,3,4,5]

L[0]
L[4]
L[-1]
L[3:]

L1 = [1, 2, 'a', True,3.56]

### Lists have an ordering, can be heterogeneous

# Tuples


T1 = (1,2,3,4,5)


T2 = (1, 2, 'a', 4, True)

T1[3] = 5

## Tuples are like lists, but immutable

# Dictionaries


D1 = {'first': ['John','Paul','Ringo'], 'last': ['Lennon','McCartney','Starr'],
     'age': [67 , 69, 61]}

## Key-value pairs
## Keys can be strings, numbers or tuples
## Values can be anything

D1.keys()
D1.values()

D1['age']
D1[0]

D1['married'] = [True, False]


'gender' in D1
'age' in D1


# Sets
L
L2 = [1,1,1,3,2,1,2,5, False, 0]

set(L2)

# Loops


for x in L:
    y = x+1
    print(x)

s = 0
for x in L:
    s = s + x
s

for i in range(10):
    for j in range(4):
        print(i + j)

L1

for element in L1:
    print(element)

for indx in range(len(L1)):
    print(L1[indx])

for element in L:
    print(element ** 2)


[element **2 for element in L] # List comprehension
squares = [element**2 for element in range(10)]

evens = [elements for elements in squares if elements % 2 == 0]


[elements for elements in squares if (elements % 2 == 0) | (elements > 50)]

# Strings

'a' + 'b'

'a' + 'b'

list('abcd')


alph = 'abcdef'

alph[0]

alph[2:]

filename = 'Abra.txt'

filename[:4]+'.csv'

filename.replace('txt','csv')

names = ['Avril','taYlor','KATE','CHRIStina']

names = [x.capitalize() for x in names]

help(str.capitalize)


dat = [29, 34, 52, 39, 34, 53]

dat2 = ','.join([str(x) for x in dat])

[float(x) for x in dat2.split(',')]

'301-345-4953'.split('-')

d = 'Jack, Ryan, 45, Male, Intelligence'

d.split(',')

[x.strip() for x in d.split(',')]

# Functions


def mysum(x):
    s = 0
    for u in x:
        s += u
    return(s)



L = [1,2,3,4,5]



mysum(L)



tuple(L)



mysum(tuple(L))



mysum([3])



mysum([1,2,'a',4,5])



def mysum(x):
    """
    Summing a list or tuple

    USAGE: mysum(x)
    """
    s = 0
    for u in x:
        if type(u) == str:
            continue
        s += u
    return(s)



help(mysum)

mysum([1,2,'a',4,5])

help(str.capitalize)

# Numpy and matplotlib


import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


A = np.array(L)
X  = np.arange(100000)
np.sum(X)
X1 = list(X)
len(X1)

%timeit np.sum(X)
%timeit mysum(X1)


x=np.linspace(0, 10, num=51)

plt.plot(x, np.sin(x))
plt.plot(x, np.sin(x),'k-', x, np.cos(x),'r--');

plt.plot(x, np.sin(x), label = 'Sine')
plt.plot(x, np.cos(x), label = 'Cosine')
plt.legend()
plt.xlabel('x')
plt.title('Trig functions');

B = np.array([[1,2,3],[4,5,6]], order='F')

B.T
B.shape
A.shape

B * A[:3]


np.zeros((5,10))
np.ones((3,2))
np.eye(5)


rng = np.random.RandomState(42)
rng.rand(10)

x = rng.rand(100) * 5

y = 3 + 2*x + rng.normal(0, 1, 100)

plt.scatter(x,y)

plt.scatter(x,y)

np.zeros_like(B)

B[:2,:2]

B1[0,0] = 100

B

B1 = B.copy()
B1[0,0] = 0


A = rng.normal(0,1, 20)

A[A > 0]


A[(A < -1) | (A > 1)]

# PyData stack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris = pd.read_csv('iris.csv')

iris.head()
type(iris)
iris.index
iris.describe()
iris.describe(include='all')

iris['Sepal.Length']
iris['Sepal.Length'].mean()
iris['Sepal.Length'].plot(kind='hist')

iris.dtypes




iris.groupby('Species').agg([np.mean, np.std])


A = pd.Series([1, np.nan, 2, None])

A.isnull()
A[~A.isnull()]
A.dropna() # inplace=True makes it stick
A.fillna(0)
A.fillna(method='ffill')
A.fillna(A.mean())



DD = pd.DataFrame({'X1': [1,2, None, 4], 'X2': [3, None, 5, 6]})
D.isnull()
D.mean()
D.fillna(D.mean())
D.fillna({'X1':0, 'X2':100})

D.iloc[:3,0]
D.loc[0,'X2']
D.index = pd.Index(['a','b','c','d'])
D.loc['c','X1']
