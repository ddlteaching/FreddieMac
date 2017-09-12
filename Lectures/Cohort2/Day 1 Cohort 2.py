
# coding: utf-8

# In[1]:

1+1


# Python works as a calculator. Yeah!!!

# In[2]:

4*6


# In[3]:

6/4


# float, int, bool, str

# In[8]:

a  = '123'
b = 123
c = 123.0
d = 123.0000000000001


# In[5]:

a == b


# In[6]:

b ==c


# In[9]:

c == d


# In[10]:

type(a)


# In[11]:

type(b)


# In[12]:

type(c)


# In[13]:

True == 1


# ## Lists, Tuples and Dictionaries

# Three kinds of brackets: [ ] , () , { }

# In[14]:

L = [1,2,3,4,5]


# In[15]:

L[0]


# In[16]:

L[4]


# In[17]:

L[-1]


# In[18]:

L[1:4]


# In[19]:

L[3:]


# In[21]:

L1 = [1, 2, 'a', True,3.56]


# In[22]:

[1, 2, ['a','b'], 3]


# In[23]:

L1[2] = 'alpha'
L1


# In[24]:

T1 = (1,2,3,4,5)


# In[25]:

type(T1)


# In[26]:

type(L)


# In[27]:

T1 == L


# In[28]:

T1[0]


# In[29]:

T1[3:]


# In[30]:

T2 = (1, 2, 'a', 4, True)


# In[31]:

T2[2] = 'alpha'


# In[32]:

{ 'first': 'John', 'last' : 'Lennon'} # key-value pairs


# Keys can be strings, numbers or tuples
# Values can be anything

# In[34]:

D1 = {'first': ['John','Paul','Ringo'], 'last': ['Lennon','McCartney','Starr'],
     'age': [67 , 69, 61]}


# In[35]:

D1


# In[36]:

D1.keys()


# In[37]:

D1.values()


# In[38]:

D1['age']


# In[39]:

D1[0]


# In[40]:

D1['married'] = [True, False]


# In[41]:

D1


# In[43]:

'gender' in D1


# In[44]:

'age' in D1


# In[45]:

'a' in T1


# In[46]:

T1


# In[47]:

L1


# In[50]:

L


# In[51]:

set(L).intersection(set(L1))


# In[52]:

set(L).difference(set(L1))


# In[61]:

L2 = [1,1,1,3,2,1,2,5, False, 0]


# In[62]:

set(L2)


# ## Loops

# In[65]:

for x in L:
    y = x+1
    print(x)


# In[66]:

s = 0
for x in L:
    s = s + x
s


# In[67]:

for i in range(10):
    for j in range(4):
        print(i + j)


# In[68]:

L1


# In[69]:

for element in L1:
    print(element)


# In[70]:

for indx in range(len(L1)):
    print(L1[indx])


# In[71]:

for element in L:
    print(element ** 2)


# In[72]:

[element **2 for element in L]


# In[73]:

squares = [element**2 for element in range(10)]


# In[74]:

squares


# In[76]:

evens = [elements for elements in squares if elements % 2 == 0]


# In[77]:

evens


# In[79]:

[elements for elements in squares if (elements % 2 == 0) | (elements > 50)]


# ##  Strings

# In[80]:

'a' + 'b'


# In[81]:

10 * 'a'


# In[82]:

list('abcd')


# In[83]:

alph = 'abcdef'


# In[84]:

alph[0]


# In[85]:

alph[2:]


# In[86]:

filename = 'Abra.txt'


# In[89]:

filename[:4]+'.csv'


# In[90]:

filename.replace('txt','csv')


# In[91]:

names = ['Avril','taYlor','KATE','CHRIStina']


# In[92]:

names = [x.capitalize() for x in names]


# In[93]:

names


# In[95]:

help(str.capitalize)


# In[96]:

dat = [29, 34, 52, 39, 34, 53]


# In[100]:

dat2 = ','.join([str(x) for x in dat])


# In[102]:

[float(x) for x in dat2.split(',')]


# In[103]:

'301-345-4953'.split('-')


# In[1]:

d = 'Jack, Ryan, 45, Male, Intelligence'


# In[2]:

d.split(',')


# In[3]:

[x.strip() for x in d.split(',')]


# ## Functions

# In[4]:

def mysum(x):
    s = 0
    for u in x:
        s += u
    return(s)


# In[6]:

L = [1,2,3,4,5]


# In[7]:

mysum(L)


# In[8]:

tuple(L)


# In[9]:

mysum(tuple(L))


# In[11]:

mysum([3])


# In[12]:

mysum([1,2,'a',4,5])


# In[16]:

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


# In[17]:

help(mysum)


# In[14]:

mysum([1,2,'a',4,5])


# In[15]:

help(str.capitalize)


# In[18]:

get_ipython().magic('pinfo mysum')


# In[20]:

get_ipython().magic('pinfo2 mysum')


# In[21]:

get_ipython().magic('pinfo2 str.replace')


# #  Numpy and Matplotlib

# In[22]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[23]:

L


# In[24]:

A = np.array(L)


# In[25]:

A


# In[26]:

np.sum(A)


# In[27]:

X  = np.arange(100000)


# In[28]:

np.sum(X)


# In[30]:

X1 = list(X)


# In[31]:

len(X1)


# In[32]:

get_ipython().magic('timeit np.sum(X)')


# In[33]:

get_ipython().magic('timeit mysum(X1)')


# In[34]:

np.median(X)


# In[35]:

X.sum()


# In[40]:

x=np.linspace(0, 10, num=51)


# In[41]:

plt.plot(x, np.sin(x))


# In[43]:

plt.plot(x, np.sin(x),'k-', x, np.cos(x),'r--');


# In[45]:

plt.plot(x, np.sin(x), label = 'Sine')
plt.plot(x, np.cos(x), label = 'Cosine')
plt.legend()
plt.xlabel('x')
plt.title('Trig functions');


# In[46]:

A


# In[52]:

B = np.array([[1,2,3],[4,5,6]], order='F')


# In[53]:

B


# In[54]:

B.T


# In[57]:

B.shape


# In[58]:

A.shape


# In[59]:

B * A[:3]


# In[61]:

np.zeros((5,10))


# In[62]:

np.ones((3,2))


# In[63]:

np.eye(5)


# In[64]:

rng = np.random.RandomState(42)
rng.rand(10)


# In[66]:

x = rng.rand(100) * 5


# In[67]:

y = 3 + 2*x + rng.normal(0, 1, 100)


# In[68]:

plt.scatter(x,y)


# In[69]:

y1 = 3 + 2*x
plt.scatter(x, y)
plt.plot(x,y1, 'r')


# In[70]:

dir(rng)


# In[71]:

B


# In[73]:

np.zeros_like(B)


# In[74]:

B[0,1]


# In[75]:

B[:2,:2]


# In[77]:

B[1,[0,2]]


# In[78]:

B1 = B


# In[79]:

B1[0,0] = 100


# In[80]:

B


# In[81]:

B1 = B.copy()
B1[0,0] = 0


# In[82]:

B1


# In[83]:

B


# In[84]:

L


# In[85]:

L1 = L
L1[0] = 100


# In[86]:

L


# In[87]:

A


# In[88]:

A = rng.normal(0,1, 20)


# In[89]:

A


# In[90]:

A[A > 0]


# In[91]:

A > 0


# In[93]:

A[(A < -1) | (A > 1)]


# #  PyData stack

# In[94]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[95]:

iris = pd.read_csv('iris.csv')


# In[96]:

pwd


# In[97]:

iris.head()


# In[98]:

type(iris)


# In[99]:

iris.index


# In[100]:

iris.describe()


# In[102]:

iris.describe(include='all')


# In[103]:

iris['Sepal.Length']


# In[104]:

iris.to_dict()


# In[105]:

iris['Sepal.Length'].mean()


# In[106]:

iris['Sepal.Length'].median()


# In[107]:

dir(iris['Sepal.Length'])


# In[109]:

iris['Sepal.Length'].plot(kind='hist')


# In[110]:

iris['Sepal.Length'].plot(kind='kde')


# In[111]:

np.mean(iris['Petal.Length'])


# In[113]:

iris.describe()


# In[114]:

iris.dtypes


# In[115]:

iris.groupby('Species').mean()


# In[118]:

iris.groupby('Species').agg([np.mean, np.std])


# In[119]:

A = pd.Series([1, np.nan, 2, None])


# In[120]:

A


# In[121]:

A.isnull()


# In[122]:

A[~A.isnull()]


# In[123]:

A.dropna() # A.dropna(inplace=True) makes it stick


# In[124]:

A


# In[125]:

A.fillna(0)


# In[126]:

A.fillna(method='ffill')


# In[127]:

A.fillna(method='bfill')


# In[128]:

A.fillna(A.mean())


# In[129]:

D = pd.DataFrame({'X1': [1,2, None, 4], 'X2': [3, None, 5, 6]})


# In[130]:

D


# In[132]:

D.isnull()


# In[133]:

D.mean()


# In[134]:

D.fillna(D.mean())


# In[135]:

d


# In[136]:

D


# In[137]:

D.fillna(0)


# In[141]:

D.fillna({'X1':0, 'X2':100})


# In[142]:

get_ipython().magic('pinfo D.fillna')


# In[143]:

D["X1"]


# In[148]:

D.iloc[:3,0]


# In[147]:

D


# In[151]:

D.loc[0,'X2']


# In[152]:

D.index = pd.Index(['a','b','c','d'])


# In[153]:

D


# In[154]:

D.loc['c','X1']


# In[155]:

D.loc[:'c',:]


# In[156]:

pwd


# In[ ]:



