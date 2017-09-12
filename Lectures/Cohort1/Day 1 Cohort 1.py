
# coding: utf-8

# # Machine Learning with Python

# ## Abhijit Dasgupta, PhD

# __WiFi__: Use the Blue network, then open browser and enter employee ID and RSA password

# Data types: float, int, str, booleans

# In[2]:

True


# In[3]:

4*5


# In[4]:

6/4


# In[5]:

a = 123
b = '123'
c = 123.0


# In[6]:

type(a)


# In[7]:

type(b)


# In[8]:

a == b


# In[9]:

a == c


# In[10]:

1 == True


# In[22]:

d = 123.000000000000001
a==d


# In[16]:

a == d


# # Lists, Tuples and Dictionaries

# There are 3 kinds of brackets in Python: [], (), {}

# In[23]:

L = [1,2,3,4,5]


# In[24]:

L[0]


# In[25]:

L[3]


# In[26]:

L[3] = 20


# In[27]:

L


# In[28]:

L[:3]


# a:b includes a, excludes b

# In[29]:

L[1:3]


# In[30]:

L[2:]


# In[31]:

L1 = [1, 'apple',True, 3.5]


# In[49]:

L2 = [[1,2], ['apple','pear'], 5.2]


# In[50]:

L2


# In[52]:

['apple', 'pear'] in L2


# In[48]:

L2 = 5


# In[ ]:




# In[34]:

L2


# In[35]:

len(L1)


# In[36]:

L


# In[37]:

L[-1]


# In[38]:

L[:-2]


# In[39]:

L[:20]


# In[40]:

L[:-10]


# In[41]:

L[:2]


# In[42]:

L[-2:]


# In[43]:

L1


# In[44]:

'apple' in L1


# In[45]:

'pear' in L1


# In[46]:

' apple' in L1


# ###  Tuples

# In[53]:

t1 = (1,2,3,4,5)


# In[55]:

t1[0]


# In[56]:

t1[2:]


# In[57]:

t1[2] = 20


# ###  Tuples are immutable, lists are not

# ## Dictionaries

# In[58]:

D1 = {'first': 'John', 'last': 'Wayne'}


# In[59]:

D1


# In[60]:

D1[0]


# In[61]:

D1['first']


# Keys can be strings, numbers or tuples
# Values can be any Python object

# In[62]:

D1[1] = [24,53]


# In[63]:

D1


# In[64]:

a = 'apple'


# In[65]:

a


# In[66]:

DIR = {'first':['John','Paul','Ringo'], 'last': ['Lennon','McCartney','Starr'], 
      'birthday': [24, 2, 14], 'month': ['Jan','Feb','July']}


# In[67]:

DIR


# In[68]:

DIR['month'][1]


# In[69]:

DIR.keys()


# In[70]:

DIR.values()


# In[71]:

[1, DIR]


# #  Loops and list comprehensions

# In[72]:

L1


# In[75]:

for kalamazoo in L1:
print(kalamazoo)


# ```
# for elem in list:
#     do something
#     do something
# ```

# In[80]:

L = [1,2,3,4,5]
mysum = 0
for x in L:
    mysum = mysum + x
    print(mysum)


# ### List comprehension

# In[81]:

[x ** 2 for x in L]


# In[82]:

DIR


# In[85]:

[DIR[x][-1] for x in DIR.keys()]


# In[86]:

L3 = [1,2,3,4,5,6,7,8,9,10]
L3sq = [x **2 for x in L3]
evens = [y for y in L3sq if y % 2 == 0]


# In[87]:

odds = [y for y in L3sq if y % 2 == 1]
odds


# #  change of direction

# In[91]:

a = '123'


# In[90]:

b = 123
c = 1.23


# In[92]:

float(a)


# In[93]:

int(a)


# In[94]:

str(c)


# ## Strings

# In[96]:

'a' + 'b'


# In[97]:

10 * 'a'


# In[98]:

list('abcde')


# In[99]:

Name = 'Johann'


# In[100]:

len(Name)


# In[101]:

Name[:3]


# In[102]:

Name[-1]


# In[103]:

filename = 'Ricardo.txt'


# In[106]:

filename[:8]+'csv'


# In[107]:

filename.replace('txt','csv')


# In[109]:

names = ['Ricardo','gUIDO','ALAN','michael']


# In[110]:

[x.capitalize() for x in names]


# In[111]:

entry = [DIR[x][-1] for x in DIR.keys()]


# In[117]:

entry = [str(x) for x in entry]


# In[118]:

entry


# In[121]:

singleentry = '|'.join(entry)


# In[122]:

singleentry.split('|')


# In[124]:

import numpy as np
import matplotlib.pyplot as plt


# In[125]:

L


# In[126]:

np.array(L)


# In[127]:

np.array(L).sum()


# ### Constructing data

# In[131]:

np.arange(10,100)


# In[142]:

x = np.linspace(0,10,num = 100)


# In[136]:

np.sin(x)


# In[137]:

np.cos(x)


# In[138]:

get_ipython().magic('matplotlib inline')


# In[145]:

plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x), 'r')


# In[147]:

plt.plot(x, np.sin(x),'k-', x, np.cos(x),'r--');


# In[151]:

plt.plot(x, np.sin(x))
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sine curve');


# In[152]:

range(len(L))


# # Functions

# In[153]:

def mysum(x):
    s = 0
    for i in range(len(x)):
        s += x[i]
    return(s)


# In[154]:

L


# In[155]:

mysum(L)


# In[156]:

mysum(np.array(L))


# In[160]:

L = [1,2,3,4,5]
L1 = L.copy()
L1[3] = 'a'
L1


# In[161]:

L


# In[162]:

mysum(L1)


# In[172]:

def mysum(x):
    """
    Sums a list or an array
    
    Input:
    x : a list or np.array of numbers
    
    Output:
    A scalar, the sum
    """
    s = 0
    for i in range(len(x)):
        if type(x[i]) == str: # avoid strings
            continue
        s += x[i]
    # Make sure to make return a function
    # 
    return(s)


# In[174]:

get_ipython().magic('pinfo mysum')


# In[175]:

get_ipython().magic('pinfo2 mysum')


# In[168]:

mysum(L1)


# ```
# if condition:
#    something
# elif condition:
#    something else
# else:
#     finally something
# ```

# In[177]:

get_ipython().magic('pinfo2 np.array')


# In[178]:

L2 = [[1,2,3],[4,5,6]]


# In[184]:

A1 = np.array(L2)


# In[180]:

np.array(L2).shape


# In[181]:

A2 = np.array([9,2,4])


# In[182]:

A2.shape


# In[185]:

A1*A2


# In[186]:

A1


# In[187]:

np.sqrt(A1)


# # Data analysis (pandas)

# In[188]:

import pandas as pd


# In[189]:

DIR


# In[191]:

DIR_df = pd.DataFrame(DIR)


# In[192]:

DIR_df


# In[193]:

DIR_df.columns


# In[194]:

DIR_df.index


# In[195]:

DIR_df.index = pd.Index(['a','b','c'])
DIR_df


# In[201]:

DIR_df['birthday':'month'] # Things to look at


# In[203]:

type(DIR_df.loc['a'])


# In[204]:

DIR_df.iloc[0]


# In[205]:

DIR_df['birthday'].describe()


# In[210]:

DIR_df.describe(include='all')


# In[207]:

DIR_df['first'].describe()


# In[208]:

DIR_df['first']


# In[212]:

iris = pd.read_csv('http://ddlteaching.github.io/FreddieMac/data/iris.csv')


# In[213]:

iris.shape


# In[214]:

iris.head()


# In[215]:

iris['Species'].value_counts()


# In[216]:

import seaborn as sns


# In[217]:

iris = sns.load_dataset('iris')


# In[218]:

iris.head()


# In[219]:

rng = np.random.RandomState(40)
rng.rand(10)


# In[221]:

dat = rng.rand(10,4)


# In[222]:

dat.shape


# In[226]:

dat = pd.DataFrame(dat, columns = ['a','b','c','d'])


# In[227]:

dat.describe()


# In[228]:

dat.dtypes


# In[229]:

DIR_df.dtypes


# In[231]:

dat['a'].plot(kind = 'hist')


# In[232]:

sns.distplot(dat['a'])


# In[233]:

iris.head()


# In[234]:

iris['species'].unique()


# In[240]:

iris.groupby('species').agg([np.mean, np.std])


# In[241]:

z = pd.Series([1, np.nan, 2, None])


# In[242]:

z


# In[243]:

z.isnull()


# In[245]:

z[~z.isnull()]


# In[246]:

z.fillna(0)


# In[248]:

z.fillna(method='ffill')


# In[249]:

z.fillna(method='bfill')


# In[250]:

z


# In[251]:

z.mean()


# In[252]:

np.mean(z)


# In[253]:

z.fillna(z.mean())


# In[254]:

dir(np)


# In[ ]:



