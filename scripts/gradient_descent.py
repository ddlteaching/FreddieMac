import numpy as np
x=[6]
cur=6

def f(x):
    return(np.array(x)**2)

def grad(x):
    return(2*x)

gamma = .9

for i in range(50):
    cur = x[-1]-gamma*grad(x[-1])
    x.append(cur)
x
%matplotlib inline
import matplotlib.pyplot as plt

y = np.linspace(-4,4,100)

x = [4]
gamma = .8
for i in range(10):
    cur = x[-1] - gamma * grad(x[-1])
    x.append(cur)

plt.scatter(x,f(x), s=30, color='red')
plt.plot(x,f(x))
plt.plot(y, f(y),'k')

#
# Another example
#

def f(x):
    return(x**4 - 3 * x**3 + 2)

def grad(x):
    return(4 * x**3 - 9 * x**2)

y = np.linspace(-3,4,201)
plt.plot(y, f(y))

gamma = 0.01
x = [4]
for i in range(10):
    cur = x[-1] - gamma * grad(x[-1])
    x.append(cur)

plt.plot(y, f(y))
plt.scatter(np.array(x), f(np.array(x)), s=50, color='red')
plt.plot(np.array(x), f(np.array(x)), color='green')





##
## Linear regression
##

rng = np.random.RandomState(24)
x = rng.standard_normal(100)
y = 3 - 2*x + rng.normal(0,0.4,100)
plt.scatter(x,y)

len(x),len(y)

a, b = [0], [0]
gamma = 0.002
def f(a,b):
    return(np.sum((y - a - b*x)**2))

def grad_a(a,b):
    return(2*np.sum((y-a-b*x)*(-1)))
def grad_b(a,b):
    return(2*np.sum((y - a - b*x)*(-x)))

grad_b(a[-1],b[-1])
for i in range(10):
    a1 = a[-1] - gamma*grad_a(a[-1],b[-1])
    b1 = b[-1] - gamma*grad_b(a[-1],b[-1])
    a.append(a1)
    b.append(b1)
len(x),len(y)
fig,ax = plt.subplots(1,1)
ax.scatter(x,y)
for i in range(10):
    ax.plot(x, a[i] + b[i]*x)


b

freqs = np.arange(5)
line, = ax.plot([],[])
ax.set_xlim([x[0], x[-1]])
ax.set_ylim([-1, 1])
ax.set_title('$\sin(x)$')
fig.canvas.draw()
