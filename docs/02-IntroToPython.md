
# Introduction to Python {#IntroToPython}

## Base Python

Python works as a calculator


```python
1+1
```

```
2
```



```python
4*6
```

```
24
```



More generally, we can assign values to _variables_. For example:


```python
a = '123'
b = 123
c = 123.0
```



Python has several _types_ of variables, but at a basic level, we will be using string or character
variables (`str`), and numerical variables (`float` or real numbers, and `int` or integers). For example,
each of a, b and c are of different _types_:


```python
print(type(a))
print(type(b))
print(type(c))
```

```
<class 'str'>
<class 'int'>
<class 'float'>
```


Even though these three values will appear the same in a spreadsheet like Excel, they are
distinct to a computer. In fact, if we test if _a_ and _b_ are the same:

```python
a == b
```

```
False
```


However, Python is smart about numbers, so,

```python
b == c
```

```
True
```



> Python can test various conditions:
>
> __Syntax__  |   __Operation__
> ------------|------------------
> `==`        |  Equality
> `!=`        | Not Equal
> `>`         | Greater than
> `>=`        | Greater than or equal
