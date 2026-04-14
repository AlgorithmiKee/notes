---
title: "Basic Python"
date: "2025"
author: "Ke Zhang"
---

# Basic Python

## Mutable and Immutable Objects

Every Python object has three properties: an **identity** (`id()`), a **type**, and a **value** (its internal state). Mutability is a property of the type:

* An object is **immutable** if its value cannot be changed after it is created. Any apparent "modification" instead produces a **new object** with a new identity.
* An object is **mutable** if its value can be changed after it is created, without producing a new object.

Common examples:

| Immutable | Mutable |
|-----------|----------|
| `int`, `float`, `bool` | `list` |
| `str` | `dict`, `set` |
| `tuple` | most user-defined classes |

### In-place modification

An **in-place modification** changes the internal state of an existing object *without* creating a new one.

> After an in-place modification, `id(obj)` is unchanged.

Only **mutable** objects support in-place modification. Common operations:

* List: `a.append(x)`, `a.sort()`, `a[i] = v`, `a.remove(x)`
* Dict: `d[key] = val`, `d.update(other)`
* Object attribute: `obj.attr = val`

### Rebinding

**Rebinding** makes a name refer to a *different* object. It does not modify the object the name previously referred to.

> After rebinding, `id(name)` changes. The original object is unchanged (and may be garbage-collected if nothing else refers to it).

Assignment (`=`) to a bare name is always rebinding. Assigning to an *element* or *attribute* is an in-place modification of the container:

```python
a = [1, 2, 3]
a = [4, 5, 6]   # rebinding: a now points to a new list
a[0] = 99       # in-place modification of the list a points to
```

### `+=` and mutability

`+=` behaves differently depending on mutability:

* For **immutable** objects, `+=` always implies rebinding (a new object is created).
* For **mutable** objects, `+=` first tries an in-place change (via `__iadd__`); if not implemented, it falls back to rebinding.

```python
x = 5
x += 1          # rebinding: x points to a new int

a = [1, 2, 3]
a += [4]        # in-place: list.__iadd__ is defined; a keeps the same id
```

## Tuple

A **tuple** is an ordered collection of elements, enclosed in parentheses `()`.

Remarks:

* The elements in a tuple can be of different types.
* Like list and string, tuples supports indexing, slicing, concatenation, and `len()`
* Unlike lists, tuples are **immutable**.
* 💣 However, if a tuple element points to a mutable object (e.g. a list), then:
  * in-place modification of that element is allowed since it does not create new objects.
  * rebinding that element is not allowed because tuples themselves are immutable.

Example:

```python
info = ('Donald Trump', 1946, ['Republican', 'Businessman'])
print(len(info))            # -> 3
print(info[2][0])           # -> Republican
info[1] = 1950              # error
info[2].append('TV star')   # allowed
info[2] = ['Republican', 'Businessman', 'TV star']  # error
```

Storing the elements of a tuple into multiple variables is called tuple **unpacking**.

Remarks:

* After unpacking, the varaibles can be changed freely.
* Parentheses can be omitted when unpacking variables.

```python
name, age = ('Donald Trump', 78)    # works without parentheses
name = 'Donald J. Trump'            # allowed
age += 1                            # allowed
```

### Typical Use Cases of Tuples

**Use case 1**: short-hand syntax for swapping variables:

```python
x, y = 1, 2     # tuple unpacking
print(x, y)     # -> 1 2
x, y = y, x     # swap
print(x, y)     # -> 2 1
```

The line `x, y = y, x` is equivalent to the classical swap syntax

```python
temp = x
x = y
y = temp
```

**Use case 2**: return multiple objects in a function

```python
def div(x, y):
    q = x // y      # integer division
    r = x % y
    return (q, r)   # works without parentheses

q, r = div(17, 5)
```

If we are only interested in part of the return value, we can use underscore `_` to discard the unwanted return value:

```python
_, r = div(17, 5)   # discards the quotient
```

**Use case 3**: tuple unpacking of enumerated iterables

```python
s = 'ABCD'
for idx, ch in enumerate(s):
    if ch == 'C':
        print(f"Found {ch} at index {idx}")
```

## Function

Basic syntax of function definition:

```python
def func_name(param1, param2, ...) -> return_type:
    # function body
    return value
```

Remarks:

* The return type annotation is optional and does not affect the behavior of the function. It is just a hint for the programmer and some static analysis tools.
* We can specify default values for parameters, which makes them optional when calling the function. For example:

    ```python
    def greet(name="world"):
        print(f"Hello, {name}!")

    greet()          # -> Hello, world!
    greet("Alice")   # -> Hello, Alice!
    ```

### Positional and Keyword Arguments

When calling a function, we can pass arguments in two ways:

1. **Positional arguments**: the arguments are passed in the order of the parameters defined in the function (just like C++). For example:

    ```python
    def wanted(name, age):
        print(f"Wanted: {name}, {age} years old.")

    wanted("Jimmy", 30)   # positional arguments
    ```

1. **Keyword arguments**: the arguments are passed by explicitly specifying the parameter name and value. The order of keyword arguments does not matter. For example:

    ```python
    def wanted(name, age):
        print(f"Wanted: {name}, {age} years old.")
    
    wanted(age=30, name="Jimmy")   # keyword arguments
    ```

In generic programming, we often use `*args` and `**kwargs` to denote any number of positional and keyword arguments, respectively.

Positional and keyword arguments can be mixed in a function call, subject to two rules:

1. All positional arguments must come before any keyword arguments.
1. A keyword argument cannot refer to a parameter that has already been assigned by a positional argument.

**Example**:

```python
def wanted(name, age, city):
    print(f"Wanted: {name}, {age} years old, from {city}.")

# valid use of mixed arguments
wanted("Jimmy", age=30, city="New York")

# also valid, since the order of keyword arguments does not matter
wanted("Jimmy", city="New York", age=30)

# invalid: positional argument must come before keyword arguments
wanted(name="Jimmy", 30, city="New York") 

# invalid: both 30 and Jimmy are mapped to the parameter 'name'
wanted(30, name="Jimmy", city="New York")
```

### Call by Sharing

In python, everything is object. When we call a function, the parameter name becomes a new local reference to the same object the caller passed. Just like copying a pointer to the argument and paste it in the function parameter.

> Basic principle of call-by-sharing:  
>
> * In-place modification of the parameter is visible in the caller.
> * Rebinding the parameter inside the function has no effect in the caller.
> * Immutable parameters can only be rebound. Effectively, their behavior is like call-by-value in C++.
> * Mutable parameters can be both in-place modified and rebound. The former behaves like call-by-reference in C++, while the latter behaves like call-by-value.

**Example**: rebinding an immutable parameter inside a function

```python
def make_zero(n):
    n = 0          # rebinding n to a new int

x = 10
make_zero(x)
print(x)            # -> 10
```

Remarks:

* `x` is a reference to the integer `10`.
* Inside `make_zero()`, `n` is a local reference to the same integer `10`. But the line `n = 0` rebinds `n` to a new integer `0`. Hence, `n` and `x` now refer to different objects.
* After `make_zero()` returns, whatever happens to `n` is invisible to the caller. The original integer that `x` refers to is unchanged.

**Example**: in-place modifying a list inside a function

```python
def f(a):       
    a.append(6)     # in-place modification

x = [1,2,3]
f(x)
print(x)            # -> [1,2,3,6]
```

Remarks:

* `x` is a reference to the number list `[1,2,3]`
* Inside `f()`, `a` is a local reference to the same number list.
* The in-place modification `a.append()` does not change the fact that `a` and `x` refer to the same list. So the appended `6` is also visible after the `f()` finishes.

**Example**: rebinding to a new list inside a function

```python
def f(a):
    a = [1,2,3,6]   # rebinding

x = [1,2,3]
f(x)
print(x)            # -> [1,2,3]
```

Remarks:

* Inside `f()`, `a` is initially a reference to the same list as `x`. But the line `a = [1,2,3,6]` rebinds `a` to a new list object. Hence, `a` and `x` now refer to different objects.
* After `f()` returns, whatever happens to `a` is invisible to the caller. The original list that `x` refers to is unchanged.

**Example**: `+=` inside a function. Mutability matters!

```python
def add_one(n):
    n += 1          # rebinds n to a new int

x = 10
add_one(x)
print(x)            # -> 10

def append_one(a):
    a += [1]        # in-place: list.__iadd__ is defined

L = [6,7,8]
append_one(L)
print(L)            # -> [6,7,8,1]
```

## Function Decorators

### Wrapper and Decorator

To understand function decorators, we first need to understand the concept of a **wrapper**. A wrapper is a function that takes another function as an argument and extends its behavior without explicitly modifying it.

**Example**: a simple wrapper that prints a message before calling the original function:

```python
def wrapper(func):
    def extended_func():
        print("Starting the function...")
        func()  # call the original function
        print("Function has finished.")
    return extended_func
```

Here, `wrapper` is a function that takes another function `func` as an argument and defines a local function `extended_func` that adds some behavior before and after calling `func`. The `wrapper` then returns this new function. We can use it like:

```python
def greet():
    print("Hello, world!")

greet = wrapper(greet)  # wrap the greet function
greet()  # calls the extended function
```

A **decorator** is a special syntax in Python that allows us to apply a wrapper to a function in a more concise way. Instead of manually wrapping the function, we can use the `@` symbol upon the function definition, e.g.

```python
@wrapper
def greet():
    print("Hello, world!")

greet()  # calls the extended function. no need to manually wrap it
```

We say that `greet` is decorated by `wrapper` in this case. The `@wrapper` syntax is just a shorthand for `greet = wrapper(greet)`.

### Decorating functions with arguments

If the original function takes arguments, the wrapper function must also accept those arguments and pass them to the original function. This can be done using `*args` and `**kwargs` to allow for any number of positional and keyword arguments.

```python
def wrapper(func):
    def extended_func(*args, **kwargs):
        print("Starting the function...")
        result = func(*args, **kwargs)  # call the original function with arguments
        print("Function has finished.")
        return result
    return extended_func
```

In this example:

* `*args` and `**kwargs` take care of any number of positional and keyword arguments, respectively, so that the extended function can work with any original function regardless of its signature.
* The wrapper can also return the result of the original function if needed.

Then we can use this wrapper to decorate functions with arguments:

```python
@wrapper
def greet(name):
    print(f"Hello, {name}!")
greet("Alice")  # calls the extended function with an argument

@wrapper
def add(x, y) -> int:
    return x + y

result = add(3, 4)  # calls the extended function with arguments
print(result)  # -> 7
```

### Multiple decorators

A function can be decorated by multiple decorators. The decorators are applied from the bottom up, meaning the decorator closest to the function definition is applied first.

```python
@decorator1
@decorator2
def func():
    pass

func()  # calls decorator1(decorator2(func))
```

Here:

* `decorator2` is applied to `func` first, and then `decorator1` is applied to the result of `decorator2(func)`.
* Calling `func()` will execute the doubly extended function `decorator1(decorator2(func))`.
* Decorators are not commutative. Their order matters, as it can affect the behavior of the final function.

**Example**: a function that requires both login and admin privileges can be decorated by two decorators:

```python
def require_login(func):
    def wrapped_func(curr_user, *args, **kwargs):
        if not curr_user.get("logged_in"):
            print("Please log in first.")
            return
        return func(curr_user, *args, **kwargs)
    return wrapped_func

def require_admin(func):
    def wrapped_func(curr_user, *args, **kwargs):
        if curr_user.get("role") != "admin":
            print("Admin privileges required.")
            return
        return func(curr_user, *args, **kwargs)
    return wrapped_func

@require_login
@require_admin
def install_sw(curr_user, app, version):
    print(f"Installing {app}, version {version}...")

current_user = {"name": "Alice", "logged_in": True, "role": "guest"}
install_sw(current_user, "VS Code", "1.113.0") 
```
