---
title: "Basic Python"
date: "2025"
author: "Kezhang"
---

# Basic Python

## Call by Sharing

In python, everything is object. When we call a function, the parameter name becomes a new local reference to the same object the caller passed. Just like copying a pointer to the argument and paste it in the function parameter.

> Basic principle of call-by-sharing:  
> 
> * In-place modification of the parameter is visible in the caller.
> * Rebinding the parameter inside the function has no affect in the caller.

In-place modification means modifying the state of an object without creating a new one:

* The object's id ($\approx$ address) stays the same after in-place modification.
* Only some part of its contents or attributes is changed.

Common in-place modification:

* List: `.append()`, `.sort()`,`.remove()`,`.insert()`, assigning elements like `a[0]=100`.
* Dict: assigning entries like `d[key] = val`.
* Objects: assigmning attributes like `dog.age = 3`.

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
* `a.append()` count as in-place modification. So the appended `6` is also visible after the `f()` finishes.

**Example**: rebinding to a new list inside a function

```python
def f(a):
    a = [1,2,3,6]   # rebinding

x = [1,2,3]
f(x)
print(x)            # -> [1,2,3]
```

Remarks:

* Inside `f()`, `a` is rebinded to a new list. It gives up the possession of the original list.
* After rebinding, whatever happens to `a` does not affect the original list, which is referenced by `x`.

### Mutable and Immutable Objects

Immutable objects (int, str, tuples) cannot be modified in-place. Hence, rebinding always happens if we change an immutable object inside a function. Hence, modification of immutable objects inside a function is never visible to the caller.

Mutable objects (list, dict, most classes) can be changed in-place. In particular, changing the attribute of an object inside a function remains visible in the caller.

Edge case: `+=` can be tricky:

* for immutable objects, `+=` always implies rebinding
* for mutable objects, `+=` first tries in-place change. If in-place change is not implemented, it tries rebinding.

**Example**: changing an immutable object inside a function implies rebinding

```python
def add_one(n):
    n += 1          # rebinds n to a new int

x = 10
add_one(x)
print(x)            # -> 10

def append_one(a):
    a += [1]        # in-place. creates no new list.

L = [6,7,8]
append_one(L)
print(L)            # -> [6,7,8,1]
```
