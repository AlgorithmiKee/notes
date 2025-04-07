---
title: "Advanced C"
author: "Ke Zhang"
date: "2024"
---

# Struct

Byte alignment.

# Union

* `struct`: allocates storage for all members.
* `union`: allocates only the storage for the largest member.

# Pointers

## ptr to const vs. const ptr

The pointer to const should be interpreted as read-only pointer.

```c
const int* ptr;			// pointer to const int
```

* The content referenced by `ptr` becomes read-only, regardless what `ptr` points to.

* `ptr` itself is not constant. i.e. It can be reassigned to reference another variable.

* `int*` can be converted to `const int*` but not vice versa. i.e.

  | may point to |                  `int`                   |               `const int`                |
  | :----------- | :--------------------------------------: | :--------------------------------------: |
  | `int*`       | <span style="color:green">**YES**</span> |  <span style="color:red">**NO**</span>   |
  | `const int*` | <span style="color:green">**YES**</span> | <span style="color:green">**YES**</span> |

* `ptr` as a function parameter is useful if we want `*ptr` to remain unchanged inside the function

  ```c
  _Bool is_multiple_of(const int* x);		// We don't expect to change *x
  ```



The const pointer is like C++ reference.

```c++
int* const ptr;
```

* `ptr` can't be reassigned to reference another variable.
* The content referenced by `ptr` may, however, be modified.



There is indeed `const int* const`.

