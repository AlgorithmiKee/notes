---
title: "Advanced C"
author: "Ke Zhang"
date: "2024"
---

# Advanced C

## Union

Union vs struct:

* `struct`: allocates storage for all members (plus padding).
  $$
  \mathtt{sizeof\big(struct\big)} \ge \sum_{\mathtt i} \mathtt{member_i}
  $$
  

* `union`: allocates only the storage for the largest member (plus padding).
  $$
  \mathtt{sizeof\big(union\big)} \ge \max_{\mathtt i} \mathtt{member_i}
  $$

### Tagged Union

Tagged union is a classic C design pattern for representing a value that may take on one of multiple different types. Typical implementation:

```c
struct {
    enum tag;
    union data;
}
```

e.g. We want to define a struct `Shape` that can represent circle, rectangle, and square. The parameters of `Shape` depends on the exact shape variant.

1. define the parameters of each shape variant:

   ```c
   typedef struct {
       int radius;
   } Circle;
   
   typedef struct {
       int width;
       int height;
   } Rectangle;
   
   typedef struct {
       int side;
   } Square;
   ```

2. define the **tag** for each shape variant:

   ```c
   typedef enum {
       SHAPE_CIRC,
       SHAPE_RECT,
       SHAPE_SQUA
   } ShapeType;
   ```

3. define `Shape` as a tagged union:

   ```c
   typedef union {
       Circle circ;
       Rectangle rect;
       Square squa;
   } ShapeData;
   
   typedef struct {
       ShapeType tag;
       ShapeData data;
   } Shape;
   ```

Remarks on `Shape` example:

* At any given time, **only one** union member is considered “active”.
* The active member is the one we most recently wrote to.
* Reading a different member than the one you last wrote is **undefined behavior** in C.

Example usage:

* Raw initialization (error-prone, not recommended)
  ```c
  Shape a = { .tag = SHAPE_CIRC, .data.circ = { .radius = 10 } };
  Shape b = { .tag = SHAPE_RECT, .data.rect = { .width = 3, .height = 4 } };
  Shape c = { .tag = SHAPE_SQUA, .data.squa = { .side = 5 } };
  // passes compilation but logically makes no sense.
  Shape d = { .tag = SHAPE_CIRC, .data.rect = {.width = 3, .height = 4} };
  printf("%d\n", d.data.rect.width);    // OK
  printf("%d\n", d.data.circ.radius);   // UNDEFINED BEHAVIOR
  ```
* Constructor for tagged union (recommended)

  ```c
  #include <assert.h>

  static inline Shape create_circle(int radius) {
      Shape s;
      s.tag = SHAPE_CIRC;
      s.data.circ = (Circle){ .radius = radius };
      return s;
  }

  static inline Shape create_rectangle(int width, int height) {
      Shape s;
      s.tag = SHAPE_RECT;
      s.data.rect = (Rectangle){ .width = width, .height = height };
      return s;
  }

  static inline Shape create_square(int side) {
      Shape s;
      s.tag = SHAPE_SQUA;
      s.data.squa = (Square){ .side = side };
      return s;
  }
  ```

* Function that flexibly adapts its behaviour on the tag:

  ```c
  double shape_area(const Shape *s) {
      assert(s);
      switch (s->tag) {
          case SHAPE_CIRC: {
              double r = (double)s->data.circ.radius;
              return M_PI * r * r;
          }
          case SHAPE_RECT:
              return (double)s->data.rect.width * (double)s->data.rect.height;

          case SHAPE_SQUA: {
              double a = (double)s->data.squa.side;
              return a * a;
          }
          default:
              // Unknown tag: decide a policy (assert, return 0, etc.)
              assert(!"Invalid ShapeType tag");
              return 0.0;
      }
  }
  ```

## Pointers

### ptr to const vs. const ptr

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

