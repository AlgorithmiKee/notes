# Compiling And Linking

## Declaration and Definitions

**Declare** `X` at code line $n$ $\iff$ Tell the compiler that

* The name `X` is available from line $n$
* The *details* of `X` might, however, be anywhere
  * If `X` is a variable, *details* mean its value
  * If `X` is a function, *details* mean its body
  * If `X` is a class, *details* mean the body of `X`'s method

**Define** `X` at code line $n$ $\iff$ Tell the compiler that

* The name `X` is available from line $n$, and
* The *details* of `X` is available from line $n$ as well

Summary:

* A pure declaration just introduces a name
* All definitions are declarations

## Build Process

Equivalent terminologies: build, compile, translate

Build process overview:

1. **Preprocessing**: Convert each `.cpp` file into `.i` file
2. **Compilation**: Translate each `.i` file into `.o` file
3. **Linking**: Combine all `.o` files into executable

### Preprocessing

* Preprocessing performs text transformation, e.g.
  * header inclusion
  * macro expansion
  * conditional compilation
* Each `.cpp` file is preprocessed separately
* The resulting `.i` file is called a **translation unit** (TU)

### Compilation

* Compilation does analysis (parsing) and code generation
* The resulting `.o` file is called a **object file**

An objective file contains

* **data**: machine instruction in binaries
* **metadata**: information which the linker needs to combine objective files into executable, including (but not limited to)
  * Function names and their addresses:
    * Some names represents definitions. i.e. This function is defined in current obj file. Other object files refering to this function can find the definition here.
    * Some names represents references. i.e. The definition of this function is missing in current obj file. It should be in other object files.
  * Object names and their address: defs and refs (similar to function names)
  * Program section names
    * Read-only data. e.g. string literals
  * Debugging information: not detailed here

### Linking

Resolves linkage. Briefly speaking, there are three types of linkages:

* external linkage: applies to global variables and global functions.
  * The definition is in external TU
* internal linkage: applies to static variables and static functions.
  * The definition is in the current TU
* no linkage: applies to local variables.

## One Definition Rule (ODR)

In following, we use the term *entity* to refer to

* variable
* function
* enumerator
* class member
* template
* template specialization

Entities are classified either as **inline** or **non-inline**

### Inline Entities

> Inline entities are entities which can be defined multiple times under certain constarints. (c.f. Next section)

Remarks:

* Not to be confused with the C++ keyword `inline`.
  * `inline` as C++ keyword: suggests the compiler to inline the code
  * An inline entity may not neccessarily be defined/delcared with the keyword `inline`
* Example of inline entities:
  * variables defined with `inline`
  * functions defined with `inline` or `constexpr`
  * class
  * enum
  * function template
  * class tempalte

### Overview

Depending whether an entity is inline or non-inline, two versions of ODR apply:

* Strict ODR: Each **non-inline** entity must have **exactly one** definition across all TUs.

* Relaxed ODR: An **inline** entity `X` is allowed to have more than one definitions across all TUs with the constraint that
  * Each TU using `X` must contain the definition of `X` **exactly once**, and
  * **All** definitions are **identical**.

The strict ODR is often covered during undergrad. However, the relaxed ODR is often less discussed in introductory programming courses.
In following sections, we will revisit the strict ODR and explain the relaxed ODR in more sophisticated examples.

### Best Practice

Works 99% of the time:

* For non-inline entities
  * declare in header file
  * define in source file

* For inline entities
  * define in header file
  * Done

The biggest challenge about ODR is that it is often not trivial to determine which entities are inline. Here, we list a few commonly encountered special cases:

* Define a method inside a class: implicitly inline
* Define a method outside of a class: non-inline
* Class template without specialization: non-inline
* Template specialization: non-inline

## Recap: ODR for Non-inline Entities

### ODR for Non-inline Variables

This example violates the ODR because the global variable `PI` is defined twice.

```c++
/// @file math_const.h 
double PI = 3.14;

/// @file geometry.cpp
#include "math_const.h"

/// @file analysis.cpp
#include "math_const.h"
```

During linking, the linker will complain that the varibale `PI` got defined twice. Two options to fix this:

* Option 1: Define & declare the variable with `const` (or `static`) in the header
  * `const` implies `static`.
  * `static` restricts the visibility of `PI` to file scope. --> internal linkage.
  * Each TU defines its own `PI` once.

  ```c++
  /// @file math_const.h 
  const double PI = 3.14;

  /// @file geometry.cpp
  #include "math_const.h"

  /// @file analysis.cpp
  #include "math_const.h"
  ```

* Option 2: Declare the variable with `extern` in the header. Define the variable in a source file.
  * `PI` is defined once in `math_const.cpp`
  * Any other TU can refer to `PI` via `extern` --> external linkage
  * `PI` is shared across all TUs referring to it

  ```c++
  /// @file math_const.h 
  extern double PI;

  /// @file math_const.cpp [NEW]
  #include "math_const.h"
  double PI = 3.14;

  /// @file geometry.cpp
  #include "math_const.h"

  /// @file analysis.cpp
  #include "math_const.h"
  ```

### ODR for Non-inline Functions

The structure of programs involving global functions is well-known from undergrad. e.g.

```c++
//-----------------------------------
/// @file math_common.h 
double abs(double x);

//-----------------------------------
/// @file math_common.cpp
#include "math_common.h"
double abs(double x){  // defined abs
  return (x >= 0) ? x : -x;
}

//-----------------------------------
/// @file lin_alg.cpp
#include "math_common.h"
{
  auto d = abs(-9.5);  // creates ref
}

//-----------------------------------
/// @file analysis.cpp
#include "math_common.h"
{
  auto u = abs(5.0);   // creates ref
}
```

This program complies to ODR

* Compiling `math_common.i`: creates the definition of `abs`
* Compiling `lin_alg.i`: creates a ref to `abs` since the definition of `abs` is not available in current TU
* Compiling `analysis.i`: creates a ref to `abs` since the definition of `abs` is not available in current TU
* Linking: resolves the reference

If we want to restrict the visibility of a function (e.g. helper function) to a specific TU, we define and declare it in that TU by adding the keyword `static`.

## ODR for Classes

TODO:

## ODR for Function Templates

A function template is not really a function, but a recepie to create a function based on the function template. The compiler will only generate code when it encounters a template instantiation, e.g. `abs<int>()`. Which code will be generated at this point?

* If `abs<int>()` is defined in the current TU, then the compiler will generate the code of the function body.
* If `abs<int>()` is not defined in the current TU, then the compiler will generate a label for it. The linker will link the label to some other TU.

A bad example

```c++
//-----------------------------------
/// @file math_common.h 
template <class T>
T abs(T x);

//-----------------------------------
/// @file math_common.cpp
#include "math_common.h"
template <class T>
T abs(T x){
  return (x >= 0) ? x : -x;
}                         // NOTHING really defined

//-----------------------------------
/// @file lin_alg.cpp
#include "math_common.h"
{
  auto d = abs(-9.5);   // creates ref to abs<double>
}

//-----------------------------------
/// @file analysis.cpp
#include "math_common.h"
{
  auto u = abs(5);      // creates ref to abs<int>
}
```

A linker error occurs

1. Compiling TU `math_common.i`:
    * The compiler won't generate code for any `abs` functions
    * There is no `abs<int>()` or `abs<double>()` defined in `math_common.o` because there no use of those instantiations in `math_common.i`.
1. Compiling TU `lin_alg.i`:
    * The compiler encounters `abs<double>()`
    * The compiler creates a reference to `abs<double>()` because the compiler can't find its definition in the current TU
1. Compiling TU `analysis.i`:
    * The compiler encounters `abs<int>()`
    * The compiler creates a reference to `abs<int>()` due to the same reason
1. Linking `analysis.o` and `lin_alg.o`:
    * Linker error: Undefined reference to `abs<int>` and `abs<double>`

Solution:

* Inclusion model (Used by STL)
  * Put both declaration and definition in the `.h` file.
  * No `.cpp` file for containing the definitions
  * Simply put, this approach inlines the complete function template into each TU
  * Pro: maximial flexibility, easy to maintain
  * Con: increases compile time since everything is inlined
* Explicit instantiation model
  * Put the declaration in the `.h` file
  * Put the definition in the `.cpp` file
  * Explicitly instantiate the template parameter in the `.cpp` file
  * Pro: less compile time
  * Con: limited flexibility

### Inclusion Model

```c++
//-----------------------------------
/// @file math_common.h 
template <class T>
T abs(T x);

template <class T>
T abs(T x){
  return (x >= 0) ? x : -x;
}                    // no math_common.cpp

//-----------------------------------
/// @file lin_alg.cpp
#include "math_common.h"
{
  auto d = abs(9);   // defines abs<int>
}

//-----------------------------------
/// @file analysis.cpp
#include "math_common.h"
{
  auto u = abs(5);   // defines abs<int>
}
```

Analysis:

1. Compiling TU `lin_alg.i`:
    * The compiler encounters `abs<int>()`
    * Template definition is found in current TU
    * The compiler defines `abs<int>()`
2. Compiling TU `analysis.i`:
    * The compiler encounters `abs<int>()`
    * Template definition is found in current TU
    * The compiler defines `abs<int>()`
3. Linking `analysis.o` and `lin_alg.o`:
    * The linker sees two definitions of `abs<int>()` but they appear in different TU and they are identical. The requirments of relaxed ODR are satisfied. Success.

### Explicit Instantiation Model

```c++
//-----------------------------------
/// @file math_common.h 
template <class T>
T abs(T x);

//-----------------------------------
/// @file math_common.cpp
template <class T>
T abs(T x){
  return (x >= 0) ? x : -x;
}

template int abs<int>(int);
template float abs<float>(float);
template double abs<double>(double);

//-----------------------------------
/// @file lin_alg.cpp
#include "math_common.h"
{
  auto d = abs(9);   // defines abs<int>
}

//-----------------------------------
/// @file analysis.cpp
#include "math_common.h"
{
  auto u = abs(5);      // defines abs<int>
}
```

Analysis:

1. Compiling TU `lin_alg.i`:
    * The compiler encounters `abs<int>()`
    * Template definition is not found in current TU
    * The compiler creates a reference to `abs<int>()`
2. Compiling TU `analysis.i`:
    * The compiler creates a reference to `abs<int>()` due to the same reason
2. Compiling TU `math_common.i`:
    * The compiler defines three instances `abs<int>`, `abs<float>` and `abs<double>`
3. Linking `analysis.o`, `lin_alg.o` and `math_common.o`:
    * The linker resolves the reference to `abs<int>()`. Successs.

Remark: Not to be confused with template specialization!!!

* **explicit instantiation**: Let the compiler generate the function for the specified type. It ensures that the template instance is available during linking. (to avoid "undefined reference to xxx"). The implementation, however, is the same across all instances.

  ```c++
  // declaration and definition
  template<class T>
  void swap (T& a, T& b){
    T temp = a;
    a = b;
    b = temp;
  }

  // explicit instantiation: shares the same implementation
  template void swap<int>(int&, int&);
  template void swap<double>(double&, double&);
  ```
  
* **template specialization**: Provides a specific implementation for a paritcular type. It makes the function non-inline.

  ```c++
  // declaration and definition: default implementation
  template<class T>
  T add (T a, T b){
    return a + b;
  }

  // explicit instantiation: specific add() in F2 field
  template<>
  F2 add<F2>(F2 a, F2 b){
    return a == b ? F2{0} : F2{1};
  }
  ```

### ODR for Template Specialization

A template specialization is no longer a template, aka a recepie which is used to generate functions or classes. Hence, we must treat template specializations as if they are non-inline entities.

```c++
//-----------------------------------
/// @file math_common.h
template<class T> // decl primary template
T add (T a, T b);

template<class T> // def primary template
T add (T a, T b){
  return a + b;
}

template<>        // decl template specialiazation
F2 add<F2>(F2 a, F2 b);

//-----------------------------------
/// @file math_common.cpp
#include "math_common.h"

template<>       // def template specialiazation
F2 add<F2>(F2 a, F2 b){
  return a == b ? F2{0} : F2{1};
}
```


## Table of Abbreviations

| Abbrv.   |             Full Name             |
|----------|-----------------------------------|
| ODR      | One Definition Rule               |
| TU       | Translation Unit                  |
| STL      | Standard Template Library         |
