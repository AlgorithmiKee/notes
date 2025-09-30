---
title: "Advanced C++"
author: "Ke Zhang"
date: "2023"
---

# Advanced C++

[toc]

## Typecast

C++ supports multiple ways of explicit typecast:

1. **C style typecast**: No type safety (Conversion between arbitray types are allowed) and hence error-prone. Not recommanded in C++ code.

   ```c++
   double pi = 3.14;
   int x = (int)pi;
   ```

2. **Static cast operator**: The default option for typecasting.

   ```c++
   double pi = 3.14;
   int x = static_cast<int>(pi);
   ```

   `static_cast` also supports casting between classes if the relevant **casting functions** are defined.

### Casting Functions

A casting function is used to convert from a class type to another type. 

Syntax:

```c++
// X.hpp:
class X{
    // Enables implicit conversion from X to T1
	operator T1();
    
    // Enables explicit conversion from X to operator T2
    explicit operator T2();  
};

// main.cpp:
X x;
T1 a1 = x;					// [OK] since X::operator T1() is implicit
T1 a2 = static_cast<T1>(x);	// [OK]
T2 b1 = x;					// [ERROR] X::operator T2() must be called explicitly
T2 b1 = static_cast<T2>(x);	// [OK] This is the only way to call X::operator T2()
```

Casting functions are often `const` since developers expect casting operatrions to be non-modifying operations.

## Type Alias

From C, we know the `typedef`

```c++
typedef unsigned int StudentID;
typedef int (*BinaryOperation)(int a, int b);

StudentID id1 = 27061108;	// equivalent: id1 = 27061108;

int max(int a, int b){
  return a>b ? a : b;
}

BinaryOperation f = max;	// equivalent: int (*f)(int, int) = max;
std::cout << f(3, 9);
```

In C++, there is another option `using`

```c++
using StudentID = unsigned int;
using BinaryOperation = int(*)(int, int);

StudentID id1 = 27061108;	// equivalent: id1 = 27061108;

int max(int a, int b){
  return a>b ? a : b;
}

BinaryOperation f = max;	// equivalent: int (*f)(int, int) = max;
std::cout << f(3, 9);
```

Why introduce `using` as we already have `typedef`?

`using` works with templates while `typedef` does not. 

## Enumeration

* Enumeration $\iff$ self-defined new type e.g. `Color`
* Enumerator $\iff$ values which an enumeration variable can take `Red`
* There are two types of enumeration: unscoped and scoped enumeration

| Unscoped Enumeration                                         | Scoped Enumeration                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Weakly typed: implicitly converted to `int`                  | Strongly typed: no implicit conversion to `int`              |
| Unscoped: enumerator and enumeration share the same scope. e.g. `Color c1 = red` | Scoped: enumerator defines a namespace. Enumerator anly accessible via `::` e.g. `Color c1 = Color::red` |
| Name collision more probable due to unscopeness              | Name collision less probable due to scopeness                |
| Comparison between enumeration variables of different enum type is allowed (although it makes no sense) | Comparison between enumeration variables of different enum type is NOT allowed |
| Declared by key word `enum`                                  | Declared by key word `enum class`                            |

**Example: Unscoped Enumeration**

```c++
enum Color{
  red, blue		
};

enum Day{
  Mon, Tue, Wed, Thu, Fri, Sat, Sun
};

Color c1 = red;	// no qualifier needed
Day d1 = Tue;		// no qualifier needed

if(c1==d1){			// OK, although non sense
  ...
}
```
Here, `Color, Day` and `red, blue, Mon, Tue, ...` share the same scope. No qulifier is needed when using the enumerator. This leads to the disadvantage that name collison becomes more probable. Another disadvange comes from weak typicality. Comparision between `c1` and `d1` is allowed since they are both implicitly converted into `int`. However, it makes no sense to do the comparison. 



```c++
enum class Color{
  red, blue
}

enum class Day{
  Mon, Tue, Wed, Thu, Fri, Sat, Sun
};

Color c1 = Color::red;	// qualifier is a must
Day d1 = Day::Tue;			// qualifier is a must

if(c1===d1){						// Compiler error
  ...
}
```

Here, the enumeration `Color` and `Day`  declared by `enum class` has the global scope. The enumerators are put in their corresponding subscopes qualified by the enumeration names. To access the color red, one must write `Color::red`. Scoped enumeration is strongly typed, meaning that comparion is only allowed among variables of the same enum type.




## OOP

###  Copy Constructor

> The source object must be passed by reference. 

Why? Suppose we pass the source object by value. Once we call the copy constructor, the program will call the copy constructor again to copy the argument, which once again calls the copy constructor... It is a dead loop. 

### Friend Methods

### Operator Overloading

Example: overloading `+` 

Example: overloading of `=` (Basic)

Example: overloading of `=` (Copy Swap)

Here, we used [copy- swap idiom](#copy-and-swap-idiom) to improve the robustness.

## Template

### Template Instantiation

Class and function templates are not classes or functions in the strict sense. Instead, they are blueprints that tell the compiler how to generate classes or functions when the template is used. We distinguish between:

* **explicit** template instantiation: explicitly specify the template parameter when using the template
* **implicit** template instantiation: Let the compiler deduce the type when using the template.

```c++
#include <iostream>

template<typename T>
class Node {
public:
    Node(T value) : _value(value) {}

private:
    T _value;
};

int main() {
    // Explicit instantiation
    Node<int> n1(42);

    // Explicit instantiation
    Node<double> n2(3.14);

    // Implicit instantiation based on initialization
    Node n3(std::string("Hello"));  // Compiler deduces T as std::string

    return 0;
}
```

### Template Specialization

Template specialization allows a custom implementation for specific template parameters. In other words: “For this type, don’t use the generic template—use my special definition.” We distinguish:

* **Full** specialization: All template parameters are specified in the custom implementation. Often used in numeric libraries where certain types deserve specialized algorithms.
* **Partial** specialization (ONLY for class templates): Only some template parameters are specified; the rest remain generic.

**Example: full template specialization**.

```c++
#include <iostream>
#include <cstring>

// Generic template
template <typename T>
bool isEqual(const T a, const T b) {
    return a == b;
}

// Special case: compare C-strings by content, not pointer value
template <>
bool isEqual(const char* a, const char* b) {
    return std::strcmp(a, b) == 0;
}

int main() {
    // instantiate generic function template. EXPECTED: True
    std::cout << isEqual(3, 3) << "\n";

    // calls template specialization.         EXPECTED: True
    std::cout << isEqual("hi", "hi") << "\n";
}
```

**Example: partial template specialization**.

```c++
#include <iostream>

// Generic template
template <typename T1, typename T2>
struct Pair {
    static void print() { std::cout << "Generic pair\n"; }
};

// Partial specialization: same types
template <typename T>
struct Pair<T, T> {
    static void print() { std::cout << "Pair of same type\n"; }
};

// Partial specialization: second is char*
template <typename T>
struct Pair<T, char*> {
    static void print() { std::cout << "Pair<T,char*>\n"; }
};

int main() {
    Pair<double, char>::print(); // Generic
    Pair<int, int>::print();     // Pair of same type
    Pair<float, char*>::print(); // Pair<T,int>
}
```

#### Calling preference

When multiple functions with the same name are available, the compiler chooses in this order of preference (from most to least preferred):

1. A non-template overload whose parameter types match exactly.
2. A fully specialized template function.
3. An instantiation of a generic template function.

**Example: calling preference**.

```c++
#include<iostream>
#include <typeinfo>
using namespace std;

// Generic template (pointer version)
template<class T>
void print(T* p){
    cout << "template T*=" << typeid(p).name() << ", "<< *p << endl;
}

// Generic template
template<class T>
void print(T x){
    cout << "template T=" << typeid(x).name() << ", "<< x << endl;
}

// template specialization for int
template<>
void print(int x){
    cout << "specialized for int, " << x << endl;
}

// non-templated overload
void print(int x){
    cout << "overload for int, " << x << endl;
}

int main(){
    int myInt = 4;
    double myDouble = 1.1;

    print(myInt);
    print<int>(myInt);
    print(myDouble);
    print(&myInt);
    return 0;
}
```

**Analysis**:

1. `print(myInt);`
	* calls the non-templated overload since non-template function is most preferred when the argument type is matched to the parameter type.
2. `print<int>(myInt);`
	* calls the template specialization since the template function has a specialized version for `int`
3.  `print(myDouble);`
	* calls an instantiation of the template function for `T = double`
4. `print(&myInt)`
	* calls an instantiation of the template function for `T = int` since the argument is of type `int*`

## Standard Template Library

### Vector

Stores an array of elements in continuous memory. The size of the array is dynamically adjustable.

* Construction:
  ```c++
  #include<vector>
  // empty vector
  std::vector<int> u;
  // a vector of 4, 5, 4, 7
  std::vector<int> v{4, 5, 4, 7};
  // a vector of length n filled with 0
  std::vector<int> w(n, 0);
  ```
* Get the number of elements:
  ```c++
  auto len = v.size();
  ```
* Iterate: There are multiple ways to iterate through an vector
  ```c++
  // classical for-loop based on indexing
  for(int i = 0; i < v.size(); i++) {
    cout << v[i] << " ";
  }
  
  // STL style for-loop based on iterator
  for(auto it = v.begin(); it != v.end(); it++) {
    cout << *it << " ";
  }
  
  // for-each style
  for(auto elem : v) {
    cout << x << " ";
  }
  ```
* Insert and remove element
  ```c++
  v.push_back(value); // insert elm. at the end
  v.pop_back(value);  // delete elm. at the end
  v.insert(v.begin() + k, val); // insert val at index k
  v.erase(v.begin() + k);       // remove elm. at index k
  ```
* Search: `std::vector` has no memeber function `.find()`. We must use `std::find()` from STL algorithm library!
  ```c++
  #include <algorithm>
  // locate the iterator pointing to val
  // returns v.end() if not found
  auto it = std::find(v.begin(), v.end(), val)
  // if found val in v
  if(it != v.end())
  ```
* Sort: `std::vector` has no memeber function `.sort()`. We must use `std::sort()` from STL algorithm library!
  ```c++
  #include <algorithm>  
  // sort 
  std::sort(v.begin(), v.end());
  ```

### 2D Array

Here, we focues on 2D array implemented by nesting `std::vector`.

* Construction:
  ```c++
  #include<vector>
  using std::vector;
  // construct with lists
  vector<vector<int>> Id3 = {
    {1, 0, 0}, {0, 1, 0}, {0, 0, 1}
  };
  // fill with zeros
  vector<vector<int>> Zeros(nrows, vector<int>(ncols, 0))
  ```
* Get the number of rows and columns:
  ```c++
  auto nrow = v.size();
  auto ncol = v[0].size();
  ```

### Unordered Map

Stores `key` to `value` pairs. Pairs are not ordered. Underlying data structure: hash table.

$O(1)$ for insert, delete, and search.

Illustration: hash table for storing goods and prices.

* Construction:
  ```c++
  #include<unordered_map>
  
  // Stores stock (string) to price (double) pairs
  std::unordered_map<std::string, double> stock_to_price;
  ```

* Insert: If the `key` does not exist in the hash table, insert the key-value pair. Otherwise, overwrite the `value` the existing key-value pair.
  ```c++
  stock_to_price["TSLA"] = 316.70;
  stock_to_price["NVDA"] = 165.55;
  ```

* Search:
  ```c++
  // if "APPL" exists in the hash map
  if(stock_to_price.find("APPL") != stock_to_price.end())
  ```

### Pointer to Functions

TODO

## Lvalues & Rvalues

<table>
  <tbody>
    <tr>
      <th> </th>
      <th align="center">lvalue</th>
      <th align="right">rvalue</th>
    </tr>
    <tr>
      <td>Informal definition</td>
      <td align="center">Values that have a name and an address and that can be assigned to some value</td>
      <td align="right">Anything that are not a lvalue</td>
    </tr>
    <tr>
      <td>Lifespan</td>
      <td align="center">long</td>
      <td align="right">short</td>
    </tr>
    <tr>
      <td align="center">Typical examples</td>
      <td>
        <ul>
          <li>variables (local, global, static)</li>
          <li>function parameters</li>
          <li>array elements</li>
          <li>deferenced pointers</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>constant literals</li>
          <li>most expressions</li>
          <li>return values of a function</li>
          <li>temporary objects (e.g. objects produced by factory class)</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

## References

Both lvalue reference and rvalue reference has a `const` version. The `const` keyword imposes read-only access to the referenced object, regardless of whether the referenced object is const or not. 

**A Quick Overview of Valid Binding**

| may bind to | `lvalue` | `const lvalue` | `rvalue` | `const rvalue` |
| ----------- | :------: | :------------: | :------: | :------------: |
| `T&`        |    ✔︎     |       ✖︎        |    ✖︎     |       ✖︎        |
| `const T&`  |    ✔︎     |       ✔︎        |    ✔︎     |       ✔︎        |
| `T&&`       |    ✖︎     |       ✖︎        |    ✔︎     |       ✖︎        |
| `const T&&` |    ✖︎     |       ✖︎        |    ✔︎     |       ✔︎        |

Before we dive in, keep in mind that

> Thou shalt not return (lvalue or rvalue) reference to local variables

as local varibales get destroyed when a function returns, leading to dangling reference. 

### Lvalue reference 

**Properties**

* An lvalue reference must be initialised when it is declared. After initialisation, it can't be reset. i.e. There is no way to rebind it to another object.
* An lvalue reference is essentially an alias of the object which it binds to. It is perticularly useful to define call-by-reference functions.
* When a call-by-reference function is invoked: No copy-constructor is invoked since the argument itself rather than a copy is passed into the function. This significantly improves efficiency when the argument is a large data structure.
* Nonconst lvalue references only bind to nonconst lvalue objects.
* Const lvalue references binds to everything, including rvalue objects.

**Examples**

```c++
// Correct binding if lvalue
int& r0 = counter;						// valid
int& r1 = arr[22];						// valid
r0 = arr[20];									// valid: this is not reset but assignment

int& r2 = var%100;						// invalid: can't bind rvalue to int&
int& r3 = arr.size();					// invalid: can't bind rvalue to int&

const int& r2 = var%100;			// valid
const int& r3 = arr.size();		// valid

// Call by reference
void swap(int& a, int& b){		// declared as call by reference
  ...
}

int main(){
  int x = 118;
  int y = 276;
  swap(x, y);									// no copying. x and y themselves are passed
}
```

|            Comparison            |           lvalue reference           |  pointer   |
| :------------------------------: | :----------------------------------: | :--------: |
|  Represents | alias | address|
| Initialisation on declaration is |                 must                 | not a must |
|          Can be reset?           |                  ❌                   |     ✅      |
|             Nesting?             | 🈚️ There is no reference to reference |     🈶 There is certainly pointer to pointer     |

### Rvalue reference

**Basic Properties**

* Nonconst rvalue references only bind to nonconst rvalues.
* Const rvalue references bind to all rvalues.
* An rvalue usually has rather short lifespan. However, once it binds to an rvalue reference, its lifespan is extended until the reference dies.
* Nonconst rvalue reference can modiy a nonconst rvalue

> 💣 An rvalue reference variable itself is an lvalue!

**Example: Value Type vs. Data Type**

```c++
int&& r1 = 5;		// r1 itself is an lvalue. Its type is rvalue reference to int
r1 = 3;					// This is not rest but modiying the rvalue it binds to
```

**Rvalue Reference as Function Parameter**

```c++
void foo(int&& x);
```

The function `foo` expects an rvalue reference to int as its parameter. Hence, we can only pass an rvalue (by reference) into `foo`. However, `x` itself is an lvalue since it has a name and an address in memory. Don't forget: Function parameters are lvalues! Hence, we can't treat `x` as an rvalue inside function `foo`. 💣

```c++
int main(){
  int   age = 72;
  int&& ref = age;
  
  foo(72);			// 1)
  foo(age);			// 2)
  foo(ref);			// 3)
}
```

1. Valid, since literal `72` is an rvalue.
2. Invalid, since `age` is an lvalue which can't be converted into rvalue. 
3. Invalid, since `ref` itself is an lvalue even though its type is `int&&`, i.e. rvalue reference to int. 💣

**Rvalue Reference in OOP**

We can overloade class method depending on whether the argument object is an lvalue or rvalue. This brings especially benifits under the situaion below

## Move Semantics

<img src="./figs/Move vs Copy.jpg" alt="Move vs Copy" style="zoom:50%;" />

**Motivation: What is move semantics? Why bother it?**

> Move $\iff$ transfer the ownership of resources 

In general, a resource can be heap memory, file descriptors or thread objects. Copying the resource to the target object is expensive. Move semantics is useful where deep copy is not necessary, since moving only requires a pointer reassignment.

**Shallow copy vs. Deep copy vs. Move. When to use which?**

Is the source managing any resources? 

* NO. $\implies$ Shallow copy. In this case, all three concepts coincide.
* YES. If the source is a ...
  			 * lvalue $\implies$ Deep copy is the safest despite the computational cost. In certain scenarios, shallow copy plus extra measures is also safe (See `shared_ptr`). Move is dangerous as the source is left with a `nullptr` (See `auto_ptr`).
      * rvalue. $\implies$ Move is the best option due to its efficiency and safety. Deep copy is OK but may be expensive. Shallow copy should be reconsidered depending on how the source handles the resource as it dies.
     

In C++, move semantics is implemeted via rvalue references (vs. copy semantics via lvalue reference)

```c++
class Student{
  int _id;
  char* _name;
  
  // Default constructor
  Student():_id(0), _name(nullptr){}
  
  // Constructor
  Student(int id, const char* name):_id(id), _name:(new char[strlen(name)]){
    strcpy(_name, name);
  }
  
  // Destructor
  ~Student():{delete[] _name;}
  
  // Copy constructor
  Student(const Student& s):_id(s.id), _name(new char[strlen(s._name)]){
    strcpy(_name, s._name);
  }
  
  // Move Constructor
  Student(Student&& s):_id(s.id), _name(s._name){
    s._name = nullptr;
  } 
};
```

## Smart Pointer

**Motivation: Why smart pointers?**

Although C++ has built-in pointers, working with raw pointers often causes problems such as

* Memory leak. This may happen when
  * Early return so that the pointers defined earlier won't be freed
  * Passing pointers between functions. Sometimes the caller or callee is responsible to free memory
* Dangling pointer. i.e. A pointer binds to a memory block which is already freed
* Two variances of freeing the memory: `delete` for non-array objects vs. `delelte []` for arrays.

Consider the example

```c++
int* foo(){
  int* p1 = new int;	// Suppose that p1 successed to allocate memory
  int* p2 = new int;	// Caution: if p2 fails to allocate memory, p1 will never be freed
  
  if(...){
    return;						// Caution: early return, both p1 and p2 won't be freed
  }
  if(...){
    throw 0;					// Caution: early return, both p1 and p2 won't be freed
  }
  
  delete p1;
  return p2;					// Caution: The caller is responsible to free the memory
}
```

Hence, it would be nicer if we can create some new class which automatically handles clean-up. Recall the destructors in C++ are automatically invoked when the object goes out of scope. How about wrapping the raw pointer into a  `auto_ptr` class? Let the destructor handles the clean-up so that we don't have to do it manually.

<img src="./figs/Smart Pointer Mindmap.jpg" alt="Smart Pointer Mindmap" style="zoom:30%;" />

### A bit history: auto pointer

A naive implementaiton of wrapped smart pointer is

```c++
template <typename T>
class auto_ptr{
  T* m_ptr;
 public:
  auto_ptr(T* ptr==nullptr):m_ptr(ptr) {}
  ~auto_ptr(){
    delete m_ptr;
  }
  T& operator*() 	const{return *m_ptr;}
  T* operator->() const{return m_ptr;}
};

int main(){
  auto_ptr<Song> s1(new Song("1234", "Feist"));
  // do something with s1
  return 0;
}
```

The constructor wraps the raw pointer which binds to external resources. Here, the external resource is a `Song` object. We call the smart pointer the **owner** since it **owns** the resource. When `s1` goes out of scope, `~auto_ptr()` is called and the `Song` object will be freed from the heap. Even if the program somewhen throw an exception, `~auto_ptr()` will be invoked anyway. So far so good.

Problem occurs when the default copy constructor is invoked. It implements shallow copy by default, leading to dangling pointer as illustrated below.

```c++
int main(){
  auto_ptr<Song> s1(new Song("Bad Guy", "Bille Eilish"));
  {
  	auto_ptr<Song> s2 = s1; // Shallow copy
  }													// s2 died. s1 becomes dangling pointer!
  return 0;									// RUNTIME ERROR: undefiend behaviour
}
```

Here,  both `s1` and `s2` own the same object. When `s2` first invokes the destructor, the object get deleted from the memory and the `s1` is left as dangling pointer. When `s1` invokes the destructor, we have undefined behaviour.

To address this problem, we must override the default copy constructor. But how? Consider three options:

* <span style="color:red">**Deep Copy:**</span> Not a good idea. Copying large objects is expensive. It also introduced inconsistency with the `=` for raw pointers.
* <span style="color:green">**Impose exclusive ownership of the raw pointer:**</span> `s2` takes the ownership of the resource while `s1` gives up its ownership. i.e. The ownership is transferred from `s1` to `s2` when the copy constructor is invoked. Later, we will see that `unique_ptr` introduced in C++11 implements this idea.
* <span style="color:green">**Implement proper shared ownership of the raw poiner:**</span> By *proper*, we mean that no smart pointers may release the resource unless no other smart pointers own it. This makes sure that no smart pointers somewhen become a dangling pointer. Later, we will see that `shared_ptr` introduced in C++11 implements this idea.

The `auto_ptr` in C++98 tries to implement the exlusiveness of the ownership but in an evil way. In particular, it allows owner `A` to quietly steal the ownership from owner `B`  when performing `=`.

```c++
class auto_ptr{
  ...
	auto_ptr(auto_ptr const & source){
  	this->m_ptr = source.m_ptr;			// destination ptr takes the ownership
  	souce.m_ptr = nullptr;					// source ptr gives up the ownership
  }
  ...
};

int main(){
	auto_ptr<Song> s1(new Song("Fly Away", "Tones and I"));
  {
  	auto_ptr<Song> s2 = s1; 				// s2 quietly steals s1's ownership
  }																	// s2 died. s1 becomes nullptr
  s1->play();												// RUNTIME ERROR: can't dereference nullptr
  ...
}
```

Stealing the resource of a lvalue causes error when we dereference that lvalue again. Thus, we need here a **move constructor** rather than a **copy constructor**. i.e. Only an rvalue smart pointer shall transfer its ownership to another smart poiner. `auto_ptr` tries to implement move semantics by overriding copy constructor, which is why it fails.

### Unique Pointer

A simplied implementaion of `unique_ptr`  is illustrated as

```c++
template <typename T>
class unique_ptr{
	T* ptr;
public:
  // empty unique pointer owns nothing
  unique_ptr() noexcept:ptr(nullptr){}
  
  // nonempty unique pointer owns the resource
  explicit unique_ptr(T* p) noexcept:ptr(p){}
  
  // automatically delete memory
  ~unique_ptr() noexcept {delete ptr}
  
  // unique pointer has NO copy constructor
  unique_ptr(const unique_ptr& src) = delete
    
  // unique pointer has NO copy assignment operator
  unique_ptr& operator=(const unique_ptr& src) = delete;
  
  // move constructor
  unique_ptr(unique_ptr&& src):noexcept{
    this->ptr = src.ptr;
    src.ptr   = nullptr;
  }
  
  // move assignment operator
  unique_ptr& operator=(unique_ptr&& src) noexcept{
    delete this->ptr;			// 'this' must first free its current resource
    this->ptr = src.ptr;
    src.ptr   = nullptr;
    return *this;
  }
  
  // give up ownership without freeing the memory
  T* release() noexcept{
    T* raw_p = this->ptr;
    this->ptr = nullptr;
    return raw_p;					// The caller must free raw_p later on
  }
  
  // free old resource and take new ownership
  void reset(T* p = nullptr) noexcept{
    delete ptr;
    ptr = p;
  }
};
```

Note: The actual implementaion of `unique_ptr` is more complex than we illustrated here. Nonetheless, the code snippets still give us some key insights of `unique_ptr` :

* `unique_ptr` has **NO** copy constructors or copy assignment operator, i.e. it is move only.
* Exclusive ownership of the resource. This is a direct consequence of move-only property
* `unique_ptr` has proper move semantics by restricting that only moving from rvalues is allowed.
* If we insist on moving from an lvalue, we must use `std::move`.

To define unique pointers, we have two options:

```c++
std::unique_ptr<Song> s1{new Song("Oh my Love", "John Lennon")};		\\ 1)
auto s1 = std::make_unique<Song>("Oh my Love", "John Lennon");			\\ 2)
```

Option 1 is straightforward. Option2 uses the factory function `make_unique`, which unifies three operations

- allocate heap memory, 
- construct the Song object with arguments 
- wrap the Song object into `unique_ptr`

> When possible, prefer `make_unique` for the sake of readability.

**Moving a unique pointer**

```c++
auto s1 = std::make_unique<Song>("Yellow Submarine", "The Beatles");			
  
// ERROR: can't move from lvalue 
// (vs. auto ptr quitely stealing the ownership)
unique_ptr<Song> s2 = s1;

// OK: std::move converts lvalue to xvalue
unique_ptr<Song> s2 = std::move(s1);		

// OK: The factory make_unique produces an rvalue
unique_ptr<Song> s1 = std::make_uniqe<Song>("High Voltage", "AC/DC"); 
```

**Support for Array Types**

Both `unique_ptr` and `make_unique` support array types (both from C++11). When applied to array-type objects, the array-type `delete[]` will be called. 

```c++
std::unique_ptr<Song[]> play_list{new Song[100]}			\\ 1)
auto play_list = std::make_unique<Song[]>(100);				\\ 2)
```

Note: In approach 2), the parameter is the number of elements in array-like object, not the constructor arguments.

**Move into a Function**

> Pass `unique_ptr` by value into a function $\iff$ Move the ownership from caller to callee

Note: Passing `unique_ptr`  by reference is not recommanded unless three is a good reason to do so.

```c++
bool is_rock(std::unique_ptr<Song> s){
  return s.get_genre() == "rock";
}
```

**Move out of a Function**

> Return `unique_ptr` by value $\iff$ Move the ownership from callee to caller

The resource allocated in the callee is guranteed to be freed by the caller.

```c++
std::unique_ptr<Song> create_song(std::string title, std::string autor){
  auto s = std::make_unique<Song>(title, author);
  return s;
}
```

**Unique Pointers and STL**

To track a set of `unique_ptr`, we can pack them into `vector`

```c++
std::vector<std::unique_ptr<T>>
```

**PITFALLS**

1. Never pass a raw pointer to $>1$ `unique_ptr` objects! 

   ```c++
   Song* caladan = new Song("Leaving Caladan", "Hans Zimmer");
   std::unique_ptr<Song> s1{caladan};
   std::unique_ptr<Song> s2{caladan};
   ```

   In this example, the `Song` object is heap is freed twice. The program will crash due to double free. 

   > Best Practice: Don't create `unique_ptr` from raw pointer unless it is stricly necessary 

2. `unique_ptr` is not immuned to dangling pointer

   ```c++
   Song* favt = nullptr;
   {
     auto s = std::make_unique<Song>("City of Stars", "Justin Hurwitz");
     s.add_lyric(text);
     favt = s.get();							// get the raw pointer
   }															// favt becomes dangling
   std::cout << favt->author;		// undefined behaviour
   ```

### Shared Pointer

Properties:

* $>1$ owners can own the resource at a time
* `shared_ptr` has copy constructors, i.e. It is copyable (vs. Unique pointer only movable)
* The resource will be freed when the **last owner** dies. This avoids dangling pointer.
* There is a control block in heap, which makes sure that no owner than the last one may free the resource.

<img src="./figs/Shared Pointer.jpg" alt="Shared Pointer" style="zoom:20%;" />

**API:**

We give a brief introduction to the API by omitting the discussion of the details.

* Default constructor: No resource is owned. No control block is allocated.

  ```c++
  shared_ptr() noexcept;
  ```

* Explicit constructor: Own the resource. Allocate control block. Set `counter=1`

  ```c++
  explicit shared_ptr(T*);
  ```

* Destructor: `this ` no longer owns the resource.  `--counter`. If `counter==0`, free the resource.

  ```c++
  ~shared_ptr();
  ```

* Copy constructor: Add `this` to the set of owners. `++counter`

  ```c++
  shared_ptr(const shared_ptr& source) noexcept;
  ```

* Move constructor 1: move the ownership from another shared pointer to `this` . `counter` remains.

  ```c++
  shared_ptr(shared_ptr&& source) noexcept;
  ```

* Move constructor 2: move from unique pointer to shared pointer (not allowed vice versa).

  ```c++
  shared_ptr(unique_ptr&& source);
  ```

* Factory:  In addition, it allocate one memeory block in heap for both the resource and the control block.

  ```c++
  shared_ptr<T> make_shared(Arg args);		// for managing an object
  shared_ptr<T> make_shared(size_t N);		// for managing an array
  ```

> ALWAYS create new shared pointer from existing shared pointers, NEVER from raw pointers! 

Bad Example: Construct shared pointers from a raw pointer

```c++
Song s = new Song("Imagine", "Lennon");
std::shared_ptr<Song> s1(s);
std::shared_ptr<Song> s2(s);	// RUNTIME ERROR as s2 dies: double free
```

This leads to runtime error. As `s1` dies, the song object is destroyed. `s2` becomes dangling pointer. As `s2` dies, it tries to destroy the song object again. $\implies$ undefined behaviour. Moreover, `s1` and `s2` have separate control blocks, which won't work together.

Good Example: Construct shared pointers from existing shared pointers

```c++
auto s1 = make_shared<Song>("Imagine", "Lennon"); // counter=1
std::shared<Song> s2(s1);													// counter=2
std::shared<Song> s3 = s1;												// counter=3
```

**Support for Array-Like**

* `shared_ptr` support array-like objects from C++17
* `make_shared` support array-like objects from C++20

**Pitfalls: Cyclic reference**

**BEST PRACTICE**

* If single owner is needed $\implies$ `unique_ptr`
* If multi owners are needed $\implies$ `shared_ptr`
* Not sure? $\implies$ Prefer `unique_ptr` since it can be transferred to `shared_ptr` later on. Hence, more flexibility.

### Weak Pointer

* Weak pointers are "slaves" of shared pointers
* Weak pointers does not claim the ownership. i.e. It has no effect on control block.
* Weak pointer can check the validality of the data. (whether it is freed.)

## Lambda

## Type deduction

## IOStream

### Ostream

### Istream

## Guidelines for Modern C++

The guidelines originate from practice and help the programmer write robust and managable software. 

The guidelines are not bibles. 

### Rule of Zero

### Copy and Swap Idiom
