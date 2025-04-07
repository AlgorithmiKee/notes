# Add-on

## Template specialization vs. template instantiation

## Initialization

You're absolutely right to feel confused—**"list initialization"** in C++ is an overloaded term that has different meanings depending on the context. Let me clarify the distinction between **list initialization** in general and the use of `std::initializer_list`. By breaking it down, I’ll also explain when each applies and how they differ.

---

### **What is List Initialization in General?**
In C++, **list initialization** broadly refers to the use of curly braces `{}` to initialize an object. This was introduced in C++11 and refined in later standards (C++14, C++17, etc.).

For example:
```cpp
MyClass obj{1, 2.5};
```

In this context:
- The term "list initialization" **does not imply `std::initializer_list` usage**.
- Instead, it means that curly braces `{}` are used to initialize the object.
- What happens internally depends on the constructors and types involved.

### Types of List Initialization
List initialization can take different forms depending on syntax and context:

1. **Direct List Initialization**:
   ```cpp
   MyClass obj{1, 2.5};
   ```
   This form directly calls a constructor of `MyClass` with the elements inside `{}`. It works if there is a constructor in `MyClass` that matches the given arguments.

2. **Copy List Initialization**:
   ```cpp
   MyClass obj = {1, 2.5};
   ```
   This form is similar to direct list initialization but uses the `=` syntax. Behind the scenes, it behaves the same as direct list initialization.

3. **Aggregate Initialization**:
   If the object being initialized is an aggregate (e.g., a `struct` or `class` with no user-defined constructors), list initialization directly assigns the values to its members:
   ```cpp
   struct MyStruct {
       int x;
       double y;
   };

   MyStruct obj{1, 2.5}; // Aggregate initialization
   ```

4. **Use of `std::initializer_list`**:
   If a constructor accepting `std::initializer_list` exists, and the braces match the signature of that constructor, it will take precedence:
   ```cpp
   class MyClass {
   public:
       MyClass(std::initializer_list<int> init) {
           // Do something with init
       }
   };

   MyClass obj{1, 2, 3}; // Calls the std::initializer_list<int> constructor
   ```

---

### **When Does `std::initializer_list` Come Into Play?**
`std::initializer_list` is a specialized feature that enables constructors to handle a **sequence of elements**. It comes into play when:
1. **A Constructor Accepting `std::initializer_list` Exists**:
   ```cpp
   class MyClass {
   public:
       MyClass(std::initializer_list<int> init) {
           // Do something with init
       }
   };

   MyClass obj{1, 2, 3}; // Calls the std::initializer_list<int> constructor
   ```

2. **The Compiler Prefers `std::initializer_list`**:
   If both a constructor taking a `std::initializer_list` and another constructor (e.g., taking multiple arguments) are viable, the compiler will prefer the `std::initializer_list` constructor.

   Example:
   ```cpp
   class MyClass {
   public:
       MyClass(int x, int y) {
           std::cout << "Two-argument constructor\n";
       }
       MyClass(std::initializer_list<int> init) {
           std::cout << "initializer_list constructor\n";
       }
   };

   MyClass obj{1, 2}; // Calls the initializer_list constructor, not the two-argument constructor.
   ```

   If you want to force the non-`std::initializer_list` constructor, you must use parentheses:
   ```cpp
   MyClass obj(1, 2); // Calls the two-argument constructor.
   ```

---

### **How List Initialization and `std::initializer_list` Differ**
The confusion arises because list initialization (`{}`) is a **syntax feature**, while `std::initializer_list` is a **type and a library feature**. Here’s how they differ:

| **Feature**               | **List Initialization**                 | **std::initializer_list**          |
|---------------------------|-----------------------------------------|-------------------------------------|
| **Purpose**               | General syntax for initialization       | Represents a fixed-size sequence of values |
| **Trigger**               | Happens whenever `{}` is used for initialization | Requires a constructor with `std::initializer_list` |
| **Behavior**              | Calls a matching constructor or aggregates | Constructs an `initializer_list` object |
| **Scope**                 | Works for any type with matching constructors | Requires explicit support in the class |
| **Precedence**            | `std::initializer_list` constructor takes precedence | Must match the type of initializer values |

---

### **Examples to Differentiate**

#### Example 1: No `std::initializer_list` Constructor
If the class has no `std::initializer_list` constructor:
```cpp
class MyClass {
public:
    MyClass(int x, double y) {
        std::cout << "Two-argument constructor\n";
    }
};

MyClass obj{1, 3.14}; // Direct list initialization, calls the (int, double) constructor.
```

Here, `{1, 3.14}` is **list initialization**, but it’s not related to `std::initializer_list`.

---

#### Example 2: With `std::initializer_list`
If a `std::initializer_list` constructor is added:
```cpp
class MyClass {
public:
    MyClass(int x, double y) {
        std::cout << "Two-argument constructor\n";
    }
    MyClass(std::initializer_list<int> init) {
        std::cout << "initializer_list constructor\n";
    }
};

MyClass obj{1, 3}; // Calls the initializer_list constructor.
```

Now, `{1, 3}` triggers the `std::initializer_list` constructor because the compiler prefers it over other constructors.

---

#### Example 3: Forcing Non-`std::initializer_list` Constructor
To force the non-`std::initializer_list` constructor:
```cpp
MyClass obj(1, 3); // Uses parentheses, so the (int, double) constructor is called.
```

---

### **Summary**
1. **List initialization** refers to the use of `{}` braces to initialize an object.
2. **`std::initializer_list`** is a specific feature that provides support for sequence initialization, which takes precedence when a matching constructor exists.
3. The confusion arises because:
   - `{}` braces are used in both contexts.
   - When a `std::initializer_list` constructor exists, it often takes precedence over other constructors.

Understanding the precedence of constructors and when `std::initializer_list` is triggered helps clarify the behavior.

## Parameterized Testing

Consider the class

```c++
template <typename E>
class Queue{
private:
  std::vector<E> _v,
public:
  void enqueue(const E& element);
  E dequeue;
  size_t size() const;
  bool isEmpty();
  bool containsValue(E targetValue)
};
```

Suppose we want to test the method `Queue::containsValue()`. A direct approach is writing multiple test cases with repeated code.

```c++
TEST(QueueContainsValueTest, NoMatch){
  Queue<int> q{1, 2, 3, 4};
  int targetValue = 5;
  bool expected = false;
  EXPECT_EQ(expected, q.containsValue(targetValue))
}

TEST(QueueContainsValueTest, SingleMatch){
  Queue<int> q{1, 2, 3, 4};
  int targetValue = 3;
  bool expected = true;
  EXPECT_EQ(expected, q.containsValue(targetValue))
}

TEST(QueueContainsValueTest, MultiMatch){
  Queue<int> q{1, 2, 3, 3};
  int targetValue = 3;
  bool expected = true;
  EXPECT_EQ(expected, q.containsValue(targetValue))
}

TEST(QueueContainsValueTest, EmptyQueue){
  Queue<int> q{};
  int targetValue = 3;
  bool expected = false;
  EXPECT_EQ(expected, q.containsValue(targetValue))
}
```

To get rid of repeated code, we use parameterized test. The basic idea is to make test inputs as parameters so that the test logic can be shared across test cases.

**STEP 1**: Declare the test suite to be parameterized.

```c++
class QueueContainsValueTest : public testing::TestWithParam<std::tuple<Queue<int>, int, bool>> {};
```

Remark:

* `QueueContainsValueTest` is a parameterized test suite. Any parameterized test suite is a subclass of `TestWithParam`.
* The class name inside the parenthesis `TestWithParam<?>` specifies the data type accepted by test cases. Here it is a 3er tuple, representing the queue object, target value and whether the target value is contained in the queue.

**STEP 2**: Define the test data instantiation

```c++
INSTANTIATE_TEST_SUITE_P(
  QueueInt,                 // instantiation name
  QueueContainsValueTest,   // test suite name
  ::testing::Values(
    std::make_tuple(Queue<int>{1, 2, 3, 4}, 5, false), // test case: no mathch
    std::make_tuple(Queue<int>{1, 2, 3, 4}, 3, true),  // test case: single mathch
    std::make_tuple(Queue<int>{1, 2, 3, 3}, 3, true),  // test case: multi mathch
    std::make_tuple(Queue<int>{}, 1, false)            // test case: empty queue
  )
);
```

Remark:

* `INSTANTIATE_TEST_SUITE_P` takes at least three arguments: instantiation name, test suite name, and parameter values.
* The **instantiation name** describes the set of parameters. Here, we group the parameters under the name `QueueInt` as the queues are all int-valued.
* The **test suite name** must match test suite declaration in the 1st step. Otherwise, Google Test won't know which test suite the parameters belong to.
* The **parameter values** are separated by `,`. Here, each parameter is a 3er tuple. Later, we can use `GetParam()` to iterate through the parameters.

**STEP 3**: Implement the parameterized test.

```c++
TEST_P(QueueContainsValueTest, WorksWithIntInputs){
  auto [q, targetValue, expected] = GetParam();      // From C++17
  EXPECT_EQ(expected, q.containsValue(targetValue));
}
```

Remark:

* The syntax `auto [q, targetValue, expected] = GetParam()`, called ***structured bindings***, is supported from C++17. When working with C++14 or lower, use `std::get()` to access elements of the tuple.

  ```c++
  TEST_P(QueueContainsValueTest, WorksWithIntInputs){
    auto param = GetParam();
    q           = std::get<0>(param);  // Backwards compatible to C++14 or lower
    targetValue = std::get<1>(param);  // Backwards compatible to C++14 or lower
    expected    = std::get<2>(param);  // Backwards compatible to C++14 or lower
    EXPECT_EQ(expected, q.containsValue(targetValue));
  }
  ```

### Test Data as Struct

So far, we grouped the test parameters as a tuple. Alternatively, we can group them in a struct. For that, we need to define the stuct additionally.

```c++
struct TestParam{
  Queue<int> q;
  int targetValue;
  bool expected;

  TestParam(Queue<int> qIn, int targetValueIn, bool expectedIn) :
  q{qIn}, targetValue{targetValueIn}, expected{expectedIn} {}
};
```

The test suite declaration should be changed so that a `TestParam` should be passed into the test instead of a tuple.

```c++
class QueueContainsValueTest : public testing::TestWithParam<TestParam> {};
```

In test data intantiation, the constructor `TestParam` should be called instead of `std::make_tuple()`.

```c++
INSTANTIATE_TEST_SUITE_P(
  QueueInt,                 // instantiation name
  QueueContainsValueTest,   // test suite name
  ::testing::Values(
    TestParam{Queue<int>{1, 2, 3, 4}, 5, false}, // test case: no mathch
    TestParam{Queue<int>{1, 2, 3, 4}, 3, true},  // test case: single mathch
    TestParam{Queue<int>{1, 2, 3, 3}, 3, true},  // test case: multi mathch
    TestParam{Queue<int>{}, 1, false}            // test case: empty queue
  )
);
```

In test implementation, access struct fields via either structured binding or conventional `.` operator.

```c++
 // From C++17
TEST_P(QueueContainsValueTest, WorksWithIntInputs){
  auto [q, targetValue, expected] = GetParam();
  EXPECT_EQ(expected, q.containsValue(targetValue));
}

// Backwards compatible to C++14 or lower
TEST_P(QueueContainsValueTest, WorksWithIntInputs){
  auto param = GetParam();
  q           = param.q;  
  targetValue = param.targetValue; 
  expected    = param.expected; 
  EXPECT_EQ(expected, q.containsValue(targetValue));
}
```
