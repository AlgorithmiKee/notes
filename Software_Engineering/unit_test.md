# Unit Test

## Test a single function

* Given the function

  ```C++
  int factorial(int n);
  ```

* Test

  ```C++
  TEST(FactorialTest, HandlesZeroInput) {
    EXPECT_EQ(Factorial(0), 1);
  }
  
  TEST(FactorialTest, HandlesPositiveInput) {
    EXPECT_EQ(Factorial(1), 1);
    EXPECT_EQ(Factorial(2), 2);
    EXPECT_EQ(Factorial(3), 6);
    EXPECT_EQ(Factorial(8), 40320);
  }
  ```

Remarks

* `FactorialTest` is called a **test suite**
* `HandlesZeroInput` is called a **test case**
* When running unit test, one can filter the test by specifying `gtest_filter`
  * `./foo_test --gtest_filter=*` runs everything
  * `./foo_test --gtest_filter=A.*` runs everything in test suite `A`
  * `./foo_test --gtest_filter=A.b` runs a specific test case `b` in test suite `A`
* Although in our the example, the test cases are quite short. In general, a test case typically consists of
  * Arrange: set things up. It can either be put in the test body or in `SetUp()` when using fixures (*will see later*).
  * Expecations (*optional*): define what to expect, e.g. what function should be called how many times.
  * Act: call some functions
  * Assert: check the result. Report failure if something went wrong.

## Test with Fixures

Basic idea: fixture groups tests that share common setup and teardown actions.

Syntax:

```C++
// define fixure
class TestFixure : public testing::Test{
protected:
  void SetUp() override {
    ...
  }

  void TearDown() override {
    ...
  }
};

// define unit tests: use TEST_F
TEST_F(TestFixure, TestCase1){
  ...
}

TEST_F(TestFixure, TestCase2){
  ...
}

TEST_F(TestFixure, TestCase3){
  ...
}
```

In this chapter, we illustrate the ideas in greater details on the following example

* Unit

  ```C++
  template <typename E>
  class Queue{
  private:
    std::vector<E> _v,
  
  public:
    void Enqueue(const E& element);
    E Dequeue;
    size_t size() const;
    bool IsEmpty();
  }
  ```

* Fixure: The fixure `QueueTest` defines common setup and teardown actions for all tests within it. Here, we initialize the queue with two elements.

  ```C++
  class QueueTest : public ::testing::Test{
  protected:
    Queue<int> q_;
      
    // we want to begin our test with a nonempty queue
    void SetUp() override{
      q_.Enqueue(1);
      q_.Enqueue(2);
    }
  
    void TearDown() override{
      std::cout << "==== Test Ended ====" << std::endl;
    }
  
  }
  ```

* Tests: Here, all three test cases share the same setup and tear down actions. Each test case gets a fresh instance of the test fixture. e.g. The modification on `q_` in one test case won't carry over to any other test cases.

  ```C++
  TEST_F(QueueTest, IsNotEmtpyInitially){
    EXPECT_EQ(q_.size(), 2);
    EXPECT_FALSE(q_.IsEmpty());
  }
  
  TEST_F(QueueTest, EnqueueWorks){
    q_.Enqueue(3);
    EXPECT_EQ(q_.Dequeue(), 1);
    EXPECT_EQ(q_.Dequeue(), 2);
    EXPECT_EQ(q_.Dequeue(), 3);
  }
  
  TEST_F(QueueTest, DequeueWorks){
    EXPECT_EQ(q_.Dequeue(), 1);
    EXPECT_EQ(q_.Dequeue(), 2);  
  }
  ```

Remarks:

* before `TEST_F` is called,
  * An object of `QueueTest` is initiated and
  * `QueueTest::SetUp()` is called
  * In general, `SetUp()` may contain constructors
* for each `TEST_F`, a separate object of `QueueTest` is initialised so that they do not interfere.
* As each `TEST_F` ended, `QueueTest::TearDown()` is called
  * In general, `TearDown()` may contain destructors

Fixure is more capable than just setting up common data.  For example, we can detect the run time of each test case. We claim a test case to be failed if it takes too long. This can be implemented by modiying our example as follows

* Fixure

  ```C++
  class QueueTest : public ::testing::Test{
    protected:
    void SetUp() override{
      auto t_start = std::chrono::high_resolution_clock::now();
      q_.Enqueue(1);
      q_.Enqueue(2);
    }
  
    void TearDown() override{
      auto t_end = std::chrono::high_resolution_clock::now();
      auto duration = 
        std::chrono::duration_cast<std::chrono::milliseconds>(
          t_end - t_start
        )
      
      std::cout << "Test took: " << duration.count() << "ms" << std::endl;
      EXPECT_TRUE(duration.count() <= 1000) << "The test took too long!";
      std::cout << "==== Test Ended ====" << std::endl;
    }
  
    Queue<int> q_;
  }
  ```

  Now, each test case taking more than 1000 ms will fail.

## Mocking

Motivation: The system under test may has collaborators (i.e. dependencies) as files, database, internet connection, external libraries. In reality, it may be difficult or too expensive to set everything up just for a unit test. To address this issue, we use a mock of each collaborator instead of the real collaborator. A mock is an object X, to which we can specify

* which methods of X should be called?
* How each method should be called?
* What each method return

### Motivation

In this chapter, we examin mocks in greater details using the ATM machine example. Suppose we have the system under test. We want to test `AtmMachine::WithDraw()`.

```C++
// AtmMachine.h
class AtmMachine {
  BankServer* _bankServer; // AtmMachine uses BankServer
  
public:
  AtmMachine(BankServer* bankServer) : _bankServer(bankServer) {}

  bool WithDraw(int accounNumberIn, int amountIn) {
    bool success = false;

    _bankserver->Connect();
    auto availableBalance = _bankServer->GetBalance(accountNumberIn);

    if (availableBalance >= amountIn){
      _bankServer->Debit(accountNumberIn, amountIn);
      success = true;
    }

    _bankServer->Disconnect();
    return success;
  }
};
```

The collaborator has the following interfaces defined

```C++
// BankServer.h 
class BankServer {
public:
  virtual ~BankServer() {};
  virtual void Connect () = 0;
  virtual void Disconnect () = 0;
  virtual void Credit(int accountNumberIn, int amountIn) = 0;
  virtual void Debit(int accountNumberIn, int amountIn) =0;
  virtual bool DoubleTransaction(int accountNumberIn, int valuel, int value2) = 0;
  virtual int GetBalance(int accountNumberIn) const = 0;
};
```

Suppose in addition that the interface is implemented y `UbsBankServer`

```C++
// UbsBankServer.h 
#include <many_dependencies.h>
#include "BankServer.h"

class UbsBankServer : public BankServer {
  // Implementation of BankServer interfaces...
};
```

Without mocking, testing becomes almost impossible because we must ensure all dependencies of UbsBankServer are ready, including having a real UBS bank server available, a stable internet connection, and other requirements. Additionally, our ATM Machine might interact with another bank server (e.g., Deutsche Bank), which could have a different implementation of the interface and potentially different dependencies, making testing even more challenging.

Our goal is to test the logic of AtmMachine::WithDraw() itself. These external dependencies are irrelevant for testing the specific method we care about. Therefore, mocking is useful as it eliminates the need for external dependencies and allows us to focus on the method in question.

### Mocked Class

What is a mock?

> A mock class is just a subclass of the interface with mocked methods.

Defining a mock class is easy using gmock framework. In our examle, simply do

```C++
// MockBankServer.h
#include "BankServer.h"
#include "gmock/gmock.h"

// No more dependencies!

class MockBankServer : public BankServer {
  MOCK_METHOD(void, Connect, (), (overrride)); // NEW SYNTAX
  MOCK_METHOD(void, Disconnect, (), (overrride));
  MOCK_METHOD(void, Credit, (int, int), (overrride));
  MOCK_METHOD(void, Debit, (int, int), (overrride));
  MOCK_METHOD(bool, DoubleTransaction, (int, int, int), (overrride));
  MOCK_METHOD(int, GetBalance, (int), (const, overrride));
};
```

Remark to mocking:

* We mock the interface `BankServer`, not the implementation `UbsBankServer`.
* The real collaborator `UbsBankServer` is used in production and has many dependencies.
* The mocked collaborator `MockBankServer` is used in testing and has no dependencies.
* Defining a mocked class as above does not yet specify the behaviour of mocked methods, which should be specified later in test definition.

A closer look at mocked method:

```c++
  MOCK_METHOD(return_type, method_name, (param_type), (specs));
```

Remark:

* `MOCK_METHOD` is a pre-defined macro and thus has limitations (*see below*)
* The field `spces` may include
  * `const`: required when overrriding const method
  * `overried`: recommended when overriding a virtual method
  * `noexcept`: required when overriding a nonexcept method

Mocking a function with template => use alias

### Testing with Mock

In testing, we use the mock instead of the real collaborator. Inside the test case, we define the behaviour of mocked methods. In our ATM example, we could define a test case by assuming sufficient fund in the banck account.

```C++
// BankServerTest.h
#include "MockBankServer.h"

TEST(AtmMachine, CanWithDraw){
  //// --- Arrange --- ////
  MockBankServer mBankServer;
  AtmMachine atm(&mBankServer); // OOD: dependency injection

  int accountNumber = 5020'9248'4090'0012;
  int withdrawAmount = 50;
  int initialBalance = 1000;

  // specify the behavior of mocked methods
  EXPECT_CALL(mBankServer, Connect());
  EXPECT_CALL(mBankServer, GetBalance(accountNumber))
      .WillOnce(Return(initialBalance));
  EXPECT_CALL(mBankServer, Debit(accountNumber, withdrawAmount));
  EXPECT_CALL(mBankServer, Disconnect());

  //// --- Act--- ////
  bool withDrawSuccess = atm.WithDraw(withdrawAmount, initialBalance)

  //// --- Assert--- ////
  EXPECT_TRUE(withDrawSuccess);
}
```

### Mock with Fixture

Mock can be combined with fixture, which is common practice in software engineering. In our ATM example, we could define a fixture `AtmMachineTest` to set up the atm machine, mocked banked server, initial balance, etc.

The fixture looks like

```c++
class AtmMachineTest : public ::testing::Test {
protected:
    MockBankServer mBankServer;
    AtmMachine atm(&mBankServer);
    int accountNumber = 5020'9248'4090'0012;
    int initialBalance = 1000;
    
    // No setup() or teardown() in this example.
    
};
```

In our simple example, there is no need to define `SetUp()` and `TearDown()`. If our `AtmMachine` requires more comoplex initialization, we might define it in `SetUp()`.

```c++
class AtmMachineTestAlternative :  public ::testing::Test {
public:
    std::shared_ptr<MockBankServer> mBankServer;
    std::shared_ptr<AtmMachine> atm;
    int accountNumber = 5020'9248'4090'0012;
    int initialBalance = 1000;
    
    void SetUp() override {
        mBankServer = std::make_shared<MockBankServer>();
        atm = std::make_shared<AtmMachine>(mBankServer.get());
    }
    
    void TearDown() override {
        // resouce clean up is done automatically by shared_ptr()
    }
};
```

Nevertheless, we stick to the 1st variant of fixture. The test case `CanWithDraw` is then

```c++
TEST_F(AtmMachineTest, CanWithDraw) {
    // Arrange: specify the behavior of mocked methods
    EXPECT_CALL(mBankServer, Connect());
    EXPECT_CALL(mBankServer, GetBalance(accountNumber))
      .WillOnce(Return(initialBalance));
    EXPECT_CALL(mBankServer, Debit(accountNumber, withdrawAmount));
    EXPECT_CALL(mBankServer, Disconnect());

    // Act
    bool withDrawSuccess = atm.withDraw(accountNumber,withDrawAmount);
    
    // Assert
    EXPECT_TRUE(withDrawSuccess);
}
```

We can also define another test case `InsufficientFund` where the withdraw is expected to fail. In this test case, we have slightly different behaviour of mocked methods. The function `MockBankServer::Debit()` should not be called in this case.

``` c++
TEST_F(AtmMachineTest, InsufficientFund) {
    // Arrange: specify the behavior of mocked methods
    EXPECT_CALL(mBankServer, Connect());
    EXPECT_CALL(mBankServer, GetBalance(accountNumber))
      .WillOnce(Return(initialBalance));
    EXPECT_CALL(mBankServer, Disconnect());

    // Act
    bool withDrawSuccess = atm.withDraw(accountNumber,withDrawAmount);
    
    // Assert
    EXPECT_FALSE(withDrawSuccess);
}
```

## Parameterized Testing

### Motivation

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

Suppose we want to test the method `Queue::containsValue()`. A direct approach involves writing multiple test cases with repeated code.

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

While this works, the code has repeated logic that can be abstracted out. Parameterized testing allows us to share test logic while using different sets of inputs.

### Steps to Create a Parameterized Test

**STEP 1**: Declare the parameterized test suite.

```c++
class QueueContainsValueTest : public testing::TestWithParam<std::tuple<Queue<int>, int, bool>> {};
```

Remarks:

* `QueueContainsValueTest` is a parameterized test suite. Any parameterized test suite must inherit from `TestWithParam`.
* The `TestWithParam` type specifies the format of the test data. Here, it’s a tuple of three elements: a `Queue<int>` object, a target value (`int`), and a boolean indicating the expected result..

**STEP 2**: Instantiate test data. i.e. Define the input values for the tests

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

Remark:`INSTANTIATE_TEST_SUITE_P` requires:

* **instantiation name**: describes the set of parameters (here, `QueueInt` since these queues contain integers).
* **test suite name**: must match test suite declaration in the 1st step. Otherwise, Google Test won't know which test suite the parameters belong to.
* **parameter values**: each parameter is a 3er tuple. Later, we can use `GetParam()` to iterate through parameters.

**STEP 3**: Implement the test logic.

```c++
TEST_P(QueueContainsValueTest, WorksWithIntInputs){
  auto [q, targetValue, expected] = GetParam();      // From C++17
  EXPECT_EQ(expected, q.containsValue(targetValue));
}
```

Remark:

* The syntax `auto [q, targetValue, expected] = GetParam()`, called ***structured bindings***, is supported from C++17.

* When working with C++14 or lower, use `std::get()` to access elements of the tuple.

```c++
TEST_P(QueueContainsValueTest, WorksWithIntInputs){
auto param = GetParam();
q           = std::get<0>(param);  // Backwards compatible to C++14 or lower
targetValue = std::get<1>(param);  // Backwards compatible to C++14 or lower
expected    = std::get<2>(param);  // Backwards compatible to C++14 or lower
EXPECT_EQ(expected, q.containsValue(targetValue));
}
```

### Alternative: Use Struct for Test Data

Instead of tuples, you can group test data into a struct for better readability:

```c++
struct TestData{
  Queue<int> q;
  int targetValue;
  bool expected;
};
```

The test suite declaration should be changed so that a `TestData` should be passed into the test instead of a tuple.

```c++
class QueueContainsValueTest : public testing::TestWithParam<TestData> {};
```

In test data intantiation, the constructor `TestData` should be called instead of `std::make_tuple()`. 

```c++
INSTANTIATE_TEST_SUITE_P(
  QueueInt,                 // instantiation name
  QueueContainsValueTest,   // test suite name
  ::testing::Values(
    // test case: no mathch
  	TestData{
        .q = Queue<int>{1, 2, 3, 4},
        .targetValue = 5, 
        .expected = false
    },
    // test case: single mathch
    TestData{
        .q = Queue<int>{1, 2, 3, 4},
        .targetValue = 3, 
        .expected = true
    },
    // test case: multi mathch
    TestData{
        .q = Queue<int>{1, 2, 3, 3},
        .targetValue = 3, 
        .expected = true
    },
	// test case: empty queue
    TestData{
        .q = Queue<int>{},
        .targetValue = 1, 
        .expected = false}
  	}
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

## Advanced Topics

### Test Private Methods

Note: Not encouraged to test private methods.

Possible choices for testing private methods:

1. use `FRIEND_TEST()`
1. declare the test fixture as a friend class
1. use **Pimpl** idiom
