---
title: "STL"
author: "Ke Zhang"
date: "2025"
---

# Standard Template Library

## Vector

Stores an array of elements in continuous memory. The size of the array is dynamically adjustable.

* Construction:
  ```c++
  #include<vector>
  // a vector of 4, 5, 4, 7
  std::vector<int> u{4, 5, 4, 7};
  // a vector of length n filled with 0
  std::vector<int> v(n, 0);
  ```
* Search: `vector` has no memeber function `.find()`. We must use `std::find()`!
  ```c++
  // locate the iterator pointing to val
  // returns v.end() if not found
  auto it = std::find(v.begin(), v.end(), val)
  // if found val in v
  if(it != v.end())
  ```
* Sort: `vector` has no memeber function `.sort()`. We must use `std::sort()`!
  ```c++
  // sort 
  std::sort(v.begin(), v.end());
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

## 2D Array

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

## Unordered Map

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
