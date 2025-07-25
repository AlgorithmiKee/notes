---
title: "Algorithms"
author: "Ke Zhang"
date: "2025"
---

# Algorithms

## Peak Element

Let array $\mathbf a = [a_0, \dots, a_{n-1}], \: n \ge 1$ be such that

$$
a_k \ne a_{k+1}, \quad \forall k = 0,\dots,n-2
$$

We call $a_k$ a ***peak element*** of $\mathbf a$ if $a_k$ is larger than its neighours, i.e.

$$
a_k > a_{k-1} \:\land\: a_k > a_{k+1}
$$

Note: we assume $a_{-1} = a_n = - \infty$ at boundaries.

**Task**: Write an algorithm to find the index of an arbitrary peak element. The time complexity should be $O(\log n)$. The index is counted from zero.

**Example**: For $\mathbf a = [2, 1, 3, 4, 6, 1]$, the peak elements are 2 and 6. The algorithm might return either 0 or 4.

### Existence of Peak Element

Before we develop the algorithm, we first show the existence of peak element.

> **Fact**: There is at least one peak element in $\mathbf a$.

*Proof*: If $n=1$, then $a_0$ is the peak element due to boundary conditions.

For $n > 1$, suppose (for the sake of contradicion) that there is no peak element in $\mathbf a$. Note that $a_0 > a_{-1}$ but $a_0$ is not a peak element. Hence, $a_1 \ge a_0$. Since $a_1 \ne a_0$, we must have $a_1 > a_0$. Similarly, $a_2 > a_1$.

Repeating the same logic, we conclude that

$$
a_{n-1} > a_{n-2} > \dots > a_1 > a_0
$$

However, $a_{n-1} > a_n = -\infty$, making $a_{n-1}$ a peak element. Contradiction. $\quad\blacksquare$

The above fact can be generalized into

> **Claim**: Suppose there exists $i,j \in\mathbb N$ such that
> $$
>   a_{i-1} < a_{i} \:\land\: a_{j} > a_{j+1}, \quad 0 \le i < j \le n-1
> $$
> Then, there exist at least a peak element $a_p$ with $i \le p \le j$

*Proof*: The logic is the same as before: The sequence can not be monotonic between $i-1$ and $j+1$. Details omitted. $\quad\blacksquare$

### Binary Search

We start by defining indices $\ell=0, r=n-1$. By boundary condition, we have the inequalities

$$
a_{\ell-1} < a_{\ell} \:\land\: a_{r} > a_{r+1}
$$

By previous analysis, we know there is at least a peak element between $a_\ell$ and $a_r$. To narrow down the range of peak elements, we consider

$$
a_m \stackrel{?}{\gtrless} a_{m+1}, \quad m = \left\lfloor \frac{r-\ell}{2}\right\rfloor
$$

* If $a_m > a_{m+1}$, we have
  $$
  a_{\ell-1} < a_{\ell}  \:\land\: a_m > a_{m+1}
  $$
  By the claim of existence of peak elements, we narrowed down the range of some peak element to $a_\ell,\dots,a_m$.

* If $a_m < a_{m+1}$, we have
  $$
  a_m < a_{m+1}  \:\land\: a_{r} > a_{r+1}
  $$
  By the claim of existence of peak elements, we narrowed down the range of some peak element to $a_{m+1},\dots,a_r$

## Coin Change Problem

Given a set $\mathcal C$ of coin denominations and a target amount $A$. What is the minimum number of coins we need to make up that amount?

**Example 1**: $\mathcal C = \{1,2,5\}, \quad A = 12$

* Solution: $3$ because $5*\underline 2 + 2* \underline 1 = 12$

**Example 2**: $\mathcal C = \{7,5\}, \quad A = 24$

* Solution: $4$ because $7*\underline 2 + 5* \underline 2 = 24$

**Example 3**: $\mathcal C = \{2\}, \quad A = 9$

* Solution: infeasible because we can never achieve 9 using \$2 coins.

To formalize, let

$$
\mathcal C = \{c_1, \dots, c_n\}
$$

We aim to solve the integer linear programming

$$
\begin{align}
\min_{x_1,\dots,x_n} \sum_{i=1} x_i
\qquad \text{s.t} \quad
\sum_{i=1} c_ix_i &= A \\
x_1, \dots, x_n &\in \mathbb N
\end{align}
$$

Boudary conditions:

* $c \ge 1, \forall c \in\mathcal C$
* $\vert\mathcal C\vert \ge 1$
* The problem could be infeasible in general.

### Greedy Approach

Consider example: $\mathcal C = \{1,5,10\}, \: A = 128$. We can construct the optimal solution as follows

1. use as many \$10 as we can. $\implies 10*12 = 120$. Next, we use \$1 and \$5 coins to make up $128-120=8$.
1. use as many \$5 as we can. $\implies 5*1 = 5$. Next, we use \$1 coins to make up $8-5=3$
1. use as many \$1 as we can. $\implies 1*3 = 3$. Next, $3-3=0$. Done.

To formalize, we assume w.l.o.g. that

$$
c_1 > c_2 > \dots > c_n
$$

The greedy algorithm is

---

Let $x_\text{total} = 0$.  
For $i = 1,\dots,n$:  
$\quad$ use as many $c_i$ as possible: $x_i \leftarrow \left\lfloor \frac{A}{c_i} \right\rfloor$.  
$\quad$ accumulate number of coins: $x_\text{total} \leftarrow x_\text{total} + x_i$.  
$\quad$ compute remainign amount: $A \leftarrow A - c_i x_i$.  
If $A = 0$:  
$\quad$ return $x_\text{total}$.  
Else:  
$\quad$ return *infeasible*.

---

Issue with greedy approach: Does not work on $\mathcal C = \{7,5\}, \: A = 24$. It woule produce infeasible but we do have $7*2 + 5*2 = 24$.

### Dynamic Programming

Let $f_{\mathcal C}(A)$ denote the minimum number of coins to make up $A$ using denominations in $\mathcal C$. We start with amount of 0. Now, we can choose any $c\in\mathcal C$. For each $c$, suppose we know $f_{\mathcal C}(A-c)$, then we can easily construct $f_{\mathcal C}(A)$ using the following optimal substructure:

$$
\begin{align}
f_{\mathcal C}(A) =
1 + \min_{c \in\mathcal C} f_{\mathcal C}(A-c)
\end{align}
$$

Remarks:

* If $f_{\mathcal C}(A-c)$ is not feasible for some $c\in\mathcal C$, then we set $f_{\mathcal C}(A-c) = \infty$.
* DP allows us to use any $c\in\mathcal C$ at any time step. In contrast, greedy algorithm shrinks $\mathcal C$ progressively.
* The optimal substracture is essentially Bellman optimality equation by considering $\mathcal C$ as the action set, and current accumulated amount as the state. The initial state is 0. Each action results in a cost of 1 (coin) regardless of current state.

*Proof*: Suppose we solved $f_{\mathcal C}(A-c)$ for all $c \in \mathcal C$. For any $c\in\mathcal C$, we need one extra coin to make up $A$ from $A-c$. Hence, we choose $c$ which minimizes $f_{\mathcal C}(A-c)$. $\quad\blacksquare$

Now, we develop the DP algorithm. Before we compute $f_{\mathcal C}(A)$, we must compute $f_{\mathcal C}(A-c)$ for all $c\in\mathcal C$ by optimal substructure. Recursively, we need to compute $f_{\mathcal C}(a)$ for even smaller $a$. Hence, we start with computing $f_{\mathcal C}(0)$, and then compute $f_{\mathcal C}(1), \dots, f_{\mathcal C}(A)$. To avoid repeated computation, we use memorization technique to store intermediate results.

---

**Input**: coin denominations $\mathcal C$, target amount $A$.  
**Output**: $f_{\mathcal C}(a)$.  
If $A=0$:  
$\quad$ return 0.  
// $A$ is at least 1.  
Let $f_{\mathcal C}(0) = 0, \: f_{\mathcal C}(1) = \dots = f_{\mathcal C}(A) = \infty$.  
For $a = 1,\dots,A$, do:  
$\quad$ For each $c \in\mathcal C$, do:  
$\qquad$ If $a-c \ge 0$:  
$\qquad\quad$ Update $f_{\mathcal C}(a) \leftarrow \min(f_{\mathcal C}(a), 1+f_{\mathcal C}(a-c))$.  
Return $f_{\mathcal C}(A)$

---

Implementation in C++: We return -1 instead of $\infty$ when the target amount is infeasible.

```c++
#include <vector>
using std::vector;

class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        if(amount < 0) {
            return -1; // infeasbile
        }
        
        if(amount == 0) {
            return 0; // trivial
        }
        
        vector<int> result(amount+1, 1e6);
        result[0] = 0;
        
        for(int a = 1; a <= amount; a++) {
            // compute the min number of coins to get a dollars
            for(const auto& c : coins) {
                int remain_amount = a - c;
                if(remain_amount < 0) {
                    continue; // infeasible
                }
                if(result[remain_amount] == 1e6) {
                    continue; // infeasible
                }
                
                // feasible
                result[a] = min(result[a], 1 + result[remain_amount]);
            }
            
        }
        
        return (result[amount] == 1e6) ? -1 : result[amount];
    }
};
```
