---
title: "Bet Sizing"
author: "Ke Zhang"
date: "2025"
fontsize: 12pt
---

# Bet Sizing

## Kelly's Optimal Bet Size

You start with $w_0$ dollars and play the following game: You choose a fixed bet size $\alpha\in[0,1]$ as the percentage of your current wealth. You bet multiple indepedent rounds. Each round has the win probability of $p_w$ and loss probability of $1-p_w$. For each win, you gain $g\in[0,1]$ of your bet. For each loss, you lose $\ell\in[0,1]$ of your bet.

We aim to address

1. The expecated wealth after $n$ rounds
2. The variance of your wealth after $n$ rounds
3. The optimal bet size $\alpha$

This problem models stock trading. The paramters $p_w$, $g$ and $\ell$ are given by the market and can not be influenced by the investor. The investor can choose a proper $\alpha$ -- the percentage of the money invested in stocks, to maximize the growth.

### Mean and Variance of Wealth

Let $W_n$ denote the wealth after $n$-th round and $X_n = \frac{W_{n}}{W_{n-1}}$ be the growth factor ($n=1,2,\dots$). Then,

$$
\begin{align*}
W_n &= w_0 \cdot \prod_{i=1}^n X_i \\
X_{n} &=
\begin{cases}
(1-\alpha) + \alpha(1+g) = 1+g\alpha       & \text{with probability } p_w \\
(1-\alpha) + \alpha(1-\ell) = 1-\ell\alpha & \text{with probability } 1-p_w
\end{cases}
\end{align*}
$$

By the iid assumption over $X_i$, the expected wealth after $n$ rounds is

$$
\begin{align*}
\mathbb E[W_n]
&= \mathbb E \left[ w_0 \cdot \prod_{i=1}^n X_i \right]
= w_0 \cdot \mathbb E \left[ \prod_{i=1}^n X_i \right]
= w_0 \cdot \prod_{i=1}^n \mathbb E[X_i]
= w_0 \cdot\mathbb E[X_1]^n
\end{align*}
$$

where

$$
\mathbb E[X_{n}] = 1+\big(p_w g - (1-p_w)\ell\big)\alpha
$$

The variance of the wealth after $n$ rounds is

$$
\begin{align*}
\mathbb V[W_n]
&= \mathbb E[W_n^2] - \mathbb E[W_n]^2 \\
&= \mathbb E\left[ w_0^2 \cdot \prod_{i=1}^n X_i^2 \right] - \Big(w_0 \cdot\mathbb E[X_1]^n\Big)^2 \\
&= w_0^2 \cdot \prod_{i=1}^n \mathbb E[X_i^2] - w_0^2 \cdot\mathbb E[X_1]^{2n} \\
&= w_0^2 \cdot \left(\mathbb E[X_1^2]^n - \mathbb E[X_1]^{2n} \right) \\
\end{align*}
$$

where

$$
\mathbb E[X_{n}^2] = p_w(1+g\alpha)^2 + (1-p_w)(1-\ell\alpha)^2
$$

### Naive Approach: Maximizing Expected Wealth

What makes an $\alpha$ good? We need to define an objective to measure the goodness of $\alpha$. At fist glance, maximzing $\mathbb E[W_n]$ over $\alpha$ seems reasonable. However, this approach would lead to all-in strategy, which exhibits very high variance (or ***volatility***), as we show as follows.

Taking the gradient

$$
\begin{align*}
\frac{d}{d\alpha}\mathbb E[W_n]
&= w_0 \cdot n\mathbb E[X_1]^{n-1} \cdot \frac{d}{d\alpha} \mathbb E[X_1] \\
&= w_0 \cdot n\left[1+\big(p_w g - (1-p_w)\ell\big)\alpha\right]^{n-1} \cdot \big(p_w g - (1-p_w)\ell\big) \\
\end{align*}
$$

Note that regardless of $\alpha$,

$$
-1 \le -\ell \le p_w g - (1-p_w)\ell \le g \le 1
$$

Thus, the base in the exponential term is always non-negative. The sign of the gradient only depends on $p_w g - (1-p_w)\ell$.

If $p_w g - (1-p_w)\ell\le0$, then $\frac{d}{d\alpha}\mathbb E[W_n] \le 0$ and $\mathbb E[W_n]$ is monotonically decreasing w.r.t. $\alpha$. Hence, the maximium is obtained at $\alpha^* = 0$. i.e. The optimal strategy is to bet nothing. The resulting mean and variance of $W_n$ are

$$
\begin{align*}
\mathbb E[W_n] = w_0, \quad \mathbb V[W_n] = 0
\end{align*}
$$

Namely, we expect zero growth rate and no volatility. Trivial.

If $p_w g - (1-p_w)\ell\ge 0$, then $\frac{d}{d\alpha}\mathbb E[W_n] \ge 0$ and $\mathbb E[W_n]$ is monotonically increasing w.r.t. $\alpha$. Hence, the maximium is obtained at $\alpha^* = 1$. i.e. The optimal strategy is to go all-in at each round. The resulting expected wealth is

$$
\begin{align*}
\mathbb E[W_n] = w_0 \left[1 + p_w g - (1-p_w)\ell \right]^n
\end{align*}
$$

which grows exponentially w.r.t $n$ with maximal growth rate. But is this really good? We must also consider the variance, which is lower bounded by

$$
\begin{align*}
\mathbb V[W_n]
&= w_0^2 \cdot \left(\mathbb E[X_1^2]^n - (\mathbb E[X_1]^2)^{n} \right)
\\
&= w_0^2 \cdot n \xi^{n-1} \left( \mathbb E[X_1^2] - \mathbb E[X_1]^2 \right)
\quad\text{where } \mathbb E[X_1]^2 \le \xi \le \mathbb E[X_1^2]
\\
&= w_0^2 \cdot n \xi^{n-1} \cdot \mathbb V[X_1]
\\
&\ge w_0^2 \cdot n \mathbb E[X_1]^{2(n-1)} \cdot \mathbb V[X_1]
\\
&= n \cdot \mathbb E[W_n]^2 \cdot \frac{\mathbb V[X_1]}{\mathbb E[X_1]^2}
\end{align*}
$$

where the 2nd step follows from mean value theorem applied to $f(x) = x^n$ (see [Appendix](#mean-value-theorem)).

Hence, maximizing $\mathbb E[W_n]$ also maximizes the lower bound of $\mathbb V[W_n]$, making the variance explode. Intuitively, $\mathbb E[W_n]$ is driven extremely rare sequences of wins, and not by typical outcomes. Intuition: If we existed in multiverse, we could just care about maximizing $\mathbb E[W_n]$ since we are playing infinitely many such games all at once. But in reality, life is only once and thus the realized wealth matters instead of expected wealth.

### Maximizing Averge Growth Factor

The average growth factor $\bar X$, defined as the geometric mean of $X_1,\dots,X_n$.

$$
\begin{align*}
\bar X = \left( \prod_{i=1}^n X_i \right)^{\frac{1}{n}}
\end{align*}
$$

Instead of maximizing the expected final wealth, we maximize the expected average growth factor.

$$
\begin{align*}
\max_{\alpha\in[0,1]} \mathbb E[\bar X]
\iff
\max_{\alpha\in[0,1]} \mathbb E\left[ \left( \prod_{i=1}^n X_i \right)^{\frac{1}{n}} \right]
\end{align*}
$$

Remark: the objective is called expected average growth factor where

* "expected" $\triangleq$ taking statistical mean over random variables $X_1,\dots,X_n$
* "average" $\triangleq$ taking geometric mean over time horizon $i=1,\dots,n$

### Simulation Results

## Appendix

### Mean Value Theorem

For $a>b$,

$$
\begin{align}
f(a) - f(b) = f'(\xi)\cdot(a-b) \quad\text{where } \xi\in(b,a) \\
\end{align}
$$

Apply to $f(x) = x^n$

$$
\begin{align}
a^n - b^n
&= n \xi^{n-1}(a-b) \\
&\ge n b^{n-1}(a-b) \\
\end{align}
$$