---
title: "Growth Rate"
author: "Ke Zhang"
date: "2025"
fontsize: 12pt
---

## Growth Factor and Grow Rate

Suppose your assets are worth $w_k$ dollars in year $k$. ($k=1,\dots,n$) Then,

The growth factor $x_k$ of year $k$ is defined as

$$
\begin{align}
x_k \triangleq \frac{w_{k+1}}{w_k}
\end{align}
$$

The growth rate $r_k$ of year $k$ is defined as

$$
\begin{align}
r_k \triangleq \frac{w_{k+1} - w_k}{w_k} = x_k-1
\end{align}
$$

## Average Growth Factor

Consider an investment with annual growth factor $x_i$ in year $i$. What is the average growth rate over $n$ years?

Let $\bar x$ be the average growht factor. Then, growing with a constant factor $\bar x$ for $n$ years should yield the same result as growing at $x_1,\dots,x_n$ over the same period.

$$
\bar x^n = \prod_{i=1}^n x_i
$$

Taking the root and we conclude that

> The average growth factor is the geometric mean of $x_1,\dots,x_n$, i.e.
>
> $$
> \begin{align}
> \bar x = \sqrt[n]{x_1\cdots x_n} = \left( \prod_{i=1}^n x_i \right)^{\frac{1}{n}}
> \end{align}
> $$

Elementary properties:

* The geometric mean is the exponential of the log-average

  > $$
  > \begin{align}
  > \bar x &= \exp\left( \frac{1}{n} \sum_{i=1}^n \ln x_i \right) \\
  > \ln\bar x &= \frac{1}{n} \sum_{i=1}^n \ln x_i
  > \end{align}
  > $$

* The geometric mean is upper bounded by the arithmetic mean

  > $$
  > \begin{align}
  > \left( \prod_{i=1}^n x_i \right)^{\frac{1}{n}} \le \frac{1}{n}\sum_{i=1}^n x_i
  > \end{align}
  > $$

## Asymptotic Property

Let the growth factor of some investment be modeled by iid random variables $X_1,\dots,X_n$. Let $\bar X$ be the geometric mean of $X_1,\dots,X_n$.

$$
\bar X = \left( \prod_{i=1}^n X_i \right)^{\frac{1}{n}} = \exp\left( \frac{1}{n} \sum_{i=1}^n \ln X_i \right)
$$

Taking the log, we can express  $\ln\bar X$ in terms of arithmetic average of $\implies \ln X_1,\dots,\ln X_n$.

$$
\ln\bar X = \frac{1}{n} \sum_{i=1}^n \ln X_i
$$

By assumption, $X_1,\dots,X_n$ are iid $\implies \ln X_1,\dots,\ln X_n$ are also iid. Hence, by the law of large numbers, $\ln\bar X$ converges in distribution to $\mathbb E[\ln X_1]$.

$$
\begin{align}
\frac{1}{n} \sum_{i=1}^n \ln X_i &\xrightarrow{d} \mathbb E[\ln X_1]
\quad\text{as } n\to\infty
\\
\lim_{n\to\infty} \ln\bar X &= \mathbb E[\ln X_1]
\end{align}
$$

Therefore, the long-term average growth factor $\bar X$ is

> $$
> \begin{align}
> \lim_{n\to\infty} \bar X &= e^{\mathbb E[\ln X_1]}
> \end{align}
> $$

The investment is worth it iff

> $$
> e^{\mathbb E[\ln X_1]} > 1 \iff \mathbb E[\ln X_1] > 0
> $$

Remark: $\mathbb E[X_1] > 1$ does **NOT** guarantee positive growth rate. The reason is that $\mathbb E[X_1]$ is an over-estimate of the true growth factor due to Jensen's inequality

$$
\mathbb E[\ln X_1] \le \ln\mathbb E[X_1]
$$

which gives the statistical version of the inequality between geometric mean and arithmetic mean

> $$
> \begin{align*}
> e^{\mathbb E[\ln X_1]} &\le \mathbb E[X_1] \\
> \end{align*}
> $$

### A "Fair" Game

Consider a game in which you have 50% chance of winning and 50% chance of losing. If you win, your money is multiplied by 1.6; if you lose, it is multiplied by 0.5. Mathematically, the multiplier can be modeled as a random variable $X$ s.t.

$$
X=
\begin{cases}
1.6 & \text{with probability 0.5} \\
0.5 & \text{with probability 0.5} \\
\end{cases}
$$

Assume each round is statistically independent. Is this game worth playing?

At first glance, this game seems to be fair since

$$
\mathbb E[X] = 1.6 \times 0.5 + 0.5 \times 0.5 = 1.05 > 1
$$

Suppose we start with \$100. The expected value of our money after playing the game $n$ rounds is

$$
\mathbb E[100X^n] = 100 \mathbb E[X]^n = 100\cdot 1.05^n
$$

But do we really likely to get this exponential growth after sufficiently many rounds?

From our previous analysis, the average growth factor is

$$
e^{\mathbb E[\ln X]} = e^{0.5\times \ln 1.6 + 0.5\times \ln 0.5} \approx 0.894
$$

which implies that we will lose $(1-0.894)\times 100\% = 10.6\%$ each round on average. After sufficiently many rounds, we are very likely to end up with \$0!

Let $Y_n=100X^n$ denote the money we have after $n$ rounds. Then $Y$ takes values in

$$
100\cdot 1.6^k 0.5^{n-k}, \quad k=0,\dots,n
$$

where $k$ represents \# wins among $n$ rounds.

The PMF of $Y_n$ is

$$
P(Y_n=100\cdot 1.6^k 0.5^{n-k}) = \binom{n}{k} 0.5^k 0.5^{n-k} = \binom{n}{k} 0.5^n
$$

One can verify that $\mathbb E[Y_n] = 100\cdot 1.05^n$ which indeed grows exponentially.

However, the most likely value of $Y$ is obtained when $k=\frac{n}{2}$, which is

$$
100\cdot 1.6^{\frac{n}{2}} 0.5^{\frac{n}{2}} = 100 \cdot 0.8^{\frac{n}{2}}
$$

which declines exponentially!
