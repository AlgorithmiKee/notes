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
  > \bar x = \exp\left( \frac{1}{n} \sum_{i=1}^n \ln x_i \right)
  > \end{align}
  > $$

* The geometric mean is upper bounded by the arithmetic mean

  > $$
  > \begin{align}
  > \left( \prod_{i=1}^n x_i \right)^{\frac{1}{n}} \le \frac{1}{n}\sum_{i=1}^n x_i
  > \end{align}
  > $$

## Asymptotic Property

Suppose the growth factors $X_1,\dots,X_n$ are iid with mean $\mu_X$ and variance $\sigma_X^2$. Let $\bar X$ be the geometric mean of $X_1,\dots,X_n$.

Let $Y_i = \ln X_i$. Then, $\ln\bar X$ becomes

$$
\ln\bar X
= \frac{1}{n} \sum_{i=1}^n \ln X_i
= \frac{1}{n} \sum_{i=1}^n Y_i
$$

By central limit theorem (CLT), the RHS converges in distribution to

$$
\frac{1}{n} \sum_{i=1}^n Y_i  \xrightarrow{d} \mathcal N\left(\mu_Y, \frac{\sigma_Y^2}{n}\right)
$$

where

$$
\begin{align*}
\mu_Y &= \mathbb E[Y_1] = \mathbb E[\ln X_1] \\
\sigma_Y^2 &= \mathbb E[Y_1^2] - \mu_Y^2 =  \mathbb E[(\ln X_1)^2] - \mathbb E[\ln X_1]^2
\end{align*}
$$

Hence, $\ln\bar X$ is approximately normally distributed for large $n$.

$$
\begin{align}
\ln\bar X \sim
\mathcal N\left(\mathbb E[\ln X_1], \frac{\mathbb E[(\ln X_1)^2] - \mathbb E[\ln X_1]^2}{n}\right)
\end{align}
$$

As $n\to\infty$, the variance goes to zero and thus $\ln\bar X$ converges to a deterministic number $\mathbb E[\ln X_1]$.

> $$
> \begin{align}
> \lim_{n\to\infty} \ln\bar X &= \mathbb E[\ln X_1] \\
> \lim_{n\to\infty} \bar X &= e^{\mathbb E[\ln X_1]}  \\
> \end{align}
> $$

Therefore, the long-term average growth rate is $e^{\mathbb E[\ln X_1]}$.

By Jensen's inequality,

$$
\lim_{n\to\infty} \ln\bar X = \mathbb E[\ln X_1] \le \ln\mathbb E[X_1] = \ln\mu_X
$$

Hence, we get the statistical version of the inequality between geometric mean and arithmetic mean

> $$
> \begin{align*}
> \lim_{n\to\infty} \bar X &\le \mu_X \\
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

Is this game worth playing?

At first glance, this game seems to be fair since

$$
\mathbb E[X] = 1.6 \times 0.5 + 0.5 \times 0.5 = 1.05 > 1
$$

Suppose you start with \$100 and play the game plenty number of times, would you end up with more than \$100?

From our previous analysis, the average growth factor is

$$
e^{\mathbb E[\ln X]} = e^{0.5\times \ln 1.6 + 0.5\times \ln 0.5} \approx 0.894
$$

which implies that we will lose $(1-0.894)\times 100\% = 10.6\%$ each round on average. After sufficiently many rounds, we are very likely to end up with \$0!