---
title: "Growth Rate"
author: "Ke Zhang"
date: "2025"
fontsize: 12pt
---

## Growth Factor and Grow Rate

Suppose your assets worth $w_k$ dollars in year $k$. ($k=1,\dots,n$) Then,

The growth factor $x_k$ of year $k$ is defined as

$$
\begin{align}
x_k \triangleq \frac{w_{k+1}}{w_k}
\end{align}
$$

The growth rate $x_k$ of year $k$ is defined as

$$
\begin{align}
r_k \triangleq \frac{w_{k+1} - w_k}{w_k} = x_k-1
\end{align}
$$

## Average Growth Factor

Consider an investment with annual growth factor $x_i$ in year $i$. What is the average growth rate over $n$ years?

Let $\bar x$ be the average growht factor. Then, growing with a constant factor $\bar x$ for $n$ years should be the same as growing $x_1,\dots,x_n$.

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

* The geometric mean is uppper bounded by arithmetic mean

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

By CLT,

$$
\frac{1}{n} \sum_{i=1}^n Y_i \sim \mathcal N\left(\mu_Y, \frac{\sigma_Y^2}{n}\right)
$$

where

$$
\begin{align*}
\mu_Y &= \mathbb E[Y_1] = \mathbb E[\ln X_1] \\
\sigma_Y^2 &= \mathbb E[Y_1^2] - \mu_Y^2 =  \mathbb E[(\ln X_1)^2] - \mathbb E[\ln X_1]^2
\end{align*}
$$

Hence, $\ln\bar X$ is normally distributed with mean $\mu_Y$ and variance $\frac{\sigma_Y^2}{n}$.

$$
\begin{align}
\ln\bar X \sim
\mathcal N\left(\mathbb E[\ln X_1], \frac{\mathbb E[(\ln X_1)^2] - \mathbb E[\ln X_1]^2}{n}\right)
\end{align}
$$

As $n\to\infty$, the variance goes to zero and thus $\ln\bar X$ converges to a deterministic number $\mathbb E[\ln X_1]$.

$$
\begin{align}
\lim_{n\to\infty} \ln\bar X &= \mathbb E[\ln X_1] \\
\lim_{n\to\infty} \bar X &= e^{\mathbb E[\ln X_1]}  \\
\end{align}
$$

By Jensen's inequality,

$$
\mathbb E[\ln X_1] \le \ln\mathbb E[X_1] = \ln\mu_X
$$

Hence,

$$
\begin{align*}
\lim_{n\to\infty} \ln\bar X &\le \ln\mu_X \\
\lim_{n\to\infty} \bar X &\le \mu_X \\
\end{align*}
$$
