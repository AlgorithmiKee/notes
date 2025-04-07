---
title: "Dollar Cost Averaging"
author: "Ke Zhang"
date: "2025"
fontsize: 12pt
---

# Dollar Cost Averaging

## Motivation

You purchase a stock twice, investing **the same dollar amount** each time. The stock prices at your first and second purchases were $p_1$ per share and $p_2$ per share, respectively. What is your average price per share?

**Solution**: Suppose you invested $m$ dollars each time. The total amount of investment is $2m$. The total number of shares is $\frac{m}{p_1} + \frac{m}{p_2}$. Hence, the average price per share is

$$
\frac{2m}{\frac{m}{p_1} + \frac{m}{p_2}} = \frac{2}{\frac{1}{p_1} + \frac{1}{p_2}}
$$

This is called the ***harmonic mean (HM)*** of $p_1$ and $p_2$.

## General Formulation

The general ***dollar cost averaging (DCA)*** can be formulated as following:  
You purchase a stock $n$ times, investing investing **the same dollar amount** each time. The stock prices at your $k$-th purchases is $p_k$ per share ($k=1,\dots,n$). Then, the average price per share is the harmonic mean of $p_1,\dots,p_n$:

$$
\frac{n}{\frac{1}{p_1} + \dots + \frac{1}{p_n}}
$$

From mathematics, we know that the harmonic mean of $p_1,\dots,p_n$ is smaller or equal to their ***arithmetic mean (AM)***. This implies:

> DCA results in a lower buy-in price compared to the time-average of the stock prices. Hence, DCA is better than purchasing a fixed number of shares each time.

### Proof: HM-AM Inequality

> Let $a_1,\dots,a_n$ be positive real numbers. Then, their HM is always smaller or equal to their AM.
>
> $$
> \begin{align}
>   \frac{n}{\frac{1}{a_1} + \dots + \frac{1}{a_n}} \le \frac{a_1 +\dots+ a_n}{n}
> \end{align}
> $$
>
> The equality holds $\iff a_1 =\dots= a_n$

*Proof*: To show the HM-AM inequality is equivalent to show
$$
n^2 \le (a_1 +\dots+ a_n) \left(\frac{1}{a_1} + \dots + \frac{1}{a_2}\right)
$$

Since all $a_k$ are positive, define

$$
\begin{align*}
\mathbf{v} &= (\sqrt{a_1} ,\dots, \sqrt{a_n})^\top \\
\mathbf{w} &= \left(\frac{1}{\sqrt{a_1}} ,\dots, \frac{1}{\sqrt{a_2}}\right)^\top
\end{align*}
$$

By Cauchy-Schwarz inequality, we get
$$
\begin{align*}
\langle\mathbf{v}, \mathbf{w}\rangle &\le  \Vert\mathbf{v}\Vert \Vert\mathbf{w}\Vert \\
n &\le \sqrt{a_1 +\dots+ a_n} \cdot \sqrt{\frac{1}{a_1} + \dots + \frac{1}{a_2}} \\
\end{align*}
$$

Taking the square on both side, we conclude the inequality.

The equal sign holds iff $\mathbf{v}$ and $\mathbf{w}$ are linearly dependent. i.e.
$$
\begin{align*}
\exists \lambda, \text{ s.t. } \forall k = 1,\dots,n, a_k &= \lambda \frac{1}{a_k} \\
a_k &= \sqrt{\lambda}
\quad\quad\square
\end{align*}
$$
