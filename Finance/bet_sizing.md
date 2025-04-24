---
title: "Bet Sizing"
author: "Ke Zhang"
date: "2025"
fontsize: 12pt
---

# Bet Sizing

## Tossing a Fair Coin

You start with 100 dollars and play the following game: You choose a fixed bet size (as a percentage of your current cash) and then toss a fair coin. If the coin lands heads, you gain 100% of your bet. If it lands tails, you lose 50% of your bet. You repeat this process over multiple rounds.

For example, if you bet 10% of your cash each time, your wealth might evlove like this:

$$
\$ 100  \xrightarrow[\text{heads}] {10\%}
\$ 120  \xrightarrow[\text{tails}] {10\%}
\$ 114  \xrightarrow[\text{heads}] {10\%}
\$ 136.8 \cdots
$$

Let's compare two strategies

1. Go all in each time -- bet 100% of your current wealth each round.
1. Bet half cash each time -- bet 50% of your current wealth each round.

Assume that each coin toss is independent. Which strategy yields higher long-term growth?

Let $W_n$ denote the wealth after $n$-th coin toss ($n=1,2,\dots$). The initial wealth is deterministic and denoted by $w_0 = 100$. By strategy 1, we have for $n=1,2,\dots$

$$
W_{n} =
\begin{cases}
2W_{n-1}           & \text{with probability } 0.5 \\
\frac{1}{2}W_{n-1} & \text{with probability } 0.5 \\
\end{cases}
$$

which motivates us to define the random variable $X_n \,(n=1,2,\dots)$ as the ratio $\frac{W_{n}}{W_{n-1}}$. Then, for $n=1,2,\dots$

$$
\begin{align*}
p(X_n = 2) &= \frac{1}{2}, \quad p(X_n = 0.5) = \frac{1}{2} \\
W_n &= w_0 \cdot \prod_{i=1}^n X_i
\end{align*}
$$

It is easy to verfiy that

$$
\begin{align*}
\mathbb E[X_n] &= 1.25 \\
\mathbb E[X_n^2] &= 2.125
\end{align*}
$$

By the iid assumption over $X_i$, we have

$$
\begin{align*}
\mathbb E[W_n]
&= \mathbb E \left[ w_0 \cdot \prod_{i=1}^n X_i \right]
= w_0 \cdot \mathbb E \left[ \prod_{i=1}^n X_i \right]
= w_0 \cdot \prod_{i=1}^n \mathbb E \left[ X_i \right]
= 1.25^n w_0
\\
\mathbb V[W_n]
&= \mathbb E[W_n^2] - \mathbb E[W_n]^2 \\
&= \mathbb E \left[ w_0^2 \cdot \prod_{i=1}^n X_i^2 \right] - w_0^2 \\
&= w_0^2 \cdot \prod_{i=1}^n \mathbb E [X_i^2] - w_0^2 \\
&= (2.125^n -1.5625^n) w_0^2 \\
\end{align*}
$$

Hence, we conclude that strategy 1 yields zero growth rate and exponentially high volatility.

Similarly, strategy 2 can be analysed as follows

Let $Y_n = \frac{W_{n}}{W_{n-1}}$ be the ratio according to strategy 2.

* If the coin lands heads, then $Y_n = 0.5 + 0.5\times 2 = 1.5$ with probability $0.5$.
* If the coin lands tails, then $Y_n = 0.5 + 0.5\times 0.5 = 0.75$ with probability $0.5$.

It is easy to verify that

$$
\begin{align*}
\mathbb E[Y_n] &= 1.125 \\
\mathbb V[Y_n^2] &= 1.40625
\end{align*}
$$

By the iid assumption over $Y_i$ and the fact that $W_n = w_0 \cdot \prod_{i=1}^n Y_i$, we have

$$
\begin{align*}
\mathbb E[W_n]
&= w_0 \cdot \prod_{i=1}^n \mathbb E \left[ Y_i \right]
= 1.125^n w_0
\\
\mathbb V[W_n]
&= w_0^2 \cdot \prod_{i=1}^n \mathbb E [Y_i^2] - \mathbb E[W_n]^2
= (1.40625^n - 1.2656^{n} ) w_0^2
\end{align*}
$$

Hence, strategy 2 has much exponential growth rate and much less volatility.

TODO: simulation result. Explain the problem lies in the variance.

## Kelly's Optimal Bet Size

General problem: You start with $w_0$ dollars and play the following game: You choose a fixed bet size $\alpha\in[0,1]$ as the percentage of your current wealth. You bet multiple indepedent rounds. Each round has the win probability of $p_w$ and loss probability of $1-p_w$. For each win, you gain $g\in[0,1]$ of your bet. For each loss, you lose $\ell\in[0,1]$ of your bet.

We aim to address

1. The expecated wealth after $n$ rounds
2. The variance of your wealth after $n$ rounds
3. The optimal bet size $\alpha$ to maximize the exptected growth rate

Let $W_n$ denote the wealth after $n$-th round and $X_n = \frac{W_{n}}{W_{n-1}}$ be the ratio ($n=1,2,\dots$). Then,

$$
\begin{align*}
W_n &= w_0 \cdot \prod_{i=1}^n X_i \\
X_{n} &=
\begin{cases}
(1-\alpha) + \alpha(1+g) = 1+g\alpha       & \text{with probability } p_w \\
(1-\alpha) + \alpha(1-\ell) = 1+\ell\alpha & \text{with probability } 1-p_w
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
= w_0 \cdot\mathbb E[X_i]^n
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

TODO: wrong intuition: $\max\mathbb E[W_n]$ over $\alpha$. Correct: $\max\mathbb E[\lim_{n\to\infty} \frac{1}{n}W_n]$