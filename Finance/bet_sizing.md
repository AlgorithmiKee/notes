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
\mathbb E[X_n] &= 1 \\
\mathbb V[X_n] &= 1.125
\end{align*}
$$

By the iid assumption over $X_i$, we have

$$
\begin{align*}
\mathbb E[W_n]
&= \mathbb E \left[ w_0 \cdot \prod_{i=1}^n X_i \right]
= w_0 \cdot \mathbb E \left[ \prod_{i=1}^n X_i \right]
= w_0 \cdot \prod_{i=1}^n \mathbb E \left[ X_i \right]
= w_0
\\
\mathbb V[W_n]
&= \mathbb E[W_n^2] - \mathbb E[W_n]^2 \\
&= \mathbb E \left[ w_0^2 \cdot \prod_{i=1}^n X_i^2 \right] - w_0^2 \\
&= w_0^2 \cdot \prod_{i=1}^n \mathbb E [X_i^2] - w_0^2 \\
&= w_0^2 \cdot \prod_{i=1}^n \left(\mathbb E[X_i]^2 + \mathbb V[X_i^2]\right) - w_0^2 \\
&= (2.125^n -1) w_0^2 \\
\end{align*}
$$

Hence, we conclude that strategy 1 yields zero growth rate and exponentially high volatility.