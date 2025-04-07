---
title: "Dynamic Programming for MDP II"
author: "Ke Zhang"
date: "2025"
fontsize: 12pt
---

# Dynamic Programming for Markov Decision Process II

[toc]

Notation:

* upercase (e.g. $R_t$): random variable

* lowercase (e.g. $s_t$): instance of random variable

* bold straight (e.g. $\mathbf{v}$): determineistic vector
  $$
  \DeclareMathOperator*{\argmax}{arg\max}
  $$

## Generalization

Stochastic policy and stochastic rewards.

A policy can also be stochastic. i.e. instead of mapping the state to a fixed action, there are multiple possible actions with different probability. A stochastic policy is described by the conditional distribution

$$
\pi(a \mid s), a \in\mathcal A, s \in\mathcal S
$$

Remarks:

* By the law of total probability,
  $$
  \sum_a \pi(a \mid s) = 1
  $$
* The determinstic policy can be seen as a special of stochastic policy by assigning $\pi(\hat a \mid s)$ to 1 for some $\hat a$.
  $$
  \pi(a \mid s) =
  \begin{cases}
    1 & a=\hat a\\
    0 & \text{else}
  \end{cases}
  $$
* Both deterministic and stochastic policy are time-invariant. i.e. The distribution of $a$ given $s$ is always the same, regardless when we arrived at $s$.
