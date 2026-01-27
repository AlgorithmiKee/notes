---
title: "Measure Theoretic Probability"
date: "2026"
author: "Ke Zhang"
---

# Measure Theoretic Probability

Preliminary: measure theory.

[toc]

Measure theoretic probability theory unifies the discrete probability distribution and continuous probability distribution under the same framework.

## Probability Space

A **probability space** consists of $(\Omega, \mathcal A, \mathbb P)$ where

* $\Omega$ is the **sample space**, which is a set of all possible outcomes of a random experiment.
* $\mathcal A$ is the **event space**, which is a $\sigma$-algebra on $\Omega$. Intuitively, $\mathcal A$ contains events we can assign a probability to.
* $\mathbb P: \mathcal A \to [0,1]$ is the **probability measure**, which is essentially a normalized measure.

All properties of general measure functions hold for probability measure. Some most important ones are:

Countable Additivity: For a sequence of disjoint events $A_1, A_2, \dots \in \mathcal{A}$:

$$
\mathbb{P}\left(\bigcup_{i=1}^\infty A_i\right) = \sum_{i=1}^\infty \mathbb{P}(A_i)
$$

Monotonicity: If $A \subseteq B$, then $\mathbb{P}(A) \le \mathbb{P}(B)$.

Continuity from Below: If $A_1 \subseteq A_2 \subseteq \dots$ is an increasing sequence of events, then:

$$
\mathbb{P}\left(\bigcup_{n=1}^\infty A_n\right) = \lim_{n \to \infty} \mathbb{P}(A_n)
$$

Continuity from Above: If $A_1 \supseteq A_2 \supseteq \dots$ and $\mathbb{P}(A_1) < \infty$, then:

$$
\mathbb{P}\left(\bigcap_{n=1}^\infty A_n\right) = \lim_{n \to \infty} \mathbb{P}(A_n)
$$

## Random Variable

Let $(S, \mathcal B, \mu)$ be another measure space where

* $S$ is called **state space**.
* $\mathcal B$ is a $\sigma$-algebra on $S$.
* $\mu: \mathcal B \to [0,\infty]$ is called the **reference measure.**

An **$S$-valued random variable** is a **measurable** function $X: \Omega \to S$. Namely, the preimage of any set $B \in \mathcal{B}$ is also in $\mathcal{A}$.

$$
\begin{align}
\forall B \in \mathcal B, \quad
X^{-1}(B) \triangleq \{ \omega: X(\omega) \in B \} \in \mathcal A
\end{align}
$$

Intuitively, the measurability of $X$ guarantees that we can measure the probability of $X^{-1}(B)$ as long as $B$ has a proper volume under $\mu$.

### Typical State Spaces

$X$ becomes a discrete random variable if we choose

* $S$ any finite or countably infinite set.
* $\mathcal B = 2^S$, i.e. the power set of $S$.
* $\mu(B) = \vert B \vert$, i.e. the counting measure

$X$ becomes a real-valued continuous random variable if we choose

* $S = \mathbb R$.
* $\mathcal B = \mathcal B(\mathbb R)$, i.e. the Borel $\sigma$-algebra on $\mathbb R$.
* $\mu(B) =  \lambda(B)$, i.e. the Lebesgue measure

$X$ becomes a real-valued random vector if we choose

* $S = \mathbb R^n$.
* $\mathcal B = \mathcal B(\mathbb R^n)$, i.e. the Borel $\sigma$-algebra on $\mathbb R^n$.
* $\mu(B) =  \lambda^n(B)$, i.e. the Lebesgue measure

### Probability Distribution

The **probability distribution (law)** of $X$, denoted by $\mathbb P_X$, is defined as the **pushforward measure** of $\mathbb P$ under $X$.

$$
\begin{align}
\mathbb P_X &= X_\sharp \mathbb P \\
\forall B \in \mathcal B, \quad \mathbb P_X(B) &= \mathbb P(\{ \omega: X(\omega) \in B \})
\end{align}
$$

Remarks:

* Short hand notation: $\mathbb P_X(B) = \mathbb P( X \in B )$.
* Equivalent notation with preimage:
    $$
    \begin{align}
    \mathbb P_X &= \mathbb P \circ X^{-1} \\
    \mathbb P_X(B) &= \mathbb P(X^{-1}(B))
    \end{align}
    $$
* Both the pushforward measure $\mathbb P_X$ and reference measure $\mu$ are defined on $\mathcal B$. They serve different purposes.
    1. $\mathbb{P}_X(B)$ tells how probable $X$ falls into a set $B$.
    1. $\mu(B)$ describes the "volume" of a set $B$.

The CDF is formally defined in terms of $\mathbb P_X$.

* For a discrete random variable $X$ with $S \subset \mathbb R$:
    $$
    \begin{align*}
    F_X(x)
    &\triangleq \mathbb P_X(\{s\in S: s \le x\}) \\
    &= \mathbb P(X \le x)
    \end{align*}
    $$

* For a continuous random variable $X$:
    $$
    \begin{align*}
    F_X(x)
    &\triangleq \mathbb P_X((-\infty,x]) \\
    &= \mathbb P(X \le x)
    \end{align*}
    $$

* For a random vector $X$:
    $$
    \begin{align*}
    F_X(x)
    &\triangleq \mathbb P_X((-\infty,x_1] \times \dots \times (-\infty,x_n]) \\
    & = \mathbb P(X_1 \le x_1, \dots, X_n \le x_n)
    \end{align*}
    $$

### Density

We say that $\mathbb P_X$ is **absolutely continuous** w.r.t. $\mu$, denoted by $\mathbb P_X \ll \mu$, iff

$$
\begin{align}
\forall B \in \mathcal B, \quad
\mu(B) = 0 \implies \mathbb P_X(B) = 0
\end{align}
$$

Intuitvely, $\mathbb P_X \ll \mu$ iff any set with zero volume (under $\mu$) also has zero probability.

**Radon-Nikodym Theorem**  
$\mathbb P_X$ is absolutely continuous w.r.t. $\mu$ iff there exists a **density** $f_X: S \to \mathbb [0,\infty)$ s.t.

$$
\begin{align}
f_X = \frac{\mathrm d\mathbb P_X}{\mathrm d\mu}
\end{align}
$$

which means

$$
\begin{align}
\forall B \in \mathcal B, \quad
\mathbb P_X(B) = \int_B f_X \,\mathrm d\mu
\end{align}
$$

Remarks:

* We say $f_X$ is the **Radon-Nikodym derivative** of $\mathbb P_X$ w.r.t. the reference measure $\mu$.
* In ML, *distribution* often refers to the density $f_X$ (often written as $p_X$) rather than the pushforward measure $\mathbb P_X$.

The density coincides with PDF or PMF under standard state space settings

* For a discrete random variable $X$ with $S \subset \mathbb R$, suppose $\mathbb P_X$ is absolutely continuous w.r.t. the counting measure $\mu$. Then, the density conincides with PMF:
    $$
    \begin{align*}
    f_X(x)
    &= \mathbb P(X = x)
    \end{align*}
    $$

* For a continuous random variable $X$, suppose $\mathbb P_X$ is absolutely continuous w.r.t. the Lebesgue measure $\lambda$. Then, the density conincides with PDF:
    $$
    \begin{align*}
    f_X(x) &= F_X'(x), \quad \mu\text{-a.e.}
    \end{align*}
    $$

* For a random vector $X$, suppose $\mathbb P_X$ is absolutely continuous w.r.t. the Lebesgue measure $\lambda^n$. Then, the density conincides with PDF:
    $$
    \begin{align*}
    f_X(x)
    & = \frac{\partial^n F_X}{\partial x_1, \dots, \partial x_n}(x),
    \quad \mu\text{-a.e.}
    \end{align*}
    $$

## Expectation

$$
\begin{align}
\mathbb E[X]
&= \int_\Omega X \,\mathrm d\mathbb P && \text{definition} \\
&= \int_S x \,\mathrm d\mathbb P_X && \text{LOTUS} \\
&= \int_S xf_X \,\mathrm d\mu && \text{Radon-Nikodym} \\
&= \int_S x f_X(x) \,\mathrm dx && \text{ML notation} \\
\end{align}
$$
