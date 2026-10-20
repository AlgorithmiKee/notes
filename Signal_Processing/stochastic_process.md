---
title: "Stochastic Process"
author: "Ke Zhang"
date: "2023"
fontsize: 12pt
---

# Stochastic Process

This note covers the general theory of stochastic processes. Special properties of weakly stationary processes (correlation analysis, spectral analysis and linear filtering) are covered in [Weakly Stationary Process](weakly_stationary_process.md).

Note: we often use the following terms interchangeably.

* Stochastic process $\iff$ stochastic signal $\iff$ signal
* $X_t, \, t\in\mathbb T \iff \{ X_t \}_{t\in \mathbb T} \iff \{ X(t) \}_{t\in \mathbb T} \iff X$

## Basic Definitions

A **stochastic process** is a family of random variables $\{ X_t \}_{t\in \mathbb T}$, all defined on the same probability space $(\Omega, \mathcal F, \mathbb P)$ and taking values in the same set $\mathbb S$.

* $\mathbb T$ is called the **index set** or **time horizon**.
* $\mathbb S$ is called the **state space**, which can be $\mathbb R$, $\mathbb C$ or $\mathbb R^n$.

Depending on the time horizon, we distinguish

* **discrete-time** processes: $\mathbb T = \mathbb Z$ or $\mathbb T = \mathbb N$. e.g. sampled signals, daily stock prices.
* **continuous-time** processes: $\mathbb T = \mathbb R$ or $\mathbb T = [0, \infty)$. e.g. thermal noise in a resistor, Brownian motion.

The theory below applies to both cases unless stated otherwise. Only the time horizon $\mathbb T$ changes. For simplicity, we consider real-valued processes, i.e. $\mathbb S = \mathbb R$.

Formally, a stochastic process is a function of two variables
$$
X:\Omega \times \mathbb T \to \mathbb S, (\omega, t)\mapsto X(\omega, t)
$$

This leads to two interpretations of $X$:

* For fixed time $t_0$, $X(\cdot, t_0)$ is a random variable.
* For fixed random outcome $\omega_0$, $X(\omega_0, \cdot)$ is a deterministic function of time, called a **realisation** or **sample path** of $X$.

In engineering, we do not define the sample space explicitly and simply write $X_t$ or $X(t)$.

**Example:** Let $X_0 \in \{-1, 1 \}$ with $\mathbb P(X_0 = -1) = \mathbb P(X_0 = 1) = \frac{1}{2}$. Define $X_t = X_0, \forall t \in\mathbb T$. This process has only two sample paths: the constant functions $-1$ and $1$.

## Finite Dimensional Distributions

To fully describe a stochastic process $X$, we need the joint distribution of all possible finite combinations of random variables in $X$.

For $n\ge 1$ and $t_1, \cdots, t_n \in\mathbb T$, the joint CDF
$$
    F_{t_1, \cdots, t_n}(x_1, \cdots, x_n) = \mathbb P\left(X_{t_1} \le x_1, \cdots, X_{t_n} \le x_n\right)
$$
is called an **$n$-th order distribution** of $X$. The collection of all of them
$$
    \left\{ F_{t_1, \cdots, t_n} \:\middle\vert\: n\ge 1, \: t_1, \cdots, t_n\in\mathbb T \right\}
$$
is called the **finite dimensional distributions (FDDs)** of $X$. This includes

* The CDFs of $X_t, \:\forall t\in\mathbb T$. (1st-order distributions)
* The joint CDFs of $X_{t_1}, X_{t_2}, \:\forall t_1,t_2 \in\mathbb T$. (2nd-order distributions)
* The joint CDFs of $X_{t_1}, X_{t_2}, X_{t_3} \:\forall t_1, t_2, t_3\in\mathbb T$. (3rd-order distributions)
* And so on... (higher-order distributions).

Remarks:

* Even for a continuous-time process, we only consider finitely many time instances at once. The infinite family $\{ X_t \}_{t\in \mathbb T}$ is described via its finite sub-families.
* Two processes are **statistically equivalent** (equal in distribution) if they have the same FDDs.

### Consistency

The FDDs of a stochastic process are not arbitrary. They satisfy the following **consistency conditions**:

1. Symmetry: permuting the time indices together with the arguments does not change the FDD. For any permutation $\pi$ of $\{1,\cdots,n\}$,
    $$
        F_{t_{\pi(1)}, \cdots, t_{\pi(n)}}(x_{\pi(1)}, \cdots, x_{\pi(n)}) = F_{t_1, \cdots, t_n}(x_1, \cdots, x_n)
    $$
2. Marginalisation: lower order FDDs are obtained from higher order ones by letting the extra arguments go to infinity.
    $$
        \lim_{x_{n+1}\to\infty} F_{t_1, \cdots, t_n, t_{n+1}}(x_1, \cdots, x_n, x_{n+1}) = F_{t_1, \cdots, t_n}(x_1, \cdots, x_n)
    $$

### iid Processes

Specifying all FDDs is impractical in general. Hence, we focus on more restrictive stochastic processes, whose FDDs can be easily described. One of the simplest models is the iid process.

A stochastic process $X$ is called **iid** if
$$
\forall n\ge 1, \forall t_1, \cdots, t_n\in\mathbb T \text{ distinct},
\:
X_{t_1}, \cdots, X_{t_n} \text{ are iid}.
$$
Equivalently, there exists a CDF $F$ such that all FDDs factorise
$$
    F_{t_1, \cdots, t_n}(x_1, \cdots, x_n) = \prod_{i=1}^n F(x_i)
$$
i.e. the whole process is described by a single 1D distribution.

**Example (thermal noise):** The voltage across a resistor fluctuates randomly due to the thermal motion of electrons. The thermal noise $\{ X_t \}_{t\in \mathbb Z}$ is commonly modelled as an iid process with
$$
    X_t \sim \mathcal N(0, \sigma^2), \quad F_{t_1, \cdots, t_n}(x_1, \cdots, x_n) = \prod_{i=1}^n \Phi\left(\frac{x_i}{\sigma}\right)
$$
where $\Phi$ is the standard normal CDF and the noise power $\sigma^2$ grows with temperature.

Note: The process $X_t = X_0$ from the first example is **not** iid, since all $X_t$ are identical (fully dependent). Its FDDs are
$$
    \mathbb P(X_{t_1} = \cdots = X_{t_n} = 1) = \mathbb P(X_{t_1} = \cdots = X_{t_n} = -1) = \frac{1}{2}
$$

### Gaussian Processes

A stochastic process $X$ is called a **Gaussian process** if all of its FDDs are (multivariate) Gaussian, i.e.
$$
\forall n\ge 1, \forall t_1, \cdots, t_n\in\mathbb T,
\:
(X_{t_1}, \cdots, X_{t_n})^\top \text{ is jointly Gaussian}.
$$

Since a multivariate Gaussian is determined by its mean vector and covariance matrix, a Gaussian process is fully described by its mean function and auto-covariance:

$$
\mu_X(t) = \mathbb E[X_t], \quad C_{XX}(t_1, t_2) = \mathbb E\left[(X_{t_1} - \mu_X(t_1)) (X_{t_2} - \mu_X(t_2))\right]
$$

## Moments

FDDs contain the full statistical information but are hard to obtain. In practice, we often work with the first and second order moments, which only depend on the 1st and 2nd order FDDs.

### Mean and Power

The **mean function** is defined by $\mu_X(t) = \mathbb E[X_t]$.

The **power** is defined by $\mathbb E[X_t^2]$.

Remarks:

1. In general, both mean and power depend on time. However, we will see that they are time-invariant for some class of stochastic signals.
2. In applications, we usually assume that $X$ has finite power, i.e. $\mathbb{E}[X_t^2] < \infty, \forall t\in\mathbb T$. Such processes are called **second-order processes**.

### Auto-Correlation and Auto-Covariance

Given a stochastic process $\{ X_t \}_{t\in \mathbb T}$,

* The **auto-correlation** is defined by
    $$
    \begin{equation}
    \fbox{$\displaystyle
        R_{XX}(t_1, t_2) = \mathbb E\left[X_{t_1}X_{t_2}\right]
    $}
    \end{equation}
    $$

* The **auto-covariance** is defined by

    $$
    \begin{split}
        C_{XX}(t_1, t_2)
        &= \mathbb E\left[(X_{t_1} - \mu_X(t_1)) (X_{t_2} - \mu_X(t_2))\right] \\
        &= R_{XX}(t_1, t_2) - \mu_X(t_1)\mu_X(t_2)
    \end{split}
    $$

Remarks:

* In general, both $R_{XX}(t_1, t_2)$ and $C_{XX}(t_1, t_2)$ depend on two time indices. Later, we will see that the dependency reduces to the time difference only for weakly stationary signals.
* The signal power coincides with $R_{XX}(t_1, t_2)$ when $t_1=t_2$.
* Both are symmetric, i.e. $R_{XX}(t_1, t_2) = R_{XX}(t_2, t_1)$ and $C_{XX}(t_1, t_2) = C_{XX}(t_2, t_1)$.
* For any $t_1, \cdots, t_n\in\mathbb T$, the matrix $[C_{XX}(t_i, t_j)]_{i,j}$ is the covariance matrix of $(X_{t_1}, \cdots, X_{t_n})^\top$ and hence positive semi-definite.
* For complex-valued processes, the second factor is conjugated, e.g. $R_{XX}(t_1, t_2) = \mathbb E\left[X_{t_1}\overline{X_{t_2}}\right]$.

### Cross-Correlation and Cross-Covariance

Given two stochastic processes $\{ X_t \}_{t\in \mathbb T}$ and $\{ Y_t \}_{t\in \mathbb T}$, we are interested in their joint statistical properties. In particular, we define

* The **cross-correlation** as
    $$
        R_{XY}(t_1, t_2) = \mathbb E\left[X_{t_1}Y_{t_2}\right]
    $$
* The **cross-covariance** as
    $$
    \begin{split}
        C_{XY}(t_1, t_2)
        &= \mathbb E\left[ (X_{t_1} - \mu_X(t_1))  (Y_{t_2} - \mu_Y(t_2))\right] \\
        &= R_{XY}(t_1, t_2) - \mu_X(t_1)\mu_Y(t_2)
    \end{split}
    $$

Remark: $R_{XY}(t_1, t_2) = R_{YX}(t_2, t_1)$.

## Stationarity

Stationarity means that certain statistical properties of a process do not change over time. It is formulated in terms of FDDs. Throughout this section, we assume $\mathbb T = \mathbb Z$ or $\mathbb T = \mathbb R$, so that time shifts $t \mapsto t + \tau$ stay in $\mathbb T$.

### $N$-th order Stationarity

Let $N\ge 1$. A stochastic process $X$ is called **$N$-th order stationary** if its $N$-th order distributions are shift-invariant, i.e.
$$
\forall t_1, \cdots, t_N, \tau \in\mathbb T,
\:
F_{t_1+\tau, \cdots, t_N+\tau} = F_{t_1, \cdots, t_N}.
$$

Equivalently, $X$ is $N$-th order stationary iff

$$
\forall t_1, \cdots, t_N, \tau \in\mathbb T, \:
(X_{t_1+\tau}, \cdots, X_{t_N+\tau}) \stackrel{d}{=} (X_{t_1}, \cdots, X_{t_N}),
$$

where $\stackrel{d}{=}$ denotes equality in distribution (see [Appendix](#equality-of-random-variables)).

**Properties:**

1. $X$ is $N$-th order stationary $\implies$ $X$ is $K$-th order stationary, $\forall K\le N$

    In other words, higher order stationarity implies lower order stationarity.

    *Proof:* Let $K < N$ and choose $t_{K+1}, \cdots, t_N\in\mathbb T$ arbitrarily. By marginalisation,
    $$
    \begin{split}
        F_{t_1+\tau, \cdots, t_K+\tau}(x_1, \cdots, x_K)
        &= \lim_{x_{K+1}, \cdots, x_N\to\infty} F_{t_1+\tau, \cdots, t_N+\tau}(x_1, \cdots, x_N) \\
        &= \lim_{x_{K+1}, \cdots, x_N\to\infty} F_{t_1, \cdots, t_N}(x_1, \cdots, x_N) \\
        &= F_{t_1, \cdots, t_K}(x_1, \cdots, x_K)
    \end{split}
    $$

2. If $X$ is 1st order stationary, then
    * all $X_t$ have the same CDF
    * all $X_t$ have the same mean and variance, i.e. $\mu_X(t) = \mu_X$ is constant
    * the signal has constant power

3. If $X$ is 2nd order stationary with finite power, then in addition, the auto-correlation is shift-invariant
    $$
        \forall t_1, t_2, \tau \in\mathbb T, \quad R_{XX}(t_1 + \tau, t_2 + \tau) = R_{XX}(t_1, t_2)
    $$
    since $(X_{t_1+\tau}, X_{t_2+\tau}) \stackrel{d}{=} (X_{t_1}, X_{t_2})$

4. If $X$ is $N$-th order stationary for some $N\ge 1$, then all conclusions in 2 also hold.

Note on terminology: In this note, 2nd order stationarity is defined in terms of the CDF, i.e. $(X_{t_1+\tau}, X_{t_2+\tau}) \stackrel{d}{=} (X_{t_1}, X_{t_2})$. In other literature, especially in time series analysis, "second-order stationary" often refers to what this note calls weak stationarity, which is defined in terms of the moments (mean and auto-correlation) only. Our 2nd order stationarity (with finite power) implies weak stationarity, but not vice versa.

### Strict Stationarity

$X$ is called **strictly stationary** (also known as strict-sense stationary) $\iff$ $\forall N\ge 1$, $X$ is $N$-th order stationary. In other words, $X$ is strictly stationary if **all** of its FDDs are shift-invariant.

$$
\forall n \ge 1, \forall t_1, \cdots, t_n, \tau \in\mathbb T, \:
F_{t_1+\tau, \cdots, t_n+\tau} = F_{t_1, \cdots, t_n}
$$

or equivalently,

$$
\forall n \ge 1, \forall t_1, \cdots, t_n, \tau \in\mathbb T, \:
(X_{t_1+\tau}, \cdots, X_{t_n+\tau}) \stackrel{d}{=} (X_{t_1}, \cdots, X_{t_n})
$$

Remarks:

* If $X$ is an iid process, then it is strictly stationary.
* The process $X_t = X_0$ from the first example is strictly stationary, although it is not iid.

However, it is nearly impossible to verify if a signal is strictly stationary in practice. This leads us to define a weaker version of stationarity.

### Weak Stationarity

$X$ is called **weakly stationary** (also known as wide-sense stationary) if

1. $X$ has **finite power**
    $$
        \forall t \in \mathbb T, \quad \mathbb E[X_t^2] < \infty
    $$
2. $X$ has **constant mean**
    $$
        \forall t \in \mathbb T, \quad \mathbb E[X_t] = \mu_X
    $$
3. The auto-correlation is **shift-invariant**
    $$
        \forall \tau, t_1, t_2 \in \mathbb T, \quad R_{XX}(t_1+\tau, t_2+\tau) = R_{XX}(t_1, t_2)
    $$

Remarks:

* In particular, setting $t_1 = t_2 = t$ and $\tau = -t$ in the 3rd condition shows that a weakly stationary process has **constant power**
    $$
        \forall t \in \mathbb T, \quad \mathbb E[X_t^2] = R_{XX}(0, 0) = \mathbb E[X_0^2]
    $$
* Unlike 2nd order stationarity, weak stationarity only constrains the first and second moments, not the FDDs. Hence, it is a weaker notion of stationarity and easier to verify in practice.
* 2nd order stationarity always implies weak stationarity, but not vice versa. However, for Gaussian processes, the two notions coincide since the FDDs are fully determined by the mean and auto-covariance.

Relations between the notions of stationarity (assuming finite power):
$$
    \text{strictly stationary}
    \implies \text{2nd order stationary}
    \implies \text{weakly stationary}
$$

In general, weak stationarity does not imply 1st order stationarity either. Consider the following counterexample.

Let $\{ X_t \}_{t\in \mathbb Z}$ be independent with $X_t \sim \mathcal N(0, 1)$ for even $t$ and $X_t$ uniformly distributed on $\{-1, 1\}$ for odd $t$. Then $\mathbb E[X_t]=0$ and $\mathbb E[X_{t_1}X_{t_2}] = \delta(t_1-t_2)$. Hence, $X$ is weakly stationary. However, the CDF of $X_t$ changes with $t$.

Additional properties of weakly stationary processes and their applications in signal processing are discussed in [Weakly Stationary Process](weakly_stationary_process.md).

## Frequently Used Stochastic Signals

### White Noise

A stochastic process $\{ X_t \}_{t\in \mathbb Z}$ is called a **white noise** if
$$
    \forall t, t_1, t_2\in\mathbb Z, \quad \mathbb E[X_t]=0 \: \land \: R_{XX} (t_1, t_2) =  \sigma^2 \delta(t_1 - t_2)
$$
where $\delta(\cdot)$ is the Kronecker delta.

Trivial facts: Let $\{ X_t \}_{t\in \mathbb Z}$ be a white noise. Then $X$

* has constant zero mean
* has constant power: $\sigma^2$
* has uncorrelated values at any two distinct time instances
* is always weakly stationary

Remarks:

* A white noise is not necessarily iid, since uncorrelated does not imply independent. Conversely, a zero-mean iid process with finite power $\sigma^2$ is a white noise.
* A Gaussian white noise is iid, since uncorrelated jointly Gaussian random variables are independent.
* In continuous time, the Kronecker delta is replaced by the Dirac delta. Such a process has infinite power and is an idealisation.

### Random Walk

Let $\{ W_t \}_{t\in \mathbb Z}$ be a white noise with power $\sigma^2$. A random walk is another stochastic process $\{ X_t \}_{t\in \mathbb Z}$, defined by
$$
    X_t =
    \begin{cases}
        0 & t\le 0 \\
        X_{t-1} + W_t & t> 0 \\
    \end{cases}
$$
Equivalently, $X_t = \sum_{k=1}^{t} W_k$ for $t > 0$.

The random walk $X$ has following properties

1. $X$ has zero mean.  $$ \forall t>0, \mathbb E[X_t] =0 $$
1. $X$ has the auto-correlation
    $$
        \forall t_1, t_2 > 0, \quad R_{XX}(t_1, t_2) = \sigma^2 \min(t_1, t_2)
    $$

    *Proof:*
    $$
        R_{XX}(t_1, t_2)
        = \sum_{k=1}^{t_1} \sum_{l=1}^{t_2} \mathbb E[W_k W_l]
        = \sum_{k=1}^{t_1} \sum_{l=1}^{t_2} \sigma^2 \delta(k-l)
        = \sigma^2 \min(t_1, t_2)
    $$
1. $X$ has increasing power.  $$ \forall t>0, \mathbb E[X_t^2] =\sigma^2 t $$
1. $X$ is <span style="color:red"> **not** </span>  weakly stationary.

Remark: The continuous-time version of the random walk is the **Wiener process** (Brownian motion), which has the same auto-correlation $\sigma^2 \min(t_1, t_2)$ for $t_1, t_2 \ge 0$.

### Auto-Regressive Signals

Let $\{ W_t \}_{t\in \mathbb Z}$ be a white noise. An **auto-regressive process of order $p$**, AR($p$), is defined by
$$
    X_t = \sum_{k=1}^{p} a_k X_{t-k} + W_t
$$
The random walk is an AR(1) process with $a_1=1$ and zero initial condition. For suitable coefficients (e.g. $p=1$, $\vert a_1\vert<1$), the AR process is weakly stationary and can be analysed as a filtered white noise. See [Weakly Stationary Process](weakly_stationary_process.md).

## Appendix

### Equality of Random Variables

Let $X$ and $Y$ be random variables. There are two common notions of "$X$ equals $Y$".

* $X$ and $Y$ are **equal almost surely (a.s.)**, written $X = Y$ a.s., if they are defined on the same probability space and
    $$
        \mathbb P\left(\{ \omega\in\Omega : X(\omega) = Y(\omega) \}\right) = 1
    $$
    i.e. $X$ and $Y$ take the same value for almost every random outcome.

* $X$ and $Y$ are **equal in distribution**, written $X \stackrel{d}{=} Y$, if they have the same CDF
    $$
        \forall x\in\mathbb R, \quad F_X(x) = F_Y(x)
    $$
    i.e. $X$ and $Y$ have the same statistical behaviour. They may even be defined on different probability spaces.

**Properties:**

1. $X = Y$ a.s. $\implies$ $X \stackrel{d}{=} Y$

    *Proof:* Let $N = \{ X \neq Y \}$ with $\mathbb P(N) = 0$. Then $\{ X \le x \} \subseteq \{ Y \le x \} \cup N$. Hence, $F_X(x) \le F_Y(x) + \mathbb P(N) = F_Y(x)$. By symmetry, $F_Y(x) \le F_X(x)$.

2. The converse is false in general.

    **Example:** Let $X$ be uniformly distributed on $\{-1, 1\}$ and $Y = -X$. Then $X \stackrel{d}{=} Y$, but $\mathbb P(X = Y) = 0$.

Remarks:

* Both notions extend to random vectors by replacing the CDF with the joint CDF.
* For stochastic processes, $X \stackrel{d}{=} Y$ means that $X$ and $Y$ have the same FDDs, i.e. they are statistically equivalent.
* Stationarity is formulated with $\stackrel{d}{=}$, not with a.s. equality. e.g. An iid process is strictly stationary, i.e. $(X_{t_1+\tau}, \cdots, X_{t_N+\tau}) \stackrel{d}{=} (X_{t_1}, \cdots, X_{t_N})$. However, $X_{t+\tau} \neq X_t$ in general, since the two samples are independent.
