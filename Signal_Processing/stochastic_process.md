---
title: "Stochastic Process"
author: "Ke Zhang"
date: "2023"
fontsize: 12pt
---

# Stochastic Process

Note: we often use the following terms interchangably.

* Stochastic process $\iff$ stochasitic signal $\iff$ signal
* $X_t, \, t\in\mathbb Z \iff \{ X_t \}_{t\in \mathbb Z} \iff \{ X(t) \}_{t\in \mathbb Z} \iff X$

## Basic Definitions

A discrete-time stochastic process is a squence of random variables  $\{ X_t \}_{t\in \mathbb Z}$, defined on the same probability space.

Strictly speaking, a stochastic process is a function of two variables
$$
X:\Omega \times \mathbb Z \to \mathbb R, (\omega, t)\mapsto X(\omega, t)
$$

This leads to two interpretations of $X$:

* For fixed random outcome $\omega_0$, $X(\omega_0, \cdot)$ is a realisation of the tranjectory.
* For fixed time index $t_0$, $X(\cdot, t_0)$ is a random variable.

In engineering, we do not define the sample space explicitly. In the upcoming sections, we write $X_t$ for the sake of simplicity. Please also note that our definition can be generalised to complex stochastic process.

**Example:** Let $X_0 \in \{-1, 1 \}$ with $P(X_0 = -1) = P(X_0 = 1) = \frac{1}{2}$. Define $X_t = X_0, \forall t \in\mathbb Z$.

To fully describe a stochastic process $X$, we need the joint distibution of all possible combinations of random varibales in $X$. i.e. We need the joint CDFs of  $X_{t_1}, \cdots, X_{t_n}$ for all $n\ge 1$ and all $t_1, \cdots, t_n\in\mathbb Z$. This includes

* The CDFs of $X_t, \:\forall t\in\mathbb Z$
* The joint CDFs of $X_{t_1}, X_{t_2}, \:\forall t_1,t_2 \in\mathbb Z$
* The joint CDFs of $X_{t_1}, X_{t_2}, X_{t_3} \:\forall t_1, t_2, t_3\in\mathbb Z$
* And so on...

Hence, we focus on more restictive stochastic processes, whose statistical properties can be easily described. One of simplest model is the iid stochastic process.

A stochastic process $X$ is called iid if
$$
\forall n\ge 1, \forall t_1, \cdots, t_n\in\mathbb Z,
\:  
X_{t_1}, \cdots, X_{t_n} \text{ are iid}.
$$

Our previsous example is an iid stochastic process.

### Mean and Power

The statistical mean is defined by $\mu_t = \mathbb E[X_t]$.

The power is defined by $\mathbb E[X_t^2]$.

Remarks:

1. In general, both mean and power denpend on time. However, we will see that they are time-invariant for some class of stochastic signals.
2. In applications, we usually assume that $X$ has finite power, i.e. $\mathbb{E}[X_t^2] \le \infty, \forall t\in\mathbb Z$.

### Auto-Corelation and Auto-Covariance

Given two random processes $\{ X_t \}_{t\in \mathbb Z}$,

* The auto correlation is defined by
    $$
    \begin{equation}
    \fbox{$\displaystyle
                R_{XX}(t_1, t_2) = \mathbb E\left[X_{t_1}X_{t_2}\right]
    $}
    \end{equation}
    $$


    $$
        R_{XX}(t_1, t_2) = \mathbb E\left[X_{t_1}X_{t_2}\right]
    $$

* The auto covariance is defined by

    $$
    \begin{split}
        C_{XX}(t_1, t_2)
        &= \mathbb E\left[(X_{t_1} - \mu_{t_1}) (X_{t_2} - \mu_{t_2})\right] \\
        &= R_{XX}(t_1, t_2) - \mu_{t_1}\mu_{t_2}
    \end{split}
    $$

Remarks:

* In general, both $R_{XX}(t_1, t_2)$ and $C_{XX}(t_1, t_2)$ depends on two time indices. Later, we will see that the dependency reduces to the time difference only for wide-sense stationary signals.
* The signal power coincides with $R_{XX}(t_1, t_2)$ when $t_1=t_2$.

### Cross-Correlation and Cross-Covariance

Given two random processes $\{ X_t \}_{t\in \mathbb Z}$ and $\{ Y_t \}_{t\in \mathbb Z}$. We are interested in their joint statistical properties. In particular, we define

* The **cross-correaltion** as
    $$
        R_{XY}(t_1, t_2) = \mathbb E\left[X_{t_1}Y_{t_2}\right]
    $$
* The **cross-corvariance** as
    $$
        C_{XY}(t_1, t_2) = \mathbb E\left[ (X_{t_1} - \mu_{X, t_1})  (Y_{t_1} - \mu_{Y, t_1})\right]
    $$

### Generalisation

The notion of stochstic process can be generalised into 
$$
X:\Omega \times \mathbb T \to \mathbb S, (\omega, t)\mapsto X(\omega, t)
$$
where

* $\mathbb T$ is called the time horizon, which can be $\mathbb N$, $\mathbb Z$ or $\mathbb R$.
* $\mathbb S$ is called the state space, which can be $\mathbb R$, $\mathbb C$ or $\mathbb R^n$.

## Stationarity

### $N$-th order Stationarity

Let $N\ge 1$. A stochastic process $X$ is called $N$-th order stationary if
$$
\forall t_1, \cdots, t_N, \tau \in\mathbb Z,
\:  
\text{the joint CDF of } X_{t_1}, \cdots, X_{t_N} \: = \: \text{the joint CDF of } X_{t_1+\tau}, \cdots, X_{t_N+\tau}.
$$

In other words,  $X$ is $N$-th order stationary if, $\forall t_1, \cdots, t_N \in\mathbb Z$, the joint distribution of $X_{t_1+\tau}, \cdots, X_{t_N+\tau}$ is independent of time shift $\tau$.

**Properties:**

1. $X$ is N-th order stationary $\implies$  $X$ is K-th order stationary, $\forall K\le N$
  
    In other words, higher order stationarity implies lower order stationarity

    *Proof*: ...

2. If $X$ is 1st order stationary, then
    * all $X_1, X_2, \cdots$ have the same CDF
    * all $X_1, X_2, \cdots$ have the same mean and variance
    * The signal has constant power

3. If $X$ is any order stationary, then all conclusion in 2 also holds.

### Strict-Sense Stationarity

$X$ is called strict-sense stationary (SSS) $\iff$ $\forall N\ge 1$, $X$ is $N$-th order stationary. In other words, $X$ is SSS if the joint CDF of arbitrarily many time instances is shift-invariant.

Remark:

* If $X$ is an iid process, then it is SSS.

However, it is nearly impossible to verify if a signal is SSS in practice. This leads us to define a weaker version of stationarity.

### Wide-Sense Stationarity

$X$ is called **wide-sense stationary (WSS)** or **weakly stationary** if both
$$
\begin{split}
    \forall \tau, t \in \mathbb Z, \quad
    & \mathbb E[X_{t+\tau}] = \mathbb E[X_{t}]\\
    \forall \tau, t_1, t_2 \in \mathbb Z, \quad
    & \mathbb E[X_{t_1+\tau} X_{t_2+\tau}] = \mathbb E[X_{t_1} X_{t_2}].
\end{split}
$$
In other words, the mean and auto-correlation don't depend on time shift.

Remarks to WSS signals:

* WSS signals have constant mean and power
* If a signal is SSS, then it is also WSS

For a WSS signal, the auto correlation depends only on the time difference, not on the location of time origin. i.e.

$$
\forall t_1, t_2, s_1, s_2 \in\mathbb Z \text{ with }  t_1 - t_2 = s_1 - s_2,
\implies
\mathbb E[X_{t_1} X_{t_2}] = \mathbb E[X_{s_1} X_{s_2}]
$$

Hence, we define the **auto correlation function**:
$$
    R_{XX}(t) = \mathbb E[X_{t+\tau} X_{\tau}]
$$

Similary, we define the **auto covariance function**:

$$
\begin{split}
    C_{XX}(t)
    &= \mathbb E\left[(X_{t+\tau} - \mu) (X_{\tau} - \mu)\right] \\
    &= R_{XX}(t) - \mu^2
\end{split}
$$


Note: Both $R_{XX}(t)$ and  $C_{XX}(t)$ are independent of $\tau$ in their definitions.

Propterties of $R_{XX}(t)$ and  $C_{XX}(t)$ of a WSS signal:

1. The signal power is $R_{XX}(0)$

2. $\vert R_{XX}(t) \vert $ is bounded by the signal power
    $$
        \forall t\in\mathbb Z, \: \left\vert R_{XX}(t) \right\vert \le  R_{XX}(0)
    $$

    *Proof:* This is followed by Cauchy-Schwarz inequality.

    $$
        \vert R_{XX}(t) \vert
        =   \left\vert \mathbb E[X_{t+\tau} X_{\tau}] \right\vert
        \le \sqrt{\mathbb E[X_{t+\tau}^2] \cdot \mathbb E[X_{\tau}^2] }
        =   \sqrt{R_{XX}(0) \cdot R_{XX}(0)}
        =   R_{XX}(0)
    $$

3. $R_{XX}(t)$ is an even function
    $$
        R_{XX}(-t) =  R_{XX}(t)
    $$

### Joint Wide-Sense Stationarity

Two stochastic processes $\{ X_t \}_{t\in \mathbb Z}$ and $\{ Y_t \}_{t\in \mathbb Z}$ are called **joint wide-sense stationary** or **jointly weakly stationary** if both

$$
\begin{split}
    1. \quad & \text{Both } X \text{ and } Y \text{ are WSS} \\
    2. \quad & \forall \tau, t_1, t_2 \in \mathbb Z, \quad
    \mathbb E[X_{t_1+\tau} Y_{t_2+\tau}] = \mathbb E[X_{t_1} Y_{t_2}].
\end{split}
$$

The second criterion means that the cross correlation does not depend on time shift.

Hence, we define the **cross correlation function**:
$$
    R_{XY}(t) = \mathbb E[X_{t+\tau} Y_{\tau}]
$$

Similary, we define the **cross covariance function**:
$$
\begin{split}
    C_{XY}(t)
    &= \mathbb E\left[(X_{t+\tau} - \mu_{X}) (Y_{\tau} - \mu_{Y})\right] \\
    &= R_{XY}(t) - \mu_X \mu_Y
\end{split}
$$

Propterties of $R_{XY}(t)$ and  $C_{XY}(t)$ of jointly WSS signals:

1. $R_{XY}(t)$ is symmetric
    $$
        R_{XY}(-t) =  R_{YX}(t)
    $$

## Frequently Used Stochastic Signals

### White Noise

A stochastic process $\{ X_t \}_{t\in \mathbb Z}$ is called a **white noise** if
$$
    \forall t\in\mathbb Z, \quad \mathbb E[X_t]=0 \: \land \: R_{XX} (t_1, t_2) =  \sigma^2 \delta(t_1 - t_2)
$$
where $\delta(\cdot)$ is the Kronecker delta.

Trivial facts: Let $\{ X_t \}_{t\in \mathbb Z}$ be a white noise

* has constant zero mean
* has constant power: $\sigma^2$
* any two distinct time instances are uncorrelated
* is always WSS and has the auto correlation function
    $$
        R_{XX} (t) =  \sigma^2 \delta(t)
    $$

### Random Walk

Let $\{ W_t \}_{t\in \mathbb Z}$ be a white noise with power $\sigma^2$. A random walk is another stochastic process $\{ X_t \}_{t\in \mathbb Z}$, defined by
$$
    X_t =
    \begin{cases}
        0 & t\le 0 \\
        X_{t-1} + W_t & t> 0 \\
    \end{cases}
$$

The random walk $X$ has following properties

1. $X$ has zero mean.  $$ \forall t>0, \mathbb E[X_t] =0 $$
1. $X$ has increasing power.  $$ \forall t>0, \mathbb E[X_t^2] =\sigma^2 t $$
1. $X$ is <span style="color:red"> **not** </span>  WSS.

### Auto-Regressive Signals

## Linear Filtering of WSS Signals

Notation: Old: $X_t$. New: $X(t)$.

Let $\{ X(t)\}_{t\in \mathbb Z}$ be a WSS stochastic signal and $\{ h(t) \}_{t\in \mathbb Z}$ be the impulse response of a discrete-time LTI system. The output signal  $\{ X(t)\}_{t\in \mathbb Z}$  is another stochastic process $\{ Y(t)\}_{t\in \mathbb Z}$, defined by the convolution

$$
    Y(t) = (X * h)(t) = \sum_{\tau = -\infty}^{\infty} X(t-\tau) h(\tau)
$$

### Time Domain Analysis

Suppose that the impluse response $h$ is summable, we can draw the following conclusion

1. $Y$ is WSS with mean
    $$
        \mu_Y = \mu_X \sum_{\tau = -\infty}^{\infty}  h(\tau)
    $$
2. $X$ and $Y$ are jointly WSS with cross correlation function
    $$
        R_{YX} (t) = (h * R_X)(t)
    $$
3. $Y$ has the autocorelation
    $$
        R_{Y} (t) = (h^r * R_{YX})(t) = (h^r * h * R_{X})(t)
    $$
    where $h^r(t) = h(-t)$

### Frequency Domain Analysis

## Dummy

hello
$$
\begin{equation}
\fbox{$\displaystyle
    \int_{\Omega_0} \zeta(\omega) d\omega
    \geq \bar{r}
$}
\end{equation}
$$
