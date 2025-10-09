---
title: "Everything about Multivariate Gaussian"
date: "2024"
author: "Ke Zhang"
---

# Everything about Multivariate Gaussian

## Basics

1. **PDF**: $X\sim\mathcal{N}(\mu, \Sigma)$. #param: $\mathcal{O}(d^2)$.

    > $$
    > p_X(x \mid \mu, \Sigma) =
    > \frac{1}{\sqrt{(2\pi)^d \vert\Sigma\vert}}
    > \exp\left[
    >   -\frac{1}{2}(x-\mu)^\top \Sigma^{-1} (x-\mu)
    > \right]
    > $$
1. **Mean**: $\mu\in\mathbb{R}^{d}$.
1. **Variance**: $\Sigma\in\mathbb{R}^{d\times d}$​. Its inverse is called **precision matrix**.
1. For two **Gaussian** random variables, **uncorrelatedness** $\iff$ **independence**.

Special cases:

* Diagonal Gaussian: $X\sim\mathcal{N}(\mu, \operatorname{diag}(\sigma_1^2, \dots, \sigma_d^2))$. #param: $\mathcal{O}(d)$.
    $$
    p_X(x \mid \mu, \sigma_{1:d}) =
    \frac{1}{\sqrt{(2\pi)^d} \sigma_1 \cdots \sigma_d}
    \exp\left(
      -\sum_{i=1}^d \frac{(x_i - \mu_i)^2}{2\sigma_i^2}
    \right)
    $$

* Spherical Gaussian: $X\sim\mathcal{N}(\mu, \sigma^2 I_d)$. #param: $\mathcal{O}(d)$. ($\mathcal{O}(1)$ if $\mu$ is known)
    $$
    p_X(x \mid \mu, \sigma) =
    \frac{1}{\sqrt{(2\pi)^d} \sigma^d}
    \exp\left(
      -\frac{\Vert x - \mu \Vert ^2}{2\sigma^2}
    \right)
    $$
* Standard Gaussian: $X\sim\mathcal{N}(0, I_d)$. No param.
    $$
    p_X(x) =
    \frac{1}{\sqrt{(2\pi)^d}}
    \exp\left(
      -\frac{\Vert x \Vert^2}{2}
    \right)
    $$

### Normalization Property

Let $A\in\mathbb{R}^{d\times d}$ be a s.p.d. matrix and $b\in\mathbb{R}^{d}$ be a vector. If $p_X(\cdot)$ is proportional to exp of the quadratic form

$$
p_X(x) \propto \exp\left( -\frac{1}{2} x^\top A x + b^\top x \right)
$$

Then, $X$ must be multivariate Gaussian with mean and variance.

$$
\begin{align*}
X &\sim \mathcal N(\mu, \Sigma) \\
\text{where}\quad
\Sigma &= A^{-1} \\
\mu &= \Sigma b = A^{-1} b
\end{align*}
$$

### Level Sets

The level sets of a multivariate Gaussian $\mathcal{N}(\mu, \Sigma)$ are ellipsoids centered at $\mu$.

Let $\lambda_i$ be eigenvalues of $\Sigma$, and $v_i$ the corresonding eigenvectors ($i=1,\dots,d$). Then:

* The ellipsoids are aligned along the directions of $v_i$.
* The semi-axis lengths of ellipsoids are proprotional to $\sqrt{\lambda_i}$.

### Moment Generating Function

The moment generating function (MGF) of a multivariate Gaussian $\mathcal{N}(\mu, \Sigma)$ is

$$
\begin{align*}
M_X(t)
&= \exp\left( t^\top \mu + \frac{1}{2} t^\top \Sigma t \right)
&& \text{for } t\in\mathbb R^d
\end{align*}
$$

Calculating moments of $X$ $\iff$ calculating derivatives of $M_X$.

$$
\begin{align*}
\mathbb E[X_i X_j]
&= \left. \frac{\partial^2 M_X(t)}{\partial t_i \partial t_j} \right|_{t=0}
\end{align*}
$$

## Invariance

The Gaussian distribution is invariant under affine transformation, summation, marginallisation, and conditioning. i.e. "Once Gaussian, (almost) always Gaussian".

* **Affine transformation** of a Gaussian rand. vec. is again a Gaussian rand. vec.
    > $$
    > X \sim \mathcal{N}(\mu, \Sigma)
    > \implies AX+b \sim \mathcal{N}(A\mu + b, A\Sigma A^\top)
    > $$
* **Sum** of two independent Gaussian rand. vec. is again a Gaussian rand. vec.
    > $$
    > \begin{rcases}
    >   X \sim \mathcal{N}(\mu_X, \Sigma_X)\\
    >   Y \sim \mathcal{N}(\mu_Y, \Sigma_Y)\\
    >   X,\, Y \:\text{ indepd.} \\
    > \end{rcases}
    > \implies X + Y \sim \mathcal{N}(\mu_X + \mu_Y, \Sigma_X + \Sigma_Y)
    > $$

* In general, if $X,Y$ are **jointly Gaussian**, their sum is always Gaussian. However, the additivity of covariance matrices no longer holds if $X,Y$ are correlated.

* Sum of two Gaussian **densities** is **NOT** a Gaussian density! Counter example: GMM.

### Jointly Gaussian Random Vectors

Let a Gaussian rand. vec. $X$ be partitioned into two sub rand. vec.s $X_A$ and $X_B$ (not necessarily of the same length). In other words, two Gaussian rand. vec.s $X_A$ and $X_B$ are jointly Gaussian.

$$
X = \begin{bmatrix} X_A \\ X_B \end{bmatrix}
\sim \mathcal{N}
\left(
\begin{bmatrix}
  \mu_A \\
  \mu_B 
\end{bmatrix},
\begin{bmatrix}
  \Sigma_{AA} & \Sigma_{AB} \\
  \Sigma_{BA} & \Sigma_{BB} \\
\end{bmatrix}
\right)
$$

Then,

* **Marginalization**: The marginal $p(x_A) = \int p_X(x_A, x_B) \mathrm d x_B$ is also Gaussian
    > $$
    > X_A \sim \mathcal{N}(\mu_A, \Sigma_{AA})
    > $$

* **Conditioning**: Conditioned on $x_B$, the rand. vec. $X_A$ is also Gaussian
    > $$
    > \begin{align*}
    > X_A \vert x_B &\sim\mathcal{N} \left( \mu_{A\vert B},\: \Sigma_{A\vert B} \right) \\
    > \mu_{A\vert B} &= \mu_A + \Sigma_{AB}\Sigma_{BB}^{-1}(x_B - \mu_B) \\
    > \Sigma_{A\vert B} &= \Sigma_{AA} - \Sigma_{AB} \Sigma_{BB}^{-1} \Sigma_{BA}
    > \end{align*}
    > $$

If $X_A$ and $X_B$ have the same length, then any **linear combination** $\alpha X_A + \beta X_B$ is Gaussian ($\alpha, \beta \in\mathbb R$). In particular:

* The sum $X_A + X_B$ is also Gaussian (invariance under affine transform):
    > $$
    > X_A + X_B \sim \mathcal{N}(\mu_A + \mu_B, \Sigma_{AA} + \Sigma_{BB} + \Sigma_{AB} + \Sigma_{BA})
    > $$
* Special case: $X_A$ and $X_B$ are in addition independent, i.e. $\Sigma_{AB} = \Sigma_{BA} = 0$
    > $$
    > X_A + X_B \sim \mathcal{N}(\mu_A + \mu_B, \Sigma_{AA} + \Sigma_{BB})
    > $$

### Product

We differentiate product of two Gaussian densities and product of two Gaussain random variables!

* Product of two Gaussian rand. vec. is generally **NOT** a Gaussian rand. vec.!
* Product of two Gaussians densities is **almost** a Gaussian density up to a normalization constant.
    > $$
    > \begin{align*}
    > \mathcal N(x \mid \mu_1, \Sigma_1) \cdot \mathcal N(x \mid \mu_2, \Sigma_2)
    > &= \mathcal N(x \mid \mu_*, \Sigma_*) \cdot \mathcal N(\mu_1 \mid \mu_2, \Sigma_1 + \Sigma_2) \\
    > &\propto \mathcal N(x \mid \mu_*, \Sigma_*)
    > \end{align*}
    > $$
  where
    > $$
    > \begin{align*}
    > \Sigma_*
    > &= \left( \Sigma_1^{-1} + \Sigma_2^{-1} \right)^{-1} \\
    > \mu_*
    > &= \Sigma_* \left( \Sigma_1^{-1} \mu_1 + \Sigma_2^{-1} \mu_2 \right)
    > \end{align*}
    > $$

## Central Limit Theorem (CLT)

**Lindeberg–Lévy CLT**  
Let $X_1, X_2, \dots$ be a sequence of iid random variables with $\mathbb E[X_i] = \mu$ and $\mathbb V[X_i] = \sigma^2$. Then, the normalized sample mean $\sqrt{n} (\bar X_n - \mu)$ converges in distribution to $\mathcal N(0, \sigma^2)$.

$$
\sqrt{n} (\bar X_n - \mu) \xrightarrow{d} \mathcal N(0, \sigma^2)
$$

Remarks:

* The unnormalized sample mean $\bar X_n$ is defined as
  $$
  \bar X_n \triangleq \frac{X_1 + \dots + X_n}{n}
  $$
* The distribution of $X_i$ can be **any** distribution.

## Derivatives

The derivative of the log PDF is also called **score function** in statistics.

### Derivatives for General Gaussian

**Log of PDF**:

> $$
> \ln p_X(x \mid \mu, \Sigma) = -\frac{d}{2}\ln(2\pi) - \frac{1}{2}\ln\vert\Sigma\vert -  \frac{1}{2}(x-\mu)^\top \Sigma^{-1} (x-\mu)
> $$

* Derivative w.r.t. $\mu$
    > $$
    > \frac{\partial \ln p_X(x \mid \mu, \Sigma)}{\partial\mu} = \Sigma^{-1}(x-\mu)
    > $$
* Derivative w.r.t. $\Sigma$
    > $$
    > \frac{\partial \ln p_X(x \mid \mu, \Sigma)}{\partial\Sigma} =
    > \frac{1}{2} \left[
    >   \Sigma^{-1}(x-\mu)(x-\mu)^\top \Sigma^{-1} - \Sigma^{-1}
    > \right]
    > $$

### Derivatives for Diagonal Gaussian

**Log of PDF**:
$$
\ln p_X(x \mid \mu, \sigma_{1:d}) = -\frac{d}{2}\ln(2\pi) - \frac{1}{2}\sum_{i=1}^d \ln\sigma_i^2 - \frac{1}{2}\sum_{i=1}^d \frac{(x_i - \mu_i)^2}{\sigma_i^2}
$$

* Derivative w.r.t. $\mu$
    $$
    \frac{\partial \ln p_X(x \mid \mu, \sigma_{1:d})}{\partial\mu} =
    \operatorname{diag}(\sigma_1^{-2}, \dots, \sigma_d^{-2}) \cdot
    (x-\mu)
    $$
* Derivative w.r.t. $\sigma_i^2$
    $$
    \frac{\partial \ln p_X(x \mid \mu, \sigma_{1:d})}{\partial\sigma_i^2} =
    -\frac{1}{2 \sigma_i^2} + \frac{(x_i - \mu_i)^2}{2 \sigma_i^4}
    $$

### Derivatives for Spherical Gaussian

**Log of PDF**:
$$
\ln p_X(x \mid \mu, \sigma) =
  -\frac{d}{2}\ln(2\pi) - \frac{d}{2} \ln\sigma^2 - \frac{\Vert x - \mu\Vert^2}{2\sigma^2}
$$

* Derivative w.r.t. $\mu$
    $$
    \frac{\partial \ln p_X(x \mid \mu, \sigma)}{\partial\mu} = \sigma^{-2} \cdot
    (x-\mu)
    $$
* Derivative w.r.t. $\sigma^2$
    $$
    \frac{\partial \ln p_X(x \mid \mu, \sigma)}{\partial\sigma^2} =
    -\frac{d}{2 \sigma^2} + \frac{\Vert x - \mu \Vert^2}{2 \sigma^4}
    $$

## MLE for Multivariate Gaussian

Suppose $x^{(1)}, \dots, x^{(N)} \in\mathbb{R}^d$ are drawn iid from $\mathcal{N}(\mu, \Sigma)$. The **log-likelihood** is then

> $$
> \begin{align*}
> \ln p\left( x^{(1)}, \dots, x^{(N)} \Big\vert  \mu, \Sigma \right)
> &= \ln \prod_{n=1}^N \mathcal{N} \left( x^{(n)} \Big\vert  \mu, \Sigma \right) \\
> &= \sum_{n=1}^N \ln \mathcal{N} \left( x^{(n)} \Big\vert  \mu, \Sigma \right) \\
> \end{align*}
> $$

### MLE for General Gaussian

Let $L(\mu, \Sigma) = \ln p\left( x^{(1)}, \dots, x^{(N)} \Big\vert  \mu, \Sigma \right)$.

* Derivative w.r.t. $\mu$
    $$
    \frac{\partial L(\mu, \Sigma)}{\partial\mu} = \sum_{n=1}^N \Sigma^{-1}\left( x^{(n)}-\mu \right)
    $$
* Derivative w.r.t. $\Sigma$
    $$
    \frac{\partial L(\mu, \Sigma)}{\partial\Sigma} =
     \frac{1}{2} \sum_{n=1}^N \left[
      \Sigma^{-1}\left( x^{(n)}-\mu \right)\left( x^{(n)}-\mu \right)^\top \Sigma^{-1} - \Sigma^{-1}
    \right]
    $$

Letting the derivatives be zero, we get

* MLE of the mean $\mu$
    > $$
    > \hat\mu = \frac{1}{N} \sum_{n=1}^N x^{(n)}
    > $$
* MLE of the covariance matrix $\Sigma$
    > $$
    > \hat\Sigma = \frac{1}{N} \sum_{n=1}^N \left( x^{(n)}-\hat\mu \right) \left( x^{(n)}-\hat\mu \right)^\top
    > $$

Note: The MLE of the covariance matrix $\Sigma$ is **biased**. The unbiased estimate is

$$
\hat\Sigma = \frac{1}{N-1} \sum_{n=1}^N \left( x^{(n)}-\hat\mu \right) \left( x^{(n)}-\hat\mu \right)^\top
$$

### MLE for Diagonal Gaussian

The MLE of $\mu$ is the same as in general Gaussian. Only the MLE of the variances are different.

* Derivative w.r.t. $\sigma_i^2$
    $$
    \frac{\partial L(\mu, \Sigma)}{\partial\sigma_i^2} =
      -\frac{N}{2\sigma_i^2} + \sum_{n=1}^N \frac{\big(x^{(n)}_i - \mu_i \big)^2}{2\sigma_i^4}
    $$

Letting the derivatives be zero, we get

* MLE of the component variance $\sigma_i^2$
    > $$
    > \hat\sigma_i^2 = \frac{1}{N} \sum_{n=1}^N \left(x^{(n)}_i - \mu_i \right)^2
    > $$

### MLE for Spherical Gaussian

The MLE of $\mu$ is the same as in general Gaussian. Only the MLE of the variances are different.

* Derivative w.r.t. $\sigma^2$
    $$
    \frac{\partial L(\mu, \Sigma)}{\partial\sigma^2} =
      -\frac{Nd}{2\sigma^2} + \sum_{n=1}^N \frac{\big\Vert x^{(n)} - \mu \big\Vert^2}{2\sigma^4}
    $$

Letting the derivatives be zero, we get

* MLE of the component variance $\sigma^2$
    > $$
    > \hat\sigma^2 = \frac{1}{Nd} \sum_{n=1}^N \left\Vert x^{(n)} - \mu \right\Vert^2
    > $$
