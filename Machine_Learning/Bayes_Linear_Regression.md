---
title: "Bayesian Linear Regression"
author: "Ke Zhang"
date: "2025"
fontsize: 12pt
---

# Bayesian Linear Regression

**Preliminary**: linear regression, Bayesian inference.

## Problem Formulation

* Given: training dataset $D=\{(\mathbf{x}_i, y_i)\}_{i=1}^n \stackrel{\text{iid}}{\sim} p(\mathbf x, y)$ where $(\mathbf{x}_i, y_i) \in \mathbb R^d \times \mathbb R$.
* Statistical model: $y_i = \mathbf{w}^\top \mathbf{x}_i + \varepsilon_i, \quad \varepsilon_i \stackrel{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2_\text{n})$
* Additional assumption: $\varepsilon_i$ and $\mathbf{x}_i$ are statistically independent
* Goal: Predict the label for a new data point $\mathbf{x}_*$ using the **full** posterior distribution $p(\mathbf w \mid D)$

## Recap: Point Estimates for Linear Regression

* Discriminative model parameterized by $\mathbf{w}$:

$$
\begin{align}
p(y_i \mid \mathbf{x}_i, \mathbf{w})
&= \mathcal{N}(y_i \mid \mathbf{w}^\top \mathbf{x}_i, \sigma^2_\text{n}) \\
&= \frac{1}{\sqrt{2\pi \sigma^2_\text{n}}} \exp\left(-\frac{(y_i - \mathbf{w}^\top \mathbf{x}_i)^2}{2\sigma^2_\text{n}}\right) \nonumber \\
\end{align}
$$

* Log likelihood:

$$
\begin{align}
\ln p(D \mid \mathbf w)
&= \ln \prod_{i=1}^n p(\mathbf{x}_i, y_i \mid\mathbf{w}) + \text{const} \\
&= -\frac{1}{2\sigma^2_\text{n}} (y_i - \mathbf{w}^\top \mathbf{x}_i)^2 + \text{const} 
\end{align}
$$

* MLE $\iff$ least square

$$
\begin{align}
\hat{\mathbf w}_\text{MLE}
&= \argmin_{\mathbf w} \sum_{i=1}^n (y_i - \mathbf{w}^\top \mathbf{x}_i)^2
\end{align}
$$

* Log posterior with Gaussian prior

$$
\begin{align}
\ln p(\mathbf w \mid D)
&= \ln \prod_{i=1}^n p(\mathbf{x}_i, y_i \mid\mathbf{w}) + \ln p(\mathbf{w}) + \text{const} \\
&= -\frac{1}{2\sigma^2_\text{n}} (y_i - \mathbf{w}^\top \mathbf{x}_i)^2 -
    \frac{1}{2\sigma^2_\text{p}} \Vert\boldsymbol{\theta}\Vert^2 +
    \text{const} 
\end{align}
$$

* MAP with Gaussian prior $\iff$ ridge regression

$$
\begin{align}
\hat{\mathbf w}_\text{MAP}
&= \argmin_{\mathbf w} \sum_{i=1}^n (y_i - \mathbf{w}^\top \mathbf{x}_i)^2 +
   \frac{\sigma^2_\text{n}}{\sigma^2_\text{p}} \Vert\boldsymbol{\theta}\Vert^2
\end{align}
$$

## Bayesian Inference for Linear Regression

Both MLE and MAP uses training dataset $D$ to compute a point estimate $\hat{\mathbf{w}}$, leading to plug-in predictive $p(y_* \mid \mathbf{x}_*, \hat{\mathbf{w}})$.

Bayesian inference uses the full posterior $p(\mathbf w \mid D)$ to highlight the importance of every  $\mathbf w\in\mathbb R^d$, leading to posterior predictive $p(y_* \mid \mathbf x_*, D)$.

$$
\begin{align}
p(y_* \mid \mathbf x_*, D)
&= \int p(y_* \mid \mathbf x_*, \mathbf w) \cdot p(\mathbf w \mid D) \: \mathrm d \mathbf w \\
&= \mathbb E_{\mathbf w \sim p(\mathbf w \mid D)} \big[ p(y_* \mid \mathbf x_*, \mathbf w) \big]
\end{align}
$$

Remarks:

* The integral on the RHS averages the conditional density $p(y_* \mid \mathbf x_*, \mathbf w)$ w.r.t. the posterior $p(\mathbf w \mid D)$. In general, this integral is intractable. However, if everthing is Gaussian, we indeed have closed-form solution.
* The plug-in predictive can be seen as a special case where we place all proability mass on a single $\hat{\mathbf{w}}$.
    $$
    \begin{align}
    p(y_* \mid \mathbf x_*, \hat{\mathbf{w}})
    &= \int p(y_* \mid \mathbf x_*, \mathbf w) \cdot \delta(\mathbf w - \hat{\mathbf{w}}) \: \mathrm d \mathbf w \\
    \end{align}
    $$

In the following, we assume a spherical Gaussian prior on $\mathbf{w}$, i.e.

$$
\begin{align}
\mathbf{w} \sim \mathcal N(\mathbf{0}, \sigma^2_\text{p}\mathbf{I})
\end{align}
$$

We will show later that the posterior is Gaussian

$$
\begin{align}
\mathbf{w} \mid D \sim \mathcal N(\boldsymbol{\mu}, \boldsymbol{\Sigma})
\end{align}
$$

where the posterior mean $\boldsymbol{\mu} \triangleq \mathbb E[\mathbf{w} \mid D]$ and posterior covariance matrix $\boldsymbol{\Sigma} \triangleq \mathbb V[\mathbf{w} \mid D]$ will be calculated later. The key idea is the sufficiency of computing the mean and variance rather than evaluating any integral if everything is Gaussian.

### The Full Posterior

By Bayesian rule,

$$
\begin{align}
p(\mathbf{w} \mid D)
&= \frac{p(D \mid \mathbf{w}) \cdot p(\mathbf{w})}{p(D)} \\
&\propto p(D \mid \mathbf{w}) \cdot p(\mathbf{w}) \\
&= \left( \prod_{i=1}^n p(\mathbf{x}_i, y_i \mid \mathbf{w}) \right) \cdot p(\mathbf{w}) \\
&= \left( \prod_{i=1}^n p(y_i \mid \mathbf{x}_i, \mathbf{w}) \, p(\mathbf{x}_i) \right) \cdot p(\mathbf{w}) \\
&\propto \left( \prod_{i=1}^n p(y_i \mid \mathbf{x}_i, \mathbf{w}) \right) \cdot p(\mathbf{w})
\end{align}
$$

For the sake of derivation, we will show that the log posterior is quadratic form of $\mathbf{w}$, which proves that the posterior is Gaussian. Recall that

$$
\begin{align*}
y_i \mid \mathbf{x}_i, \mathbf{w} \sim \mathcal N(\mathbf{w}^\top \mathbf{x}_i, \sigma^2_\text{n})
&\implies
\ln p(y_i \mid \mathbf{x}_i, \mathbf{w}) = -\frac{1}{2\sigma^2_\text{n}} (y_i - \mathbf{w}^\top \mathbf{x}_i)^2 + \text{const.}
\\
\mathbf{w} \sim \mathcal N(\mathbf{0}, \sigma^2_\text{p}\mathbf{I})
&\implies
\ln p(\mathbf{w}) = -\frac{1}{2\sigma^2_\text{p}} \Vert\mathbf{w}\Vert^2 + \text{const.}
\end{align*}
$$

Hence, the log posterior is

$$
\begin{align}
\ln p(\mathbf{w} \mid D)
&= \sum_{i=1}^n \ln p(y_i \mid \mathbf{x}_i, \mathbf{w}) + \ln p(\mathbf{w}) + \text{const.}
\\
&= -\frac{1}{2\sigma^2_\text{n}} \sum_{i=1}^n (y_i - \mathbf{w}^\top \mathbf{x}_i)^2
   -\frac{1}{2\sigma^2_\text{p}} \Vert\mathbf{w}\Vert^2 + \text{const.}
\\
&= -\frac{1}{2\sigma^2_\text{n}} \Vert\mathbf{y} - \mathbf{X}\mathbf{w}\Vert^2
   -\frac{1}{2\sigma^2_\text{p}} \Vert\mathbf{w}\Vert^2 + \text{const.}
\\
&= -\frac{1}{2} \left[
      \sigma^{-2}_\text{n}(\mathbf{y} - \mathbf{X}\mathbf{w})^\top (\mathbf{y} - \mathbf{X}\mathbf{w})
      +\sigma^{-2}_\text{p} \mathbf{w}^\top \mathbf{w}
    \right] + \text{const.}
\\
&= -\frac{1}{2} \left[
      \sigma^{-2}_\text{n} \left( \mathbf{w}^\top \mathbf{X}^\top \mathbf{X} \mathbf{w} - 2 \mathbf{y}^\top \mathbf{X} \mathbf{w} \right)
      +\sigma^{-2}_\text{p} \mathbf{w}^\top \mathbf{w}
    \right] + \text{const.}
\\
&= -\frac{1}{2} \left[
       \mathbf{w}^\top \left( \sigma^{-2}_\text{n} \mathbf{X}^\top \mathbf{X} + \sigma^{-2}_\text{p}\mathbf{I} \right) \mathbf{w} - 2 \sigma^{-2}_\text{n} \mathbf{y}^\top \mathbf{X} \mathbf{w}
    \right] + \text{const.}
\\
\end{align}
$$

where

$$
\begin{align}
\mathbf{X} =
\begin{bmatrix}
\mathbf{x}_1^\top \\ \vdots \\ \mathbf{x}_n^\top
\end{bmatrix}
\in\mathbb R^{n \times d}
\quad
\mathbf{y} =
\begin{bmatrix}
y_1 \\ \vdots \\ y_n
\end{bmatrix}
\in\mathbb R^{n}
\end{align}
$$

By the property of multivariate Gaussian, we conclude that the posterior is also Gaussian

$$
\begin{align}
\mathbf{w} \mid D \sim \mathcal N(\boldsymbol{\mu}, \boldsymbol{\Sigma})
\end{align}
$$

with posterior mean and posterior variance

$$
\begin{align}
\boldsymbol{\Sigma}
&= \left( \sigma^{-2}_\text{n} \mathbf{X}^\top \mathbf{X} + \sigma^{-2}_\text{p} \mathbf{I} \right)^{-1} \\
\boldsymbol{\mu}
&= \sigma^{-2}_\text{n} \boldsymbol{\Sigma} \mathbf{X}^\top \mathbf{y}
\end{align}
$$

Remarks:

* We avoid using $\boldsymbol{\mu}_{\mathbf{w} \mid D}, \boldsymbol{\Sigma}_{\mathbf{w} \mid D}$ for the posterior mean and covariance matrix for the sake of clean notation.
* If we have a strong prior (small $\sigma_\text{p}$), then $\boldsymbol{\Sigma}\approx\sigma^{2}_\text{p} \mathbf{I}$ and $\boldsymbol{\mu}\approx\mathbf{0}$, which drives the weights $\mathbf{w}$ close to zero.
* If the label noise is low (small $\sigma_\text{n}$), then $\boldsymbol{\Sigma}\approx\sigma^{2}_\text{n} \left( \mathbf{X}^\top \mathbf{X} \right)^{-1}$ and $\boldsymbol{\mu}\approx\left( \mathbf{X}^\top \mathbf{X} \right)^{-1} \mathbf{X}^\top \mathbf{y}$ which drives the weights $\mathbf{w}$ close to the least square estimate.
* Compared to MAP estimate (which only computes the mode $\boldsymbol{\mu}$), Bayesian inference also computes $\boldsymbol{\Sigma}$, which quantifies the uncertainty of the weights.

### The Posterior Predictive

Having derived the posterior $p(\mathbf{w} \mid D)$, we are ready to derive the posterior predictive $p(y_* \mid \mathbf{x}_*, D)$. Recall the probabilitstic model

$$
\begin{align}
y_* = \mathbf{w}^\top \mathbf{x}_* + \varepsilon, \quad
\varepsilon \sim \mathcal N(\mathrm{0}, \sigma^2_\text{n})
\end{align}
$$

Given training data $D$ and new input $\mathbf{x}_*$, the label $y_*$ is Gaussian because it is a sum of two indepdent Gaussian random variables $\mathbf{w}^\top \mathbf{x}_*$ and $\varepsilon$. Hence, it is sufficient to compute the mean and variance of $y_*$ (given $\mathbf{x}_*$ and $D$) in order to obtain the posterior predictive.

$$
\begin{align*}
\mathbb E[y_* \mid \mathbf{x}_*, D]
&= \mathbb E[\mathbf{w}^\top \mathbf{x}_* + \varepsilon \mid \mathbf{x}_*, D] \\
&= \mathbb E[\mathbf{w}^\top \mathbf{x}_* \mid \mathbf{x}_*, D] + \mathbb E[\varepsilon \mid \mathbf{x}_*, D] \\
&= \mathbb E[\mathbf{w} \mid D]^\top \mathbf{x}_* + \mathbb E[\varepsilon] \\
&= \boldsymbol{\mu}^\top \mathbf{x}_*
\end{align*}
$$

$$
\begin{align*}
\mathbb V[y_* \mid \mathbf{x}_*, D]
&= \mathbb V[\mathbf{w}^\top \mathbf{x}_* + \varepsilon \mid \mathbf{x}_*, D] \\
&= \mathbb V[\mathbf{w}^\top \mathbf{x}_* \mid \mathbf{x}_*, D] + \mathbb V[\varepsilon \mid \mathbf{x}_*, D] \\
&= \mathbf{x}_*^\top \mathbb V[\mathbf{w} \mid D] \mathbf{x}_* + \mathbb V[\varepsilon] \\
&= \mathbf{x}_*^\top \boldsymbol{\Sigma} \mathbf{x}_* + \sigma^2_\text{n}
\end{align*}
$$

Remarks:

* The uncertainty in $y_*$ consists of two parts.
  * ***Predictive variance*** $\mathbf{x}_*^\top \boldsymbol{\Sigma} \mathbf{x}_*$, which arises due to lack of data.
  * ***Irreducible noise*** $\sigma^2_\text{n}$.
* The predictive variance $\mathbf{x}_*^\top \boldsymbol{\Sigma} \mathbf{x}_*$ tends to be (proof omitted)
  * low when $\mathbf{x}_*$ lies near the subspace spanned by the training data.
  * large when $\mathbf{x}_*$ is far from or orthogonal to the training data.

Therefore, the posterior predicitve is

$$
\begin{align}
y_* \mid \mathbf{x}_*, D \sim \mathcal N(
    \boldsymbol{\mu}^\top \mathbf{x}_*, \:
    \mathbf{x}_*^\top \boldsymbol{\Sigma} \mathbf{x}_* + \sigma^2_\text{n}
)
\end{align}
$$

with

$$
\begin{align}
\boldsymbol{\Sigma}
&= \left( \sigma^{-2}_\text{n} \mathbf{X}^\top \mathbf{X} + \sigma^{-2}_\text{p} \mathbf{I} \right)^{-1} \\
\boldsymbol{\mu}
&= \sigma^{-2}_\text{n} \boldsymbol{\Sigma} \mathbf{X}^\top \mathbf{y}
\end{align}
$$

## Appendix

### Conditional Probability

Let $X,Y,Z$ be random variables. Then,

$$
\begin{align}
p(z \mid x,y) = \frac{p(z \mid x) \cdot p(y \mid z,x)}{p(y \mid x)}
\end{align}
$$

If $X$ and $Y$ are independent, then

$$
\begin{align}
p(z \mid x,y) = \frac{p(z \mid x) \cdot p(y \mid z,x)}{p(y)}
\end{align}
$$

If $X$ and $Y$ are conditionally independent given $Z$ , then

$$
\begin{align}
p(z \mid x,y) = \frac{p(z \mid x) \cdot p(y \mid z)}{p(y \mid x)}
\end{align}
$$

If $X$ and $Y$ are both independent and conditionally independent given $Z$ , then

$$
\begin{align}
p(z \mid x,y) = \frac{p(z \mid x) \cdot p(y \mid z)}{p(y)}
\end{align}
$$

### Multivariant Gaussian

Let $X: \Omega \to \mathbb R^{n+m}$ be a Gaussian random vector.

$$
X \sim \mathcal{N}\left( \boldsymbol\mu, \boldsymbol\Sigma \right)
$$

where

$$
\boldsymbol\mu\in\mathbb{R}^{n+m}, \quad \boldsymbol\Sigma\in\mathbb{R}^{(n+m)\times(n+m)}
$$

We partition $X$ into two subvectors $X_A$ and $X_B$

* $X_A = [X_1, \dots, X_n]^\top$
* $X_B = [X_{n+1}, \dots, X_{n+m}]^\top$

The overall mean and covariance can be partitioned into

$$
\begin{bmatrix}  X_A \\  X_B \end{bmatrix}
\sim\mathcal{N}
\left(
  \begin{bmatrix} \boldsymbol\mu_A \\ \boldsymbol\mu_B \end{bmatrix},
  \begin{bmatrix}
    \boldsymbol\Sigma_{AA} & \boldsymbol\Sigma_{AB} \\
    \boldsymbol\Sigma_{BA} & \boldsymbol\Sigma_{BB}
  \end{bmatrix}
\right)
$$

where

$$
\begin{matrix}
  \boldsymbol\mu_A\in\mathbb{R}^n \\  \boldsymbol\mu_B\in\mathbb{R}^m
\end{matrix}
\quad\text{ and }\quad
\begin{matrix}
  \boldsymbol\Sigma_{AA}\in\mathbb{R}^{n \times n} & \boldsymbol\Sigma_{AB}\in\mathbb{R}^{n \times m} \\
  \boldsymbol\Sigma_{BA}\in\mathbb{R}^{m \times n} & \boldsymbol\Sigma_{BB}\in\mathbb{R}^{m \times m}
\end{matrix}
$$

#### Marginalized Gaussian

One can show that the marginal distributions

$$
\begin{align}
p(\mathbf x_A) = \int p(\mathbf x_A, \mathbf x_B ; \boldsymbol\mu, \boldsymbol\Sigma) \:\mathrm{d}\mathbf x_B \\
p(\mathbf x_B) = \int p(\mathbf x_A, \mathbf x_B ; \boldsymbol\mu, \boldsymbol\Sigma) \:\mathrm{d}\mathbf x_A \\
\end{align}
$$

are also Gaussian

$$
\begin{align}
X_A &\sim \mathcal{N}\left( \boldsymbol\mu_A, \boldsymbol\Sigma_{AA}\right) \\
X_B &\sim \mathcal{N}\left( \boldsymbol\mu_B, \boldsymbol\Sigma_{BB}\right)
\end{align}
$$

#### Gaussian Conditioning

Conditioned on $X_B = \mathbf x_B$, the distribution of $X_A$
$$
\begin{align}
p(\mathbf x_A \mid \mathbf x_B) 
&= \frac{p(\mathbf x_A, \mathbf x_B ; \boldsymbol\mu, \boldsymbol\Sigma)}{p(\mathbf x_B ; \boldsymbol\mu, \boldsymbol\Sigma)}  \\
&= \frac{p(\mathbf x_A, \mathbf x_B ; \boldsymbol\mu, \boldsymbol\Sigma)}{\int p(\mathbf x_A, \mathbf x_B ; \boldsymbol\mu, \boldsymbol\Sigma) \:\mathrm{d}\mathbf x_A}
\end{align}
$$

is also Gaussian
$$
\begin{align}
X_A \,\mid\, \mathbf x_B &\sim \mathcal{N}(\boldsymbol\mu_{A \mid B}, \boldsymbol\Sigma_{A \mid B}) \\
\boldsymbol\mu_{A \mid B} &= \boldsymbol\mu_{A} + \boldsymbol\Sigma_{AB} \boldsymbol\Sigma_{BB}^{-1} (\mathbf x_B - \boldsymbol\mu_B)\\
\boldsymbol\Sigma_{A \mid B} &= \boldsymbol\Sigma_{AA} - \boldsymbol\Sigma_{AB}\boldsymbol\Sigma_{BB}^{-1} \boldsymbol\Sigma_{BA}
\end{align}
$$

#### Convolution Rule

$$
\begin{align*}
\int \mathcal N(\mathbf x_* ; \boldsymbol\mu, \boldsymbol\Sigma) \cdot
     \mathcal N(\boldsymbol\mu ; \boldsymbol\mu_n, \boldsymbol\Sigma_n) \:\mathrm{d}\boldsymbol\mu
=
\mathcal N(\mathbf x_* ; \boldsymbol\mu_n, \boldsymbol\Sigma + \boldsymbol\Sigma_n)
\end{align*}
$$