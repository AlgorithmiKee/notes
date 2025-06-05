---
title: "Bayesian Linear Regression"
author: "Ke Zhang"
date: "2025"
fontsize: 12pt
---

# Bayesian Linear Regression

[toc]

$$
\DeclareMathOperator*{\argmax}{argmax}
\DeclareMathOperator*{\argmin}{argmin}
$$

**Preliminary**: linear regression, Bayesian inference.

## Problem Formulation

* Given: training dataset $D=\{(\mathbf{x}_i, y_i)\}_{i=1}^n \stackrel{\text{iid}}{\sim} p(\mathbf x, y)$ where $(\mathbf{x}_i, y_i) \in \mathbb R^d \times \mathbb R$.
* Statistical model: $y_i = \mathbf{w}^\top \mathbf{x}_i + \varepsilon_i, \quad \varepsilon_i \stackrel{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2)$
* Additional assumption: $\varepsilon_i$ and $\mathbf{x}_i$ are statistically independent
* Goal: Predict the label for a new data point $\mathbf{x}_*$ using the **full** posterior distribution $p(\mathbf w \mid D)$

## Recap: Point Estimates for Model Parameters

* Discriminative model parameterized by $\mathbf{w}$:

$$
\begin{align}
p(y_i \mid \mathbf{x}_i, \mathbf{w})
&= \mathcal{N}(y_i \mid \mathbf{w}^\top \mathbf{x}_i, \sigma^2) \\
&= \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(-\frac{(y_i - \mathbf{w}^\top \mathbf{x}_i)^2}{2\sigma^2}\right) \nonumber \\
\end{align}
$$

* Log likelihood:

$$
\begin{align}
\ln p(D \mid \mathbf w)
&= \ln \prod_{i=1}^n p(\mathbf{x}_i, y_i \mid\mathbf{w}) + \text{const} \\
&= -\frac{1}{2\sigma^2} (y_i - \mathbf{w}^\top \mathbf{x}_i)^2 + \text{const} 
\end{align}
$$

* MLE $\iff$ least square

$$
\begin{align}
\hat{\mathbf w}_\text{MLE}
&= \argmin_{\mathbf w} \sum_{i=1}^n (y_i - \mathbf{w}^\top \mathbf{x}_i)^2
\end{align}
$$

* Log posterior with Gaussian prior $\mathbf{w} \sim \mathcal N(\mathbf{0}, \sigma^2_\text{p}\mathbf{I})$

$$
\begin{align}
\ln p(\mathbf w \mid D)
&= \ln \prod_{i=1}^n p(\mathbf{x}_i, y_i \mid\mathbf{w}) + \ln p(\mathbf{w}) + \text{const} \\
&= -\frac{1}{2\sigma^2} (y_i - \mathbf{w}^\top \mathbf{x}_i)^2 -
    \frac{1}{2\sigma^2_\text{p}} \Vert\boldsymbol{\theta}\Vert^2 +
    \text{const} 
\end{align}
$$

* MAP with Gaussian prior $\iff$ ridge regression

$$
\begin{align}
\hat{\mathbf w}_\text{MAP}
&= \argmin_{\mathbf w} \sum_{i=1}^n (y_i - \mathbf{w}^\top \mathbf{x}_i)^2 +
   \frac{\sigma^2}{\sigma^2_\text{p}} \Vert\boldsymbol{\theta}\Vert^2
\end{align}
$$

## Offline Bayesian Linear Regression

Both MLE and MAP use training dataset $D$ to compute a point estimate $\hat{\mathbf{w}}$, leading to plug-in predictive $p(y_* \mid \mathbf{x}_*, \hat{\mathbf{w}})$.

Bayesian inference uses the full posterior $p(\mathbf w \mid D)$ to account for the contribution of every  $\mathbf w\in\mathbb R^d$, leading to posterior predictive $p(y_* \mid \mathbf x_*, D)$.

$$
\begin{align}
p(y_* \mid \mathbf x_*, D)
&= \int p(y_* \mid \mathbf x_*, \mathbf w) \cdot p(\mathbf w \mid D) \: \mathrm d \mathbf w \\
&= \mathbb E_{\mathbf w \sim p(\mathbf w \mid D)} \big[ p(y_* \mid \mathbf x_*, \mathbf w) \big]
\end{align}
$$

Remarks:

* The integral on the RHS averages the conditional density $p(y_* \mid \mathbf x_*, \mathbf w)$ w.r.t. the posterior $p(\mathbf w \mid D)$. In general, this integral is intractable. However, if everything is Gaussian, we indeed have closed-form solution.
* The plug-in predictive can be seen as a special case where we place all probability mass on a single $\hat{\mathbf{w}}$.
    $$
    \begin{align}
    p(y_* \mid \mathbf x_*, \hat{\mathbf{w}})
    &= \int p(y_* \mid \mathbf x_*, \mathbf w) \cdot \delta(\mathbf w - \hat{\mathbf{w}}) \: \mathrm d \mathbf w \\
    \end{align}
    $$

In the following, we assume a Gaussian prior on $\mathbf{w}$, i.e.

$$
\begin{align}
p(\mathbf{w}) = \mathcal N(\mathbf{w} \mid \mathbf{w}_0, \mathbf{P}_0)
\end{align}
$$

Remarks:

* $\mathbf{w}_0$ represents our initial estimate of weight vector $\mathbf{w}$.
* $\mathbf{P}_0$ quantifies the uncertainty about $\mathbf{w}$ before we observe any data.
* In standard MAP estimation, the prior is chosen s.t. $\mathbf{w}_0 = \mathbf{0}$ and $\mathbf{P}_0 = \sigma^2_\text{p}\mathbf{I}$. Here, we use a slightly more general Gaussian prior.

We will show later that the posterior is Gaussian

$$
\begin{align}
p(\mathbf{w} \mid D) = \mathcal N(\mathbf{w} \mid \mathbf{w}_n, \mathbf{P}_n)
\end{align}
$$

Remarks:

* We use the subscript $n$ to hightlight the size of our data set $D$.
* $\mathbf{w}_n$ is the MAP estimate of weight vector $\mathbf{w}$ after we observe $D$.
* $\mathbf{P}_n$ quantifies the uncertainty about $\mathbf{w}$ after we observe $D$.

Recall the Bayesian rule

$$
\begin{align}
p(\mathbf{w} \mid D)
&= \frac{p(D \mid \mathbf{w}) \cdot p(\mathbf{w})}{p(D)}
= \frac{p(D \mid \mathbf{w}) \cdot p(\mathbf{w})}{\int p(D \mid \mathbf{w}) \cdot p(\mathbf{w}) \:\mathrm d \mathbf{w}}
\end{align}
$$

Due to Gaussian-ness, we only need to compute $\mathbf{w}_n$ and $\mathbf{P}_n$ to obtain the full posterior rather than evaluating the high-dimensional integral in the denominator.

### The Full Posterior

We now show that the posterior is Guassian by showing that the log posterior is a quadratic function of $\mathbf{w}$.

Recall the statistical model

$$
\begin{align}
y_i &= \mathbf{w}^\top \mathbf{x}_i + \varepsilon_i, \quad
\varepsilon_i \sim \mathcal N(0, \sigma^2)
\\
\mathbf{y} &= \mathbf{X}^\top \mathbf{w} + \boldsymbol{\varepsilon}, \quad
\boldsymbol{\varepsilon} \sim \mathcal N(\mathbf{0}, \sigma^2\mathbf{I})
\end{align}
$$

where

$$
\begin{align}
\mathbf{y} =
\begin{bmatrix}
  y_1 \\ \vdots \\ y_n
\end{bmatrix}
\in\mathbb R^{n},
%%%%%%%%%%%%
\quad
%%%%%%%%%%%%
\mathbf{X} =
\begin{bmatrix}
  \mathbf{x}_1^\top \\ \vdots \\ \mathbf{x}_n^\top
\end{bmatrix}
\in\mathbb R^{n \times d},
%%%%%%%%%%%%
\quad
%%%%%%%%%%%%
\boldsymbol{\varepsilon} =
\begin{bmatrix}
  \varepsilon_1 \\ \vdots \\ \varepsilon_n
\end{bmatrix}
\in\mathbb R^{n}
\end{align}
$$

The likelihood is thus

$$
\begin{align}
p(D \mid \mathbf{w}) 
&= p(\mathbf{x}_{1:n}, y_{1:n} \mid \mathbf{w}) \\
&= p(y_{1:n} \mid \mathbf{x}_{1:n}, \mathbf{w}) \cdot p(\mathbf{x}_{1:n}) \\
&\propto p(y_{1:n} \mid \mathbf{x}_{1:n}, \mathbf{w}) \\
&= \mathcal N(\mathbf{y} \mid \mathbf{X}^\top \mathbf{w}, \sigma^2\mathbf{I})
\end{align}
$$

The posterior is

$$
\begin{align}
p(\mathbf{w} \mid D)
&\propto p(D \mid \mathbf{w}) \cdot p(\mathbf{w}) \\
&\propto \mathcal N(\mathbf{y} \mid \mathbf{X}^\top \mathbf{w}, \sigma^2\mathbf{I}) \cdot \mathcal N(\mathbf{w} \mid \mathbf{w}_0, \mathbf{P}_0) \\
\end{align}
$$

Taking the log, we obtain a quadratic log posterior

$$
\begin{align}
\ln p(\mathbf{w} \mid D)
&= -\frac{1}{2\sigma^2} \big\Vert\mathbf{y} - \mathbf{X}\mathbf{w} \big\Vert^2
   -\frac{1}{2} (\mathbf{w} - \mathbf{w}_0)^\top \mathbf{P}_0^{-1} (\mathbf{w} - \mathbf{w}_0) + \text{const.}
\\
&= -\frac{1}{2} \left[
      \sigma^{-2}(\mathbf{y} - \mathbf{X}\mathbf{w})^\top (\mathbf{y} - \mathbf{X}\mathbf{w})
      +(\mathbf{w} - \mathbf{w}_0)^\top \mathbf{P}_0^{-1} (\mathbf{w} - \mathbf{w}_0)
    \right] + \text{const.}
\\
&= -\frac{1}{2} \left[
      \sigma^{-2} \left( \mathbf{w}^\top \mathbf{X}^\top \mathbf{X} \mathbf{w} - 2 \mathbf{y}^\top \mathbf{X} \mathbf{w} \right)
      + \mathbf{w}^\top \mathbf{P}_0^{-1} \mathbf{w} - 2 \mathbf{w}_0^\top \mathbf{P}_0^{-1} \mathbf{w}
    \right] + \text{const.}
\\
&= -\frac{1}{2} \left[
       \mathbf{w}^\top \left( \sigma^{-2} \mathbf{X}^\top \mathbf{X} + \mathbf{P}_0^{-1} \right) \mathbf{w} -
       2 \left( \sigma^{-2} \mathbf{y}^\top \mathbf{X} + \mathbf{w}_0^\top \mathbf{P}_0^{-1} \right) \mathbf{w}
    \right] + \text{const.}
\\
\end{align}
$$

Hence, the posterior is also Gaussian (by normalization property of multivariate Gaussian )

> $$
> \begin{align}
> p(\mathbf{w} \mid D) = \mathcal N(\mathbf{w} \mid \mathbf{w}_n, \mathbf{P}_n)
> \end{align}
> $$

The posterior mean and posterior variance can be read-off in the log posterior:

> $$
> \begin{align}
> \mathbf{P}_n^{-1}
> &= \sigma^{-2} \mathbf{X}^\top \mathbf{X} + \mathbf{P}_0^{-1}
> \\[6pt]
> \mathbf{w}_n
> &= \mathbf{P}_n \left( \sigma^{-2} \mathbf{X}^\top \mathbf{y} + \mathbf{P}_0^{-1} \mathbf{w}_0 \right)
> \end{align}
> $$

The posterior mean $\mathbf{w}_n$ can be refumulated as a sum of the prior mean $\mathbf{w}_0$ and a correction term.

> $$
> \begin{align}
> \mathbf{w}_n
> &= \mathbf{w}_0 + \sigma^{-2} \mathbf{P}_n \mathbf{X}^\top \left( \mathbf{y} - \mathbf{X} \mathbf{w}_0 \right)
> \end{align}
> $$

Remarks:

* $\mathbf{X} \mathbf{w}_0$ contains label predictions using the prior mean $\mathbf{w}_0$ on $\mathbf{x}_{1:n}$ before we observe $y_{1:n}$.
* $\mathbf{y} - \mathbf{X} \mathbf{w}_0$ is called ***residual***. It reflects the difference between the observed labels and predicted labels based on prior mean. It is similar to the ***innovation*** in Kalman filter.
* $\sigma^{-2} \mathbf{P}_n \mathbf{X}^\top$ is called ***gain***. It reflects how strong we respond to the residual, similar to ***Kalman gain***.

Let $\mathbf{K}_n \triangleq \sigma^{-2} \mathbf{P}_n \mathbf{X}^\top$, we can express $\mathbf{w}_n$ as a weighted average of $\mathbf{w}_0$ and $\mathbf{y}$.

> $$
> \begin{align}
> \mathbf{w}_n
> &= \left( \mathbf{I} - \mathbf{K}_n\mathbf{X} \right) \mathbf{w}_0 + \mathbf{K}_n \mathbf{y}
> \end{align}
> $$

If the data set contains only one sample $D = \{ \mathbf x_1, y_1 \}$ (i.e. $n=1$), then $\mathbf{w}_n$ and $\mathbf{P}_n$ simplify to

$$
\begin{align*}
\mathbf{P}_1^{-1}
&= \sigma^{-2} \mathbf{x}_1 \mathbf{x}_1^\top + \mathbf{P}_0^{-1}
\\[6pt]
\mathbf{w}_1
&= \mathbf{P}_1 \left( \sigma^{-2} \mathbf{x}_1 y_1 + \mathbf{P}_0^{-1} \mathbf{w}_0 \right)
\\
&= \mathbf{w}_0 + \sigma^{-2} \mathbf{P}_1 \mathbf{x}_1 \left( y_1 - \mathbf{w}_0^\top \mathbf{x}_1 \right)
\end{align*}
$$

In standard MAP estimation, we assume $\mathbf{w}_0 = \mathbf{0}$ and $\mathbf{P}_0 = \sigma^2_\text{p}\mathbf{I}$, i.e.

$$
\begin{align}
p(\mathbf{w}) = \mathcal N(\mathbf{w} \mid \mathbf{0}, \sigma_{0}^{2}\mathbf{I})
\end{align}
$$

The posterior mean and posterior variance simplify to

$$
\begin{align}
\mathbf{P}_n^{-1}
&= \sigma^{-2} \mathbf{X}^\top \mathbf{X} + \sigma_{0}^{-2} \mathbf{I}
\\[6pt]
\mathbf{w}_n
&= \left( \mathbf{X}^\top \mathbf{X} + \lambda\mathbf{I} \right)^{-1} \mathbf{X}^\top \mathbf{y},
\qquad \lambda \triangleq \sigma^{2} / \sigma_{0}^{2}
\end{align}
$$

Remarks:

* $\mathbf{w}_n$ is exactly the MAP estimate aka solution of Ridge regression.
* In Bayesian inference, we also compute $\mathbf{P}_n$, which quantifies the uncertainty of $\mathbf{w}_n$.

### The Posterior Predictive

Having derived the posterior $p(\mathbf{w} \mid D)$, we are ready to derive the posterior predictive $p(y_* \mid \mathbf{x}_*, D)$. Recall the probabilistic model

$$
\begin{align}
y_* = \mathbf{w}^\top \mathbf{x}_* + \varepsilon, \quad
\varepsilon \sim \mathcal N(\mathrm{0}, \sigma^2)
\end{align}
$$

Given training data $D$ and new input $\mathbf{x}_*$, the label $y_*$ is Gaussian because it is a sum of two independent Gaussian random variables $\mathbf{w}^\top \mathbf{x}_*$ and $\varepsilon$. Hence, it is sufficient to compute the mean and variance of $y_*$ (given $\mathbf{x}_*$ and $D$) in order to obtain the posterior predictive.

$$
\begin{align*}
\mathbb E[y_* \mid \mathbf{x}_*, D]
&= \mathbb E[\mathbf{w}^\top \mathbf{x}_* + \varepsilon \mid \mathbf{x}_*, D] \\
&= \mathbb E[\mathbf{w}^\top \mathbf{x}_* \mid \mathbf{x}_*, D] + \mathbb E[\varepsilon \mid \mathbf{x}_*, D] \\
&= \mathbb E[\mathbf{w} \mid D]^\top \mathbf{x}_* + \mathbb E[\varepsilon] \\
&= \mathbf{w}_n^\top \mathbf{x}_*
\end{align*}
$$

$$
\begin{align*}
\mathbb V[y_* \mid \mathbf{x}_*, D]
&= \mathbb V[\mathbf{w}^\top \mathbf{x}_* + \varepsilon \mid \mathbf{x}_*, D] \\
&= \mathbb V[\mathbf{w}^\top \mathbf{x}_* \mid \mathbf{x}_*, D] + \mathbb V[\varepsilon \mid \mathbf{x}_*, D] \\
&= \mathbf{x}_*^\top \mathbb V[\mathbf{w} \mid D] \mathbf{x}_* + \mathbb V[\varepsilon] \\
&= \mathbf{x}_*^\top \mathbf{P}_n \mathbf{x}_* + \sigma^2
\end{align*}
$$

Therefore, the posterior predictive is

> $$
> \begin{align}
> y_* \mid \mathbf{x}_*, D \sim \mathcal N(
>     \mathbf{w}_n^\top \mathbf{x}_*, \:
>     \mathbf{x}_*^\top \mathbf{P}_n \mathbf{x}_* + \sigma^2
> )
> \end{align}
> $$

Remarks:

* The uncertainty in $y_*$ consists of two parts.
  * ***Predictive variance*** $\mathbf{x}_*^\top \mathbf{P}_n \mathbf{x}_*$, which arises due to lack of data.
  * ***Irreducible noise*** $\sigma^2$.
* The predictive variance $\mathbf{x}_*^\top \mathbf{P}_n \mathbf{x}_*$ tends to be (proof omitted)
  * low when $\mathbf{x}_*$ lies near the subspace spanned by the training data.
  * large when $\mathbf{x}_*$ is far from or orthogonal to the training data.

## Online Bayesian Linear Regression

What if the data is collected sequentially instead of all at once? We can apply ***recursive Bayesian inference*** for $\mathbf{w}$ in our linear model. This procedure is called ***online Bayesian linear regression***.

Before we see any data, we put a prior on $\mathbf{w}$ like before:

$$
\begin{align}
p(\mathbf{w}) = \mathcal N(\mathbf{w} \mid \mathbf{w}_0, \mathbf{P}_0)
\end{align}
$$

For each $t=1,2,\dots$, we use $\mathcal N(\mathbf{w} \mid \mathbf{w}_{t-1}, \mathbf{P}_{t-1})$ as the prior. After observing $(\mathbf{x}_t, y_t)$, we update this prior to the posterior. Because all distributions involved are Gaussian, it suffices to update only the posterior mean and covariance:

$$
\begin{align}
\mathbf{P}_{t}
&= \left( \mathbf{P}_{t-1}^{-1} + \sigma^{-2} \mathbf{x}_{t} \mathbf{x}_{t}^\top \right)^{-1}
\\[6pt]
\mathbf{w}_{t}
&= \mathbf{P}_{t} \left(  \mathbf{P}_{t-1}^{-1} \mathbf{w}_{t-1} + \sigma^{-2} \mathbf{x}_{t} y_{t} \right)
\\
&= \mathbf{w}_{t-1} + \sigma^{-2} \mathbf{P}_{t} \mathbf{x}_{t} \left( y_{t} - \mathbf{w}_{t-1}^\top \mathbf{x}_{t} \right)
\end{align}
$$

Remarks:

* Updating $\mathbf{P}_t \in\mathbb R^{d\times d}$ requires matrix inversion at each step, which costs $O(d^3)$. We will address this issue in the next section.
* $y_{t} - \mathbf{w}_{t-1}^\top \mathbf{x}_{t} \in\mathbb R$ is the residual. It is the difference between observed label $y_{t}$ and the prediction $\mathbf{w}_{t-1}^\top \mathbf{x}_{t}$ based on the prior mean.
* The update to $\mathbf{w}_t$ corrects $\mathbf{w}_{t-1}$ in the direction of $\mathbf{P}_{t}\mathbf{x}_{t}$. A larger residual or lower observation noise (i.e. smaller $\sigma$) results in a larger correction.

*Proof*: The update rule follows directly from the offline Bayesian linear regression

$$
\begin{align*}
\mathbf{P}_n
&= \left( \mathbf{P}_0^{-1} + \sigma^{-2} \mathbf{X}^\top \mathbf{X} \right)^{-1}
\\[6pt]
\mathbf{w}_n
&= \mathbf{P}_n \left( \mathbf{P}_0^{-1} \mathbf{w}_0 + \sigma^{-2} \mathbf{X}^\top \mathbf{y} \right)
\\
&= \mathbf{w}_0 + \sigma^{-2} \mathbf{P}_n \mathbf{X}^\top \left( \mathbf{y} - \mathbf{X} \mathbf{w}_0 \right)
\end{align*}
$$

by assuming

$$
\begin{align*}
&\text{prior: }
\cancel{\mathbf{w}_0} \to \mathbf{w}_{t-1}, \quad
\cancel{\mathbf{P}_0} \to \mathbf{P}_{t-1}
\\
&\text{data: }
D=\{\mathbf x_t, y_t\} \implies n=1, \: \mathbf X = \mathbf x_t^\top, \: \mathbf{y} = y_t
\\
&\text{posterior: }
\cancel{\mathbf{w}_n} \to \mathbf{w}_t, \quad
\cancel{\mathbf{P}_n} \to \mathbf{P}_t
\tag*{$\blacksquare$}
\end{align*}
$$

### Avoiding Matrix Inversion

Can we compute $\mathbf{P}_t \in\mathbb R^{d\times d}$ without matrix inversion which requires $O(d^3)$ at every time step?

Applying the **matrix inversion lemma** (see Appendix) to

$$
\mathbf{P}_{t} = \left( \mathbf{P}_{t-1}^{-1} + \sigma^{-2} \mathbf{x}_{t} \mathbf{x}_{t}^\top \right)^{-1},
$$

we obtain

$$
\begin{align}
\mathbf{P}_{t}
&= \mathbf{P}_{t-1} - \frac{\mathbf{P}_{t-1} \mathbf{x}_{t} \mathbf{x}_{t}^\top \mathbf{P}_{t-1}}{\sigma^2 + \mathbf{x}_{t}^\top \mathbf{P}_{t-1} \mathbf{x}_{t}}
\\
&= \mathbf{P}_{t-1} - \frac{(\mathbf{P}_{t-1} \mathbf{x}_{t}) (\mathbf{P}_{t-1} \mathbf{x}_{t})^\top}{\sigma^2 + \mathbf{x}_{t}^\top \mathbf{P}_{t-1} \mathbf{x}_{t}}
\end{align}
$$

which only requires computing matrix-vector multiplication, vector outerproduct, and scalar division. Hence, the computational complexity is reduced to $O(d^2)$.

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

### Matrix Inversion Lemma

Let $\mathbf{A}\in\mathbb R^{n \times n}$ and $\mathbf{C}\in\mathbb R^{m \times m}$ be invertible. Let $\mathbf{U}\in\mathbb R^{n \times m}$ and $\mathbf{V}\in\mathbb R^{m \times n}$.

$$
\begin{align}
\left( \mathbf{A} + \mathbf{U} \mathbf{C} \mathbf{V} \right)^{-1}
= \mathbf{A}^{-1} - \mathbf{A}^{-1} \mathbf{U} \left( \mathbf{C}^{-1} + \mathbf{V} \mathbf{A}^{-1} \mathbf{U} \right)^{-1} \mathbf{V} \mathbf{A}^{-1}
\end{align}
$$

Special cases:

* Let $m=1$, $\mathbf{C} = 1$, $\mathbf{U} = \mathbf{u} \in\mathbb R^{n \times 1}$ and $\mathbf{V} = \mathbf{v}^\top \in\mathbb R^{1 \times n}$. Then,

  $$
  \begin{align}
  \left( \mathbf{A} + \mathbf{u} \mathbf{v}^\top \right)^{-1}
  = \mathbf{A}^{-1} - \frac{\mathbf{A}^{-1} \mathbf{u} \mathbf{v}^\top \mathbf{A}^{-1}}{1 + \mathbf{v}^\top \mathbf{A}^{-1} \mathbf{u}}
  \end{align}
  $$

  If $\mathbf{A}$ is symmetric in addition, then,

  $$
  \begin{align}
  \left( \mathbf{A} + \mathbf{u} \mathbf{v}^\top \right)^{-1}
  = \mathbf{A}^{-1} - \frac{(\mathbf{A}^{-1} \mathbf{u}) (\mathbf{A}^{-1} \mathbf{u})^\top}{1 + \mathbf{v}^\top \mathbf{A}^{-1} \mathbf{u}}
  \end{align}
  $$

* Let $m=n$, $\mathbf{C} = \mathbf{B}$, $\mathbf{U} = \mathbf{I}$ and $\mathbf{V} = \mathbf{I}$. Then,

  $$
  \begin{align}
  \left( \mathbf{A} + \mathbf{B} \right)^{-1}
  = \mathbf{A}^{-1} - \mathbf{A}^{-1} \left( \mathbf{C}^{-1} + \mathbf{A}^{-1} \right)^{-1} \mathbf{A}^{-1}
  \end{align}
  $$
