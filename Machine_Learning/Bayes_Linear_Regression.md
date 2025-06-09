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

### Recap: Point Estimates

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

Both MLE and MAP use training dataset $D$ to compute a point estimate $\hat{\mathbf{w}}$, leading to plug-in predictive distribution $p(y_* \mid \mathbf{x}_*, \hat{\mathbf{w}})$.

Bayesian inference uses the full posterior $p(\mathbf w \mid D)$ to account for the contribution of every  $\mathbf w\in\mathbb R^d$, leading to posterior predictive distribution $p(y_* \mid \mathbf x_*, D)$.

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

In the following, we assume a Gaussian prior on $\mathbf{w} \in\mathbb R^d$, i.e.

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

* We use the subscript $n$ to highlight the size of our data set $D$.
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

We now show that the posterior is Gaussian by showing that the log posterior is a quadratic function of $\mathbf{w}$.

Recall the statistical model

$$
\begin{align}
y_i &= \mathbf{w}^\top \mathbf{x}_i + \varepsilon_i, \quad
\varepsilon_i \sim \mathcal N(0, \sigma^2), \quad
i=1,\dots,n
\end{align}
$$

In vector form, it becomes

$$
\begin{align}
\mathbf{y} &= \mathbf{X} \mathbf{w} + \boldsymbol{\varepsilon}, \quad
&\boldsymbol{\varepsilon} \sim \mathcal N(\mathbf{0}, \sigma^2\mathbf{I})
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
&= \mathcal N(\mathbf{y} \mid \mathbf{X} \mathbf{w}, \sigma^2\mathbf{I})
\end{align}
$$

The posterior is

$$
\begin{align}
p(\mathbf{w} \mid D)
&\propto p(D \mid \mathbf{w}) \cdot p(\mathbf{w}) \\
&\propto \mathcal N(\mathbf{y} \mid \mathbf{X} \mathbf{w}, \sigma^2\mathbf{I}) \cdot \mathcal N(\mathbf{w} \mid \mathbf{w}_0, \mathbf{P}_0) \\
\end{align}
$$

Taking the log, we see that the log posterior is a quadratic function of $\mathbf{w}$:

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
      \sigma^{-2} \left( \mathbf{w}^\top \mathbf{X}^\top \mathbf{X} \mathbf{w} - 2 \mathbf{w}^\top \mathbf{X}^\top \mathbf{y} \right)
      + \mathbf{w}^\top \mathbf{P}_0^{-1} \mathbf{w} - 2 \mathbf{w}^\top \mathbf{P}_0^{-1} \mathbf{w}_0
    \right] + \text{const.}
\\
&= -\frac{1}{2} \left[
       \mathbf{w}^\top \left( \sigma^{-2} \mathbf{X}^\top \mathbf{X} + \mathbf{P}_0^{-1} \right) \mathbf{w} -
       2 \mathbf{w} \left( \sigma^{-2}\mathbf{X}^\top \mathbf{y} + \mathbf{P}_0^{-1} \mathbf{w}_0 \right)
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
> \mathbf{P}_n
> &= \left( \sigma^{-2} \mathbf{X}^\top \mathbf{X} + \mathbf{P}_0^{-1} \right)^{-1}
> \\[6pt]
> \mathbf{w}_n
> &= \mathbf{P}_n \left( \sigma^{-2} \mathbf{X}^\top \mathbf{y} + \mathbf{P}_0^{-1} \mathbf{w}_0 \right)
> \end{align}
> $$

The posterior mean $\mathbf{w}_n$ can be reformulated as a sum of the prior mean $\mathbf{w}_0$ and a correction term.

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
* Computing $\mathbf{X}^\top \mathbf{X}$ requires $O(nd^2)$. Computing $\mathbf{P}_n$ requires inverting a $d\times d$ matrix and thus $O(d^3)$. Hence, the overall complexity is $O(nd^2 + d^3)$.

Let $\mathbf{K}_n \triangleq \sigma^{-2} \mathbf{P}_n \mathbf{X}^\top \in\mathbb R^{d \times n}$, we can express $\mathbf{w}_n$ as a weighted average of $\mathbf{w}_0$ and $\mathbf{y}$.

> $$
> \begin{align}
> \mathbf{w}_n
> &= \left( \mathbf{I} - \mathbf{K}_n\mathbf{X} \right) \mathbf{w}_0 + \mathbf{K}_n \mathbf{y}
> \end{align}
> $$

If the data set contains only one sample $D = \{ \mathbf x_1, y_1 \}$ (i.e. $n=1$), then $\mathbf{w}_n$ and $\mathbf{P}_n$ simplify to

$$
\begin{align*}
\mathbf{P}_1
&= \left( \sigma^{-2} \mathbf{x}_1 \mathbf{x}_1^\top + \mathbf{P}_0^{-1} \right)^{-1}
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
\mathbf{P}_n
&= \left( \sigma^{-2} \mathbf{X}^\top \mathbf{X} + \sigma_{0}^{-2} \mathbf{I} \right)^{-1}
\\[6pt]
\mathbf{w}_n
&= \left( \mathbf{X}^\top \mathbf{X} + \lambda\mathbf{I} \right)^{-1} \mathbf{X}^\top \mathbf{y},
\qquad \lambda \triangleq \sigma^{2} / \sigma_{0}^{2}
\end{align}
$$

Remarks:

* $\mathbf{w}_n$ is exactly the MAP estimate aka solution of Ridge regression.
* In Bayesian inference, we also compute $\mathbf{P}_n$, which quantifies the uncertainty of $\mathbf{w}_n$.

### The Posterior Predictive Distribution

Having derived the posterior $p(\mathbf{w} \mid D)$, we are ready to derive the posterior predictive $p(y_* \mid \mathbf{x}_*, D)$. Recall the probabilistic model

$$
\begin{align*}
y_* = \mathbf{w}^\top \mathbf{x}_* + \varepsilon, \quad
\varepsilon \sim \mathcal N(\mathrm{0}, \sigma^2)
\end{align*}
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

* The mean $\mathbf{w}_n^\top \mathbf{x}_*$ coincides with the label prediction by plugging the MAP estimate $\hat{\mathbf{w}}_\text{MAP} = \mathbf{w}_n$ into $\mathbf{w}^\top \mathbf{x}_*$.
* The variance $\mathbf{x}_*^\top \mathbf{P}_n \mathbf{x}_* + \sigma^2$ quantifies the uncertainty about $y_*$, which consists of two parts:
  * ***Predictive variance*** $\mathbf{x}_*^\top \mathbf{P}_n \mathbf{x}_*$, which arises due to lack of data.
  * ***Irreducible noise*** $\sigma^2$.

## Online Bayesian Linear Regression

What if the data is collected sequentially instead of all at once? We can apply ***recursive Bayesian inference*** for $\mathbf{w}$ in our linear model. This procedure is called ***online Bayesian linear regression***. For cleaner notation, we define

$$
\begin{align}
D_{1:t} &\triangleq
\begin{cases}
  \{ \mathbf{x}_1, y_1, \dots, \mathbf{x}_t, y_t \}, & t\ge 1
  \\
  \varnothing, & t= 0
\end{cases}
\end{align}
$$

Before we see any data, we put a prior on $\mathbf{w}$ like before:

$$
\begin{align}
p(\mathbf{w}) = \mathcal N(\mathbf{w} \mid \mathbf{w}_0, \mathbf{P}_0)
\end{align}
$$

Remarks:

* Here, we simply write $p(\mathbf{w})$ instead of $p(\mathbf{w} \mid D_{1:0})$ or $p(\mathbf{w} \mid \varnothing)$ to denote the prior before we see any data.
* In practice, we typically choose $\mathbf{w}_0 = \mathbf{0}$ and $\mathbf{P}_0 = \sigma_0^2 \mathbf{I}$ as in standard MAP estimation.

For each $t=1,2,\dots$, we use the old posterior as the current prior:

$$
\begin{align}
p(\mathbf{w} \mid D_{t-1})  = \mathcal N(\mathbf{w} \mid \mathbf{w}_{t-1}, \mathbf{P}_{t-1})
\end{align}
$$

Remarks:

* By $y_t = \mathbf{w}^\top \mathbf{x}_t + \varepsilon_t$, the prior predictive distribution is
  > $$
  > \begin{align}
  > p(y_t \mid \mathbf{x}_t, D_{t-1}) = \mathcal N(y_t \mid \mathbf{w}_{t-1}^\top \mathbf{x}_{t},\: \mathbf{x}_{t}^\top \mathbf{P}_{t-1} \mathbf{x}_{t} + \sigma^2)
  > \end{align}
  > $$
* Namely, we can predict $y_t$ by multiplying the prior mean $\mathbf{w}_{t-1}$ (i.e. old MAP estimate of $\mathbf{w}$) with current input $\mathbf{x}_t$.
* The uncertainty in $y_t$ comes from the uncertainty in $\mathbf{w}$ and the observation noise $\varepsilon_t$.

After observing $y_t$, we update this prior to the posterior:

$$
\begin{align}
p(\mathbf{w} \mid D_{t})  = \mathcal N(\mathbf{w} \mid \mathbf{w}_{t}, \mathbf{P}_{t})
\end{align}
$$

where the posterior mean and covariance are updated as

> $$
> \begin{align}
> \mathbf{P}_{t}
> &= \left( \mathbf{P}_{t-1}^{-1} + \sigma^{-2} \mathbf{x}_{t} \mathbf{x}_{t}^\top \right)^{-1}
> \\[6pt]
> \mathbf{w}_{t}
> &= \mathbf{P}_{t} \left(  \mathbf{P}_{t-1}^{-1} \mathbf{w}_{t-1} + \sigma^{-2} \mathbf{x}_{t} y_{t} \right)
> \\
> &= \mathbf{w}_{t-1} + \sigma^{-2} \mathbf{P}_{t} \mathbf{x}_{t} \left( y_{t} - \mathbf{w}_{t-1}^\top \mathbf{x}_{t} \right)
> \end{align}
> $$

Remarks:

* Updating $\mathbf{P}_t \in\mathbb R^{d\times d}$ requires matrix inversion at each step, which costs $O(d^3)$. We will address this issue in the next section.
* $y_{t} - \mathbf{w}_{t-1}^\top \mathbf{x}_{t} \in\mathbb R$ is the residual. It is the difference between observed label $y_{t}$ and the prediction $\mathbf{w}_{t-1}^\top \mathbf{x}_{t}$ based on the prior mean.
* The update to $\mathbf{w}_t$ corrects $\mathbf{w}_{t-1}$ in the direction of the gain $\mathbf{k}_t = \sigma^{-2} \mathbf{P}_{t} \mathbf{x}_{t} \in\mathbb R^{d}$. A larger residual or lower observation noise (i.e. smaller $\sigma$) results in a larger correction.

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

Remarks:

* The 2nd equality follows from the symmetry of the covariance matrix $\mathbf{P}_{t-1}$.
* This update rule only requires computing matrix-vector multiplication, outer product, and scalar division. Hence, the computational complexity is reduced to $O(d^2)$.

### Gain Vector

Recall the gain vector is defined as

$$
\begin{align}
\mathbf{k}_t = \sigma^{-2} \mathbf{P}_{t} \mathbf{x}_{t} \in\mathbb R^{d}
\end{align}
$$

Using the inversion-free update rule for $\mathbf{P}_t$, we can express the gain as

> $$
> \begin{align}
> \mathbf{k}_t = \frac{\mathbf{P}_{t-1} \mathbf{x}_{t}}{\sigma^{2} + \mathbf{x}_{t}^\top \mathbf{P}_{t-1} \mathbf{x}_{t}}
> \end{align}
> $$

Remarks:

* The denominator $\sigma^2 + \mathbf{x}_{t}^\top \mathbf{P}_{t-1} \mathbf{x}_{t}$ is the variance of the predictive distribution $p(y_t \mid \mathbf{x}_t, D_{t-1})$ before we observe $y_t$.
* Comparing both expressions of $\mathbf{k}_t$, we see that the correction direction $\mathbf{P}_{t} \mathbf{x}_{t}$ is the same as $\mathbf{P}_{t-1} \mathbf{x}_{t}$.

*Proof*: The 2nd expression follows from some simple algebra:
$$
\begin{align*}
\mathbf{k}_t
&= \sigma^{-2} \left( \mathbf{P}_{t-1} - \frac{\mathbf{P}_{t-1} \mathbf{x}_{t} \, \mathbf{x}_{t}^\top \mathbf{P}_{t-1}}{\sigma^2 + \mathbf{x}_{t}^\top \mathbf{P}_{t-1} \mathbf{x}_{t}} \right) \mathbf{x}_{t}
\\
&= \sigma^{-2} \left( \mathbf{P}_{t-1} \mathbf{x}_{t} - \frac{\mathbf{P}_{t-1} \mathbf{x}_{t} \cdot \mathbf{x}_{t}^\top \mathbf{P}_{t-1} \mathbf{x}_{t}}{\sigma^2 + \mathbf{x}_{t}^\top \mathbf{P}_{t-1} \mathbf{x}_{t}} \right)
\\
&= \sigma^{-2} \left( 1 - \frac{\mathbf{x}_{t}^\top \mathbf{P}_{t-1} \mathbf{x}_{t}}{\sigma^2 + \mathbf{x}_{t}^\top \mathbf{P}_{t-1} \mathbf{x}_{t}} \right) \mathbf{P}_{t-1} \mathbf{x}_{t}
\\
&= \frac{1}{\sigma^{2} + \mathbf{x}_{t}^\top \mathbf{P}_{t-1} \mathbf{x}_{t}} \mathbf{P}_{t-1} \mathbf{x}_{t}
\tag*{$\blacksquare$}
\end{align*}
$$

Note that the update to $\mathbf{P}_{t}$ requires computing the outer product of $\mathbf{P}_{t-1} \mathbf{x}_{t}$, which is just a scalar mulitple of the gain $\mathbf{k}_t$. Hence, we can express $\mathbf{P}_{t}$ explicitly in terms of $\mathbf{k}_t$.

> $$
> \begin{align}
> \mathbf{P}_{t}
> &= \mathbf{P}_{t-1} - \left( \sigma^2 + \mathbf{x}_{t}^\top \mathbf{P}_{t-1} \mathbf{x}_{t} \right) \mathbf{k}_t \mathbf{k}_t^\top
> \end{align}
> $$

* The correction term is positive semidefinite becuase it is the outer product $ \mathbf{k}_t \mathbf{k}_t^\top$ scaled by the prior predictive variance $\sigma^2 + \mathbf{x}_{t}^\top \mathbf{P}_{t-1} \mathbf{x}_{t}$ (a non negative number). See Appendix for the semi positive definitenss of outer product.
* Thus, $\mathbf{P}_{t}$ is computed by substracting a positive semidefinite matrix from $\mathbf{P}_{t-1}$. Intuitively, this means the uncertainty in $\mathbf{w}$ shrinks over time as more data is assimilated.

### The Complete Algorithm

> Init: $\mathbf{w}_0 \in\mathbb R^{d}, \mathbf{P}_0 \in\mathbb R^{d \times d}$  
> For $t=1,2,\dots$, do:  
> $\qquad$ Prior prediction (mean and variance) of $y_t$:
> $$
> \begin{align}
> \hat{y}_t &=  \mathbf{w}_{t-1}^\top \mathbf{x}_{t} \\
> s^2_{t} &= \sigma^2 + \mathbf{x}_{t}^\top \mathbf{P}_{t-1} \mathbf{x}_{t}
> \end{align}
> $$
>
> $\qquad$ Compute the gain vector $\mathbf{k}_t$:  
>
> $$
> \begin{align}
> \mathbf{k}_t = \frac{\mathbf{P}_{t-1} \mathbf{x}_{t}}{s^2_{t}}
> \end{align}
> $$
>
> $\qquad$ Update the posterior mean of $\mathbf{w}$:  
>
> $$
> \begin{align}
> \mathbf{w}_{t}
> &= \mathbf{w}_{t-1} + \mathbf{k}_{t} \left( y_{t} - \hat{y}_t \right)
> \end{align}
> $$
>
> $\qquad$ Update the posterior variance of $\mathbf{w}$:  
>
> $$
> \begin{align}
> \mathbf{P}_{t}
> &= \mathbf{P}_{t-1} - s^2_{t} \, \mathbf{k}_t \mathbf{k}_t^\top
> \end{align}
> $$

Remarks:

* Online Bayesian linear regression requires $O(d^2)$ computation per time step. Over $n$ time steps, the total computational cost is $O(nd^2)$.
* The algorithm strongly parallels the Kalman filter. In fact, online Bayesian linear regression is a special case of Kalman filter, as we detail below.

From Kalman filter pespective, the Bayesian linear regression model can be reformulated into
$$
\begin{align}
\mathbf{w}_t &= \mathbf{w}_{t-1} \\
y_t &= \mathbf{x}_t^\top \mathbf{w}_t + \varepsilon_t, \quad
\varepsilon_t \sim \mathcal N(0, \sigma^2)
\end{align}
$$
Remarks:

* $\mathbf{w}_t$ is treated as the latent state vector, evolving deterministically (identity mapping without process noise).
* $y_t$ is the scalar observation, and $\mathbf{x}_t^\top$ serves as the observation matrix.. The observation noise is modeled by $\varepsilon_t$.

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
  = \mathbf{A}^{-1} - \mathbf{A}^{-1} \left( \mathbf{A}^{-1} + \mathbf{B}^{-1} \right)^{-1} \mathbf{A}^{-1}
  \end{align}
  $$

### Outer Products

The outer product of two vectors $\mathbf u \in\mathbb R^m, \mathbf v \in\mathbb R^n$ is a matrix, defined by $\mathbf u \mathbf v^\top \in\mathbb R^{m \times n}$.

**Fact**: For $\mathbf u = \mathbf v$, the outerproduct $\mathbf u \mathbf u^\top$ is always postive semi definite.

*Proof*: For any $\mathbf x \ne \mathbf 0$, we see that the quadratic form is non negative as
$$
\mathbf x^\top (\mathbf u \mathbf u^\top)\mathbf x
= (\mathbf x^\top \mathbf u) (\mathbf u^\top\mathbf x)
= (\mathbf x^\top \mathbf u)^2
\ge 0
\tag*{$\blacksquare$}
$$
