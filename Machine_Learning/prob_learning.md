---
title: "Probabilistic Learning"
author: "Ke Zhang"
date: "2025"
fontsize: 12pt
---

[toc]

# Probabilistic Learning

This notes give a high level overview of probabilistic learning.

Notations:

* $\mathcal X$: the data space. Common choices: $\mathcal X = \mathbb R^d$ or a finite discrete set.
* $\mathcal P(\mathcal X)$: the set of all distributions on $\mathcal X$.
  * If $\mathcal X = \mathbb R^d$, then $\mathcal P(\mathcal X)$ consists of all PDFs $p: \mathcal X \to \mathbb R_{\ge 0}$
  * If $\mathcal X$ is a finite discrete set, then $\mathcal P(\mathcal X)$ consists of all PMFs $p: \mathcal X \to [0,1]$.
* $\theta \in \mathbb R^p$: parameter vector.  e.g. For univariate Gaussian, $\theta = (\mu, \sigma^2) \in \mathbb R^2$.
* $\mathcal P_\theta(\mathcal X)$: the set of all parameterized distributions on $\mathcal X$.
  * Clearly, $\mathcal P_\theta(\mathcal X) \subseteq \mathcal P(\mathcal X)$
  * e.g. $\mathcal X=\mathbb R$ and $\mathcal P_\theta(\mathbb R)$ could be the set of all univariate Gaussian distributions with $\theta = (\mu, \sigma^2)$. Clearly, an exponential distribution is not in $\mathcal P_\theta(\mathbb R)$.

## Learning a Marginal Distribution

Consider the random variable $X\in\mathcal X$ generated from the unknown ground truth distribution $p^* \in\mathcal P(\mathcal X)$. Probabilistic learning can be understood as finding a model distribution $p_\theta \in \mathcal P_\theta(\mathcal X)$ which is as close to (ideally identical to) $p^*$ as possible.

We say that the model is correctly specified when $p^* \in \mathcal P_\theta(\mathcal X)$, misspecified otherwise. e.g. The ground truth distribution of a DC voltage is Gaussian due to thermal noise. The model distribution $p_\theta$ is also Gaussian, living in the same distribution space as the $p^*$.

The cross entropy loss is commonly used to quantify the loss of using $p_\theta$ while the ground truth is $p^*$:

$$
\begin{align}
H(p^*, p_\theta) = \mathbb E_{x \sim p^*} [-\log p_\theta(x)]
\end{align}
$$

Probabilistic learning can thus be formulated as minimizing the cross entropy loss of $p_\theta$ w.r.t. $p^*$:

$$
\begin{align}
\min_{p_\theta \in \mathcal P_\theta(\mathcal X)} H(p^*, p_\theta)
\end{align}
$$

Equivalent formulations:

$$
\begin{align}
\min_{\theta \in \mathbb R^p} \mathbb E_{x \sim p^*(x)} [-\log p_\theta(x)]
\end{align}
$$

Two concerns:

1. The ground truth distribution $p^*$ is unknown in the first place.
2. Potential model misspecification: $p^*$ might lie outside of $\mathcal P_\theta(\mathcal X)$

The 1st concern can be resolved by Monte Carlo (MC) estimation. $\to$ See section [MC estimation of cross entropy](#MC-estimation-of-cross-entropy)

The 2nd concern is not always an issue unless $p_\theta$ substancially poor (?) TODO: full explanation.

To obtain the optimal parameter, we often need to compute the gradient of cross entropy. Since the ground truth does not depend on $\theta$, we can move the gradient inside the expectation:

$$
\begin{align}
\nabla_\theta H(p^*, p_\theta) = \mathbb E_{x \sim p^*} [-\log p_\theta(x)]
\end{align}
$$

Again, the graident is approximated by MC estimation in practice. $\to$ See section [MC estimation of cross entropy](#MC-estimation-of-cross-entropy)

### Theoretical Lower Bound of Cross Entropy

By properties of cross entropy (see notes *Math Toolbox for ML*), we have

$$
\begin{align}
H(p^*, p_\theta)
&= H(p^*) + D_\text{KL}(p^* \parallel p_\theta) \\
&\ge H(p^*)
\end{align}
$$

Remarks:

* The inequality in the 2nd line becomes equality iff $p_\theta = p^*$ a.e.
* We cannot make the cross entropy loss to lower than $H(p^*)$, which represents the inhert uncertainty in the data generating distribution.
* Minimizing the cross entropy $H(p^*, p_\theta)$ is equivalent to minimizing the KL divergence $D_\text{KL}(p^* \parallel p_\theta)$.

### MC Estimation of Cross Entropy

Both the cross entropy and its gradient can be approximated by

$$
\begin{align}
x_i &\stackrel{\text{iid}}{\sim} p^*, \quad i=1,\dots,N \\
H(p^*, p_\theta) &\approx -\frac{1}{N} \sum_{i=1}^N \log p_\theta(x_i) \\
\nabla_\theta H(p^*, p_\theta) &\approx -\frac{1}{N} \sum_{i=1}^N \nabla_\theta \log p_\theta(x_i)
\end{align}
$$

Remarks:

* Both approxiamtions are unbiased.
* The term $\log p_\theta(x_i)$ is exactly the log likelihood of $\theta$ for observation $x_i$.
* The cross entropy is approximated with the negative log likelihood of $\theta$ for the whole dataset $x_1,\dots,x_N$.

> Minimising the corss entropy is asymptotically equivalent to maximizing the log likelihood.

Hence, the minimizer of the cross entropy is also the maximizer of the log likelihood.

$$
\begin{align}
\sum_{i=1}^N \left. \nabla_\theta \log p_\theta(x_i) \right|_{\theta = \hat{\theta}} = 0
\end{align}
$$
