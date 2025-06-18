---
title: "Probabilistic Learning"
author: "Ke Zhang"
date: "2025"
fontsize: 12pt
---

# Probabilistic Learning

## The Theoretical Framework

Notations:

* $\mathcal X$: the data space. Common choices: $\mathcal X = \mathbb R^d$ or a finite discrete set.
* $\mathcal P(\mathcal X)$: the set of all distributions on $\mathcal X$.
  * If $\mathcal X = \mathbb R^d$, then $\mathcal P(\mathcal X)$ consists of all PDFs $p: \mathcal X \to \mathbb R_{\ge 0}$
  * If $\mathcal X$ is a finite discrete set, then $\mathcal P(\mathcal X)$ consists of all PMFs $p: \mathcal X \to [0,1]$.
* $\mathcal P_\theta(\mathcal X)$: the set of all parameterized distributions on $\mathcal X$.
  * Clearly, $\mathcal P_\theta(\mathcal X) \subseteq \mathcal P(\mathcal X)$
  * e.g. $\mathcal X=\mathbb R$ and $\mathcal P_\theta(\mathbb R)$ could be the set of all univariate Gaussian distributions with $\theta = (\mu, \sigma^2)$. Clearly, an exponential distribution is not in $\mathcal P_\theta(\mathbb R)$.

Consider the random variable $X\in\mathcal X$ generated from the unknown ground truth distribution $p^* \in\mathcal P(\mathcal X)$. Probabilistic learning can be understood as finding a model distribution $p_\theta \in \mathcal P_\theta(\mathcal X)$ which is as close to (ideally identical to) $p^*$ as possible.

The cross entropy loss is commonly used to quantify the loss of using $p_\theta$ while the ground truth is $p^*$:

$$
H(p^*, p_\theta) = \mathbb E_{x \sim p^*(x)} [-\log p_\theta(x)]
$$

In general,

$$
\min_{p_\theta \in \mathcal P_\theta(\mathcal X)} H(p^*, p_\theta)
$$

is a very hard problem as $p^*$ might lie outside of $\mathcal P_\theta(\mathcal X)$, which is called model misspecification. TODO: but seems that we can nevertheless apply Monte Carlo approximation?

An easier version would be the case where $p^* \in \mathcal P_\theta(\mathcal X)$. We say that the model is correctly specified in such cases. Estimating the whole distribution boils down to estimating the ground truth parameter.

e.g. The ground truth distribution of a DC voltage is Gaussian due to thermal noise. The model distribution $p_\theta$ is also Gaussian, living in the same distribution space as the $p^*$.

TODO:

* under correct model specification: min. cross entropy loss equivalent to max. log likelihood
