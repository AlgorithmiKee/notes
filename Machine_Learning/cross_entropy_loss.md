---
title: "Cross Entropy Loss"
author: "Ke Zhang"
date: "2025"
fontsize: 12pt
---

[toc]

# Cross Entropy Loss

These notes provide a high-level overview of probabilistic learning from the perspective of cross entropy.

**Notations:**

* $\mathcal X$: the data space. Common choices include $\mathcal X = \mathbb R^d$ or a finite discrete set.
* $\mathcal P(\mathcal X)$: the set of all probability distributions on $\mathcal X$.
  * If $\mathcal X = \mathbb R^d$, then $\mathcal P(\mathcal X)$ includes all PDFs $p: \mathcal X \to \mathbb R_{\ge 0}$ such that $\int p(x) \, dx = 1$.
  * If $\mathcal X$ is finite and discrete, then $\mathcal P(\mathcal X)$ consists of all PMFs $p: \mathcal X \to [0,1]$ with $\sum_x p(x) = 1$.
* $\theta \in \mathbb R^p$: parameter vector. For example, for a univariate Gaussian, $\theta = (\mu, \sigma^2) \in \mathbb R^2$.
* $\mathcal P_\theta(\mathcal X)$: a parameterized family of distributions on $\mathcal X$.
  * Clearly, $\mathcal P_\theta(\mathcal X) \subseteq \mathcal P(\mathcal X)$.
  * Example: If $\mathcal X=\mathbb R$, then $\mathcal P_\theta(\mathbb R)$ could be the set of all univariate Gaussians parameterized by $\theta = (\mu, \sigma^2)$. In this case, the exponential distribution is not in $\mathcal P_\theta(\mathbb R)$.

## Learning a Marginal Distribution

Suppose $X \in \mathcal X$ is drawn from an unknown true distribution $p^* \in \mathcal P(\mathcal X)$. Probabilistic learning can be seen as the process of finding a model distribution $p_\theta \in \mathcal P_\theta(\mathcal X)$ that is as close to $p^*$ as possible.

We say the model is **well-specified** if $p^* \in \mathcal P_\theta(\mathcal X)$, and **misspecified** otherwise.  
For example, the voltage of a DC signal subject to thermal noise may follow a Gaussian distribution. If we model it with another Gaussian, the model is well-specified.

To quantify the difference between $p_\theta$ and $p^*$, we often use **cross entropy loss**:

$$
\begin{align}
H(p^*, p_\theta) = \mathbb E_{x \sim p^*} [-\log p_\theta(x)]
\end{align}
$$

Learning then becomes an optimization problem:

$$
\begin{align}
\min_{p_\theta \in \mathcal P_\theta(\mathcal X)} H(p^*, p_\theta)
\end{align}
$$

Or equivalently, in terms of parameters:

$$
\begin{align}
\min_{\theta \in \mathbb R^p} \mathbb E_{x \sim p^*} [-\log p_\theta(x)]
\end{align}
$$

To optimize the cross entropy, we often need its gradient. Since $p^*$ does not depend on $\theta$, we can move the gradient into the expectation:

$$
\begin{align}
\nabla_\theta H(p^*, p_\theta) = \mathbb E_{x \sim p^*} [-\nabla_\theta \log p_\theta(x)]
\end{align}
$$

**Two practical concerns:**

1. The true distribution $p^*$ is unknown.
2. The model may be misspecified: $p^*$ might not lie within $\mathcal P_\theta(\mathcal X)$.

The first issue can be addressed using **Monte Carlo estimation**.  
→ See section [MC Estimation of Cross Entropy](#mc-estimation-of-cross-entropy)

The second issue (misspecification) is more nuanced:  
If the model is **severely misspecified**, minimizing cross entropy may still yield the best approximation within the model family, but the performance is limited the expressiveness of $\mathcal P_\theta$. A richer family $\mathcal P_\theta$ allows better approximation of $p^*$. However, a richer $\mathcal P_\theta$ is also more prone to overfitting.


### Theoretical Lower Bound of Cross Entropy

From information theory, we have:

$$
\begin{align}
H(p^*, p_\theta)
&= H(p^*) + D_\text{KL}(p^* \parallel p_\theta) \\
&\ge H(p^*)
\end{align}
$$

Remarks:

* The inequality becomes equality iff $p_\theta = p^*$ almost everywhere.
* $H(p^*)$ represents the inherent uncertainty in the tue data-generating distribution. It is the theoretical minimum achievable cross entropy.
* Minimizing cross entropy $H(p^*, p_\theta)$ is equivalent to minimizing KL divergence $D_\text{KL}(p^* \parallel p_\theta)$.

### MC Estimation of Cross Entropy

Both the cross entropy and its gradient can be approximated as:

$$
\begin{align}
x_i &\stackrel{\text{iid}}{\sim} p^*, \quad i=1,\dots,N \\
H(p^*, p_\theta) &\approx -\frac{1}{N} \sum_{i=1}^N \log p_\theta(x_i) \\
\nabla_\theta H(p^*, p_\theta) &\approx -\frac{1}{N} \sum_{i=1}^N \nabla_\theta \log p_\theta(x_i)
\end{align}
$$

Remarks:

* Both approximations are unbiased estimators.
* The term $\log p_\theta(x_i)$ is the log-likelihood of observation $x_i$ under model $p_\theta$.
* Therefore, the cross entropy is approximated by the negative log-likelihood (NLL) over the dataset $x_1, \dots, x_N$.

> **Minimizing cross entropy** is asymptotically equivalent to **maximizing the log-likelihood**.

Hence, the optimal parameter $\hat{\theta}$ satisfies:

$$
\begin{align}
\sum_{i=1}^N \left. \nabla_\theta \log p_\theta(x_i) \right|_{\theta = \hat{\theta}} = 0
\end{align}
$$
