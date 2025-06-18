---
title: "Gradient Approximation"
date: "2025"
author: "Ke Zhang"
---

# Gradient Approximation

## Motivation

Many ML problems involve optimizing an objective of the form

$$
J(\theta) = \mathbb E_{x \sim p^*} [f_\theta(x)]
$$

where $p^*$ is the unknown ground truth distribution of $X$, which does **not** depend on $\theta$.

To optimize $J(\theta) $, we often need to compute its gradient. Under standard regularity conditions, we can move the gradient into the expectation because $p^*$ does not depend on $\theta$.

$$
\begin{align*}
\nabla_{\theta} J(\theta) = \mathbb E_{x \sim p^*} [\nabla_{\theta} f_\theta(x)]
\end{align*}
$$

In practice, both the objective function and its gradient are approximated by Monte Carlo (MC) sampling due to the unknown nature of $p^*$.

$$
\begin{align*}
x^{(i)} &\stackrel{\text{iid}}{\sim} p^*, \quad i=1, \dots, N \\
J(\theta) &\approx \frac{1}{N} \sum_{i=1}^N f_\theta(x^{(i)}) \\
\nabla_{\theta} J(\theta) &\approx \frac{1}{N} \sum_{i=1}^N \nabla_{\theta} f_\theta(x^{(i)})
\end{align*}
$$

e.g. In standard MLE, we use the model distribution $p(\cdot \mid \theta)$ to approximate $p^*(\cdot)$. For this purpose, we need to maximize the expected log-likelihood by setting $f_\theta(x) = \ln p(x \mid \theta)$. The objective thus becomes

$$
J(\theta) = \mathbb E_{x \sim p^*} [\ln p(x \mid \theta)]
$$

The corresponding MC approximation is

$$
\begin{align*}
x^{(i)} &\stackrel{\text{iid}}{\sim} p^*, \quad i=1, \dots, N \\
J(\theta) &\approx \frac{1}{N} \sum_{i=1}^N \ln p(x^{(i)}  \mid \theta) \\
\nabla_{\theta} J(\theta) &\approx \frac{1}{N} \sum_{i=1}^N \nabla_{\theta} \ln p(x^{(i)}  \mid \theta)
\end{align*}
$$

The key assumption of the framework so far is that $x$ is sampled from a distribution that does not depend on the optimization variable $\theta$. But what if the distribution of $x$ does depend on $\theta$? How can we approximate the gradient under this new setting?

## Problem Formulation

Let $X \in\mathbb R^d$ be a random vector with PDF $p_\theta(\cdot)$ where $\theta\in\mathbb R^p$ is the parameter vector. We aim to optimize

$$
J(\theta) = \mathbb E_{x \sim p_\theta} [f(x)]
$$

where $f: \mathbb R^d \to \mathbb R$ is a deterministic function -- typically a loss or reward function.

Unlike the previous setting, the PDF of $X$ now dependens on optimization variable $\theta$. We can no longer move the gradient inside the expectation (without modification). Starting from first principles:

$$
\begin{align*}
\nabla_{\theta} J(\theta)
&= \nabla_{\theta}\mathbb E_{x \sim p_\theta} [f(x)] \\
&= \nabla_{\theta} \int_x f(x) p_\theta(x) \,\mathrm dx \\
&= \int_x f(x) \nabla_{\theta} p_\theta(x) \,\mathrm dx \\
\end{align*}
$$

We see two main challenges here:

* The integral is generally intractable, especially when the dimensionality of $x$ is high.
* The expression is not in the form $\mathbb E_{x \sim p_\theta} [\cdot]$. Hence, we cannot directly apply MC sampling to approximate the gradient $\nabla_{\theta} J(\theta)$.

Two major techniques are used to approximate the gradient:

1. Score function method: Express the gradient into the form of $\mathbb E_{x \sim p_\theta} [\cdot]$ so that we can apply MC approximation. Widely used in reinforcement learning.
1. Reparameterization trick: Express $X$ as the transformation of another base random variable whose density does not depend on $\theta$. Commonly used in variational inference.

## Score Function Method

The gradient $\nabla_{\theta} \mathbb E_{x \sim p_\theta} [f(x)]$ can be expressed as

$$
\begin{align}
\nabla_{\theta} \mathbb E_{x \sim p_\theta} [f(x)]
&=  \mathbb E_{x \sim p_\theta} \left[f(x) \nabla_{\theta} \big(\ln p_\theta(x) \big) \right]
\end{align}
$$

Remarks:

* It’s called the score function because $\nabla_{\theta} \big(\ln p_\theta(x) \big)$ is the ***score function*** in statistics (the gradient of the log-likelihood). This method is also known as ***REINFORCE***.
* Score function method does not require differentiability of $f$ even if $X$ is a continuous random variable.
* Now, we expressed $\nabla_{\theta} \mathbb E_{x \sim p_\theta} [f(x)]$ in the form of $\mathbb E_{x \sim p_\theta} [\cdot]$. The true gradient can thus be approximated by MC.

*Proof*: We assume regularity conditions (convergence, differentiable, etc.)

$$
\begin{align*}
\nabla_{\theta} \mathbb E_{x \sim p_\theta} [f(x)]
&= \nabla_{\theta} \int_x f(x) p_\theta(x) \,\mathrm dx \\
&= \int_x f(x) \nabla_{\theta} p_\theta(x) \,\mathrm dx \\
&= \int_x f(x) \, p_\theta(x) \frac{1}{p_\theta(x)} \nabla_{\theta} p_\theta(x) \,\mathrm dx \\
&= \int_x f(x) \, p_\theta(x) \nabla_{\theta} \big(\ln p_\theta(x) \big) \,\mathrm dx \\
&=  \mathbb E_{x \sim p_\theta} \left[f(x) \nabla_{\theta} \big(\ln p_\theta(x) \big) \right]
\tag*{$\blacksquare$}
\end{align*}
$$

MC approximation of the gradient $\nabla_{\theta} \mathbb E_{x \sim p_\theta} [f(x)]$:

$$
\begin{align}
\nabla_{\theta} \mathbb E_{x \sim p_\theta} [f(x)]
&\approx \frac{1}{N} \sum_{i=1}^N f(x^{(i)}) \nabla_{\theta} \big(\ln p_\theta(x^{(i)}) \big)
\end{align}
$$

where $x^{(1)}, \dots, x^{(N)} \stackrel{\text{iid}}{\sim} p_\theta$.

Assuming we are minimizing $\mathbb E_{x \sim p_\theta} [f(x)]$ over $\theta$, the SGD becomes

> **Algorithm (SGD with score function gradient)**  
> Init $\theta^{(0)} \in\mathbb R^p$  
> For $t=0,1,2,\dots$ until convergence, do  
> $\qquad$ Draw a sample $x^{(1)}, \dots, x^{(N)} \stackrel{\text{iid}}{\sim} p_\theta$  
> $\qquad$ Update parameters
> $$
> \theta^{(t+1)} = \theta^{(t)} - \eta^{(t)} \frac{1}{N}\sum_{i=1}^N f(x^{(i)}) \nabla_{\theta} \left.\big(\ln p_\theta(x^{(i)}) \big)\right|_{\theta = \theta^{(t)}}
> $$

### Example: Score Function for Gaussian

Consider the multivariate Gaussian $p_\theta(x) = \mathcal N(x; \mu, \Sigma)$ with $\theta = (\mu, \Sigma)$. Then, the gradient

$$
\begin{align*}
\nabla_{\mu, \Sigma} \, \mathbb E_{x \sim \mathcal N(\mu, \Sigma)} [f(x)]
\end{align*}
$$

can be approximated by

$$
\begin{align}
\nabla_{\mu, \Sigma} \, \mathbb E_{x \sim \mathcal N(\mu, \Sigma)} [f(x)]
&= \mathbb E_{x \sim \mathcal N(\mu, \Sigma)} \left[ f(x) \nabla_{\mu, \Sigma}(\ln \mathcal N(x; \mu, \Sigma)) \right]
\\
&\approx \frac{1}{N} \sum_{i=1}^N \left[ f(x^{(i)}) \nabla_{\mu, \Sigma}(\ln \mathcal N(x^{(i)}; \mu, \Sigma)) \right]
\end{align}
$$

where (see *everthing about Gaussian*) $\forall i=1,\dots,n$:

$$
\begin{align}
x^{(i)} &\stackrel{\text{iid}}{\sim} \mathcal N(\mu, \Sigma)
\\
\nabla_{\mu}(\ln \mathcal N(x^{(i)}; \mu, \Sigma))
&= \Sigma^{-1}(x^{(i)}-\mu)
\\
\nabla_{\Sigma}(\ln \mathcal N(x^{(i)}; \mu, \Sigma))
&= \frac{1}{2} \left[ \Sigma^{-1}(x^{(i)}-\mu)(x^{(i)}-\mu)^\top \Sigma^{-1} - \Sigma^{-1} \right]
\end{align}
$$

## Reparameterization Trick

We say the distribution $x\sim p_\theta$ ***reparameterizable*** if there exists a random vector $\epsilon\in\mathbb R^d$ and a bijective function $g_{\theta}: \mathbb R^d \to \mathbb R^d$, s.t.

$$
\begin{align}
\epsilon \sim p(\epsilon),
\quad
x = g_{\theta}(\epsilon)
\end{align}
$$

Remarks:

* The PDF of $\epsilon$ is called ***reference density*** and it must **not** depend on the optimization variable $\theta$.
* Instead of sampling $x$ directly from $p_\theta$, we obtain $x$ by first sampling $\epsilon$ from $p$ and then applying $g_{\theta}$.
* The function $g_\theta$ defines a differentiable (w.r.t $\theta$) transformation of random vectors.

Suppose $x\sim p_\theta$ is reparameterizable with reference density $p$ and transform function $g_{\theta}$. Then,

$$
\begin{align}
\mathbb E_{x \sim p_\theta} [f(x)]
&= \mathbb E_{\epsilon \sim p} \left[ f(g_{\theta}(\epsilon)) \right]
\\[6pt]
\nabla_{\theta} \mathbb E_{x \sim p_\theta} [f(x)]
&= \mathbb E_{\epsilon \sim p} \left[ \nabla_{\theta} f(g_{\theta}(\epsilon)) \right]
\end{align}
$$

Remarks:

* The 2nd equation suggests that differentiability of $f$ is required to apply reparameterization trick.
* It is essential that the reference density does not depend on $\theta$. Otherwise, we can not move the gradient operator into expectation.

*Proof*: The 1st equality follows from the [law of the unconscious statistian (LOTUS)](https://en.wikipedia.org/wiki/Law_of_the_unconscious_statistician). The 2nd equality follows from the linearity of expectation and the fact that the reference density does not depend on $\theta$. $\quad\blacksquare$

MC approximation of $\nabla_{\theta} \mathbb E_{x \sim p_\theta} [f(x)]$:

$$
\begin{align}
\nabla_{\theta} \mathbb E_{x \sim p_\theta} [f(x)]
&\approx \frac{1}{N} \sum_{i=1}^N \nabla_{\theta} f(g_{\theta}(\epsilon^{(i)}))
\end{align}
$$

where $\epsilon^{(1)}, \dots, \epsilon^{(N)} \stackrel{\text{iid}}{\sim} p$.

Assuming we are minimizing $\mathbb E_{x \sim p_\theta} [f(x)]$ over $\theta$, the SGD becomes

> **Algorithm (SGD with reparameterization trick)**  
> Init $\theta^{(0)} \in\mathbb R^p$  
> For $t=0,1,2,\dots$ until convergence, do  
> $\qquad$ Draw a sample $\epsilon^{(1)}, \dots, \epsilon^{(N)} \stackrel{\text{iid}}{\sim} p$  
> $\qquad$ Update parameters
> $$
> \theta^{(t+1)} = \theta^{(t)} - \eta^{(t)} \frac{1}{N}\sum_{i=1}^N \left. \nabla_{\theta} f(g_{\theta}(\epsilon^{(i)})) \right|_{\theta = \theta^{(t)}}
> $$

### Example: Reparameterization for Gaussian

Consider the multivariate Gaussian $p_\theta(x) = \mathcal N(x; \mu, \Sigma)$ with $\theta = (\mu, C)$ where $C \triangleq \Sigma^{1/2}$ is the Cholesky factor of $\Sigma$.

We would like to approximate

$$
\begin{align}
\nabla_{\mu, C} \, \mathbb E_{x \sim \mathcal N(\mu, \Sigma)} [f(x)]
\end{align}
$$

Remarks:

* We consider $\Sigma^{1/2}$ instead of $\Sigma$ as part of $\theta$ because the reparameterization function directly depends on $\Sigma^{1/2}$. Otherwise, we would have to compute $\nabla_{\mu, \Sigma} \Sigma^{1/2}$, which is very hard in practice.
* In the following, we perform SGD for parameters $\theta = (\mu, C)$. After SGD converges, we can simply recover the covariance matrix by computing $\Sigma = CC^\top$

One reparameterization is

$$
\begin{align}
\epsilon &\sim p(\epsilon) = \mathcal N(\epsilon; 0, I),
\\
x &= g_{\mu,C}(\epsilon) = C \epsilon + \mu
\end{align}
$$

Hence, the gradient $\nabla_{\mu, C} \, \mathbb E_{x \sim \mathcal N(\mu, \Sigma)} [f(x)]$ can be approximated by

$$
\begin{align}
\nabla_{\mu, C} \, \mathbb E_{x \sim \mathcal N(\mu, \Sigma)} [f(x)]
&= \nabla_{\mu, C} \, \mathbb E_{\epsilon \sim \mathcal N(0, I)} [f(C \epsilon + \mu)]
&& \text{reparam. trick}
\\
&= \mathbb E_{\epsilon \sim \mathcal N(0, I)} \left[ \nabla_{\mu, C} f(C \epsilon + \mu) \right]
&& \text{move $\nabla$ inside}
\\
&\approx \frac{1}{N} \sum_{i=1}^N \nabla_{\mu, C} f(C \epsilon^{(i)} + \mu)
&& \epsilon^{(i)} \stackrel{\text{iid}}{\sim} \mathcal N(0, I)
\end{align}
$$

Modern ML frameworks (e.g. `torch.autograd`) automatically compute $\nabla_{\mu, C} f(C \epsilon^{(i)} + \mu)$ using chain rule. In practice, we do not need to manually code the detailed computations like

$$
\begin{align*}
\nabla_{\mu} f(C \epsilon^{(i)} + \mu)
&= \frac{\partial f}{\partial x} \cdot \frac{\partial x}{\partial \mu}
= \frac{\partial f}{\partial x} \cdot I \\
&= \left. \nabla_x f(x) \right|_{x = C \epsilon^{(i)} + \mu}
\\
\nabla_{C} f(C \epsilon^{(i)} + \mu)
&= \frac{\partial f}{\partial x} \cdot \frac{\partial x}{\partial C}
= \frac{\partial f}{\partial x} \cdot (\epsilon^{(i)})^\top \\
&= \left. \nabla_x f(x) \right|_{x = C \epsilon^{(i)} + \mu} \otimes \epsilon^{(i)}
\end{align*}
$$
