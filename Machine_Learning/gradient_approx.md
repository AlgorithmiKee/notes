---
title: "Gradient Approximation"
date: "2025"
author: "Ke Zhang"
---

# Gradient Approximation

## Problem Formulation

Let $X \in\mathbb R^d$ be a random vector with PDF $p_\theta(\cdot)$ where $\theta\in\mathbb R^k$ is the parameter vector. In many ML problems, we need to minimize or maximize

$$
J(\theta) = \mathbb E_{x \sim p_\theta} [f(x)]
$$

where $f: \mathbb R^d \to \mathbb R$ is a deterministic function.

To obtain the solution, we often need to compute the gradient

$$
\nabla_{\theta} J(\theta) = \nabla_{\theta}\mathbb E_{x \sim p_\theta} [f(x)]
$$

A direct computation of the gradient would be

$$
\begin{align*}
\nabla_{\theta} \mathbb E_{x \sim p_\theta} [f(x)]
&= \nabla_{\theta} \int_x f(x) p_\theta(x) \,\mathrm dx \\
&= \int_x f(x) \nabla_{\theta} p_\theta(x) \,\mathrm dx
\end{align*}
$$

However, the above computation is often challenging due to

* The integral is generally intractable due to high dimensionality of $x$.
* We can neither directly use Monte Carlo (MC) sampling to approximate the gradient since the above expression is not in the form of $\mathbb E_{x \sim p_\theta} [\cdot]$.

There are two main methods to address this problem:

1. Score function method: Express the gradient into the form of $\mathbb E_{x \sim p_\theta} [\cdot]$ so that we can apply MC. Often used in reinforcement learning.
1. Reparameterization trick: Use transform of random variables so that $\theta$ only appears inside the expectation. Often used in variational inference.

## Score Function Method

The gradient $\nabla_{\theta} \mathbb E_{x \sim p_\theta} [f(x)]$ can be expressed as

$$
\begin{align}
\nabla_{\theta} \mathbb E_{x \sim p_\theta} [f(x)]
&=  \mathbb E_{x \sim p_\theta} \left[f(x) \nabla_{\theta} \big(\ln p_\theta(x) \big) \right]
\end{align}
$$

Remarks:

* It’s called the score function because $\nabla_{\theta} \big(\ln p_\theta(x) \big)$ is the ***score function*** in statistics (the gradient of the log-likelihood).
* This method is also known as ***REINFORCE***.
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
> Init $\theta^{(0)} \in\mathbb R^k$  
> For $t=0,2,\dots$ until convergence, do  
> $\qquad$ Draw a sample $x^{(1)}, \dots, x^{(N)} \stackrel{\text{iid}}{\sim} p_\theta$  
> $\qquad$ Update parameters
> $$
> \theta^{(t+1)} = \theta^{(t)} - \eta^{(t)} \frac{1}{N}\sum_{i=1}^N f(x^{(i)}) \nabla_{\theta} \left.\big(\ln p_\theta(x^{(i)}) \big)\right|_{\theta = \theta^{(t)}}
> $$

TODO:

* variance reduction technique (e.g. baselines)
* example.

## Reparameterization Trick

We call $x\sim p_\theta$ ***reparameterizable*** if there exists a random vector $\epsilon\in\mathbb R^d$ and a bijective function $g_{\theta}: \mathbb R^d \to \mathbb R^d$, s.t.

$$
\begin{align}
\epsilon \sim p(\epsilon),
\quad
x = g_{\theta}(\epsilon)
\end{align}
$$

Remarks:

* The PDF of $\epsilon$ is called reference density and it must not depend on the optimization variable $\theta$.
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
> Init $\theta^{(0)} \in\mathbb R^k$  
> For $t=0,2,\dots$ until convergence, do  
> $\qquad$ Draw a sample $\epsilon^{(1)}, \dots, \epsilon^{(N)} \stackrel{\text{iid}}{\sim} p$  
> $\qquad$ Update parameters
> $$
> \theta^{(t+1)} = \theta^{(t)} - \eta^{(t)} \frac{1}{N}\sum_{i=1}^N \left. \nabla_{\theta} f(g_{\theta}(\epsilon^{(i)})) \right|_{\theta = \theta^{(t)}}
> $$

TODO:

* variance reduction technique (e.g. baselines)
* example.
