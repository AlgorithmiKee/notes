---
title: "KL Divergence"
author: "Ke Zhang"
date: "2025"
fontsize: 12pt
---

# KL Divergence

## Probabilistic Learning

Problem formulation:

* Unknown true distribution $p_\star$
* Design a probabilistic model: $p_\theta$
* Train the probabilistic model so that $p_\theta \approx p_\star$

How to measure the *distance* between two distributions? A popular choice is KL divergence. For two distributions $p$ and $q$, the KL divergence is defined as

> $$
> \begin{align}
> D_\text{KL}(p \| q)
> &= \mathbb E_{x \sim p} \left[ \log \frac{p(x)}{q(x)} \right]
> \end{align}
> $$

KL divergence is not symmetric. In probabilistic learning, we distinguish

* **forward** KL divergence

    > $$
    > \begin{align}
    > D_\text{KL}(p_\star \| p_\theta)
    > &= \mathbb E_{x \sim p_\star} \left[ \log \frac{p_\star(x)}{p_\theta(x)} \right]
    > \end{align}
    > $$

* **reverse** KL divergence

    > $$
    > \begin{align}
    > D_\text{KL}(p_\theta \| p_\star)
    > &= \mathbb E_{x \sim p_\theta} \left[ \log \frac{p_\theta(x)}{p_\star(x)} \right]
    > \end{align}
    > $$
    >

The learning problem can be formulated as minimizing either the forward or the reverse KL divergence. Each choice leads to different learning behaviors and practical consequences.

## Forward KL-Divergence

By def.,
$$
\begin{align}
D_\text{KL}(p_\star \| p_\theta)
&= \mathbb E_{x \sim p_\star} [ \log p_\star(x) ] + \mathbb E_{x \sim p_\star} [ -\log p_\theta(x) ]\\
\end{align}
$$
Remarks:

* The expectation is taken over the true data distribution.
* The 1st term is independent of parameter $\theta$ and can be ignored during optimization.
* Minimizing the 2nd term encourages $p_\theta$ to place probability mass wherever $p_\star$ has support. (support seeking) This often makes $p_\theta$ more “spread out,” covering all modes of $p_\star$ rather than collapsing on a single one.

Hence, the following optimization problems are equivalent 
$$
\begin{align}
&\min_\theta D_\text{KL}(p_\star \| p_\theta)
&& \blacktriangleright \, \text{min. forward KL divergence} \\
\iff &\min_\theta \mathbb E_{x \sim p_\star} [ - \log p_\theta(x) ]
&& \blacktriangleright \, \text{min. cross-entropy loss} \\
\iff &\max_\theta \mathbb E_{x \sim p_\star} [ \log p_\theta(x) ]
&& \blacktriangleright \,\text{max. expected log-liklihood} \\
\end{align}
$$
In practice, we do not know $p_\star$. However, if we can sample from it, the expected log-likelihood can be approximed by empirical mean.
$$
\mathbb E_{x \sim p_\star} [ \log p_\theta(x) ]
\approx \frac{1}{n} \sum_{i=1}^n \log p_\theta(x_i), \quad
x_1,\dots,x_n \stackrel{\text{iid}}{\sim} p_\star
$$

> Minimizing the forward KL is then asymptotically equivalent to maximum likelihood estimation.
> $$
> \min_\theta D_\text{KL}(p_\star \| p_\theta)
> \xLeftrightarrow{\, n \to \infty \,}
> \max_\theta \sum_{i=1}^n \log p_\theta(x_i), \quad
> x_1,\dots,x_n \stackrel{\text{iid}}{\sim} p_\star
> $$

This answers the question: Why is $\log$ introduced to MLE?

* The classical explanation of MLE introduces the $\log$ for algebraic convenience.
* From an information-theoretic view, the $\log$ ensures that MLE corresponds to minimizing forward KL.

## Reverse KL Divergence

By def.
$$
\begin{align}
D_\text{KL}(p_\theta \| p_\star)
&= \mathbb E_{x \sim p_\theta} [ \log p_\theta(x) ] - \mathbb E_{x \sim p_\theta} [ \log p_\star(x) ] \nonumber \\
&= - H(p_\theta) - \mathbb E_{x \sim p_\theta} [ \log p_\star(x) ] 
\end{align}
$$
Hence, minimizing the reverse KL is equivalent to
$$
\begin{align}
\max_\theta \mathbb E_{x \sim p_\theta} [\log p_\star(x)] + H(p_\theta)
\end{align}
$$
Remarks:

* The expectation is taken over the modeled distribution.
* The 1st term encourages $p_\theta$ to place all mass at the mode of $p_\star$. (mode seeking)
* The 2nd term encourages $p_\theta$ to spread out. (regularization)

Here, the expectation is over $p_\theta$ (model distribution). We can sample from $p_\theta$, but evaluating $\log p_\star(x)$ requires knowing $p_\star$. In practice, we have additional assumption making the optimzaition viable.

Suppose $p_\star$ is known up to a normalization constant, which is the common scenario in Bayesian inference.
$$
p_\star(x) = \frac{\tilde p(x)}{\aleph}
$$
Remarks:

* $\tilde p$ is the unnormalized density, which integrates to unknown normalization constant $\aleph$. In latent variable model, it represents the *evidence*.
* In practice, we can not compute $\aleph$ as high dimensional integration is intractable.
* In ML literatures, the normalizaiton constant is often written as $Z$, which is confusing as $Z$ also denotes latent variables.

Then, miminizng the reverse KL is equivalent to
$$
\begin{align}
\max_\theta \mathbb E_{x \sim p_\theta} [\log \tilde p(x)] + H(p_\theta) - \log\aleph
\end{align}
$$
In Bayesian inference, the first term is called ***evidence lower bound*** (***ELBO***). Hence:

> Minimizing the reverse KL is equivalent to maximizing the evidence lower bound.
> $$
> \begin{align}
> D_\text{KL}(p_\theta \| p_\star)
> \iff
> \max_\theta \mathbb E_{x \sim p_\theta} [\log \tilde p(x)] + H(p_\theta)
> \end{align}
> $$

Since the expectation depends on $\theta$, gradient approximations are required:

* score function method
* reparameterization trick

$\to$ See separate notes <u>gradient approximation</u>.

## Summary

| **Forward KL Minimization**                 | **Reverse KL Minimization**                  |
| ------------------------------------------- | -------------------------------------------- |
| Support seeking                             | Mode seeking                                 |
| Unknown $p_\star$ but we can sample from it | Known $p_\star$ up to a normalization factor |
| equivalent to: MLE                          | equivalent to: max. ELBO                     |
