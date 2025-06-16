---
title: "ELBO"
date: "2025"
author: "Ke Zhang"
---

# The Evidence Lower Bound

## Motivation

Consider an observable random variable $X$, explained by some latent random variable $Z$. In general, both variables can be high dimensional. Assume we know the joint distribution $p(x,z)$, which can be very complex in general.

For a given instance $x$, we would like to compute the posterior distribution of the latent variable.

$$
p(z \mid x)
= \frac{p(x,z)}{p(x)}
= \frac{p(x,z)}{\int p(x,z) \,\mathrm dz}
$$

Even though we know the joint distribution, the above computation is still intractable in general.

We choose a tractable distribution family $\mathcal Q$ and use a surrogate distribution $q\in\mathcal Q$ to approximate the true posterior $p(z \mid x)$. To assess how well $q$ approximates the true posterior, we minimize the KL divergence:

$$
\min_{q \in\mathcal Q} D_\text{KL}(q(z) \parallel p(z \mid x))
$$

Remarks:

* Now, we turned the inference problem (a high dimensional integral) into an optimization problem.
* In practice, $\mathcal Q$ is a parameterized family (e.g. Gaussian). Computing the optimal $q$ is equivalent to computing the optimal parameters.

However, minimizing the KL divergence still requires knowledge of the posterior. Next, we will make the above optimization problem tractable.

> Final goal: Approximate the intractable posterior $p(z \mid x)$ with a tractable $q(z)$ by maximing the ELBO.

## Theoretical Foundations of ELBO

For a given instance $x$, we call $\log p(x)$ the ***evidence***. Then:

For any surrogate $q$, it holds that

$$
\begin{align}
\log p(x)
\ge \mathbb E_{z \sim q} \left[ \log\frac{p(x,z)}{q(z)} \right]
\end{align}
$$

where the RHS is called ***evidence lower bound*** (or ***ELBO***), denoted by $\mathcal L(q,x)$:

$$
\begin{align}
\mathcal L(q,x)
\triangleq \mathbb E_{z \sim q} \left[ \log\frac{p(x,z)}{q(z)} \right]
\end{align}
$$

Remarks:

* Other common notations for ELBO: $\mathcal L(q)$, $\mathcal L$ or simply $\mathrm{ELBO}$
* Computing the ELBO is **tractable** since it does not require evaluating high dimensional integral.

*Proof*: First, we express $p(x)$ as

$$
\begin{align*}
p(x)
= \int p(x,z) \,\mathrm dz
= \int q(z) \frac{p(x,z)}{q(z)} \,\mathrm dz
= \mathbb E_{z \sim q} \left[ \frac{p(x,z)}{q(z)}\right]
\end{align*}
$$

Taking the log and applying Jensen's inequality, we conclude

$$
\begin{align*}
\log p(x)
&= \log \mathbb E_{z \sim q} \left[ \frac{p(x,z)}{q(z)}\right] \\
&\ge \underbrace{\mathbb E_{z \sim q} \left[ \log \frac{p(x,z)}{q(z)}\right]}_{\mathcal L(q,x)}
\tag*{$\blacksquare$}
\end{align*}
$$

There are three ways to decompose the ELBO $\mathcal L(q,x)$ as we will show below. Each decomposation gives us insights from different perspectives.

$$
\begin{align*}
\mathcal L(q,x)
&= \log p(x) - D_\text{KL}(q(z) \parallel p(z \mid x)) \\
&= \mathbb E_{z \sim q} \left[ \log p(x \mid z) \right] - D_\text{KL}(q(z) \parallel p(z)) \\
&= \mathbb E_{z \sim q} \left[ \log p(x,z) \right] + H(q)
\end{align*}
$$

### ELBO as evidence minus variational gap

The 1st decomposition of ELBO is

$$
\begin{align}
\mathcal L(q,x)
&= \log p(x) - D_\text{KL}(q(z) \parallel p(z \mid x))
\end{align}
$$

or equivalently

$$
\begin{align}
\underbrace{\log p(x)}_\text{evidence}
&= \underbrace{\mathcal L(q,x)}_\text{ELBO} + \underbrace{D_\text{KL}(q(z) \parallel p(z \mid x))}_\text{variational gap}
\end{align}
$$

Remarks:

* The gap between the evidence and ELBO is exactly the KL divergence we want to minimize earlier (also known as ***variational gap***). Minimizing the variational gap is equivalent to maximizing the ELBO, which is the key idea of variational inference.
* The ELBO becomes tight (or maximized) iff $q(z) = p(z \mid x)$, which is typically not achievable in practice due to the restricted distribution class $\mathcal Q$.

*Proof*: Substitute $p(x,z) = p(z \mid x) \cdot p(x)$ into the definition of ELBO, we conclude

$$
\begin{align*}
\mathcal L(q,x)
&= \mathbb E_{z \sim q} \left[ \log\frac{p(z \mid x) \cdot p(x)}{q(z)} \right] \\
&= \mathbb E_{z \sim q} \left[ \log\frac{p(z \mid x)}{q(z)} + \log p(x) \right] \\
&= \underbrace{\mathbb E_{z \sim q} \left[ \log\frac{p(z \mid x)}{q(z)} \right]}_{- D_\text{KL}(q(z) \parallel p(z \mid x))} + \log p(x)
\tag*{$\blacksquare$}
\end{align*}
$$

Now, we are able to approximate $p(z \mid x)$ with $q(z)$ by

$$
\begin{align}
\max_{q \in\mathcal Q}
\mathcal L(q,x) = \mathbb E_{z \sim q} \left[ \log\frac{p(x,z)}{q(z)} \right]
\end{align}
$$

In practice, this optimization problem is solved by parameterizing $q$ and applying gradient methods (detailed later).

### ELBO as regularized reconstruction

The 2nd decomposition of ELBO is

$$
\begin{align}
\mathcal L(q,x)
&= \mathbb E_{z \sim q} \left[ \log p(x \mid z) \right] - D_\text{KL}(q(z) \parallel p(z))
\end{align}
$$

Remarks:

* The 1st term is expected log likelihood w.r.t. the surrogate. It measures the average goodness of reconstruction, assuming that $z \sim q$.
* The 2nd term is the KL divergence of the surrogate $q(z)$ w.r.t. the prior $p(z)$. i.e. We penalize those surrogates far from the prior. Thus, this term has a regularization effect.
* Maximizing the ELBO is a trade-off between maximizing the reconstruction fidelity and keeping surrogate close to the prior. This is the key idea behind variational autoencoders (VAEs).

*Proof*: Substitute $p(x,z) = p(x \mid z) \cdot p(z)$ into the definition of ELBO, we conclude

$$
\begin{align*}
\mathcal L(q,x)
&= \mathbb E_{z \sim q} \left[ \log\frac{p(x \mid z) \cdot p(z)}{q(z)} \right] \\
&= \mathbb E_{z \sim q} \left[ \log\frac{p(z)}{q(z)} + \log p(x \mid z) \right] \\
&= \underbrace{\mathbb E_{z \sim q} \left[ \log\frac{p(z)}{q(z)} \right]}_{- D_\text{KL}(q(z) \parallel p(z))} + \mathbb E_{z \sim q} \left[ \log p(x \mid z) \right]
\tag*{$\blacksquare$}
\end{align*}
$$

### ELBO as average log joint plus entropy

The 3rd decomposition of ELBO is

$$
\begin{align}
\mathcal L(q,x)
&= \mathbb E_{z \sim q} \left[ \log p(x,z) \right] + H(q)
\end{align}
$$

TODO: any insights?

*Proof*:

$$
\begin{align*}
\mathcal L(q,x)
&= \mathbb E_{z \sim q} \left[ \log \frac{p(x,z)}{q(z)} \right] \\
&= \mathbb E_{z \sim q} \left[ \log p(x,z) + \log \frac{1}{q(z)} \right] \\
&= \mathbb E_{z \sim q} \left[ \log p(x,z) \right] + \underbrace{\mathbb E_{z \sim q} \left[ \log \frac{1}{q(z)} \right]}_{H(q)}
\tag*{$\blacksquare$}
\end{align*}
$$

## ELBO for a Dataset

Consider the unspervised learning with latent variables:

* Model: $p(x,z)$
* Given: training data $D = \{ x_1, \cdots,  x_n\}$.

Can we derive the ELBO for the dataset?

depends on the form of $q(z_{1:n})$

### Parameterization

TODO

## Optional? relation bw. ELBO and EM

TODO
