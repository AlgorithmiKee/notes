---
title: "ELBO"
date: "2025"
author: "Ke Zhang"
---

[toc]

$$
\DeclareMathOperator*{\argmax}{argmax}
\DeclareMathOperator*{\argmin}{argmin}
$$

# The Evidence Lower Bound

## Motivation

Consider an observable random variable $X\in\mathbb R^d$, explained by some latent random variable $Z\in\mathbb R^\ell$. In general, both variables can be high dimensional. Assume we know the joint distribution $p(x,z)$.

For a given instance $x$, we would like to compute the posterior distribution of the latent variable.

$$
p(z \mid x)
= \frac{p(x,z)}{p(x)}
= \frac{p(x,z)}{\int p(x,z) \,\mathrm dz}
$$

Even though we know the joint distribution, the above computation is still intractable in general due to the high dimensional integral in the denominator.

In variational inference, we choose a tractable distribution family $\mathcal Q$ and use a surrogate distribution $q\in\mathcal Q$ to approximate the true posterior $p(z \mid x)$. To assess how well $q$ approximates the true posterior, we minimize the KL divergence:

$$
\begin{align}
\min_{q \in\mathcal Q} D_\text{KL}(q(z) \parallel p(z \mid x))
\end{align}
$$

Remarks:

* The optimal approximation $q^*$ implicitly depends on $x$. Given another observation $x$, we typically end up with another $q^*$, as illustrated below.
* In practice, $\mathcal Q$ is a parameterized family (e.g. Gaussian). Computing the optimal $q$ is equivalent to computing the optimal parameters.
* Here, we slightly abuse notation for clarity. Strictly speaking, the KL divergence should be written as $D_\text{KL}(q(\cdot) \parallel p(\cdot \mid x))$ where $\cdot$ is the place holder for $z$.

<img src="./figs/vi_illustration.pdf" alt="elbo maximizer" style="zoom:67%;" />

We have transformed the inference problem — a high-dimensional integration — into an optimization problem. However, minimizing the KL divergence still requires knowledge of the posterior. Next, we will make the above optimization problem tractable.

> Final goal: Approximate the intractable posterior $p(z \mid x)$ with a tractable $q(z)$ by maximing the ELBO.

## ELBO for a Single Observation

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

* Other common notations for ELBO: $\mathcal L(q)$, $\mathcal L$ or simply $\mathrm{ELBO}$.
* For a fixed $x$, the ELBO is a functional of $q$. The higher the ELBO, the better $q$ approximates the true posterior.
* The ELBO also depends on the observation $x$. For a fixed $q$, evaluating ELBO at another $x'$ will give another bound (for another posterior $p(z \mid x')$).
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
* The ELBO becomes tight (or maximized) iff $q(z) = p(z \mid x)$, which is typically not achievable in practice due to the limited expressiveness of the distribution class $\mathcal Q$.

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

In practice, this optimization problem is usually solved by parameterizing $q$ and applying gradient methods (instead of applying calculus of variations).

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

Remarks:

* The 1st term is known as negative free energy in statistical physics. It rewards $q$ that explain the data well.
* The 2nd term is the entropy of the surrogate. It rewards $q$ with higher uncertainty.
* Maximizing the ELBO can be interpreted as minimizing the free energy while maintaining high entropy in the surrogate distribution.

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

Without the entropy term, maximizing the ELBO would result in a point mass for $z$ at the mode of $p(x,z)$ (recall: $x$ is fixed):

$$
q^*(z) = \delta(z - \hat{z}), \quad \text{where } \hat{z} = \argmax_z p(x,z)
$$

The entropy term, however, favors $q$ with higher entropy. In contrast, the Dirac delta is infinitely narrow and thus has extremely low entropy. Therefore, maximizing the ELBO seeks a balance between those two aspects.

<img src="./figs/elbo_maximizer.pdf" alt="elbo maximizer" style="zoom:67%;" />

## ELBO for a Dataset

Previously, we derived the ELBO $\mathcal L(q,x)$ for a single observation $x$. From now on, let's call it **per-sample** ELBO (or **per-observation** ELBO).

**Question**: What if we have a dataset consisting of multiple iid observations? Can we lower-bound the evidence of the whole dataset?

Consider the unspervised learning with latent variables:

* Model: $p(x,z)$
* Given: training data $D = \{ x_1, \cdots,  x_n\} \stackrel{\text{iid}}{\sim} p(x) = \int_z p(x,z) \:\mathrm dz$.

We refer to the lower bound on $\log p(D)$ as the **dataset ELBO**. In fact, dataset ELBO does exist since the evidence of the dataset is the sum of evidence of observations

$$
\begin{align}
\log p(D) = \log \prod_{i=1}^n p(x_i) = \sum_{i=1}^n \log p(x_i)
\end{align}
$$

Each $\log p(x_i)$ can be lower bounded by its individual per-sample ELBO. Therefore, $\log p(D)$ can also be lower bounded. The remaining question is how to design the surrogate.

### Per-Sample Surrogate

**Per-sample surrogate** means: For each $x_i$, we use $q_i(z)$ as the surrogate of the true posterior $p(z \mid x_i)$.

For any combination of surrogates $q_1, \dots, q_n \in \mathcal Q$, it holds that

$$
\begin{align}
\log p(D) &\ge \mathcal L(q_1, \dots, q_n, D)
\end{align}
$$

where the RHS is the dataset ELBO, defined as

$$
\begin{align}
\mathcal L(q_1, \dots, q_n, D)
&\triangleq \sum_{i=1}^n \mathcal L(q_i, x_i) \\
&= \sum_{i=1}^n \mathbb E_{z \sim q_i} \left[ \log\frac{p(x_i, z)}{q_i(z)} \right]
\end{align}
$$

Remarks:

* The (dataset) ELBO is a functional of $q_1, \dots, q_n$.
* We assume that $q_1, \dots, q_n \in \mathcal Q$. i.e. All surrogates belong to the same distribution class.
* Per-sample surrogates are used in classical variational inference.

*Proof*: By our previous discussion, each $\log p(x_i)$ is lowered bounded by its per-sample ELBO:
$$
\begin{align}
\mathcal L(q_i, x_i) = \mathbb E_{z \sim q_i} \left[ \log\frac{p(x_i, z)}{q_i(z)} \right]
\end{align}
$$

Summing over all observations, we obtain the dataset ELBO for $\log p(D)$:

$$
\begin{align*}
\log p(D)
= \sum_{i=1}^n \log p(x_i)
\ge \sum_{i=1}^n \mathcal L(q_i, x_i)
= \sum_{i=1}^n \mathbb E_{z \sim q_i} \left[ \log\frac{p(x_i, z)}{q_i(z)} \right]
\tag*{$\blacksquare$}
\end{align*}
$$

The optimal surrogates can be obtained by separately solving the functional optimization problems

$$
\begin{align}
q_i^* = \argmax_{q_i \in \mathcal Q} \mathcal L(q_i, x_i), \quad i=1,\dots,n
\end{align}
$$

Remarks:

* Due to the additive structure of dataset ELBO, each $q_i$ can be optimized independently of each other.
* In practice, $\mathcal Q$ is a parametric distribution class, e.g. Gaussian. Therefore, we turn this functional optimization problem into a parameter optimization problem.
* The idea of per-sample surrogate allows very high flexibility. Consider $z\in\mathbb R$ (1D latent space) and $\mathcal Q$ as the set of all univariate Gaussians. Per-sample surrogate assumption allows that each $q_i$ has its own mean and variance, i.e. $q_i(z) = \mathcal N(z; \mu_i, \sigma^2_i)$.
* The drawback of per-sample surrogate is that the number of variational parameters $\{\mu_i, \sigma^2_i\}_{i=1}^n$ grows as dataset becoming large. Poor scalability.

Again, the dataset ELBO also has three popular decompositions

$$
\begin{align}
\mathcal L(q_1, \dots, q_n, D)
&= \log p(D) - \sum_{i=1}^n D_\text{KL}(q_i(z) \parallel p(z \mid x_i)) \\
&= \sum_{i=1}^n \Big\{ \mathbb E_{z \sim q_i} \left[ \log p(x_i \mid z) \right] - D_\text{KL}(q_i(z) \parallel p(z)) \Big\} \\
&= \sum_{i=1}^n \Big\{ \mathbb E_{z \sim q_i} \left[ \log p(x_i, z) \right] + H(q_i) \Big\}
\end{align}
$$

*Proof*: The decomposition of dataset ELBO follows immediately by summing the decomposition equalities of per-sample ELBO:
$$
\begin{align*}
\mathcal L(q_i, x_i)
&= \log p(x_i) - D_\text{KL}(q_i(z) \parallel p(z \mid x_i)) \\
&= \mathbb E_{z \sim q_i} \left[ \log p(x_i \mid z) \right] - D_\text{KL}(q_i(z) \parallel p(z)) \\
&= \mathbb E_{z \sim q_i} \left[ \log p(x_i, z) \right] + H(q_i)
\tag*{$\blacksquare$}
\end{align*}
$$

### Global Surrogate

Under per-sample surrogate settings, the number of surrogate distributions scales linearly w.r.t. the size of the dataset $D$. In practice, the number of variational parameters also grows with the dataset size. With a global surrogate, we make the variational inference scalable.

Essentially, we want to learn the mapping $\mathcal F$
$$
\mathcal F: \mathbb R^d \to \mathcal Q, x \mapsto q(\cdot \mid x)
$$
where $\cdot$ is the placeholder for $z$, s.t. $\forall x_i \in D$
$$
q(\cdot \mid x_i) \approx p(\cdot \mid x_i)
$$
Remarks:

* The abstract mapping $\mathcal F$ maps each data point to a surrogate distribution. Mathematically, it is a complex point-to-function mapping.
* In practice, $\mathcal Q$ is a parameteric distribution class (e.g. univariate Gaussian). Each surrogate distribution is represented by its parameter: $\mu$ and $\sigma$. The point-to-function mapping $\mathcal F$ boils down to a mapping from $x$ to $[\mu, \sigma]^\top$. Since both the input and output are points (aka vectors), we can represent $\mathcal F$ with a neural net.
* Global surrogate is used in amortized variational inference (e.g. VAE). The term "amortized" hightlights the fact that all $x\in\mathbb R^d$ share the mapping rule $\mathcal F: x \mapsto q(\cdot \mid x)$.

For each sample $x_i$, the per-sample ELBO is
$$
\begin{align}
\mathcal L(\mathcal F(x_i), x_i)
&= \mathcal L(q(\cdot \mid x_i), x_i) \\
&= \mathbb E_{z \sim q(\cdot \mid x_i)} \left[ \log\frac{p(x_i, z)}{q(z \mid x_i)} \right]
\end{align}
$$


Summing over all samples, we obtain the dataset ELBO
$$
\begin{align}
\mathcal L(\mathcal F, D)
&\triangleq \sum_{i=1}^n \mathcal L(\mathcal F(x_i), x_i)\\
&= \sum_{i=1}^n \mathbb E_{z \sim q(\cdot \mid x_i)} \left[ \log\frac{p(x_i, z)}{q(z \mid x_i)} \right]
\end{align}
$$
Remarks:

* Not to be confused by the notation: $\mathcal F(x_i) = q(\cdot \mid x_i) \in \mathcal Q$, i.e. $\mathcal F(x_i)$ is a (proability density) function.
* Comparing to per-sample surrogate scheme, there is a key distinction between $q_i(\cdot)$ and $q(\cdot \mid x_i)$:
  * Per-sample surrogate: We choose each $q_i(\cdot) \in \mathcal Q$ freely.
  * Global surrogate: Each $q(\cdot \mid x_i)$ are determined by plugging $x_i$ into the point-to-function mapping $\mathcal F$, which shared by all $x_i \in D$.

To maximize the dataset ELBO, we aim to solve
$$
\begin{align}
\mathcal F^* = \argmax_{\mathcal F} \mathcal L(\mathcal F, D)
\end{align}
$$
This is again a functional optimization problem. In practice, we avoid dealing directly with a functional optimization problem by characterizing $\mathcal F$ with its parameters.

**Example: Global variational Gaussian**  
Consider 1D latent space ($z\in\mathbb R$). A global variational Gaussian chooses $\mathcal Q$ as the set of all univariate Gaussians.
$$
\mathcal Q = \{\mathcal N(\cdot; \mu,\sigma^2) \mid \mu,\sigma\in\mathbb R\}
$$
The goal is to learn the mapping
$$
\mathcal F: \mathbb R^d \to \mathbb R^2, x \mapsto [\mu, \sigma]^\top
$$
Where $\mu$ and $\sigma$ depends on $x$ in a complex way (e.g. neural net), s.t.
$$
q(z \mid x_i) = \left.\mathcal N(z ; \mu_i, \sigma_i)\right|_{[\mu_i, \sigma_i] = \mathcal F(x)} \approx p(z \mid x_i)
$$
In this settings, all $x\in\mathbb R^d$ share the same mapping rule $\mathcal F: x \to [\mu, \sigma]^\top$.
