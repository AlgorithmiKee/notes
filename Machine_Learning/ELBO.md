---
title: "VI & ELBO"
date: "2025"
author: "Ke Zhang"
---

[toc]

$$
\DeclareMathOperator*{\argmax}{argmax}
\DeclareMathOperator*{\argmin}{argmin}
$$

TODO: move BBVI into section VI. move dataset ELBO into section ELBO.

# Variational Inference and Evidence Lower Bound

## Overarching Goal of Variational Inference

Consider an observable random variable $X\in\mathbb R^d$, explained by some latent random variable $Z\in\mathbb R^\ell$. In general, both variables can be high dimensional. Assume we know the joint distribution $p(x,z)$.

For a given instance $x$, we would like to compute the posterior distribution of the latent variable.

$$
p(z \mid x)
= \frac{p(x,z)}{p(x)}
= \frac{p(x,z)}{\int p(x,z) \,\mathrm dz}
$$

Even though we know the joint distribution, the above computation is still intractable in general due to the high dimensional integral in the denominator.

In variational inference, we choose a tractable variational family $\mathcal Q$ and use a variational distribution $q\in\mathcal Q$ to approximate the true posterior $p(z \mid x)$. To assess how well $q$ approximates the true posterior, we minimize the KL divergence:

$$
\begin{align}
\min_{q \in\mathcal Q} D_\text{KL}(q(\cdot) \parallel p(\cdot \mid x))
\end{align}
$$

where $\cdot$ is the place holder for $z$.

Remarks:

* The optimal approximation $q^*$ implicitly depends on $x$. Given another observation $x$, we typically end up with another $q^*$, as illustrated below.
* In practice, $\mathcal Q$ is a parameterized family (e.g. Gaussian). Computing the optimal $q$ is equivalent to computing the optimal parameters.

<img src="./figs/vi_illustration.pdf" alt="elbo maximizer" style="zoom:67%;" />

We have transformed the inference problem — a high-dimensional integration — into an optimization problem. However, minimizing the KL divergence still requires knowledge of the posterior. Next, we will make the above optimization problem tractable.

> **Core idea of variational inference**:  
> Approximate the intractable posterior $p(z \mid x)$ with a tractable $q(z)$ by maximing the [ELBO](#the-evidence-lower-bound).

## The Evidence Lower Bound

For a given instance $x$, we call $\log p(x)$ the ***evidence*** (or log of evidence).

For any variational distribution $q$, it holds that

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

[Later](#elbo-as-evidence-minus-variational-gap), we will show that the best approximation of the true posterior in distribution class $\mathcal Q$ is the maximizer of the ELBO

$$
\begin{align}
q^*
&= \argmax_{q \in\mathcal Q} \mathcal L(q,x) \\
&= \argmin_{q \in\mathcal Q} D_\text{KL}(q(\cdot) \parallel p(\cdot \mid x))
\end{align}
$$

However, by definition, the ELBO is an expectation w.r.t. $q$ which again requires integrating in latent space. This issue can be addressed in two ways:

* In traditional variational inference (not covered here): the ELBO can be expressed in closed form if we restrict to the exponential family and apply conjugacy. $\to$ Not suitable for modeling complex distributions. ❌
* In ***black-box variatioanl inferece (BBVI)***, the ELBO is estimated by **Monte Carlo (MC)** sampling rather than computed exactly.

MC estimation of ELBO:

$$
\begin{align}
\tilde{\mathcal L}(q,x)
&= \frac{1}{M} \sum_{k=1}^M \log\frac{p(x, z^{(k)})}{q(z^{(k)})}, \quad z^{(k)} \sim q
\end{align}
$$

### Equivalent Reformulation of ELBO

There are three equivalent reformulations of the ELBO $\mathcal L(q,x)$. Each reformulations provides insights from a different perspective.

$$
\begin{align*}
\mathcal L(q,x)
&\triangleq \mathbb E_{z \sim q} \left[ \log\frac{p(x,z)}{q(z)} \right] \\
&= \log p(x) - D_\text{KL}(q(\cdot) \parallel p(\cdot \mid x)) \\
&= \mathbb E_{z \sim q} \Big[ \log p(x \mid z) \Big] - D_\text{KL}(q(z) \parallel p(z)) \\
&= \mathbb E_{z \sim q} \Big[ \log p(x,z) \Big] + H(q)
\end{align*}
$$

#### ELBO as evidence minus variational gap

The 1st reformulation of ELBO is

$$
\begin{align}
\mathcal L(q,x)
&= \log p(x) - D_\text{KL}(q(\cdot) \parallel p(\cdot \mid x))
\end{align}
$$

or equivalently

$$
\begin{align}
\underbrace{\log p(x)}_\text{evidence}
&= \underbrace{\mathcal L(q,x)}_\text{ELBO} + \underbrace{D_\text{KL}(q(\cdot) \parallel p(\cdot \mid x))}_\text{variational gap}
\end{align}
$$

Remarks:

* The gap between the evidence and ELBO is exactly the KL divergence we want to minimize earlier (also known as ***variational gap***). Minimizing the variational gap is equivalent to maximizing the ELBO, which captures the core idea of variational inference.
* The ELBO becomes tight (or maximized) iff $q(z) = p(z \mid x)$, which is typically not achievable in practice due to the limited expressiveness of the distribution class $\mathcal Q$.

*Proof*: Substitute $p(x,z) = p(z \mid x) \cdot p(x)$ into the definition of ELBO, we conclude

$$
\begin{align*}
\mathcal L(q,x)
&= \mathbb E_{z \sim q} \left[ \log\frac{p(z \mid x) \cdot p(x)}{q(z)} \right] \\
&= \mathbb E_{z \sim q} \left[ \log\frac{p(z \mid x)}{q(z)} + \log p(x) \right] \\
&= \underbrace{\mathbb E_{z \sim q} \left[ \log\frac{p(z \mid x)}{q(z)} \right]}_{- D_\text{KL}(q(\cdot) \parallel p(\cdot \mid x))} + \log p(x)
\tag*{$\blacksquare$}
\end{align*}
$$

Therefore, the best approximation $p(z \mid x)$ in $\mathcal Q$ is the solution of

$$
\begin{align}
\max_{q \in\mathcal Q}
\mathcal L(q,x) = \mathbb E_{z \sim q} \left[ \log\frac{p(x,z)}{q(z)} \right]
\end{align}
$$

In practice, this optimization problem is usually solved by parameterizing $q$ and applying gradient methods (instead of applying calculus of variations).

#### ELBO as regularized reconstruction

The 2nd reformulation of ELBO is

$$
\begin{align}
\mathcal L(q,x)
&= \mathbb E_{z \sim q} \left[ \log p(x \mid z) \right] - D_\text{KL}(q(z) \parallel p(z))
\end{align}
$$

Remarks:

* The 1st term is expected log likelihood w.r.t. the variational distribution. It measures the average goodness of reconstruction, assuming that $z \sim q$.
* The 2nd term is the KL divergence of the variational distribution $q(z)$ w.r.t. the prior $p(z)$. i.e. We penalize those variational distributions that significantly deviate from the prior. $\to$ regularization effect.
* Maximizing the ELBO is a trade-off between maximizing the reconstruction fidelity and keeping variational distribution close to the prior. This is the key idea behind variational autoencoders (VAEs).

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

#### ELBO as entropy minus free energy

The 3rd reformulation of ELBO is

$$
\begin{align}
\mathcal L(q,x)
&= \mathbb E_{z \sim q} \left[ \log p(x,z) \right] + H(q)
\end{align}
$$

Remarks:

* The 1st term is known as negative free energy in statistical physics. It rewards $q$ that explain the data well.
* The 2nd term is the entropy of the variational distribution. It rewards $q$ with higher uncertainty.
* Maximizing the ELBO can be interpreted as minimizing the free energy while maintaining high entropy in the variational distribution.

*Proof*: This reformulation follows directly from the definition of the ELBO:

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

## Black-Box Variational Inference

We use multivariate Gaussian as the variational distribution

$$
\begin{align*}
q(z) &= \mathcal N(z ; \mu, \Sigma)
\end{align*}
$$

The corresponding distribution class $\mathcal Q$ is

$$
\begin{align*}
\mathcal Q
&= \{\mathcal N(\mu, \Sigma) \mid
  \mu\in\mathbb R^\ell, \Sigma\in\mathbb R^{\ell \times \ell}, \Sigma \text{ is s.p.d.}
\}
\end{align*}
$$

Remarks:

* "s.p.d." is short for *symmetric positive definite*.
* Alternatively, we can shrink the distribution class $\mathcal Q$ by restricting it to diagonal Gaussian (also known as ***mean-field*** Gaussian) or even spherical Gaussian (also known as ***isotrophic*** Gaussian).

Each variational distribution $q$ is represented by its parameters $(\mu, \Sigma)$. The ELBO, previously defined as a functional of $q$, now becomes a function of $(\mu, \Sigma)$.

$$
\begin{align}
\mathcal L(\mu, \Sigma,x)
&= \mathbb E_{z \sim \mathcal N(\mu, \Sigma)} \left[ \log\frac{p(x,z)}{\mathcal N(z ; \mu, \Sigma)} \right]
\end{align}
$$

or equivalently using the [3rd ELBO reformulation](#elbo-as-entropy-minus-free-energy)

$$
\begin{align}
\mathcal L(\mu, \Sigma,x)
&= \mathbb E_{z \sim \mathcal N(\mu, \Sigma)} \left[ \log p(x,z) \right] + H(\mathcal N(\mu, \Sigma))
\end{align}
$$

The entropy of multivariate Gaussian has closed-form solution.

$$
\begin{align*}
H(\mathcal N(\mu, \Sigma))
&= \frac{1}{2} \log\vert\Sigma\vert + \underbrace{\frac{\ell}{2}\log(2\pi e)}_{\text{const.}}
\end{align*}
$$

The ELBO thus becomes

$$
\begin{align}
\mathcal L(\mu, \Sigma,x)
&= \mathbb E_{z \sim \mathcal N(\mu, \Sigma)} \left[ \log p(x,z) \right] + \frac{1}{2} \log\vert\Sigma\vert + \text{const.}
\end{align}
$$

The best Gaussian variational distribution is obtained by maximizing the ELBO, or equivalently

$$
\begin{align}
\max_{\mu, \Sigma}
\mathbb E_{z \sim \mathcal N(\mu, \Sigma)} \left[ \log p(x,z) \right] + \frac{1}{2} \log\vert\Sigma\vert
\end{align}
$$

Note that the expectation is taken w.r.t. $z \sim \mathcal N(\mu, \Sigma)$, which depends on the optimization variable. Hence, we need reparameterization trick by expressing $z$ as a deterministic transformation of a standard Gaussian:

$$
z = \mu + L\epsilon
$$

where $\epsilon \sim \mathcal N(0, I)$ and $L$ is the Cholesky factor of $\Sigma$. (i.e. $\Sigma = LL^\top$)

The optimization problem then becomes

$$
\begin{align}
\max_{\mu, L}
\mathbb E_{\epsilon \sim \mathcal N(0, I)} \left[ \log p(x,z) \mid_{z = \mu + L\epsilon} \right] + \log\vert L \vert
\end{align}
$$

Remarks:

* After reparameterization, we treat the Cholesky factor $L$ as the optimization variable instead of $\Sigma$.

The objective function can be approximated by Monte Carlo sampling. Hence, we solve

$$
\begin{align}
\max_{\mu, L}
\frac{1}{M} \sum_{k=1}^M \log p(x, z^{(k)}) \mid_{z^{(k)} = \mu + L\epsilon^{(k)}} + \log\vert L \vert
\quad \text{ where } \epsilon^{(k)} \sim \mathcal N(0, I)
\end{align}
$$

## ELBO for a Dataset

[Previously](#the-evidence-lower-bound), we derived the ELBO $\mathcal L(q,x)$ for a single observation $x$. From now on, let's call it **per-sample** ELBO (or **per-observation** ELBO).

**Question**: What if we have a dataset consisting of multiple iid observations? Can we lower-bound the evidence of the whole dataset?

Problem formulation:

* Known: generative model $p(x,z) = p(z) \, p(x \mid z)$.
* Given: training data $D = \{ x_1, \cdots,  x_n\} \stackrel{\text{iid}}{\sim} p(x) = \int_z p(x,z) \:\mathrm dz$.
* Select: variational family $\mathcal Q$.
* Goal: derive a lower bound on $\log p(D)$.

We refer to the lower bound on $\log p(D)$ as the **dataset ELBO**. In fact, dataset ELBO does exist since

$$
\begin{align}
\log p(D) = \log \prod_{i=1}^n p(x_i) = \sum_{i=1}^n \log p(x_i)
\end{align}
$$

Each $\log p(x_i)$ can be lower bounded by its individual per-sample ELBO. Therefore, $\log p(D)$ can also be lower bounded. The remaining question is how to design the variational distribution for each $x_i$.

### Local Variational Distribution

A natural extension of per-sample ELBO to dataset ELBO is choosing variational distribution independently for each observation. Formally:

For each $x_i$, we choose $q_i \in\mathcal Q$ independently to approximate the true posterior $p(z \mid x_i)$. This gives per-sample ELBO

$$
\begin{align}
\mathcal L(q_i, x_i)
&= \sum_{i=1}^n \mathbb E_{z \sim q_i} \left[ \log\frac{p(x_i, z)}{q_i(z)} \right]
\end{align}
$$

For any combination of variational distributions $q_1, \dots, q_n \in \mathcal Q$, it holds that

$$
\begin{align*}
\log p(D)
= \sum_{i=1}^n \log p(x_i)
\ge \underbrace{\sum_{i=1}^n \mathcal L(q_i, x_i)}_{\mathcal L(q_1, \dots, q_n, D)}
\end{align*}
$$

Hence, we obtain the dataset ELBO:

$$
\begin{align}
\mathcal L(q_1, \dots, q_n, D)
\triangleq \sum_{i=1}^n \mathcal L(q_i, x_i)
= \sum_{i=1}^n \mathbb E_{z \sim q_i} \left[ \log\frac{p(x_i, z)}{q_i(z)} \right]
\end{align}
$$

Remarks:

* The (dataset) ELBO is a functional of $q_1, \dots, q_n$, which are **freely** chosen.
* We assume that $q_1, \dots, q_n \in \mathcal Q$. i.e. All variational distributions belong to the same distribution class.
* Local variational distributions are used in classical variational inference.

The optimal variational distributions are obtained by solving the functional optimization problems

$$
\begin{align}
q_i^* = \argmax_{q_i \in \mathcal Q} \mathcal L(q_i, x_i), \quad i=1,\dots,n
\end{align}
$$

Remarks:

* Due to the additive structure of dataset ELBO, each $q_i$ can be optimized independently of each other.
* In practice, $\mathcal Q$ is a parametric distribution class, e.g. Gaussian. Therefore, we turn this functional optimization problem into a parameter optimization problem.

Drawbacks:

1. **Poor scalability**: The number of optimization problems scales linearly with the size of the dataset. If we have $n$ observations, we have solve $n$ independent optimization problems.
1. **No generalization**: Given a new observation $x_*$, we have to solve the optimization problem again. We cannot forge the variational distribution for $x_*$ from $q_1^*, \dots, q_n^*$.

### Global Inference Model

Instead of learning $q_i$ for each $x_i$ individually, we learn a **global inference model**, conceptually defined as a mapping $f$

$$
f: \mathbb R^d \to \mathcal Q, x \mapsto q(\cdot \mid x)
$$

where $\cdot$ is the placeholder for $z$, such that $\forall x \in \mathbb R^d$

$$
q(\cdot \mid x) \approx p(\cdot \mid x)
$$

Remarks:

* The abstract mapping $f$ maps each data point to a variational distribution. Mathematically, it is a complex point-to-function mapping.
* ⚠️ To reduce visual clutter, we write $f(x) = q(\cdot \mid x)$ rather than $f(x) = q_{f}(\cdot \mid x)$
* The globalness hightlights the fact that $f$ is shared by all $x\in\mathbb R^d$.
* Once we learned such $f$ on training data $D$, not only can we plug in $x_i$ and use $q(\cdot \mid x_i) \approx p(\cdot \mid x_i)$, but also we can plug in any unseen $x_*$ and get $q(\cdot \mid x_*) \approx p(\cdot \mid x_*)$.

For each sample $x_i$, the per-sample ELBO is

$$
\begin{align}
\mathcal L(f(x_i), x_i)
&= \mathcal L(q(\cdot \mid x_i), x_i) \\
&= \mathbb E_{z \sim q(\cdot \mid x_i)} \left[ \log\frac{p(x_i, z)}{q(z \mid x_i)} \right]
\end{align}
$$

Summing over all samples, we obtain the dataset ELBO

$$
\begin{align}
\mathcal L(f, D)
&\triangleq \sum_{i=1}^n \mathcal L(f(x_i), x_i)\\
&= \sum_{i=1}^n \mathbb E_{z \sim q(\cdot \mid x_i)} \left[ \log\frac{p(x_i, z)}{q(z \mid x_i)} \right]
\end{align}
$$

Remarks:

* Not to be confused by the notation: $f(x_i) = q(\cdot \mid x_i) \in \mathcal Q$, i.e. $f(x_i)$ is a (probability density) function.
* Comparing to local variational distribution scheme, there is a key distinction between $q_i(\cdot)$ and $q(\cdot \mid x_i)$:
  * Local variational distribution: We choose each $q_i(\cdot) \in \mathcal Q$ freely.
  * Global variational distribution: Each $q(\cdot \mid x_i)$ are determined by plugging $x_i$ into the point-to-function mapping $f$, which shared by all $x_i \in D$.

To maximize the dataset ELBO, we aim to solve

$$
\begin{align}
f^* = \argmax_{f} \mathcal L(f, D)
\end{align}
$$

This is again a functional optimization problem. In practice, we avoid dealing directly with a functional optimization problem by

1. using parametric family $\mathcal Q$, e.g. multivariate Gaussian with parameter $(\mu, \Sigma)$
1. designing $f$ as a neural net with $x$ as its input layer, and $(\mu, \Sigma)$ as its output layer.

Learning $f$ boils down to training such a neual net.

## Variational Inference

Unless otherwise specified, we use a **Gaussian variational distribution** to approximate the true posterior.

Previously, we have seen [variational inference for a single observation](#elbo-maximization-for-gaussian-variational distribution). Now, we extend variational inference to the case of multiple observations.

Problem formulation:

* Known: generative model $p(x,z) = p(z) \, p(x \mid z)$
* Given: training data $D = \{ x_1, \cdots,  x_n\} \stackrel{\text{iid}}{\sim} p(x) = \int_z p(x,z) \:\mathrm dz$.
* Select: Gaussian variational family $\mathcal Q = \{ \mathcal N(\mu, \Sigma) \}$.
* Goal: maximize the dataset ELBO

In the following, we will consider local variational distribution scheme and global variational distribution scheme. The dataset ELBO, previously defined as a functional over variational distributions, will be reformulated as a scalar-valued function of parameter vectors.

### Classical Variational Inference

For each observation $x_i$, we use $\mathcal N(z; \mu_i, \Sigma_i)$ to approximate the true posterior $p(z \mid x_i)$.

* Classical: We use local variational distributions, i.e. we choose $\mu_i,\Sigma_i$ independently for each $x_i$.

The dataset ELBO, previously as a functional of $\{q_i\}_{i=1}^n$, now becomes a function of $\{\mu_i, \Sigma_i\}_{i=1}^n$:

$$
\begin{align}
\mathcal L(\mu_1, \Sigma_1, \dots, \mu_n, \Sigma_n, D)
&\triangleq \sum_{i=1}^n \mathcal L(\mu_i, \Sigma_i, x_i) \\
&= \sum_{i=1}^n \mathbb E_{z \sim \mathcal N(\mu_i, \Sigma_i)} \left[ \log\frac{p(x_i, z)}{\mathcal N(z; \mu_i, \Sigma_i)} \right] \\
&= \sum_{i=1}^n \mathbb E_{z \sim \mathcal N(\mu_i, \Sigma_i)} \left[ \log p(x_i, z) \right] + H(\mathcal N(\mu_i, \Sigma_i)) + \text{const} \\
&= \sum_{i=1}^n \mathbb E_{z \sim \mathcal N(\mu_i, \Sigma_i)} \left[ \log p(x_i, z) \right] + \frac{1}{2} \log\vert\Sigma_i\vert + \text{const}
\end{align}
$$

The dataset ELBO can be maximized sample-wise as follows

$$
\begin{align}
\forall x_i \in D: \quad
\max_{\mu_i, \Sigma_i} \mathbb E_{z \sim \mathcal N(\mu_i, \Sigma_i)} \left[ \log p(x_i, z) \right] + \frac{1}{2} \log\vert\Sigma_i\vert
\end{align}
$$

Again, we apply reparameterization trick to allow Monte Carlo estimation of the objective

$$
\begin{align}
z
&= \mu_i + L_i \epsilon, \quad \epsilon \sim \mathcal N(0,I)
\\
\max_{\mu_i, L_i}\:
&\mathbb E_{\epsilon \sim \mathcal N(0, I)} \left[ \log p(x_i, \mu_i + L_i \epsilon) \right] + \log\vert L_i \vert
\\
\max_{\mu_i, L_i}\:
&\frac{1}{M} \sum_{k=1}^M \log p(x_i, z^{(k)}) |_{z^{(k)} = \mu_i + L_i\epsilon^{(k)}} + \log\vert L_i \vert
\quad \text{ where } \epsilon^{(k)} \sim \mathcal N(0, I)
\end{align}
$$

The complete algorithm is summarized below:

---
**Algorithm: classical variational inference with Gaussian variational distributions**  
**Input**: $x_1, \dots, x_n \in\mathbb R^d$  
**Output**: $\mu_1,\dots,\mu_n\in\mathbb R^\ell, \Sigma_1,\dots,\Sigma_n\in\mathbb R^{\ell \times \ell}$  
**Goal**: use $\mathcal N(z; \mu_i,\Sigma_i)$ to approximate $p(z \mid x_i)$

For each $i=1,\dots,n$: do  
$\quad$ Init $\mu_i \in \mathbb R^\ell$ and $L_i \in \mathbb R^{\ell \times \ell}$  
$\quad$ While the SGD for $\mu_i$ and $L_i$ is not converged: do  
$\qquad$ Sample a mini-batch $\epsilon^{(1)}, \dots, \epsilon^{(M)} \sim \mathcal N(0, I_{\ell})$  
$\qquad$ Compute the objective $\mathcal L(\mu_i, L_i)$ and its gradient

$$
\mathcal L(\mu_i, L_i) \triangleq
\frac{1}{M} \sum_{k=1}^M \log p(x_i, z^{(k)}) + \log\vert L_i \vert,
\quad z^{(k)} = \mu_i + L_i \epsilon^{(k)}
$$

$\qquad$ Update $\mu_i$ and $L_i$

$$
\begin{align*}
\mu_i &\leftarrow \mu_i + \eta_t \nabla_{\mu_i} \mathcal L(\mu_i, L_i) \\
L_i   &\leftarrow L_i + \eta_t \nabla_{L_i} \mathcal L(\mu_i, L_i) \\
\end{align*}
$$

$\quad$ Set $\Sigma_i = L_iL_i^\top$

return $\mu_1,\dots,\mu_n, \Sigma_1,\dots,\Sigma_n$

---

Remarks:

* This approach is essentially performing BBVI for each observation.
* The total \# parameters to be learned is $O(n\ell^2)$, which scales linearly with the dataset size $n$. This limits scalability for large datasets.
* Local variational distributions provide high flexibility, as each variational distribution's mean and covariance are learned independently.

### Amortized Variational Inference

For each observation $x_i$, we again use $\mathcal N(z; \mu_i, \Sigma_i)$ to approximate the true posterior $p(z \mid x_i)$, but now:

* Amortized: The mapping rule $f: x_i \mapsto (\mu_i, \Sigma_i)$ is now shared by all $x_i\in D$. Instead of learning each $(\mu_i, \Sigma_i)$ individually, we learn the shared function $f$.

Basic idea of amortized variational inference:

$$
x \longrightarrow \boxed{ \text{Neural Net } f_\phi \vphantom{\int} } \longrightarrow
\begin{bmatrix} \mu_\phi(x) \\ \Sigma_\phi(x) \end{bmatrix}
\longrightarrow q_\phi(z \mid x) \approx p(z \mid x)
$$

The mapping $f$ is typically implemented as a neural net (NN) parameterized by $\phi$:

$$
f_\phi: \mathbb R^d \to \mathbb R^\ell \times \mathbb R^{\ell\times\ell}, x \mapsto (\mu_\phi(x), \Sigma_\phi(x))
$$

The resulting variational distribution for each $x\in\mathbb R^d$ becomes

$$
q_\phi(z \mid x) = \mathcal N(z ; \mu_\phi(x), \Sigma_\phi(x))
$$

Remarks:

* The output $(\mu_\phi(x), \Sigma_\phi(x))$ of the NN depends on both the observation $x$ and the network parameter $\phi$.
* In practice, the output of the NN is $(\mu_\phi(x), L_\phi(x))$ where $L_\phi(x)$ is the Cholesky factor of $\Sigma_\phi(x)$. This design later helps letting the gradient flow from ELBO to $x$ by reparameterization trick.

Goal: Train the NN (aka learn $\phi$) so that

$$
q_\phi(z \mid x) \approx p(z \mid x), \forall x \in D
$$

To achieve this goal, we need to maximize the ELBO. Previously, the ELBOs are defined as an abstract functional of $f$. Now, they become a function of $\phi$.

For each $x \in D$, the per-sample ELBO is

$$
\begin{align}
\mathcal L(\phi, x)
&= \mathbb E_{z \sim q_\phi(\cdot \mid x)} \left[ \log\frac{p(x, z)}{q_\phi(z \mid x)} \right]
\\
&= \mathbb E_{z \sim q_\phi(\cdot \mid x)} \left[ \log p(x,z) \right] + H(q_\phi(\cdot \mid x))
\\
&= \mathbb E_{\epsilon \sim \mathcal N(0, I)} \left[ \log p(x,z) \vert_{z = \mu_\phi(x) + L_\phi(x)\epsilon} \right] + \log\vert L_\phi(x) \vert
\end{align}
$$

where the last step follows from:

* Entropy of multivariate Gaussian
  $$
  H(q_\phi(\cdot \mid x)) = \log\vert L_\phi(x) \vert + \text{const}
  $$

* Reparameterization trick

  $$
  \begin{align*}
  z = \mu_\phi(x) + L_\phi(x) \cdot \epsilon, \quad
  \epsilon \sim \mathcal N(0,I), \quad
  \Sigma_\phi(x) = L_\phi(x) L_\phi(x)^\top
  \end{align*}
  $$

The dataset ELBO becomes

$$
\begin{align}
\mathcal L(\phi, D)
&= \sum_{x\in D} \mathcal L(\phi, x) \\
&= \sum_{x\in D} \mathbb E_{z \sim q_\phi(\cdot \mid x)} \left[ \log\frac{p(x, z)}{q_\phi(z \mid x)} \right] \\
&= \sum_{x\in D} \mathbb E_{\epsilon \sim \mathcal N(0, I)} \left[ \log p(x,z)  \right] + \log\vert L_\phi(x) \vert,
\quad z = \mu_\phi(x) + L_\phi(x) \epsilon
\\
&= |D| \cdot \sum_{x\in D} \frac{1}{|D|} \left[\mathbb E_{\epsilon \sim \mathcal N(0, I)} \left[ \log p(x,z) \right] + \log\vert L_\phi(x) \vert \right],
\quad z = \mu_\phi(x) + L_\phi(x) \epsilon
\nonumber \\
&= |D| \cdot \mathbb E_{x \sim \mathrm{Unif}(D)} \left[\mathbb E_{\epsilon \sim \mathcal N(0, I)} \left[ \log p(x,z) \right] + \log\vert L_\phi(x) \vert \right],
\quad z = \mu_\phi(x) + L_\phi(x) \epsilon
\end{align}
$$

Now, we apply MC estimation for $\mathcal L(\phi, D)$ and its gradient

$$
\begin{align}
\mathcal L(\phi, D)
&\approx \frac{|D|}{|B|} \sum_{x \in B} \left[\frac{1}{m} \sum_{k=1}^m \log p(x,z^{(k)}) + \log\vert L_\phi(x) \vert \right]
\\
\nabla_\phi \mathcal L(\phi, D)
&\approx \frac{|D|}{|B|} \sum_{x \in B} \left[\frac{1}{m} \sum_{k=1}^m \nabla_\phi \log p(x,z^{(k)}) + \nabla_\phi \log\vert L_\phi(x) \vert \right]
\end{align}
$$

where

* $B$ is a mini-batch sampled from the whole dataset: $B \subseteq D = \{x_1, \dots, x_n\}$
* $\epsilon^{(k)} \stackrel{\text{iid}}{\sim} \mathcal N(0,I), \quad k=1,\dots,m$
* $z^{(k)} = \mu_\phi(x) + L_\phi(x) \epsilon^{(k)}, \quad k=1,\dots,m$

The complete algorith of amortized variatioal inference is summarized below:

---

**Algorithm: amortized variational inference with Gaussian variational distributions**  
**Input**: $D = \{x_1, \dots, x_n \in\mathbb R^d\}$  
**Output**: $\phi$  
**Goal**: train a $\mathrm{NN}_\phi: x \mapsto (\mu_\phi(x),\Sigma_\phi(x))$ s.t. $\mathcal N(z; \mu_\phi(x),\Sigma_\phi(x)) \approx p(z \mid x)$

While SGD for $\phi$ is not converged: do  
$\quad$ Sample a mini-batch: $B \subseteq D$  
$\quad$ For each $x\in B$: do  
$\qquad$ Forward-pass: compute $\mu_\phi(x), L_\phi(x)$  
$\qquad$ For $k = 1,\dots,m$: do  
$\qquad\quad$ Sampling: $\epsilon^{(k)} \stackrel{\text{iid}}{\sim} \mathcal N(0,I)$  
$\qquad\quad$ Reparamterization: $z^{(k)} = \mu_\phi(x) + L_\phi(x) \epsilon^{(k)}$  

$\qquad$ Compute the per-sample ELBO:

$$
\begin{align*}
\mathcal L(\phi, x)
&= \frac{1}{m} \sum_{k=1}^m \log p(x,z^{(k)}) + \log\vert L_\phi(x) \vert
\end{align*}
$$

$\quad$ Compute the dataset ELBO:

$$
\begin{align*}
\mathcal L(\phi, B)
&= \frac{|D|}{|B|} \sum_{x \in B} \mathcal L(\phi, x)
\end{align*}
$$

$\quad$ Backward-pass: compute the gradient $\nabla_\phi \mathcal L(\phi, B)$  
$\quad$ Update: $\phi \leftarrow \phi + \eta_t \nabla_\phi \mathcal L(\phi, B)$

Return $\phi$

---

Remarks:

* The \# parameters is now fully determined by the architecture of NN (or the \# scalars in $\phi$). No longer scales up with the dataset size.
* Once, we trained the NN. We can compute $q_\phi(z \mid x_*)$ for unseen data $x_*$ by simply performing a forward pass. This allows us effortless generalization for inference. In contrast, classical VI has no generalizaiton ability.

## Appendix

### Entropy of Gaussian

For a multivariate Gaussian $p(x) = \mathcal N(x ; \mu, \Sigma), \, x \in \mathbb R^d$, the differential entropy is:

$$
\begin{align}
H(p)
&= -\int p(x) \log p(x) \, dx \\
&= \frac{1}{2} \log \left[ (2\pi e)^d \det(\Sigma) \right] \\
&= \frac{1}{2} \log \left[ \det(\Sigma) \right] + \frac{d}{2} \log (2\pi e) \\
\end{align}
$$

Let $L$ (lower triangular) be the Cholesky factor of the covariance matrix, i.e. $\Sigma = L L^\top$. Then,

$$
\begin{align*}
\log \left[ \det(\Sigma) \right]
&= \log \left[ \det(L L^\top) \right] \\
&= \log \left[ \det(L) \cdot \det(L^\top) \right] \\
&= \log \left[ \det(L)^2 \right] \\
&= 2\log \left[ \det(L) \right] \\
\end{align*}
$$

Therefore, We can express $H(p)$ as

$$
\begin{align}
H(p)
&= \log \left[ \det(L) \right] + \frac{d}{2} \log (2\pi e) \\
\end{align}
$$
