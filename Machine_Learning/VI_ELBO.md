---
title: "VI & ELBO"
date: "2025"
author: "Ke Zhang"
---
# Variational Inference and Evidence Lower Bound

[toc]

$$
\DeclareMathOperator*{\argmax}{argmax}
\DeclareMathOperator*{\argmin}{argmin}
$$

## Motivation

Consider an observable random variable $\mathbf{x}\in\mathbb R^d$, explained by some latent random variable $\mathbf{z}\in\mathbb R^\ell$. In general, both variables can be high dimensional. Assume we know the joint distribution $p(\mathbf{x},\mathbf{z})$.

For a given instance $\mathbf{x}$, we would like to compute the posterior distribution of the latent variable.

$$
p(\mathbf{z} \mid \mathbf{x})
= \frac{p(\mathbf{x},\mathbf{z})}{p(\mathbf{x})}
= \frac{p(\mathbf{x},\mathbf{z})}{\int p(\mathbf{x},\mathbf{z}) \,\mathrm dz}
$$

Even though we know the joint distribution, the above computation is still intractable in general due to the high dimensional integral in the denominator.

In variational inference, we choose a tractable variational family $\mathcal Q$ and use a variational distribution $q\in\mathcal Q$ to approximate the true posterior $p(\mathbf{z} \mid \mathbf{x})$. To assess how well $q$ approximates the true posterior, we minimize the KL divergence:

$$
\begin{align}
\min_{q \in\mathcal Q} D_\text{KL}(q(\cdot) \parallel p(\cdot \mid \mathbf{x}))
\end{align}
$$

where $\cdot$ is the place holder for $\mathbf{z}$.

Remarks:

* The optimal approximation $q^*$ implicitly depends on $\mathbf{x}$. Given another observation $\mathbf{x}$, we typically end up with another $q^*$, as illustrated below.
* In practice, $\mathcal Q$ is a parameterized family (e.g. Gaussian). Computing the optimal $q$ is equivalent to computing the optimal parameters.

<img src="./figs/vi_illustration.pdf" alt="elbo maximizer" style="zoom:67%;" />

We have transformed the inference problem — a high-dimensional integration — into an optimization problem. However, minimizing the KL divergence still requires knowledge of the posterior. Next, we will make the above optimization problem tractable.

> **Core idea of variational inference**:  
> Approximate the intractable posterior $p(\mathbf{z} \mid \mathbf{x})$ with a tractable $q(\mathbf{z})$ by maximing the [evidence lower bound](#the-evidence-lower-bound).

## The Evidence Lower Bound

For a given instance $\mathbf{x}$, we call $\log p(\mathbf{x})$ the ***evidence*** (or log of evidence).

For any variational distribution $q$, it holds that

$$
\begin{align}
\log p(\mathbf{x})
\ge \mathbb E_{\mathbf{z} \sim q} \left[ \log\frac{p(\mathbf{x},\mathbf{z})}{q(\mathbf{z})} \right]
\end{align}
$$

where the RHS is called ***evidence lower bound*** (or ***ELBO***), denoted by $\mathcal L(q,\mathbf{x})$:

$$
\begin{align}
\mathcal L(q,\mathbf{x})
\triangleq \mathbb E_{\mathbf{z} \sim q} \left[ \log\frac{p(\mathbf{x},\mathbf{z})}{q(\mathbf{z})} \right]
\end{align}
$$

Remarks:

* Other common notations for ELBO: $\mathcal L(q)$, $\mathcal L$ or simply $\mathrm{ELBO}$.
* For a fixed $\mathbf{x}$, the ELBO is a functional of $q$. The higher the ELBO, the better $q$ approximates the true posterior.
* The ELBO also depends on the observation $\mathbf{x}$. For a fixed $q$, evaluating ELBO at another $\mathbf{x}'$ will give another bound (for another posterior $p(\mathbf{z} \mid \mathbf{x}')$).

*Proof*: First, we express $p(\mathbf{x})$ as
$$
\begin{align*}
p(\mathbf{x})
= \int p(\mathbf{x},\mathbf{z}) \,\mathrm dz
= \int q(\mathbf{z}) \frac{p(\mathbf{x},\mathbf{z})}{q(\mathbf{z})} \,\mathrm dz
= \mathbb E_{\mathbf{z} \sim q} \left[ \frac{p(\mathbf{x},\mathbf{z})}{q(\mathbf{z})}\right]
\end{align*}
$$

Taking the log and applying Jensen's inequality, we conclude

$$
\begin{align*}
\log p(\mathbf{x})
&= \log \mathbb E_{\mathbf{z} \sim q} \left[ \frac{p(\mathbf{x},\mathbf{z})}{q(\mathbf{z})}\right] \\
&\ge \underbrace{\mathbb E_{\mathbf{z} \sim q} \left[ \log \frac{p(\mathbf{x},\mathbf{z})}{q(\mathbf{z})}\right]}_{\mathcal L(q,\mathbf{x})}
\tag*{$\blacksquare$}
\end{align*}
$$

[Later](#elbo-as-evidence-minus-variational-gap), we will show that the best approximation of the true posterior in variational family $\mathcal Q$ is the maximizer of the ELBO

$$
\begin{align}
q^*
&= \argmax_{q \in\mathcal Q} \mathcal L(q,\mathbf{x}) \\
&= \argmin_{q \in\mathcal Q} D_\text{KL}(q(\cdot) \parallel p(\cdot \mid \mathbf{x}))
\end{align}
$$

However, by definition, the ELBO is an expectation w.r.t. $q$ which again requires integrating in latent space. This issue can be addressed in two ways:

* In traditional variational inference (not covered here): the ELBO can be expressed in closed form if we restrict to the exponential family and apply conjugacy. $\to$ Not suitable for modeling complex distributions. ❌
* In ***black-box variatioanl inferece (BBVI)***, the ELBO is estimated by **Monte Carlo (MC)** sampling rather than computed exactly.

MC estimation of ELBO:

$$
\begin{align}
\tilde{\mathcal L}(q,\mathbf{x})
&= \frac{1}{M} \sum_{k=1}^M \log\frac{p(\mathbf{x}, \mathbf{z}^{(k)})}{q(\mathbf{z}^{(k)})}, \quad \mathbf{z}^{(k)} \sim q
\end{align}
$$

Remarks:

* This illustrates MC estimation of the **entire** ELBO.
* Later, we will see different reformulation of the ELBO. When the variational distribution $q$ is Gaussian, certain terms of ELBO can be computed in closed-form (entropy of $q$, KL divergence to Gaussian prior). Therefore, we only need to apply MC to estimate the remaining terms.

There are three equivalent reformulations of the ELBO $\mathcal L(q,\mathbf{x})$. Each reformulations provides insights from a different perspective.

$$
\begin{align*}
\mathcal L(q,\mathbf{x})
&\triangleq \mathbb E_{\mathbf{z} \sim q} \left[ \log\frac{p(\mathbf{x},\mathbf{z})}{q(\mathbf{z})} \right] \\
&= \log p(\mathbf{x}) - D_\text{KL}(q(\cdot) \parallel p(\cdot \mid \mathbf{x})) \\
&= \mathbb E_{\mathbf{z} \sim q} \Big[ \log p(\mathbf{x} \mid \mathbf{z}) \Big] - D_\text{KL}(q(\mathbf{z}) \parallel p(\mathbf{z})) \\
&= \mathbb E_{\mathbf{z} \sim q} \Big[ \log p(\mathbf{x},\mathbf{z}) \Big] + H(q)
\end{align*}
$$

### ELBO as evidence minus variational gap

The 1st reformulation of ELBO is

$$
\begin{align}
\mathcal L(q,\mathbf{x})
&= \log p(\mathbf{x}) - D_\text{KL}(q(\cdot) \parallel p(\cdot \mid \mathbf{x}))
\end{align}
$$

or equivalently

$$
\begin{align}
\underbrace{\log p(\mathbf{x})}_\text{evidence}
&= \underbrace{\mathcal L(q,\mathbf{x})}_\text{ELBO} + \underbrace{D_\text{KL}(q(\cdot) \parallel p(\cdot \mid \mathbf{x}))}_\text{variational gap}
\end{align}
$$

Remarks:

* The gap between the evidence and ELBO is exactly the KL divergence we want to minimize earlier (also known as ***variational gap***). Minimizing the variational gap is equivalent to maximizing the ELBO, which captures the core idea of variational inference.
* The ELBO becomes tight (or maximized) iff $q(\mathbf{z}) = p(\mathbf{z} \mid \mathbf{x})$, which is typically not achievable in practice due to the limited expressiveness of the variational family $\mathcal Q$.

*Proof*: Substitute $p(\mathbf{x},\mathbf{z}) = p(\mathbf{z} \mid \mathbf{x}) \cdot p(\mathbf{x})$ into the definition of ELBO, we conclude

$$
\begin{align*}
\mathcal L(q,\mathbf{x})
&= \mathbb E_{\mathbf{z} \sim q} \left[ \log\frac{p(\mathbf{z} \mid \mathbf{x}) \cdot p(\mathbf{x})}{q(\mathbf{z})} \right] \\
&= \mathbb E_{\mathbf{z} \sim q} \left[ \log\frac{p(\mathbf{z} \mid \mathbf{x})}{q(\mathbf{z})} + \log p(\mathbf{x}) \right] \\
&= \underbrace{\mathbb E_{\mathbf{z} \sim q} \left[ \log\frac{p(\mathbf{z} \mid \mathbf{x})}{q(\mathbf{z})} \right]}_{- D_\text{KL}(q(\cdot) \parallel p(\cdot \mid \mathbf{x}))} + \log p(\mathbf{x})
\tag*{$\blacksquare$}
\end{align*}
$$

Therefore, the best approximation $p(\mathbf{z} \mid \mathbf{x})$ in $\mathcal Q$ is the solution of

$$
\begin{align}
\max_{q \in\mathcal Q}
\mathcal L(q,\mathbf{x}) = \mathbb E_{\mathbf{z} \sim q} \left[ \log\frac{p(\mathbf{x},\mathbf{z})}{q(\mathbf{z})} \right]
\end{align}
$$

In practice, this optimization problem is usually solved by parameterizing $q$ and applying gradient methods (instead of applying calculus of variations).

### ELBO as regularized reconstruction

The 2nd reformulation of ELBO is

$$
\begin{align}
\mathcal L(q,\mathbf{x})
&= \mathbb E_{\mathbf{z} \sim q} \left[ \log p(\mathbf{x} \mid \mathbf{z}) \right] - D_\text{KL}(q(\mathbf{z}) \parallel p(\mathbf{z}))
\end{align}
$$

Remarks:

* The 1st term is expected log likelihood w.r.t. the variational distribution. It measures the average goodness of reconstruction, assuming that $\mathbf{z} \sim q$.
* The 2nd term measures the proximity of $q(\mathbf{z})$ to the prior $p(\mathbf{z})$. i.e. We penalize those variational distributions that significantly deviate from the prior. $\to$ regularization effect.
* Maximizing the ELBO is a trade-off between maximizing the reconstruction fidelity and keeping variational distribution close to the prior. This is the key idea behind variational autoencoders (VAEs).

*Proof*: Substitute $p(\mathbf{x},\mathbf{z}) = p(\mathbf{x} \mid \mathbf{z}) \cdot p(\mathbf{z})$ into the definition of ELBO, we conclude

$$
\begin{align*}
\mathcal L(q,\mathbf{x})
&= \mathbb E_{\mathbf{z} \sim q} \left[ \log\frac{p(\mathbf{x} \mid \mathbf{z}) \cdot p(\mathbf{z})}{q(\mathbf{z})} \right] \\
&= \mathbb E_{\mathbf{z} \sim q} \left[ \log\frac{p(\mathbf{z})}{q(\mathbf{z})} + \log p(\mathbf{x} \mid \mathbf{z}) \right] \\
&= \underbrace{\mathbb E_{\mathbf{z} \sim q} \left[ \log\frac{p(\mathbf{z})}{q(\mathbf{z})} \right]}_{- D_\text{KL}(q(\mathbf{z}) \parallel p(\mathbf{z}))} + \mathbb E_{\mathbf{z} \sim q} \left[ \log p(\mathbf{x} \mid \mathbf{z}) \right]
\tag*{$\blacksquare$}
\end{align*}
$$

If both the prior $p(\mathbf{z})$ and the variational distribution $q(\mathbf{z})$ are Gaussian, the KL divergence in the 2nd term can be computed in closed form. To compute the ELBO, we only need to apply MC esimation to the 1st term.

$$
\begin{align*}
\mathcal L(q,\mathbf{x})
&= \underbrace{
  \mathbb E_{\mathbf{z} \sim q} \left[ \log p(\mathbf{x} \mid \mathbf{z}) \right]
}_{\text{MC estimation}}
- \underbrace{D_\text{KL}(q(\mathbf{z}) \parallel p(\mathbf{z}))}_{\text{closed-form}}
\end{align*}
$$

### ELBO as entropy minus free energy

The 3rd reformulation of ELBO is

$$
\begin{align}
\mathcal L(q,\mathbf{x})
&= \mathbb E_{\mathbf{z} \sim q} \left[ \log p(\mathbf{x},\mathbf{z}) \right] + H(q)
\end{align}
$$

Remarks:

* The 1st term is known as negative free energy in statistical physics. It rewards $q$ that explain the data well.
* The 2nd term is the entropy of the variational distribution. It rewards $q$ with higher uncertainty.
* Maximizing the ELBO can be interpreted as minimizing the free energy while maintaining high entropy in the variational distribution.

*Proof*: This reformulation follows directly from the definition of the ELBO:

$$
\begin{align*}
\mathcal L(q,\mathbf{x})
&= \mathbb E_{\mathbf{z} \sim q} \left[ \log \frac{p(\mathbf{x},\mathbf{z})}{q(\mathbf{z})} \right] \\
&= \mathbb E_{\mathbf{z} \sim q} \left[ \log p(\mathbf{x},\mathbf{z}) + \log \frac{1}{q(\mathbf{z})} \right] \\
&= \mathbb E_{\mathbf{z} \sim q} \left[ \log p(\mathbf{x},\mathbf{z}) \right] + \underbrace{\mathbb E_{\mathbf{z} \sim q} \left[ \log \frac{1}{q(\mathbf{z})} \right]}_{H(q)}
\tag*{$\blacksquare$}
\end{align*}
$$

Without the entropy term, maximizing the ELBO would result in a point mass for $\mathbf{z}$ at the mode of $p(\mathbf{x},\mathbf{z})$ (recall: $\mathbf{x}$ is fixed):

$$
q^*(\mathbf{z}) = \delta(\mathbf{z} - \hat{\mathbf{z}}), \quad \text{where } \hat{\mathbf{z}} = \argmax_z p(\mathbf{x},\mathbf{z})
$$

The entropy term, however, favors $q$ with higher entropy. In contrast, the Dirac delta is infinitely narrow and thus has extremely low entropy. Therefore, maximizing the ELBO seeks a balance between those two aspects.

<img src="./figs/elbo_maximizer.pdf" alt="elbo maximizer" style="zoom:67%;" />

Again, if the variational distribution $q(\mathbf{z})$ is Gaussian, the entropy in the 2nd term can be computed in closed form. To compute the ELBO, we only need to apply MC esimation to the 1st term.

$$
\begin{align*}
\mathcal L(q,\mathbf{x})
&= \underbrace{\mathbb E_{\mathbf{z} \sim q} \left[ \log p(\mathbf{x},\mathbf{z}) \right]}_{\text{MC estimation}}
+ \underbrace{H(q)}_{\text{closed form}}
\end{align*}
$$

## Gaussian Variational Distribution

We use multivariate Gaussian as the variational distribution

$$
\begin{align*}
q(\mathbf{z}) &= \mathcal N(\mathbf{z} ; \boldsymbol{\mu}, \boldsymbol{\Sigma})
\end{align*}
$$

The corresponding variational family $\mathcal Q$ is

$$
\begin{align*}
\mathcal Q
&= \{\mathcal N(\boldsymbol{\mu}, \boldsymbol{\Sigma}) \mid
  \boldsymbol{\mu}\in\mathbb R^\ell, \boldsymbol{\Sigma}\in\mathbb R^{\ell \times \ell}, \boldsymbol{\Sigma} \text{ is s.p.d.}
\}
\end{align*}
$$

Remarks:

* "s.p.d." stands for *symmetric positive definite*.
* Here, we use full covariance matrix in $\mathcal Q$. Alternatively, we can shrink the variational family $\mathcal Q$ by restricting it to diagonal Gaussian (also known as ***mean-field*** Gaussian) or even spherical Gaussian (also known as ***isotropic*** Gaussian).

Each variational distribution $q$ is represented by its parameters $(\boldsymbol{\mu}, \boldsymbol{\Sigma})$. The ELBO, previously defined as a functional of $q$, now becomes a function of $(\boldsymbol{\mu}, \boldsymbol{\Sigma})$.
$$
\begin{align}
\mathcal L(\boldsymbol{\mu}, \boldsymbol{\Sigma},\mathbf{x})
&= \mathbb E_{\mathbf{z} \sim \mathcal N(\boldsymbol{\mu}, \boldsymbol{\Sigma})} \left[ \log\frac{p(\mathbf{x},\mathbf{z})}{\mathcal N(\mathbf{z} ; \boldsymbol{\mu}, \boldsymbol{\Sigma})} \right]
\end{align}
$$

Next, we derive the ELBO into a form that can be optimized efficiently by exploiting properties of Gaussian distributions:

1. Closed-form expression for entropy and KL divergence. $\to$ Only part of the ELBO needs to be estimated via MC sampling, reducing the variance.
1. Allows reparameterization. $\to$ Enables gradient approximation in SGD.

For a multivariate Gaussian $q(\mathbf{z}) = \mathcal N(\mathbf{z} ; \boldsymbol{\mu}, \boldsymbol{\Sigma}), \, \mathbf{z} \in \mathbb R^\ell$, the differential entropy is:

$$
\begin{align}
H(q)
&= -\mathbb E_{q(\mathbf{z})} \left[ \log q(\mathbf{z}) \right] \nonumber \\
&= \frac{1}{2} \log \left[ (2\pi e)^\ell \det(\boldsymbol{\Sigma}) \right] \\
&= \frac{1}{2} \log \left[ \det(\boldsymbol{\Sigma}) \right] + \frac{\ell}{2} \log (2\pi e) \\
&= \log \left[ \det(\mathbf{L}) \right] + \frac{\ell}{2} \log (2\pi e)
\end{align}
$$

where $\mathbf{L}$ is the Cholesky factor of $\boldsymbol{\Sigma}$, i.e. $\boldsymbol{\Sigma} = \mathbf{LL}^\top$.

Using this entropy expression, we [reformulate the ELBO](#elbo-as-entropy-minus-free-energy) as

$$
\begin{align}
\mathcal L(\boldsymbol{\mu}, \boldsymbol{\Sigma},\mathbf{x})
&= \mathbb E_{\mathbf{z} \sim \mathcal N(\boldsymbol{\mu}, \boldsymbol{\Sigma})} \left[ \log p(\mathbf{x},\mathbf{z}) \right] + H(\mathcal N(\boldsymbol{\mu}, \boldsymbol{\Sigma}))
\nonumber \\
&= \mathbb E_{\mathbf{z} \sim \mathcal N(\boldsymbol{\mu}, \boldsymbol{\Sigma})} \left[ \log p(\mathbf{x},\mathbf{z}) \right] + \frac{1}{2} \log \left[ \det(\boldsymbol{\Sigma}) \right] + \text{const.}
\end{align}
$$

### Reparameterization Trick

The optimal Gaussian variational distribution is obtained by maximizing the ELBO:

$$
\begin{align*}
\max_{\boldsymbol{\mu}, \boldsymbol{\Sigma}} \:
\mathbb E_{\mathbf{z} \sim \mathcal N(\boldsymbol{\mu}, \boldsymbol{\Sigma})} \left[ \log p(\mathbf{x},\mathbf{z}) \right] + \frac{1}{2} \log \left[ \det(\boldsymbol{\Sigma}) \right]
\end{align*}
$$

To perform the optimization, we need the gradient of the ELBO. However, the expectation is taken w.r.t. $\mathbf{z} \sim \mathcal N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, which depends on the optimization variables. Hence, we cannot directly move the gradient inside the expectation and then compute the MC estimation of the gradient. This issue can be addressed in two ways:

1. score function method: not covered here. $\to$ see notes *Gradient Approximation*.
1. reparameterization trick: commonly used in variational inference.

We reparameterize $\mathbf{z}$ as a deterministic transformation of a standard Gaussian:

$$
\begin{align}
\mathbf{z} = \boldsymbol{\mu} + \mathbf{L}\boldsymbol{\epsilon},
\quad \boldsymbol{\epsilon} \sim \mathcal N(\mathbf{0}, \mathbf{I})
\end{align}
$$

where $\mathbf{L}$ is the Cholesky factor of $\boldsymbol{\Sigma}$, i.e. $\boldsymbol{\Sigma} = \mathbf{LL}^\top$.

Remarks:

* The Cholesky factor $\mathbf{L}$ is a **lower triangular** matrix with **positive diagonal elements**.
* After reparameterization, we treat $\boldsymbol{\mu}$ and $\mathbf{L}$ as the optimization variables instead of $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$.

The reparameterized ELBO and its gradient become:

$$
\begin{align}
\mathcal L(\boldsymbol{\mu}, \mathbf{L},\mathbf{x})
&= \mathbb E_{\boldsymbol{\epsilon} \sim \mathcal N(\mathbf{0}, \mathbf{I})} \left[ \log p(\mathbf{x},\mathbf{z}) \big|_{\mathbf{z} = \boldsymbol{\mu} + \mathbf{L}\boldsymbol{\epsilon}} \right] + \log \left[ \det(\mathbf{L}) \right]
\\
\nabla_{\boldsymbol{\mu}, \mathbf{L}} \mathcal L(\boldsymbol{\mu}, \mathbf{L},\mathbf{x})
&= \mathbb E_{\boldsymbol{\epsilon} \sim \mathcal N(\mathbf{0}, \mathbf{I})} \left[ \nabla_{\boldsymbol{\mu}, \mathbf{L}} \log p(\mathbf{x},\mathbf{z}) \big|_{\mathbf{z} = \boldsymbol{\mu} + \mathbf{L}\boldsymbol{\epsilon}} \right] + \nabla_{\boldsymbol{\mu}, \mathbf{L}} \log \left[ \det(\mathbf{L}) \right]
\end{align}
$$

Remarks:

* The constant term $\frac{\ell}{2} \log (2\pi e)$ from in the entropy $H(q)$ is discarded.
* The determinant in $\log \left[ \det(\mathbf{L}) \right]$ is simply the product of diagonal elements of $\mathbf{L}$ because it is lower triangular.
  $$
  \begin{align}
  \log \left[ \det(\mathbf{L}) \right]
  &= \log \left(\prod_{i=1}^\ell L_{ii}\right) = \sum_{i=1}^\ell \log(L_{ii})
  \end{align}
  $$
* Only the 1st term (negative free energy) in ELBO and its gradient require MC estimation.
* We do not expand the chain rules of the gradient as they are handled by autodiff tools in practice.

MC estimation of the ELBO and its gradient:

$$
\begin{align}
\boldsymbol{\epsilon}^{(k)}
&\stackrel{\text{iid}}{\sim} \mathcal N(\mathbf{0}, \mathbf{I}), \quad k = 1,\dots,m
\\
\mathbf{z}^{(k)}
&= \boldsymbol{\mu} + \mathbf{L}\boldsymbol{\epsilon}^{(k)}, \quad k = 1,\dots,m
\\
\tilde{\mathcal L}(\boldsymbol{\mu}, \mathbf{L},\mathbf{x})
&=
\frac{1}{m} \sum_{k=1}^m \log p(\mathbf{x}, \mathbf{z}^{(k)}) + \log \left[ \det(\mathbf{L}) \right]
\\
\nabla_{\boldsymbol{\mu}, \mathbf{L}} \tilde{\mathcal L}(\boldsymbol{\mu}, \mathbf{L},\mathbf{x})
&=
\frac{1}{m} \sum_{k=1}^m \nabla_{\boldsymbol{\mu}, \mathbf{L}} \log p(\mathbf{x}, \mathbf{z}^{(k)}) + \nabla_{\boldsymbol{\mu}, \mathbf{L}} \log \left[ \det(\mathbf{L}) \right]
\end{align}
$$

### Black-Box Variational Inference

Putting everything together, we are able to maximize the ELBO using stochastic gradient descent. The resulting algorithm is called black box variational inference. The name reflects the black box nature that we do not compute the ELBO and its gradient in closed-form, but estimate them using MC sampling.

---

**Algorithm: BBVI with Gaussian variational distributions**  
**Input**: observation $\mathbf{x} \in\mathbb R^d$, generative model $p(\mathbf{x}, \mathbf{z})$, learning rate $\{ \eta_t \}_{t=1}^{\infty}$  
**Output**: $\boldsymbol{\mu} \in\mathbb R^\ell, \boldsymbol{\Sigma} \in\mathbb R^{\ell \times \ell}$  
**Goal**: use $\mathcal N(\mathbf{z}; \boldsymbol{\mu},\boldsymbol{\Sigma})$ to approximate $p(\mathbf{z} \mid \mathbf{x})$

Init $\boldsymbol{\mu} \in \mathbb R^\ell$ and $\mathbf{L} \in \mathbb R^{\ell \times \ell}$  
While the SGD for $\boldsymbol{\mu}$ and $\mathbf{L}$ is not converged: do  
$\quad$ Sample a mini-batch: $\boldsymbol{\epsilon}^{(1)}, \dots, \boldsymbol{\epsilon}^{(m)} \sim \mathcal N(0, I_{\ell})$  
$\quad$ Reparameterization: $\mathbf{z}^{(k)} = \boldsymbol{\mu} + \mathbf{L}\boldsymbol{\epsilon}^{(k)}, \quad k = 1,\dots,m$  
$\quad$ Estimate the ELBO and its gradient:

$$
\begin{align*}
\tilde{\mathcal L}(\boldsymbol{\mu}, \mathbf{L})
&=
\frac{1}{m} \sum_{k=1}^m \log p(\mathbf{x}, \mathbf{z}^{(k)}) + \log \left[ \det(\mathbf{L}) \right]
\\
\nabla_{\boldsymbol{\mu}, \mathbf{L}} \tilde{\mathcal L}(\boldsymbol{\mu}, \mathbf{L})
&=
\frac{1}{m} \sum_{k=1}^m \nabla_{\boldsymbol{\mu}, \mathbf{L}} \log p(\mathbf{x}, \mathbf{z}^{(k)}) + \nabla_{\boldsymbol{\mu}, \mathbf{L}} \log \left[ \det(\mathbf{L}) \right]
\end{align*}
$$

$\quad$ Update $\boldsymbol{\mu}$ and $\mathbf{L}$:

$$
\begin{align*}
\boldsymbol{\mu} &\leftarrow \boldsymbol{\mu} + \eta_t \nabla_{\boldsymbol{\mu}} \mathcal L(\boldsymbol{\mu}, \mathbf{L}) \\
\mathbf{L}   &\leftarrow \mathbf{L} + \eta_t \nabla_{\mathbf{L}} \mathcal L(\boldsymbol{\mu}, \mathbf{L}) \\
\end{align*}
$$

Set $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$

return $\boldsymbol{\mu}, \boldsymbol{\Sigma}$

---

## Dataset-Level ELBO

[Previously](#the-evidence-lower-bound), we derived the ELBO $\mathcal L(q,\mathbf{x})$ for a single observation $\mathbf{x}$. From now on, let's call it **per-sample** ELBO (or **per-observation** ELBO).

**Question**: What if we have a dataset consisting of multiple iid observations? Can we lower-bound the evidence of the whole dataset?

Problem formulation:

* Known: generative model $p(\mathbf{x},\mathbf{z}) = p(\mathbf{z}) \, p(\mathbf{x} \mid \mathbf{z})$.
* Given: training data $D = \{ \mathbf{x}_1, \cdots,  \mathbf{x}_n\} \stackrel{\text{iid}}{\sim} p(\mathbf{x}) = \int_z p(\mathbf{x},\mathbf{z}) \:\mathrm dz$.
* Select: variational family $\mathcal Q$.
* Goal: derive a lower bound on $\log p(D)$.

We refer to the lower bound on $\log p(D)$ as the **dataset-level ELBO** (or simply **dataset ELBO**). In fact, dataset ELBO does exist since

$$
\begin{align}
\log p(D) = \log \prod_{i=1}^n p(\mathbf{x}_i) = \sum_{i=1}^n \log p(\mathbf{x}_i)
\end{align}
$$

Each $\log p(\mathbf{x}_i)$ can be lower bounded by its individual per-sample ELBO. Therefore, $\log p(D)$ can also be lower bounded. The remaining question is how to design the variational distribution for each $\mathbf{x}_i$.

### Local Inference Model

A natural extension of per-sample ELBO to dataset ELBO is choosing variational distribution independently for each observation. Formally:

For each $\mathbf{x}_i$, we choose $q_i \in\mathcal Q$ independently to approximate the true posterior $p(\mathbf{z} \mid \mathbf{x}_i)$. This gives per-sample ELBO

$$
\begin{align}
\mathcal L(q_i, \mathbf{x}_i)
&= \sum_{i=1}^n \mathbb E_{\mathbf{z} \sim q_i} \left[ \log\frac{p(\mathbf{x}_i, \mathbf{z})}{q_i(\mathbf{z})} \right]
\end{align}
$$

For any combination of variational distributions $q_1, \dots, q_n \in \mathcal Q$, it holds that

$$
\begin{align*}
\log p(D)
= \sum_{i=1}^n \log p(\mathbf{x}_i)
\ge \underbrace{\sum_{i=1}^n \mathcal L(q_i, \mathbf{x}_i)}_{\mathcal L(q_1, \dots, q_n, D)}
\end{align*}
$$

Hence, we obtain the dataset ELBO:

$$
\begin{align}
\mathcal L(q_1, \dots, q_n, D)
\triangleq \sum_{i=1}^n \mathcal L(q_i, \mathbf{x}_i)
= \sum_{i=1}^n \mathbb E_{\mathbf{z} \sim q_i} \left[ \log\frac{p(\mathbf{x}_i, \mathbf{z})}{q_i(\mathbf{z})} \right]
\end{align}
$$

Remarks:

* The (dataset) ELBO is a functional of $q_1, \dots, q_n$, which are **freely** chosen.
* We assume that $q_1, \dots, q_n \in \mathcal Q$. i.e. All variational distributions belong to the same variational family.
* Local variational distributions are used in classical variational inference.

The optimal variational distributions are obtained by solving the functional optimization problems

$$
\begin{align}
q_i^* = \argmax_{q_i \in \mathcal Q} \mathcal L(q_i, \mathbf{x}_i), \quad i=1,\dots,n
\end{align}
$$

Remarks:

* Due to the additive structure of dataset ELBO, each $q_i$ can be optimized independently of each other.
* In practice, $\mathcal Q$ is a parametric variational family, e.g. Gaussian. Therefore, we turn this functional optimization problem into a parameter optimization problem.

Drawbacks:

1. **Poor scalability**: The number of optimization problems scales linearly with the size of the dataset. If we have $n$ observations, we have solve $n$ independent optimization problems.
1. **No generalization**: Given a new observation $\mathbf{x}_*$, we have to solve the optimization problem again. We cannot forge the variational distribution for $\mathbf{x}_*$ from $q_1^*, \dots, q_n^*$.

### Global Inference Model

Instead of learning $q_i$ for each $\mathbf{x}_i$ individually, we learn a **global inference model**, conceptually defined as a mapping $f$

$$
f: \mathbb R^d \to \mathcal Q, \mathbf{x} \mapsto q(\cdot \mid \mathbf{x})
$$

where $\cdot$ is the placeholder for $\mathbf{z}$, such that $\forall \mathbf{x} \in \mathbb R^d$

$$
q(\cdot \mid \mathbf{x}) \approx p(\cdot \mid \mathbf{x})
$$

Remarks:

* The abstract mapping $f$ maps each data point to a variational distribution. Mathematically, it is a complex point-to-function mapping.
* ⚠️ To reduce visual clutter, we write $f(\mathbf{x}) = q(\cdot \mid \mathbf{x})$ rather than $f(\mathbf{x}) = q_{f}(\cdot \mid \mathbf{x})$
* The globalness hightlights the fact that $f$ is shared by all $\mathbf{x}\in\mathbb R^d$.
* Once we learned such $f$ on training data $D$, not only can we plug in $\mathbf{x}_i$ and use $q(\cdot \mid \mathbf{x}_i) \approx p(\cdot \mid \mathbf{x}_i)$, but also we can plug in any unseen $\mathbf{x}_*$ and get $q(\cdot \mid \mathbf{x}_*) \approx p(\cdot \mid \mathbf{x}_*)$.

For each sample $\mathbf{x}_i$, the per-sample ELBO is

$$
\begin{align}
\mathcal L(f(\mathbf{x}_i), \mathbf{x}_i)
&= \mathcal L(q(\cdot \mid \mathbf{x}_i), \mathbf{x}_i) \\
&= \mathbb E_{\mathbf{z} \sim q(\cdot \mid \mathbf{x}_i)} \left[ \log\frac{p(\mathbf{x}_i, \mathbf{z})}{q(\mathbf{z} \mid \mathbf{x}_i)} \right]
\end{align}
$$

Summing over all samples, we obtain the dataset ELBO

$$
\begin{align}
\mathcal L(f, D)
&\triangleq \sum_{i=1}^n \mathcal L(f(\mathbf{x}_i), \mathbf{x}_i)\\
&= \sum_{i=1}^n \mathbb E_{\mathbf{z} \sim q(\cdot \mid \mathbf{x}_i)} \left[ \log\frac{p(\mathbf{x}_i, \mathbf{z})}{q(\mathbf{z} \mid \mathbf{x}_i)} \right]
\end{align}
$$

Remarks:

* Not to be confused by the notation: $f(\mathbf{x}_i) = q(\cdot \mid \mathbf{x}_i) \in \mathcal Q$, i.e. $f(\mathbf{x}_i)$ is a (probability density) function.
* Comparing to local variational distribution scheme, there is a key distinction between $q_i(\cdot)$ and $q(\cdot \mid \mathbf{x}_i)$:
  * Local variational distribution: We choose each $q_i(\cdot) \in \mathcal Q$ freely.
  * Global variational distribution: Each $q(\cdot \mid \mathbf{x}_i)$ are determined by plugging $\mathbf{x}_i$ into the point-to-function mapping $f$, which shared by all $\mathbf{x}_i \in D$.

To maximize the dataset ELBO, we aim to solve

$$
\begin{align}
f^* = \argmax_{f} \mathcal L(f, D)
\end{align}
$$

This is again a functional optimization problem. In practice, we avoid dealing directly with a functional optimization problem by

1. using parametric family $\mathcal Q$, e.g. multivariate Gaussian with parameter $(\boldsymbol{\mu}, \boldsymbol{\Sigma})$
1. designing $f$ as a neural net with $\mathbf{x}$ as its input layer, and $(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ as its output layer.

Learning $f$ boils down to training such a neual net.

## Variational Inference

Unless otherwise specified, we use a **Gaussian variational distribution** to approximate the true posterior.

Previously, we have seen [variational inference for a single observation](#elbo-maximization-for-gaussian-variational distribution). Now, we extend variational inference to the case of multiple observations.

Problem formulation:

* Known: generative model $p(\mathbf{x},\mathbf{z}) = p(\mathbf{z}) \, p(\mathbf{x} \mid \mathbf{z})$
* Given: training data $D = \{ \mathbf{x}_1, \cdots,  \mathbf{x}_n\} \stackrel{\text{iid}}{\sim} p(\mathbf{x}) = \int_z p(\mathbf{x},\mathbf{z}) \:\mathrm dz$.
* Select: Gaussian variational family $\mathcal Q = \{ \mathcal N(\boldsymbol{\mu}, \boldsymbol{\Sigma}) \}$.
* Goal: maximize the dataset ELBO

In the following, we will consider local variational distribution scheme and global variational distribution scheme. The dataset ELBO, previously defined as a functional over variational distributions, will be reformulated as a scalar-valued function of parameter vectors.

### Classical Variational Inference

For each observation $\mathbf{x}_i$, we use $\mathcal N(\mathbf{z}; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)$ to approximate the true posterior $p(\mathbf{z} \mid \mathbf{x}_i)$.

* Classical: We use local variational distributions, i.e. we choose $\boldsymbol{\mu}_i,\boldsymbol{\Sigma}_i$ independently for each $\mathbf{x}_i$.

The dataset ELBO, previously as a functional of $\{q_i\}_{i=1}^n$, now becomes a function of $\{\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i\}_{i=1}^n$:

$$
\begin{align}
\mathcal L(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1, \dots, \boldsymbol{\mu}_n, \boldsymbol{\Sigma}_n, D)
&\triangleq \sum_{i=1}^n \mathcal L(\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i, \mathbf{x}_i) \\
&= \sum_{i=1}^n \mathbb E_{\mathbf{z} \sim \mathcal N(\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)} \left[ \log\frac{p(\mathbf{x}_i, \mathbf{z})}{\mathcal N(\mathbf{z}; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)} \right] \\
&= \sum_{i=1}^n \mathbb E_{\mathbf{z} \sim \mathcal N(\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)} \left[ \log p(\mathbf{x}_i, \mathbf{z}) \right] + H(\mathcal N(\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)) + \text{const} \\
&= \sum_{i=1}^n \mathbb E_{\mathbf{z} \sim \mathcal N(\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)} \left[ \log p(\mathbf{x}_i, \mathbf{z}) \right] + \frac{1}{2} \log\vert\boldsymbol{\Sigma}_i\vert + \text{const}
\end{align}
$$

The dataset ELBO can be maximized sample-wise as follows

$$
\begin{align}
\forall \mathbf{x}_i \in D: \quad
\max_{\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i} \mathbb E_{\mathbf{z} \sim \mathcal N(\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)} \left[ \log p(\mathbf{x}_i, \mathbf{z}) \right] + \frac{1}{2} \log\vert\boldsymbol{\Sigma}_i\vert
\end{align}
$$

Again, we apply reparameterization trick to allow MC estimation of the objective

$$
\begin{align}
\mathbf{z}
&= \boldsymbol{\mu}_i + \mathbf{L}_i \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal N(\mathbf{0}, \mathbf{I})
\\
\max_{\boldsymbol{\mu}_i, \mathbf{L}_i}\:
&\mathbb E_{\boldsymbol{\epsilon} \sim \mathcal N(\mathbf{0}, \mathbf{I})} \left[ \log p(\mathbf{x}_i, \boldsymbol{\mu}_i + \mathbf{L}_i \boldsymbol{\epsilon}) \right] + \log\vert \mathbf{L}_i \vert
\\
\max_{\boldsymbol{\mu}_i, \mathbf{L}_i}\:
&\frac{1}{M} \sum_{k=1}^M \log p(\mathbf{x}_i, \mathbf{z}^{(k)}) |_{\mathbf{z}^{(k)} = \boldsymbol{\mu}_i + \mathbf{L}_i\boldsymbol{\epsilon}^{(k)}} + \log\vert \mathbf{L}_i \vert
\quad \text{ where } \boldsymbol{\epsilon}^{(k)} \sim \mathcal N(\mathbf{0}, \mathbf{I})
\end{align}
$$

The complete algorithm is summarized below:

---

**Algorithm: classical variational inference with Gaussian variational distributions**  
**Input**: $\mathbf{x}_1, \dots, \mathbf{x}_n \in\mathbb R^d$  
**Output**: $\boldsymbol{\mu}_1,\dots,\boldsymbol{\mu}_n\in\mathbb R^\ell, \boldsymbol{\Sigma}_1,\dots,\boldsymbol{\Sigma}_n\in\mathbb R^{\ell \times \ell}$  
**Goal**: use $\mathcal N(\mathbf{z}; \boldsymbol{\mu}_i,\boldsymbol{\Sigma}_i)$ to approximate $p(\mathbf{z} \mid \mathbf{x}_i)$

For each $i=1,\dots,n$: do  
$\quad$ Init $\boldsymbol{\mu}_i \in \mathbb R^\ell$ and $\mathbf{L}_i \in \mathbb R^{\ell \times \ell}$  
$\quad$ While the SGD for $\boldsymbol{\mu}_i$ and $\mathbf{L}_i$ is not converged: do  
$\qquad$ Sample a mini-batch $\boldsymbol{\epsilon}^{(1)}, \dots, \boldsymbol{\epsilon}^{(M)} \sim \mathcal N(0, I_{\ell})$  
$\qquad$ Compute the objective $\mathcal L(\boldsymbol{\mu}_i, \mathbf{L}_i)$ and its gradient

$$
\mathcal L(\boldsymbol{\mu}_i, \mathbf{L}_i) \triangleq
\frac{1}{M} \sum_{k=1}^M \log p(\mathbf{x}_i, \mathbf{z}^{(k)}) + \log\vert \mathbf{L}_i \vert,
\quad \mathbf{z}^{(k)} = \boldsymbol{\mu}_i + \mathbf{L}_i \boldsymbol{\epsilon}^{(k)}
$$

$\qquad$ Update $\boldsymbol{\mu}_i$ and $\mathbf{L}_i$

$$
\begin{align*}
\boldsymbol{\mu}_i &\leftarrow \boldsymbol{\mu}_i + \eta_t \nabla_{\boldsymbol{\mu}_i} \mathcal L(\boldsymbol{\mu}_i, \mathbf{L}_i) \\
\mathbf{L}_i   &\leftarrow \mathbf{L}_i + \eta_t \nabla_{\mathbf{L}_i} \mathcal L(\boldsymbol{\mu}_i, \mathbf{L}_i) \\
\end{align*}
$$

$\quad$ Set $\boldsymbol{\Sigma}_i = \mathbf{L}_i\mathbf{L}_i^\top$

return $\boldsymbol{\mu}_1,\dots,\boldsymbol{\mu}_n, \boldsymbol{\Sigma}_1,\dots,\boldsymbol{\Sigma}_n$

---

Remarks:

* This approach is essentially performing BBVI iteratively for each observation.
* The total \# parameters to be learned is $O(n\ell^2)$, which scales linearly with the dataset size $n$. This limits scalability for large datasets.
* Local variational distributions provide high flexibility, as each variational distribution's mean and covariance are learned independently.

### Amortized Variational Inference

For each observation $\mathbf{x}_i$, we again use $\mathcal N(\mathbf{z}; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)$ to approximate the true posterior $p(\mathbf{z} \mid \mathbf{x}_i)$, but now:

* Amortized: The mapping rule $f: \mathbf{x}_i \mapsto (\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)$ is now shared by all $\mathbf{x}_i\in D$. Instead of learning each $(\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)$ individually, we learn the shared function $f$.

Basic idea of amortized variational inference:

$$
\mathbf{x} \longrightarrow \boxed{ \text{Neural Net } f_{\boldsymbol{\phi}} \vphantom{\int} } \longrightarrow
\begin{bmatrix} \boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}) \\ \boldsymbol{\Sigma}_{\boldsymbol{\phi}}(\mathbf{x}) \end{bmatrix}
\longrightarrow q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \approx p(\mathbf{z} \mid \mathbf{x})
$$

The mapping $f$ is typically implemented as a neural net (NN) parameterized by ${\boldsymbol{\phi}}$:

$$
f_{\boldsymbol{\phi}}: \mathbb R^d \to \mathbb R^\ell \times \mathbb R^{\ell\times\ell}, \mathbf{x} \mapsto (\boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}), \boldsymbol{\Sigma}_{\boldsymbol{\phi}}(\mathbf{x}))
$$

The resulting variational distribution for each $\mathbf{x}\in\mathbb R^d$ becomes

$$
q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) = \mathcal N(\mathbf{z} ; \boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}), \boldsymbol{\Sigma}_{\boldsymbol{\phi}}(\mathbf{x}))
$$

Remarks:

* The output $(\boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}), \boldsymbol{\Sigma}_{\boldsymbol{\phi}}(\mathbf{x}))$ of the NN depends on both the observation $\mathbf{x}$ and the network parameter ${\boldsymbol{\phi}}$.
* In practice, the output of the NN is $(\boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}), \mathbf{L}_{\boldsymbol{\phi}}(\mathbf{x}))$ where $\mathbf{L}_{\boldsymbol{\phi}}(\mathbf{x})$ is the Cholesky factor of $\boldsymbol{\Sigma}_{\boldsymbol{\phi}}(\mathbf{x})$. This design later helps letting the gradient flow from ELBO to $\mathbf{x}$ by reparameterization trick.

Goal: Train the NN (aka learn ${\boldsymbol{\phi}}$) so that

$$
q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \approx p(\mathbf{z} \mid \mathbf{x}), \forall \mathbf{x} \in D
$$

To achieve this goal, we need to maximize the ELBO. Previously, the ELBOs are defined as an abstract functional of $f$. Now, they become a function of ${\boldsymbol{\phi}}$.

For each $\mathbf{x} \in D$, the per-sample ELBO is

$$
\begin{align}
\mathcal L({\boldsymbol{\phi}}, \mathbf{x})
&= \mathbb E_{\mathbf{z} \sim q_{\boldsymbol{\phi}}(\cdot \mid \mathbf{x})} \left[ \log\frac{p(\mathbf{x}, \mathbf{z})}{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})} \right]
\\
&= \mathbb E_{\mathbf{z} \sim q_{\boldsymbol{\phi}}(\cdot \mid \mathbf{x})} \left[ \log p(\mathbf{x},\mathbf{z}) \right] + H(q_{\boldsymbol{\phi}}(\cdot \mid \mathbf{x}))
\\
&= \mathbb E_{\boldsymbol{\epsilon} \sim \mathcal N(\mathbf{0}, \mathbf{I})} \left[ \log p(\mathbf{x},\mathbf{z}) \right] + \log\left[ \det \mathbf{L}_{\boldsymbol{\phi}}(\mathbf{x}) \right],
\quad \mathbf{z} = \boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}) + \mathbf{L}_{\boldsymbol{\phi}}(\mathbf{x}) \cdot \boldsymbol{\epsilon}
\end{align}
$$

With $\mathbf{z} = \boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}) + \mathbf{L}_{\boldsymbol{\phi}}(\mathbf{x}) \cdot \boldsymbol{\epsilon}$ for each $\mathbf{x} \in D$, the dataset ELBO is

$$
\begin{align}
\mathcal L({\boldsymbol{\phi}}, D)
&= \sum_{\mathbf{x}\in D} \mathcal L({\boldsymbol{\phi}}, \mathbf{x}) \\
&= \sum_{\mathbf{x}\in D} \mathbb E_{\mathbf{z} \sim q_{\boldsymbol{\phi}}(\cdot \mid \mathbf{x})} \left[ \log\frac{p(\mathbf{x}, \mathbf{z})}{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})} \right] \\
&= \sum_{\mathbf{x}\in D} \mathbb E_{\boldsymbol{\epsilon} \sim \mathcal N(\mathbf{0}, \mathbf{I})} \left[ \log p(\mathbf{x},\mathbf{z})  \right] + \log\left[ \det \mathbf{L}_{\boldsymbol{\phi}}(\mathbf{x}) \right]
\\
&= |D| \cdot \sum_{\mathbf{x}\in D} \frac{1}{|D|} \left[\mathbb E_{\boldsymbol{\epsilon} \sim \mathcal N(\mathbf{0}, \mathbf{I})} \left[ \log p(\mathbf{x},\mathbf{z}) \right] + \log\left[ \det \mathbf{L}_{\boldsymbol{\phi}}(\mathbf{x}) \right] \right]
\nonumber
\\
&= |D| \cdot \mathbb E_{\mathbf{x} \sim \mathrm{Unif}(D)} \left[\mathbb E_{\boldsymbol{\epsilon} \sim \mathcal N(\mathbf{0}, \mathbf{I})} \left[ \log p(\mathbf{x},\mathbf{z}) \right] + \log\left[ \det \mathbf{L}_{\boldsymbol{\phi}}(\mathbf{x}) \right] \right]
\end{align}
$$

Again, the dataset ELBO and its gradient can be estimated via MC sampling.

$$
\begin{align}
\tilde{\mathcal L}({\boldsymbol{\phi}}, D)
&= \frac{|D|}{|B|} \sum_{\mathbf{x} \in B} \left[\frac{1}{m} \sum_{k=1}^m \log p(\mathbf{x},\mathbf{z}^{(k)}) + \log\left[ \det \mathbf{L}_{\boldsymbol{\phi}}(\mathbf{x}) \right] \right]
\\
\nabla_{\boldsymbol{\phi}} \tilde{\mathcal L}({\boldsymbol{\phi}}, D)
&= \frac{|D|}{|B|} \sum_{\mathbf{x} \in B} \left[\frac{1}{m} \sum_{k=1}^m \nabla_{\boldsymbol{\phi}} \log p(\mathbf{x},\mathbf{z}^{(k)}) + \nabla_{\boldsymbol{\phi}} \log\left[ \det \mathbf{L}_{\boldsymbol{\phi}}(\mathbf{x}) \right] \right]
\end{align}
$$

where

* $B$ is a mini-batch sampled from the whole dataset: $B \subseteq D = \{\mathbf{x}_1, \dots, \mathbf{x}_n\}$
* $\boldsymbol{\epsilon}^{(k)} \stackrel{\text{iid}}{\sim} \mathcal N(\mathbf{0}, \mathbf{I}), \quad k=1,\dots,m$
* $\mathbf{z}^{(k)} = \boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}) + \mathbf{L}_{\boldsymbol{\phi}}(\mathbf{x}) \boldsymbol{\epsilon}^{(k)}, \quad k=1,\dots,m$

The complete algorith of amortized variatioal inference is summarized below:

---

**Algorithm: amortized variational inference with Gaussian variational distributions**  
**Input**: $D = \{\mathbf{x}_1, \dots, \mathbf{x}_n \in\mathbb R^d\}$  
**Output**: ${\boldsymbol{\phi}}$  
**Goal**: train a $\mathrm{NN}_{\boldsymbol{\phi}}: \mathbf{x} \mapsto (\boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}),\boldsymbol{\Sigma}_{\boldsymbol{\phi}}(\mathbf{x}))$ s.t. $\mathcal N(\mathbf{z}; \boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}),\boldsymbol{\Sigma}_{\boldsymbol{\phi}}(\mathbf{x})) \approx p(\mathbf{z} \mid \mathbf{x})$

While SGD for ${\boldsymbol{\phi}}$ is not converged: do  
$\quad$ Sample a mini-batch: $B \subseteq D$  
$\quad$ For each $\mathbf{x}\in B$: do  
$\qquad$ Forward-pass: compute $\boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}), \mathbf{L}_{\boldsymbol{\phi}}(\mathbf{x})$  
$\qquad$ For $k = 1,\dots,m$: do  
$\qquad\quad$ Sampling: $\boldsymbol{\epsilon}^{(k)} \stackrel{\text{iid}}{\sim} \mathcal N(\mathbf{0}, \mathbf{I})$  
$\qquad\quad$ Reparamterization: $\mathbf{z}^{(k)} = \boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}) + \mathbf{L}_{\boldsymbol{\phi}}(\mathbf{x}) \boldsymbol{\epsilon}^{(k)}$  

$\qquad$ Compute the per-sample ELBO:

$$
\begin{align*}
\tilde{\mathcal L}({\boldsymbol{\phi}}, \mathbf{x})
&= \frac{1}{m} \sum_{k=1}^m \log p(\mathbf{x},\mathbf{z}^{(k)}) + \log\left[ \det \mathbf{L}_{\boldsymbol{\phi}}(\mathbf{x}) \right]
\end{align*}
$$

$\quad$ Compute the dataset ELBO:

$$
\begin{align*}
\tilde{\mathcal L}({\boldsymbol{\phi}}, B)
&= \frac{|D|}{|B|} \sum_{\mathbf{x} \in B} \mathcal L({\boldsymbol{\phi}}, \mathbf{x})
\end{align*}
$$

$\quad$ Backward-pass: compute the gradient $\nabla_{\boldsymbol{\phi}} \tilde{\mathcal L}({\boldsymbol{\phi}}, B)$  
$\quad$ Update: ${\boldsymbol{\phi}} \leftarrow {\boldsymbol{\phi}} + \eta_t \nabla_{\boldsymbol{\phi}} \mathcal L({\boldsymbol{\phi}}, B)$

Return ${\boldsymbol{\phi}}$

---

Remarks:

* The \# parameters is now fully determined by the architecture of NN (or the \# scalars in ${\boldsymbol{\phi}}$). No longer scales up with the dataset size.
* Once, we trained the NN. We can compute $q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}_*)$ for unseen data $\mathbf{x}_*$ by simply performing a forward pass. This allows us effortless generalization for inference. In contrast, classical VI has no generalizaiton ability.

### Summary of Terminologies

Variational Inference (VI): A class of optimization-based methods that approximate the true posterior within a variational family by optimzing the ELBO. We distinguish different types of VI based on the following criteria:

How is the ELBO and its gradient computed?

* Traditional variational inference: ELBO and its gradient are computed analytically in close-form. Requires conjugacy.
* Black-box variational inference: ELBO and its gradient are estimated by MC sampling. Does not require conjugacy.

Additional assumption on the structure of variational distribution?

* Mean-field variatioal inference: The variational distribution can be factorized component-wise, i.e. no correlation among latent dimensions.

How is inference performed across observations?

* Classical variational inference: optimize a separate variatioanl distribution for each observation.
* Amortized variational inference: train a global inference model that is shared by all observations.

These types can be mixed depending on the modeling goal. e.g. One might use an amortized mean-field Gaussian variational distribution.

## Appendix

### Entropy of Gaussian

For a multivariate Gaussian $p(\mathbf{x}) = \mathcal N(\mathbf{x} ; \boldsymbol{\mu}, \boldsymbol{\Sigma}), \, \mathbf{x} \in \mathbb R^d$, the differential entropy is:

$$
\begin{align}
H(p)
&= -\mathbb E_{\mathbf{x} \sim p} \left[ \log p(\mathbf{x}) \right] \\
&= \frac{1}{2} \log \left[ (2\pi e)^d \det(\boldsymbol{\Sigma}) \right] \\
&= \frac{1}{2} \log \left[ \det(\boldsymbol{\Sigma}) \right] + \frac{d}{2} \log (2\pi e) \\
\end{align}
$$

Let $\mathbf{L}$ (lower triangular) be the Cholesky factor of the covariance matrix, i.e. $\boldsymbol{\Sigma} = \mathbf{LL}^\top$. Then,

$$
\begin{align*}
\log \left[ \det(\boldsymbol{\Sigma}) \right]
&= \log \left[ \det(\mathbf{LL}^\top) \right] \\
&= \log \left[ \det(\mathbf{L}) \cdot \det(\mathbf{L}^\top) \right] \\
&= \log \left[ \det(\mathbf{L})^2 \right] \\
&= 2\log \left[ \det(\mathbf{L}) \right] \\
\end{align*}
$$

Therefore, We can express $H(p)$ as

$$
\begin{align}
H(p)
&= \log \left[ \det(\mathbf{L}) \right] + \frac{d}{2} \log (2\pi e) \\
\end{align}
$$
