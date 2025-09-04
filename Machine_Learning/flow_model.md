---
title: "Flow Models"
date: "2025"
author: "Kezhang"
---

# Flow Models

[toc]

$$
\DeclareMathOperator*{\argmax}{argmax}
\DeclareMathOperator*{\argmin}{argmin}
$$

## Motivation

How do we model a complex distribution $p_\text{data}$?

A variational autoencoder (VAE) models $p_\text{data}$ with deep latent varible models:

1. start from simple prior distribution $z \sim \mathcal N( 0,  I)$.
1. transform via decoder net $x \mid z \sim \mathcal N(\mu_{\theta}(z), \Sigma_{\theta}(z))$.
1. yields a complex/flexible marginal distribution $p_\theta(x) = \int p_\theta(x,z) \,\mathrm dz \approx p_\text{data}(x)$.

A flow model is similar to a VAE in the sense that it transforms simple distribution to a complex one. However, flow models use a different approach -- by modeling a flow.

1. start from simple base distribution $z \sim \mathcal N( 0,  I)$.
1. transform via a flow $x = f(z)$.
1. yields a complex/flexible push-forward distribution $p(x) = (f_\sharp p_Z)(x) \approx p_\text{data}(x)$.

## Basic Idea of Flow Models

### Flow

A ***flow*** is a vector field

$$
\begin{align}
f: \mathbb R^d \to \mathbb R^d, z \mapsto x = f(z)
\end{align}
$$

such that

1. $f$ is **differentiable** and **invertible**
1. The inverse $f^{-1}$ is also **differentiable**

Remarks:

* The differentiability of the inverse is later required by the change of variable rule.
* A flow model can be viewed as a deterministic and invertible VAE, such that each $x$ corresponds to a unique $z$. Conditioned on $z$, there is no stochasticity in $x$.
* Unlike VAEs, a flow model does not compress information into a lower-dimensional latent space since $x$ and $z$ have the **same** dimension.

A collection of flows $f_1,\dots,f_T$ can be composited into a new flow.

$$
\begin{align}
f = f_T \circ \dots \circ f_1
\end{align}
$$

Since each flow is invertible, the composite flow is also invertible

$$
\begin{align}
f = f_1^{-1} \circ \dots \circ f_T^{-1}
\end{align}
$$

Illustration:

$$
z \triangleq
x_0    \xrightleftharpoons[f_1^{-1}]{f_1}
x_1    \xrightleftharpoons[f_2^{-1}]{f_2}
x_2    \xrightleftharpoons[f_3^{-1}]{f_3}
\cdots \xrightleftharpoons[f_T^{-1}]{f_T}
x_T \triangleq x
$$

For each $t=1,\dots,T$, let $D_{f_t}(u)$ denote the Jacobian of $f_t$ evaluated at $u\in\mathbb R^d$. By the chain rule, the composite flow is also differentiable with the Jacobian

$$
\begin{align}
D_{f}(z) = \prod_{t=1}^T D_{f_t}(x_{t-1}), \quad x_0 \triangleq z
\end{align}
$$

The inverse $f^{-1}$ of the composition is also invertible with the Jacobian

$$
\begin{align}
D_{f^{-1}}(x)
&= [D_{f}(z)]^{-1} \\
&= \prod_{t=1}^T [ D_{f_t}(x_{t-1}) ]^{-1}, && x_0 \triangleq z \\
&= \prod_{t=1}^T D_{f_t^{-1}}(x_{t}),     && x_T \triangleq x
\end{align}
$$

### Push-Forward Distribution

Consider random vector $Z$ with a simple **base distribution** $p_Z$. Let random vector $X$ be defined by as $X = f(Z)$. The distribution $p_X = f_\sharp p_Z$ is the **push-forward distribution** of $p_Z$ under $f$. By change of variables rule, we have

$$
\begin{align}
p_X(x)
&= p_Z\left( f^{-1}(x) \right) \cdot \left\vert \det \Big( D_{f^{-1}}(x) \Big) \right\vert \\
&= p_Z\left( z \right) \cdot \left\vert \det \Big( D_{f}(z) \Big) \right\vert^{-1}, \quad
z = f^{-1}(x)
\end{align}
$$

where $D_{f^{-1}}(x)$ is the Jacobian matrix of $f^{-1}$ evaluated at $x$.

Remarks:

* The 2nd equation follows from the fact $\det(A^{-1}) = \det(A)^{-1}$.
* In 1D case, the formula simplifies to
    $$
    p_X(x)
    = p_Z\left( z \right) \cdot \left\vert f'(z) \right\vert^{-1}, \quad
    z = f^{-1}(x)
    $$
* Assume $f$ is sufficiently expressive. Then, $f$ pushes the simple base distribution to a more complex distribution. Conversely, the inverse function $f^{-1}$ pushes a complex distribution to a simpler one.
    $$
    \begin{align}
    p_X = f_\sharp p_Z, \quad p_Z = [f^{-1}]_\sharp p_X
    \end{align}
    $$

Applying change of variable rule requires

* Computing the inverse $f^{-1}$.
* Computing the Jacobian determinant, which typically costs $O(d^3)$ time unless $f$ has special structure. The computation is infeasible in higher dimension.

Therefore, while every flow induces a valid push-forward distribution, a flow is called a ***normalizing flow*** only when its inverse and Jacobian determinant can be evaluated efficiently. For exmample, normalizing flows have diagonal or low-rank Jacobians.

### The Learning Problem

Given a simple base distribution $p_Z$, how can we learn a flow $f$ so that the resulting push-forward distribution $f_\sharp p_Z$ matches the true data distribution $p_\text{data}$? Formally, we would like to minimize the **forward** KL divergence

$$
\begin{align}
\min_{f} D_\mathrm{KL}(p_\text{data} \| f_\sharp p_Z)
\end{align}
$$

Remarks:

* In practice, the flow $f$ is a deep neural network. Learning $f$ boilds down to learning network parameters.
* Flow models are generative in nature. Once we learned $f$, we can generate samples $x$ by first sampling $z \sim p_Z$ and then applying $f$.

Theoretically, there always exists flows pushing $p_Z$ exactly to $p_\text{data}$ if $p_\text{data}$ is known.

> Under regularity assumptions, there exists infinitely many flows $f$ s.t. $f_\sharp p_Z = p_\text{data}$ for given $p_Z$ and $p_\text{data}$.

The proof of existence requires tranport theory -- very hardcore math. Here we only illustrate the non-uniqueness of such flows. Suppose $Z\sim\mathcal N(0, I_d)$ and $f$ pushes $p_Z$ forward to $p_\text{data}$. Then for any orthogonal linear map $g: \mathbb R^d \to \mathbb R^d$ (e.g. rotation), the composition flow $f \circ g$ also pushes $p_Z$ forward to $p_\text{data}$.

$$
g_\sharp p_Z = p_Z \implies
(f \circ g)_\sharp p_Z = f_\sharp(g_\sharp p_Z) = f_\sharp p_Z = p_\text{data}
$$

However, the transport theory does not directly give an algorithm to compute such flows. In practice, we can only approximate one of these flows.

The learning problem is equivalent to maximizing the expected log likelihood under true data distribution.

$$
\begin{align}
\min_{f} D_\mathrm{KL}(p_\text{data} \| f_\sharp p_Z)
\iff
\max_{f} \mathbb E_{x \sim p_\text{data}} \left[ \log p_X(x) \right], \quad p_X \triangleq f_\sharp p_Z
\end{align}
$$

*Proof*: Let $p_X \triangleq f_\sharp p_Z$. By definition of forward KL, we have

$$
\begin{align*}
D_\mathrm{KL}(p_\text{data} \| p_X)
&= \mathbb E_{x \sim p_\text{data}} \left[ \log\frac{p_\text{data}(x)}{p_X(x)} \right] \\
&= \mathbb E_{x \sim p_\text{data}} \left[ \log p_\text{data}(x) - \log p_X(x) \right] \\
&= \underbrace{\mathbb E_{x \sim p_\text{data}} [\log p_\text{data}(x)]}_{\text{independent of } f} -
   \underbrace{\mathbb E_{x \sim p_\text{data}} [\log p_X(x)]}_{p_X = f_\sharp p_Z \text{ depends on } f} \\
\min_f D_\mathrm{KL}(p_\text{data} \| p_X) &\iff \max_f \mathbb E_{x \sim p_\text{data}} [\log p_X(x)]
\tag*{$\blacksquare$}
\end{align*}
$$

Practical challenges:

1. Trade-off between expressiveness and efficiency:

    * On one side, we want  $f$ is sufficiently expressive to make the resulting push-forward distribution flexible/complex.
    * On the other side, arbitrary flow typically has full-rank Jacobian, which is hard to compute. Hence, we need to impose special structure on $f$ to compute the Jacobian efficiently, limiting the expressiveness of $f$.

2. Difficult to enforce invertibility on $f$ represented by a neural network.

    * How do we enforce invertibility on $f$?
    * How to compute the inverse $f^{-1}$ efficiently?

## Normalizing Flow

Suppose $f_\theta$ is nn representing a normalizing flow.

Let $p_\theta \triangleq [f_\theta]_\sharp p_Z$. Then,

$$
\begin{align}
\min_{\theta} D_\mathrm{KL}(p_\text{data} \| p_\theta)
\iff
\max_{\theta} \mathbb E_{x \sim p_\text{data}} \left[ \log p_\theta(x) \right]
\end{align}
$$

Let $x^{(1)}, \dots, x^{(n)} \stackrel{\text{iid}}{\sim} p_\theta$. Then minimizing forward KL corresponds to maximizing the log likelihood

$$
L(\theta) = \sum_{i=1}^n \log p_\theta(x^{(i)})
$$

TODO: apply the change of var rule. derive the gradient.

## Discrete Normalizing Flow

## Continuous Normalizing Flow

## Appendix

### Examples of Simple Flow Models

**Example 1: rotational flow preseves spherical Gaussian**  
We start with a 2D spherical Gaussian

$$
Z \sim \mathcal{N}(0, I_2)
$$

Then we apply a rotation transformation

$$
f(z) = R_\theta z, \quad
R_\theta =
\begin{pmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{pmatrix}.
$$

Note that the rotation matrix is orthogonal and thus

$$
\begin{align*}
R_\theta^{-1} &= R_\theta^\top
&&\text{invertible}
\\
\vert \det(R_\theta) \vert &= 1
&&\text{vol. preserving}
\\
\forall x \in\mathbb R^2, \: \Vert R_\theta x \Vert &= \Vert x \Vert,
&&\text{norm preserving}
\end{align*}
$$

Therefore, the resulting push-forward distribution is again spherical Gaussian.

$$
\begin{align*}
p_X(x)
&= p_Z\left( R_\theta^{-1}x \right) \cdot \left\vert \det ( R_\theta ) \right\vert^{-1}
\\
&= \frac{1}{2\pi} \exp\left( -\frac{\Vert R_\theta^\top x \Vert^2}{2}\right)
\\
&= \frac{1}{2\pi} \exp\left( -\frac{\Vert x \Vert^2}{2}\right)
\\
\end{align*}
$$

Geometrically, rotating the circular level sets of $p_Z$ does not alter their shapes.

**Example 2: Nonlinear flow from uniform distribution to standard Gaussian**  
TODO: Derive the flow function f which pushes a uniform distribution on [0,1] to standard normal distribution. Give your derivation step by step
