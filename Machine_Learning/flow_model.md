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

Recall: How does a variational autoencoder (VAE) model a complex distribution?

1. start from simple prior $z \sim \mathcal N( 0,  I)$.
1. transform via decoder net $x \mid z \sim \mathcal N(\mu_{\theta}(z), \Sigma_{\theta}(z))$.
1. yields a complex/flexible marginal $p_\theta(x)$.

A flow model is similar to a VAE in the sense that it transforms simple distribution to a complex one. However, flow models use a different approach to achieve this goal.

## Basic Idea of Flow Models

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
* A flow model can be viewed as a deterministic and invertible VAE, such that each $x$ corresponds to a unique $z$.
* Unlike VAEs, a flow model does not compress information into a lower-dimensional latent space since $x$ and $z$ have the **same** dimension.

Consider random vector $Z$ with PDF $p_Z$. Let random vector $X$ be defined by as $X = f(Z)$. By change of variables rule, the PDF of $X$ is

$$
\begin{align}
p_X(x) = p_Z\left( f^{-1}(x) \right) \cdot \left\vert \det \Big( D_{f^{-1}}(x) \Big) \right\vert
\end{align}
$$

where $D_{f^{-1}}(x)$ is the Jacobian matrix of $f^{-1}$ evaluated at $x$.

Using the fact $\det(A^{-1}) = \det(A)^{-1}$, we can rewrite $p_X$ as

$$
\begin{align}
p_X(x) = p_Z\left( z \right) \cdot \left\vert \det \Big( D_{f}(z) \Big) \right\vert^{-1}, \quad
z = f^{-1}(x)
\end{align}
$$

Remarks:

* The PDF $p_X$ is called the **push-forward distribution** of $p_Z$ by $f$, denoted by $p_X = f_\sharp p_Z$. The push-forward distribution can model complex data patterns provided that $f$ is sufficiently expressive.
* Computing the Jacobian determinant typically costs $O(d^3)$ time unless $f$ has special structure. The computation is infeasible in higher dimension.
* While every flow induces a valid push-forward distribution, a flow is called a ***normalizing flow*** only when its Jacobian determinant can be evaluated efficiently.
* In 1D case, the formula simplifies to
    $$
    p_X(x)
    = p_Z\left( z \right) \cdot \left\vert f'(z) \right\vert^{-1}, \quad
    z = f^{-1}(x)
    $$

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

### Central Tasks Around Flow Models

Flow models are generative in nature. Given $p_Z$ and a flow model $f$, we can generate samples of $X$ by first sampling $z \sim p_Z$ and then applying $f$. Computing the push-forward distribution $f_\sharp p_Z = p_X$ is also straight foward.

However, the inverse problems are challenging

1. Given $p_Z$ and $p_X$, how to design a flow model $f$ such that $f_\sharp p_Z \approx p_X$ or even $f_\sharp p_Z = p_X$?
1. In more practical settings, $p_X$ is unknown and we can only sample from it. Given $p_Z$ and samples from $p_X$, how to design/learn a flow model $f$ such that $f_\sharp p_Z \approx p_X$ ?

Before we answer the 1st question, we acknowledge that under regularity assumptions:

> There exists infinitely many flows $f$ s.t. $f_\sharp p_Z = p_X$ for given $p_Z$ and $p_X$.

The proof of existence requires tranport theory -- very hardcore math. Here we only illustrate the non-uniqueness of such flows. Suppose $Z\sim\mathcal N(0, I_d)$ and $f$ pushes $p_Z$ forward to $p_X$. Then for any orthogonal linear map $g: \mathbb R^d \to \mathbb R^d$ (e.g. rotation), the composition flow $f \circ g$ also pushes $p_Z$ forward to $p_X$.

$$
g_\sharp p_Z = p_Z \implies
(f \circ g)_\sharp p_Z = f_\sharp(g_\sharp p_Z) = f_\sharp p_Z = p_X
$$

However, the transport theory does not directly give an algorithm to compute such flows. In practice, we can only approximate one of these flows. Problem 1 then becomes

$$
\min_{f} D_\mathrm{KL}(p_X \| f_\sharp p_Z)
$$

Remarks:

* $p_X$ denotes the unknown true distribution of $X$. In ML papers, it is also denoted by $p_\text{data}$.
* We optimize the forward KL, which corresponds to maximum likelihood principle.

TODO:

* stop using $p_X$.
* push forward by flow:
  * $(f_\sharp p_Z)(x)$ before nn
  * $p_{\theta}(x)$ after nn
* ground truth: $p_\text{data}(x)$

## Discrete Normalizing Flow

## Continuous Normalizing Flow
