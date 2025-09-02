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

A flow is a **differentiable** and **invertible** vector field $f: \mathbb R^d \to \mathbb R^d$ such that

$$
\begin{align}
x = f(z)
\end{align}
$$

Remarks:

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

* The PDF $p_X$ is called the **push-forward distribution** of $p_Z$ by $f$, denoted by $p_X = f_\sharp p_Z$.
* Computing the Jacobian determinant typically costs $O(d^3)$ time unless $f$ has special structure.
* While every flow induces a valid push-forward distribution, a flow is called a ***normalizing flow*** only when its Jacobian determinant can be evaluated efficiently.
* In 1D case, the formula simplifies to
    $$
    p_X(x)
    = p_Z\left( z \right) \cdot \left\vert f'(z) \right\vert^{-1}, \quad
    z = f^{-1}(x)
    $$

---

**Example: rotational flow applied to a spherical Gaussian**  

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

---

Given $p_Z$ and a flow model $f$, it is easy to compute the push-forward distribution $f_\sharp p_Z$. However, the inverse problems are challenging

1. Given $p_Z$ and $p_X$, how to design a flow model $f$ such that $f_\sharp p_Z \approx p_X$ ? 
1. In more practical settings, $p_X$ is unknown and we can only sample from it. Given $p_Z$ and samples of $X$, how to design/learn a flow model $f$ such that $f_\sharp p_Z \approx p_X$ ? 

### Existence and Non-Uniqueness of Optimal Flow

TODO: Derive the flow function f which pushes a uniform distribution on [0,1] to standard normal distribution. Give your derivation step by step

## Discrete Normalizing Flow

## Continuous Normalizing Flow
