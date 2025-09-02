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

## Normalizing Flow

A flow model is similar to variational autoencoder (VAE), which

1. start from simple prior $z \sim \mathcal N( 0,  I)$.
1. transform via decoder net $x \mid z \sim \mathcal N(\mu_{\theta}(z), \Sigma_{\theta}(z))$.
1. yields a complex/flexible marginal $p_\theta(x)$.

A flow is a **differentiable** and **invertible** vector field $f: \mathbb R^d \to \mathbb R^d$ such that

$$
\begin{align}
x = f(z)
\end{align}
$$

Remarks:

* what does the normalizing in the name mean?
* A flow can be viewed as a deterministic and invertible VAE, such that each $x$ is corresponded with a unique $z$.
* There is no more compressed latent representation in flow model as $x$ and $z$ have the same dimension.

Consider random vector $Z$ with PDF $p_Z$. Let random vector $X$ be defined by as $X = f(Z)$. By change of variables rule, the PDF of $X$ is

$$
\begin{align}
p_X(x) = p_Z\left( f^{-1}(x) \right) \cdot \left\vert \det \Big( D_{f^{-1}}(x) \Big) \right\vert
\end{align}
$$

where $D_{f^{-1}}(x)$ is the Jacobian matrix of $f^{-1}$ evaluated at $x$.

By the fact that $\det(A^{-1}) = \det(A)^{-1}$, the push forward density can be written as

$$
\begin{align}
p_X(x) = p_Z\left( z \right) \cdot \left\vert \det \Big( D_{f}(z) \Big) \right\vert^{-1}, \quad
z = f^{-1}(x)
\end{align}
$$

Remarks:

* The PDF $p_X$ is called the push-forward distribution of $p_Z$ by $f$, denoted by $p_X = f_\sharp p_Z$.
* In 1D case, the formula simplifies to
    $$
    \begin{align}
    p_X(x)
    &= p_Z\left( z \right) \cdot \left\vert f'(z) \right\vert^{-1}, \quad
    z = f^{-1}(x)
    \end{align}
    $$