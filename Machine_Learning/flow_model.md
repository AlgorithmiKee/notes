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

A variational autoencoder (VAE) models $p_\text{data}$ using deep latent varible models:

1. start from simple prior distribution $z \sim \mathcal N( 0,  I)$.

1. transform via decoder net $x \mid z \sim \mathcal N(\mu_{\theta}(z), \Sigma_{\theta}(z))$.

1. yields a complex/flexible marginal distribution
   $$
   p_\theta(x) = \int p_\theta(x,z) \,\mathrm dz \approx p_\text{data}(x)
   $$

A flow model is similar in spirit, as it also transforms a simple distribution into a complex one. However, the mechanism is different: flows model the transformation directly as an invertible mapping.

1. start from simple base distribution $z \sim \mathcal N( 0,  I)$.

1. transform via a flow $x = f(z)$.

1. yields a complex/flexible push-forward distribution
   $$
   p(x) = (f_\sharp p_Z)(x) \approx p_\text{data}(x)
   $$

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

A collection of flows $f_1,\dots,f_T$ can be composed into a new flow.

$$
\begin{align}
f = f_T \circ \dots \circ f_1
\end{align}
$$

Since each flow is invertible, the composition is also invertible
$$
\begin{align}
f^{-1} = f_1^{-1} \circ \dots \circ f_T^{-1}
\end{align}
$$

Illustration:

$$
z \triangleq
x_0    \rightarrow \boxed{\phantom{X} f_1^{\phantom{-1}}} \rightarrow
x_1    \rightarrow \boxed{\phantom{X} f_2^{\phantom{-1}}} \rightarrow
x_2    \rightarrow \boxed{\phantom{X} f_3^{\phantom{-1}}} \rightarrow
\cdots \rightarrow \boxed{\phantom{X} f_T^{\phantom{-1}}} \rightarrow
x_T \triangleq x
\\[8pt]
z \triangleq
x_0    \leftarrow \boxed{\phantom{X} f_1^{-1}} \leftarrow
x_1    \leftarrow \boxed{\phantom{X} f_2^{-1}} \leftarrow
x_2    \leftarrow \boxed{\phantom{X} f_3^{-1}} \leftarrow
\cdots \leftarrow \boxed{\phantom{X} f_T^{-1}} \leftarrow
x_T \triangleq x
$$

For each $t=1,\dots,T$, let $D_{f_t}(u)$ denote the Jacobian of $f_t$ evaluated at $u\in\mathbb R^d$. By the chain rule, the composition is also differentiable with the Jacobian

$$
\begin{align}
D_{f}(z) = D_{f_T}(x_{T-1}) \cdots D_{f_2}(x_{1}) \cdot D_{f_1}(x_{0}),
\quad x_0 \triangleq z
\end{align}
$$

The inverse $f^{-1}$ is also differentiable with the Jacobian

$$
\begin{align}
D_{f^{-1}}(x)
&= [D_{f}(z)]^{-1} \\
&= [D_{f_1}(x_{0})]^{-1} \cdot [D_{f_2}(x_{1})]^{-1} \cdots [D_{f_T}(x_{T-1})]^{-1} ,
&& x_0 \triangleq z \\
&= [D_{f_1^{-1}}(x_{1})]^{-1} \cdot [D_{f_2^{-1}}(x_{2})]^{-1} \cdots [D_{f_T^{-1}}(x_T)]^{-1} ,
&& x_T \triangleq x
\end{align}
$$

### Push-Forward Distribution

Let $Z$ be a random vector with a simple **base distribution** $p_Z$. Define $X = f(Z)$. The distribution $p_X = f_\sharp p_Z$ is the **push-forward distribution** of $p_Z$ under $f$. By change of variables formula,

$$
\begin{align}
p_X(x)
&= p_Z\left( f^{-1}(x) \right) \cdot \left\vert \det( D_{f^{-1}}(x)) \right\vert \\
&= p_Z\left( z \right) \cdot \Big\vert \det( D_{f}(z)) \Big\vert^{-1}, \quad
z = f^{-1}(x)
\end{align}
$$

where $D_{f^{-1}}(x)$ is the Jacobian matrix of $f^{-1}$ evaluated at $x$.

Remarks:

* The 2nd equation follows from $\det(A^{-1}) = \det(A)^{-1}$.
* In 1D case, the formula simplifies to
    $$
    p_X(x)
    = p_Z\left( z \right) \cdot \left\vert f'(z) \right\vert^{-1}, \quad
    z = f^{-1}(x)
    $$
* If $f$ is sufficiently expressive, it pushes the simple base distribution to a complex one. Conversely, $f^{-1}$ pushes a complex distribution back to a simpler one.
    $$
    \begin{align}
    p_X = f_\sharp p_Z, \quad p_Z = [f^{-1}]_\sharp p_X
    \end{align}
    $$

Applying change of variable rule requires

* Computing the inverse $f^{-1}$.
* Computing the Jacobian determinant, which typically costs $O(d^3)$ time unless $f$ has special structure. The computation is infeasible in higher dimension.

Therefore, while every flow induces a valid push-forward distribution, a flow is called a ***normalizing flow*** only when its inverse and Jacobian determinant can be evaluated efficiently. Normalizing flows often use flows with diagonal, triangular, or other structured Jacobians.

### The Learning Problem

Given a simple base distribution $p_Z$, how can we learn a flow $f$ so that the resulting push-forward distribution $f_\sharp p_Z$ matches the true data distribution $p_\text{data}$? Formally, we want $f$ to minimize the **forward** KL divergence

$$
\begin{align}
\min_{f} D_\mathrm{KL}(p_\text{data} \| f_\sharp p_Z)
\end{align}
$$

Remarks:

* In practice, the flow $f$ is a deep neural network. Learning $f$ boils down to learning network parameters.
* Flow models are generative: Once we learned $f$, we can generate samples $x$ by first sampling $z \sim p_Z$ and then applying $f$.

> Theoretically, under regularity assumptions, there always exist infinitely many flows pushing $p_Z$ forward to $p_\text{data}$. 

The proof of existence requires transport theory -- very hardcore math. Here, we only illustrate the non-uniqueness of such flows. Suppose $Z\sim\mathcal N(0, I_d)$ and $f$ pushes $p_Z$ forward to $p_\text{data}$. Then for any orthogonal linear map $g: \mathbb R^d \to \mathbb R^d$ (e.g. rotation), the composition flow $f \circ g$ also pushes $p_Z$ forward to $p_\text{data}$.

$$
(f \circ g)_\sharp p_Z = f_\sharp(g_\sharp p_Z) = f_\sharp p_Z = p_\text{data}
$$

However, the transport theory does not provide an explicit algorithm to compute such flows. In practice, we can only approximate them.

**Likelihood Formulation**  
The learning problem is equivalent to maximizing the expected log-likelihood under true data distribution.
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
\end{align*}
$$

The first term is independent of $f$, so minimizing the KL divergence is equivalent to maximizing $\mathbb E_{x \sim p_\text{data}} [\log p_X(x)]$. $\;\blacksquare$

**Practical challenges**:

1. Trade-off between expressiveness and efficiency:

    * Expressiveness requires $f$ to be flexible enough to yield complex distributions.
    * Efficiency requires structured Jacobians so that $\det(D_f)$ can be computed quickly. Arbitrary flows usually have full-rank Jacobians, which are expensive to compute.

2. Invertibility of neural nets

    * Enforcing invertibility in a neural network is nontrivial.
    * Efficient computation of $f^{-1}$ is another challenge.

## Normalizing Flow

Suppose $f_\theta$ is a neural network parameterizing a normalizing flow. Let

$$
p_\theta \triangleq [f_\theta]_\sharp p_Z
$$
Then,
$$
\begin{align}
\min_{\theta} D_\mathrm{KL}(p_\text{data} \| p_\theta)
\iff
\max_{\theta} \mathbb E_{x \sim p_\text{data}} \left[ \log p_\theta(x) \right]
\end{align}
$$

Let $x^{(1)}, \dots, x^{(n)} \stackrel{\text{iid}}{\sim} p_\text{data}$. The expected log-likelihood can be approximated by empirical average:
$$
\mathbb E_{x \sim p_\text{data}} \left[ \log p_\theta(x) \right]
\approx \frac{1}{n} \sum_{i=1}^n \log p_\theta(x^{(i)})
$$
Hence, minimizing the forward KL is equivalent to maximizing the log-likelihood of the dataset
$$
\begin{align}
L(\theta)
&= \sum_{i=1}^n \log p_\theta(x^{(i)}) \\
&\triangleq \log p_{\theta}(x^{(1)}, \dots, x^{(n)})
\end{align}
$$

By change of variable, the log-likelihood of each sample $x^{(i)}$ is
$$
\begin{align}
\log p_\theta(x^{(i)})
&= \log \left[ p_Z( z^{(i)}) \cdot \left\vert \det(D_{f_\theta}(z^{(i)})) \right\vert^{-1} \right],
&& z^{(i)} = f_\theta^{-1}(x^{(i)}) \nonumber
\\[6pt]
&= \log p_Z( z^{(i)}) - \log \left\vert \det(D_{f_\theta}(z^{(i)})) \right\vert,
&& z^{(i)} = f_\theta^{-1}(x^{(i)}) \nonumber
\end{align}
$$
Hence, the objective function becomes
$$
\begin{align}
L(\theta)
&= \sum_{i=1}^n \log p_Z( z^{(i)}) - \log \left\vert \det(D_{f_\theta}(z^{(i)})) \right\vert,
&& z^{(i)} = f_\theta^{-1}(x^{(i)})
\end{align}
$$
The gradient is
$$
\begin{align}
\nabla_\theta L = \sum_{i=1}^n \nabla_\theta\left(
	\log p_Z( z^{(i)}) - \log \left\vert \det(D_{f_\theta}(z^{(i)})) \right\vert
\right)
\end{align}
$$

### Efficient Jacobian Determinant

The central challenge is that computing $\det D_{f_\theta}(z)$ is $O(d^3)$ in general. Normalizing flows impose special structure on $f_\theta$ so that:

* $f_\theta$ is invertible with easy-to-compute inverse
* **$\det(D_{f_\theta})$ **(or its log) is cheap to compute

Popular design patterns:

1. **Volume-Preserving Flows**: $\det D_{f_\theta}(z) = 1$, so the Jacobian determinant is trivial.
2. **Triangular / Affine Coupling Layers**: Impose block triangular structure on $D_{f_\theta}(z)$.
3. **Autoregressive Flows**: Impose lower triangular structure on $D_{f_\theta}(z)$.
4. **Continuous Normalizing Flows**: Avoid explicit determinants but requires solving ODEs numerically.

## Discrete Normalizing Flow

From a complex flow model by composing

* Volume-Preserving Flows
* Triangular / Affine Coupling Layers
* Autoregressive Flows

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
