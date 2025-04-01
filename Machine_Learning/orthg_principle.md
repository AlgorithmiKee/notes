---
title: "Orthogonality Principle"
author: "Ke Zhang"
date: "2025"
fontsize: 12pt
---

**TODO**: move this file to folder *signal processing* after migraion.

# Orthogonality Principle

Let $(H,\mathbb R)$ be a Hilber space, i.e. it satisfies all of

1. $(H,\mathbb R)$ is a vector space.
1. $H$ is equipped with inner product $\langle\cdot,\cdot\rangle: H\times H \to \mathbb R$.
1. $H$ is complete w.r.t. the distance induced by $\langle\cdot,\cdot\rangle$.

Remark: Every inner product induces a norm and thus a distance.

$$
\Vert \mathbf x \Vert = \sqrt{\langle\mathbf x,\mathbf x\rangle},
\quad
d(\mathbf x, \mathbf y) = \Vert \mathbf x - \mathbf y \Vert
$$

Commonly used Hilbert spaces and their inner products

* Euclidean space $\mathbb R^n$ where
  $$
  \langle\mathbf x, \mathbf y\rangle = \mathbf x^\top \mathbf y
  $$

* Space of square integrable functions $L^2(\mathbb R,\mathbb R)=\{f:\mathbb R \to\mathbb R \mid \int_{-\infty}^{\infty} \vert f(t) \vert^2 \,\mathrm dt <\infty \}$ where
  $$
  \langle f,g \rangle = \int_{\mathbb R} f(t)g(t) \,\mathrm dt
  $$

* Space of random variables with finite 2nd order moment $L^2(\Omega,\mathbb R)=\{X:\Omega\to\mathbb R\mid \mathbb E[X^2] <\infty \}$ where
  $$
  \langle X,Y\rangle = \mathbb E[XY] = \int_{\Omega} X(\omega)Y(\omega) \,\mathrm d\mathbb P
  $$

**Subspace Approximation Problem**  
Let $\mathbf x_1, \dots, \mathbf x_n, \mathbf y\in H$. We woule like to find a vector $\hat{\mathbf y}\in\operatorname{span}(\mathbf x_1, \dots, \mathbf x_n)$ s.t. $\hat{\mathbf y}$ is as close to $\mathbf y$ as possible. Formally, we would like to solve

> $$
> \begin{align}
> \min_{\hat{\mathbf y}\in\operatorname{span}(\mathbf x_1, \dots, \mathbf x_n)}
> \Vert \hat{\mathbf y} - \mathbf y \Vert^2
> \end{align}
> $$

or equivalently

> $$
> \begin{align}
> \min_{w_1, \dots, w_n\in\mathbb R}
> \left\Vert \sum_{k=1}^n  w_k\mathbf x_k - \mathbf y \right\Vert^2
> \end{align}
> $$

Remark:

* The equivalent formulation is straight forward as $\hat{\mathbf y}=\sum_{k=1}^n  w_k\mathbf x_k$.
* The spanning vectors $\mathbf x_1, \dots, \mathbf x_n$ are not necessarily linearly independent.

**Orthogonality Principle**  
Let $U=\operatorname{span}(\mathbf x_1, \dots, \mathbf x_n)$. The optimal solution of the subspace approximation problem is the orthogonal projection of $\mathbf y$ to $U$. The approximation error $\hat{\mathbf y} - \mathbf y$ lies in the orthogonal complement of $U$. In particular,

> $$
> \begin{align}
> \langle \hat{\mathbf y} - \mathbf y, \mathbf x_k \rangle = 0, \: \forall > k=1,\dots,n
> \end{align}
> $$

Now, we calculate the optimal coefficients as follows. Reformulate the orthogonality principle into
$$
\begin{align*}
\langle \hat{\mathbf y} - \mathbf y, \mathbf x_k \rangle
&= 0
\\
\langle \hat{\mathbf y},\mathbf x_k \rangle
&= \langle\mathbf y,\mathbf x_k \rangle
\\
\left\langle \sum_{\ell=1}^n w_\ell\mathbf x_\ell, \mathbf x_k\right\rangle
&= \langle\mathbf y,\mathbf x_k \rangle
\\
\sum_{\ell=1}^n w_\ell \left\langle\mathbf x_\ell, \mathbf x_k\right\rangle
&= \langle\mathbf y,\mathbf x_k \rangle
\end{align*}
$$

For inner product with codomain $\mathbb R$, we have $\langle\mathbf u,\mathbf v\rangle = \langle\mathbf v,\mathbf u\rangle$. Thus, the orthogonality principle is equivalent to
> $$
> \begin{align}
> \sum_{\ell=1}^n w_\ell \left\langle\mathbf x_k, \mathbf x_\ell\right\rangle
> = \langle\mathbf x_k,\mathbf y \rangle,
> \:\forall k=1,\dots,n
> \end{align}
> $$

which is a linear system of $n$ equations with $n$ unknown coefficients.

> $$
> \begin{align}
> \underbrace{
> \begin{bmatrix}
> \langle\mathbf x_1,\mathbf x_1\rangle & \langle\mathbf x_1,\mathbf x_2\rangle & \cdots & \langle\mathbf x_1,\mathbf x_n\rangle
> \\
> \langle\mathbf x_2,\mathbf x_1\rangle & \langle\mathbf x_2,\mathbf x_2\rangle & \cdots & \langle\mathbf x_2,\mathbf x_n\rangle
> \\
> \vdots &\vdots &\ddots &\vdots
> \\
> \langle\mathbf x_n,\mathbf x_1\rangle & \langle\mathbf x_n,\mathbf x_2\rangle & \cdots & \langle\mathbf x_n,\mathbf x_n\rangle
> \end{bmatrix}
> }_{\mathbf G\in\mathbb R^{n\times n}}
> %%%
> \underbrace{
> \begin{bmatrix}
> w_1 \\ w_2 \\ \vdots \\ w_n
> \end{bmatrix}
> }_{\mathbf w \in\mathbb R^n}
> =%%%
> \underbrace{
> \begin{bmatrix}
> \langle\mathbf x_1,\mathbf y\rangle \\ \langle\mathbf x_2,\mathbf y\rangle \\ \vdots \\ \langle\mathbf x_n,\mathbf y\rangle
> \end{bmatrix}
> }_{\mathbf r\in\mathbb R^n}
> 
> \end{align}
> $$

Remarks:

* The matrix $\mathbf G$ is called ***Gram matrix*** (or ***kernel matrix***) of $\mathbf x_1, \dots, \mathbf x_n$. The element $G_{ij}=\langle\mathbf x_i,\mathbf x_j\rangle$ describes the similarity between $\mathbf x_i$ and $\mathbf x_j$. Likewise, the term $r_{i}=\langle\mathbf x_i,\mathbf y\rangle$ on the RHS describes similarity between $\mathbf x_i$ and $\mathbf y$. 
* The Gram matrix $\mathbf G$ is invertible iff $\mathbf x_1, \dots, \mathbf x_n$ is linearly independent vectors in $H$. (c.f. Appendix for proof.) Hence, in general, the optimal $\mathbf w$ is not unique. However, the optimal approximation $\hat{\mathbf y}$ is always unique due to uniqueness of orthogonal projection.

Special case: $\mathbf x_1, \dots, \mathbf x_n$ are nonzero and orthogonal to each other. The Gram matrix becomes a diagonal matrix and thus invertible
$$
\mathbf G = \operatorname{diag}(
\langle\mathbf x_1,\mathbf x_1\rangle,
\langle\mathbf x_2,\mathbf x_2\rangle,
\dots,
\langle\mathbf x_n,\mathbf x_n\rangle
)
$$
The optimal coefficents becomes
$$
w_k
= \frac{\langle\mathbf x_k,\mathbf y\rangle}{\langle\mathbf x_k,\mathbf x_k\rangle}
= \frac{\langle\mathbf x_k,\mathbf y\rangle}{\Vert\mathbf x_k\Vert^2}
$$
The orthogonal projection becomes
$$
\hat{\mathbf y} = \sum_{k=1}^n \frac{\langle\mathbf x_k,\mathbf y\rangle}{\Vert\mathbf x_k\Vert^2} \mathbf x_k
$$
If $\mathbf x_1, \dots, \mathbf x_n$ are in addition orthonormal (i.e. $\langle\mathbf x_i,\mathbf x_j\rangle = \mathbb I[i=j]$), the results simplifies further to
$$
w_k = \langle\mathbf x_k,\mathbf y\rangle,
\quad
\hat{\mathbf y} = \sum_{k=1}^n \langle\mathbf x_k,\mathbf y\rangle \mathbf x_k
$$

## Connection to Least Square

## Connection to Fourier Series

## Connection to LMMSE

## Appendix

### Invertibility of Gram Matrix

TODO
