---
title: "Orthogonality Principle"
author: "Ke Zhang"
date: "2025"
fontsize: 12pt
---

**TODO**: move this file to folder *signal processing* after migraion.

# Orthogonality Principle

[toc]

$$
\newcommand{\herm}{\mathsf{H}}
$$

Let $\mathbb F$ denote either $\mathbb R$ or $\mathbb C$. Then, $(H,\mathbb F)$ is a Hilber space iff all of the following are true

1. $(H,\mathbb F)$ is a vector space.
1. $H$ is equipped with inner product $\langle\cdot,\cdot\rangle: H\times H \to \mathbb F$.
1. $H$ is complete w.r.t. the distance induced by $\langle\cdot,\cdot\rangle$.

Remark:

* The axioms of inner product are listed in appendix. Here, we highlight that $\langle\cdot,\cdot\rangle$ is **conjugate** linear w.r.t. the **1st** argument and **linear** w.r.t. the **2nd** argument. This is consistent with physics/engineering convention. NumPy and MATLAB also follows this convention.

    $$
    \langle\lambda\mathbf x, \mathbf y\rangle = \overline\lambda\langle\mathbf x, \mathbf y\rangle,
    \quad
    \langle\mathbf x, \lambda\mathbf y\rangle = \lambda\langle\mathbf x, \mathbf y\rangle
    $$

    ```python
    import numpy as np
    a = np.array([1+2j, 3+4j])
    b = np.array([5+6j, 7+8j])
    print(np.vdot(a, b))  # Output: (70-8j)
    ```

* If $\mathbb F = \mathbb R$, then $\langle\cdot,\cdot\rangle$ becomes a bilinear form, i.e. linear w.r.t. both arguments.
* Every inner product induces a norm and thus a distance.

$$
\Vert \mathbf x \Vert = \sqrt{\langle\mathbf x,\mathbf x\rangle},
\quad
d(\mathbf x, \mathbf y) = \Vert \mathbf x - \mathbf y \Vert
$$

Commonly used Hilbert spaces and their inner products

* Euclidean space $\mathbb R^n$.
  $$
  \langle\mathbf x, \mathbf y\rangle = \mathbf x^\top \mathbf y = \sum_{i=1}^n x_i y_i
  $$

* Complex coordinate space $\mathbb C^n$.
  $$
  \langle\mathbf x, \mathbf y\rangle = \mathbf x^\herm \mathbf y = \sum_{i=1}^n \overline{x_i} y_i
  $$

* Space of square integrable functions $L^2(\mathbb R,\mathbb F)=\{f:\mathbb R \to\mathbb F \mid \int_{-\infty}^{\infty} \vert f(t) \vert^2 \,\mathrm dt <\infty \}$.
  $$
  \langle f,g \rangle = \int_{\mathbb R} \overline{f(t)}g(t) \,\mathrm dt
  $$

* Space of random variables with finite 2nd order moment $L^2(\Omega,\mathbb F)=\{X:\Omega\to\mathbb F\mid \mathbb E[\vert X\vert^2] <\infty \}$.
  $$
  \langle X,Y\rangle = \mathbb E[\overline{X}Y] = \int_{\Omega} \overline{X(\omega)}Y(\omega) \,\mathrm d\mathbb P
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
> \min_{w_1, \dots, w_n\in\mathbb F}
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
> \langle \hat{\mathbf y} - \mathbf y, \mathbf x_k \rangle = 0, \: \forall k=1,\dots,n
> \end{align}
> $$

Now, we calculate the optimal coefficients as follows. Reformulate the orthogonality principle into
$$
\begin{align*}
\langle \hat{\mathbf y},\mathbf x_k \rangle
&= \langle\mathbf y,\mathbf x_k \rangle
\\
\left\langle \sum_{\ell=1}^n w_\ell\mathbf x_\ell, \mathbf x_k\right\rangle
&= \langle\mathbf y,\mathbf x_k \rangle
\\
\sum_{\ell=1}^n \overline{w_\ell} \left\langle\mathbf x_\ell, \mathbf x_k\right\rangle
&= \langle\mathbf y,\mathbf x_k \rangle
\end{align*}
$$

Taking the complex conjugate on both sides, we get a linear system of $n$ equations with $n$ unknown coefficients.

> $$
> \begin{align}
> \sum_{\ell=1}^n \left\langle\mathbf x_k, \mathbf x_\ell\right\rangle w_\ell
> = \langle\mathbf x_k,\mathbf y \rangle,
> \:\forall k=1,\dots,n
> \end{align}
> $$

Matrix form:

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
> }_{\mathbf G\in\mathbb F^{n\times n}}
> %%%
> \underbrace{
> \begin{bmatrix}
> w_1 \\ w_2 \\ \vdots \\ w_n
> \end{bmatrix}
> }_{\mathbf w \in\mathbb F^n}
> =%%%
> \underbrace{
> \begin{bmatrix}
> \langle\mathbf x_1,\mathbf y\rangle \\ \langle\mathbf x_2,\mathbf y\rangle \\ \vdots \\ \langle\mathbf x_n,\mathbf y\rangle
> \end{bmatrix}
> }_{\mathbf r\in\mathbb F^n}
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

## Ordinary Least Square

Consider $H=\mathbb R^m$ and $\mathbb F=\mathbb R$. We would like to approximate a vector $\mathbf y$ in the span of $\mathbf x_1, \dots, \mathbf x_n$.

Note that the approximation $\hat{\mathbf y}$ can be written as matrix vector product.

$$
\hat{\mathbf y} = \sum_{k=1}^n  w_k\mathbf x_k = \mathbf{Xw},
\:\text{ where }\:
\mathbf X \triangleq [\mathbf x_1, \dots, \mathbf x_n] \in\mathbb R^{m\times n}

$$

The subspace approximation problem

$$
\min_{\hat{\mathbf y}\in\operatorname{span}(\mathbf x_1, \dots, \mathbf x_n)}
\Vert \hat{\mathbf y} - \mathbf y \Vert^2
$$

becomes **ordinary least square**

$$
\begin{align}
\min_{\mathbf w\in\mathbb R^n}
\left\Vert \mathbf{Xw} - \mathbf y \right\Vert^2
\end{align}
$$

For $\forall k=1,\dots,n$, the orthogonality principle $\langle \hat{\mathbf y} - \mathbf y, \mathbf x_k \rangle = 0$ becomes

$$
\begin{align}
(\hat{\mathbf y} - \mathbf y)^\top \mathbf x_k &= 0 \\
(\mathbf{Xw} - \mathbf y)^\top \mathbf x_k &= 0 \\
\end{align}
$$

which can be written more compactly as

$$
\begin{align}
(\mathbf{Xw} - \mathbf y)^\top [\mathbf x_1,\dots,\mathbf x_n] &= [0,\dots,0] \nonumber
\\
(\mathbf{Xw} - \mathbf y)^\top \mathbf X &= \mathbf 0^\top \nonumber
\\
\mathbf X^\top (\mathbf{Xw} - \mathbf y) &= \mathbf 0
\iff \nabla_{\mathbf w} \Vert \mathbf{Xw} - \mathbf y \Vert^2 =0
\end{align}
$$

Hence, we can the optimal weights by solving

$$
\begin{align}
\mathbf X^\top \mathbf{Xw} = \mathbf {X^\top y}
\end{align}
$$

Remarks:

* One can verify that $\mathbf X^\top\mathbf X = \mathbf G$ with. $G_{ij} = \mathbf x_i^\top \mathbf x_j$ and that $\mathbf X^\top\mathbf y=\mathbf r$ with $r_{i} = \mathbf x_i^\top \mathbf y$.
* If $\mathbf X^\top\mathbf X$ is invertible, we have unique solution $\mathbf{w} = (\mathbf X^\top\mathbf X)^{-1}\mathbf {X^\top y}$.

Relation to linear regression: Recall that $\mathbf X\in\mathbb R^{m\times n}$. Consider each row of $\mathbf X$ as a data point in $\mathbb R^n$. Then, $\mathbf X$ represents $m$ data points in $\mathbb R^n$ while $\mathbf y\in\mathbb R^{m}$ represents their corresponding labels.

* \#training samples: $m$.
* \#features per sample: $n$.
* $i$-th tranning sample: $i$-th row of $\mathbf X$ and $y_i$.

## LMMSE

## Connection to Fourier Series

## Appendix

### Axiomic Definition of Inner Product

Let $(V,\mathbb F)$ be a vector space. Then, $\langle\cdot,\cdot\rangle: V\times V \to \mathbb F$ is called a inner product iff all of

1. conjugate symmetry: $\langle\mathbf x,\mathbf y\rangle = \overline{\langle\mathbf x,\mathbf y\rangle}$
1. linear w.r.t. the 2nd argument:  $\langle\mathbf x, \lambda\mathbf y\rangle = \lambda\langle\mathbf x, \mathbf y\rangle$ and $\langle\mathbf x, \mathbf y + \mathbf z\rangle = \langle\mathbf x,\mathbf y\rangle + \langle\mathbf x,\mathbf z\rangle$
1. positive definite: $\langle\mathbf x,\mathbf x\rangle \ge 0$ with equality iff $\mathbf x = \mathbf 0$

Elementary properties:

* $\langle\lambda\mathbf x,\mathbf y\rangle = \overline\lambda\langle\mathbf x,\mathbf y\rangle$
* $\langle\mathbf x + \mathbf y, \mathbf z\rangle = \langle\mathbf x,\mathbf y\rangle + \langle\mathbf x,\mathbf z\rangle$
* $\langle\mathbf x, \mathbf 0\rangle = 0$

Remarks:

* In general, $\langle\mathbf x,\mathbf y\rangle \ne \langle\mathbf y,\mathbf x\rangle$ unless $\mathbb F = \mathbb R$.
* In math literatures, the inner product is often defined such that it is linear w.r.t the 1st argument.

### Invertibility of Gram Matrix

TODO
