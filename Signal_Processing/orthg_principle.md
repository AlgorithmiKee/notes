---
title: "Orthogonality Principle"
author: "Ke Zhang"
date: "2025"
fontsize: 12pt
---

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
d(\mathbf x, \mathbf y) = \Vert \mathbf x - \mathbf y \Vert = \sqrt{\langle\mathbf x - \mathbf y,\mathbf x - \mathbf y\rangle}
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

* Space of square integrable functions $L^2(I,\mathbb F)=\{f:I \to\mathbb F \mid \int_{t\in I} \vert f(t) \vert^2 \,\mathrm dt <\infty \}$ where $I\subseteq\mathbb R$.
  $$
  \langle f,g \rangle = \int_{I} \overline{f(t)}g(t) \,\mathrm dt
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
* The  Gram matrix $\mathbf G$ is Hermitian, i.e. $\mathbf G^\herm = \mathbf G$. If $\mathbb F = \mathbb R$, then $\mathbf G$ becomes symmetric.
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

For all $k=1,\dots,n$, the orthogonality principle $\langle \hat{\mathbf y} - \mathbf y, \mathbf x_k \rangle = 0$ becomes

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

## LMMSE Estimation

Consider $H=\{X:\Omega\to\mathbb F\mid \mathbb E[\vert X\vert^2] <\infty \}$ where $\mathbb F$ is either $\mathbb R$ or $\mathbb C$. We would like to approximate a random variable $Y$ using a linear combination of $X_1,\dots,X_n$.

$$
\hat Y = \sum_{k=1}^n w_k X_k, \quad w_1,\dots,w_n\in\mathbb F
$$

Recall for two random variables $U,V\in H$, we have $\langle U,V\rangle = \mathbb E[\overline{U}V]$ and $\Vert U-V \Vert^2 = \mathbb E[\vert U-V \vert^2]$.

The subspace approximation problem

$$
\min_{\hat{Y}\in\operatorname{span}(X_1, \dots, X_n)}
\Vert \hat{Y} - Y \Vert^2
$$

becomes **Linear Minimum Mean Square Error (LMMSE)** estimation:

$$
\begin{align}
\min_{\hat{Y}\in\operatorname{span}(X_1, \dots, X_n)} & \mathbb E
\left[
  \big\vert \hat{Y} - Y \big\vert^2
\right]
\\
\min_{w_1,\dots,w_n\in\mathbb F} & \mathbb E
\left[
  \left\vert \sum_{k=1}^n w_k X_k - Y \right\vert^2
\right]
\end{align}
$$

For all $k=1,\dots,n$, the orthogonality principle $\langle \hat{Y} - Y, X_k \rangle = 0$ becomes

$$
\begin{align}
\langle X_k, \hat{Y} - Y \rangle &= 0
\\
\mathbb E\left[\overline{X_k} \cdot ({\hat{Y} - Y})\right]  &= 0
\\
\mathbb E\left[\overline{X_k} \cdot \left(\sum_{\ell=1}^n w_\ell X_\ell - Y \right)\right]  &= 0
\\
\sum_{\ell=1}^n \mathbb E\left[\overline{X_k}X_\ell \right] w_\ell &=  \mathbb E[\overline{X_k}Y]
\end{align}
$$

Matrix form:

> $$
> \begin{align}
> \begin{bmatrix}
>  \mathbb E[\overline{X_1}X_1] & \mathbb E[\overline{X_1}X_2] & \cdots & \mathbb E[\overline{X_1}X_n]
> \\
>  \mathbb E[\overline{X_2}X_1] & \mathbb E[\overline{X_2}X_2] & \cdots & \mathbb E[\overline{X_2}X_n]
> \\
> \vdots &\vdots &\ddots &\vdots
> \\
>  \mathbb E[\overline{X_n}X_1] & \mathbb E[\overline{X_n}X_2] & \cdots & \mathbb E[\overline{X_n}X_n]
> \end{bmatrix}
> \cdot
> \begin{bmatrix}
> w_1 \\ w_2 \\ \vdots \\ w_n
> \end{bmatrix}
> =%%%
> \begin{bmatrix}
> \mathbb E[\overline{X_1}Y] \\ \mathbb E[\overline{X_2}Y] \\ \vdots \\ \mathbb E[\overline{X_n}Y]
> \end{bmatrix}
> \tag{$\star$}
> \end{align}
> $$

Define the random vector $\mathbf X \triangleq [X_1,\dots,X_n]^\top\in\mathbb C^n$. Recall the auto-correlation matrix and cross-correlation matrix are defined as

$$
\begin{align*}
\mathbf R_{XX} = \mathbb E[\mathbf{XX}^\herm] &\iff (\mathbf R_{XX})_{ij} = \mathbb E[X_i\overline{X_j}]
\\
\mathbf r_{XY} = \mathbb E[\mathbf{X}\overline{Y}] &\iff (\mathbf r_{XY})_{i} = \mathbb E[X_i\overline{Y}]
\end{align*}
$$

Let $\mathbf w \triangleq [w_1,\dots,w_n]^\top$. Equation $(\star)$ becomes

$$
\begin{align}
\overline{\mathbf R_{XX}} \cdot \mathbf w = \overline{\mathbf r_{XY}}
\end{align}
$$

If $\mathbb F = \mathbb R$, then equation $(\star)$ reduces further to

$$
\begin{align}
\mathbf R_{XX} \cdot \mathbf w = \mathbf r_{XY}
\end{align}
$$

### OLS converges to LMMSE estimation

Let $\{(\mathbf x_i, y_i)\}_{i=1}^m \subset \mathbb R^n \times \mathbb R$ be iid from $p(\mathbf x, y)$. In OLS, we aim to fit the model $y = \mathbf w^\top \mathbf x$. Define the data matrix

$$
\begin{align*}
\mathbf X_\text{train} &= [\mathbf x_1, \dots, \mathbf x_m] \in\mathbb R^{n\times m} \\
\mathbf y_\text{train} &= [y_1, \dots, y_m]^\top \in\mathbb R^{m}
\end{align*}
$$

**Note**: There are two ways to define the data matrix. In OLS section, the data matrix was defined s.t. each row is a data point. Here, $\mathbf X_\text{train}$ is defined s.t. each column is a data point. The advantage of such definition will be clear later.

By OLS, the optimal weight vector satisfies

$$
\mathbf{X}_\text{train}\mathbf{X_\text{train}^\top}\mathbf{w} = \mathbf{X_\text{train}}\mathbf{y_\text{train}}
$$

Multiplying both sides with sample size $m$ yields

$$
\frac{1}{m}\mathbf{X}_\text{train}\mathbf{X_\text{train}^\top}\mathbf{w} = \frac{1}{m}\mathbf{X_\text{train}}\mathbf{y_\text{train}}
$$

Note that

$$
\begin{align*}
\frac{1}{m}\mathbf{X}_\text{train}\mathbf{X_\text{train}^\top}
&= \frac{1}{m}[\mathbf x_1,\dots, \mathbf x_m] \cdot
   \begin{bmatrix}
   \mathbf x_1^\top \\ \vdots \\ \mathbf x_m^\top
   \end{bmatrix}
= \frac{1}{m}\sum_{i=1}^m \mathbf x_i \mathbf x_i^\top
\\
\frac{1}{m}\mathbf{X_\text{train}}\mathbf{y_\text{train}}
&= \frac{1}{m}[\mathbf x_1,\dots, \mathbf x_m] \cdot
   \begin{bmatrix}
   y_1 \\ \vdots \\ y_m
   \end{bmatrix}
= \frac{1}{m}\sum_{i=1}^m \mathbf x_i y_i^\top
\end{align*}
$$

By the law of large number,

$$
\begin{align*}
\lim_{m\to\infty}\frac{1}{m}\sum_{i=1}^m \mathbf x_i \mathbf x_i^\top
&= \mathbb E[\mathbf x \mathbf x^\top] = \mathbf{R}_{XX}
\\
\lim_{m\to\infty}\frac{1}{m}\sum_{i=1}^m \mathbf x_i y_i^\top
&= \mathbb E[\mathbf x y] = \mathbf{r}_{XY}
\end{align*}
$$

Theorefore, as the sample size $m\to\infty$, OLS becomes LMMSE estimation.

$$
\begin{align*}
\underbrace{
  \frac{1}{m}\mathbf{X}_\text{train}\mathbf{X_\text{train}^\top}
}_{\to \mathbf{R}_{XX} \text{ as } m\to\infty}
\mathbf{w} &=
\underbrace{
  \frac{1}{m}\mathbf{X_\text{train}}\mathbf{y_\text{train}}
}_{\to \mathbf{r}_{XY} \text{ as } m\to\infty}
\\
\mathbf{R}_{XX}\mathbf{w} &= \mathbf{r}_{XY}
\end{align*}
$$

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

Let $\mathbf A\in\mathbb R^{m\times n}$, then

1. $\ker(\mathbf A^\top)$ and $\operatorname{ran}(\mathbf A)$ are orthogonal complements. (and thus their intersection is $\{\mathbf 0\}$)
2. $\ker(\mathbf A^\top \mathbf A) = \ker(\mathbf A)$.
3. $\mathbf A^\top \mathbf A$ is invertible $\iff \mathbf A$ has linearly independent columns.

*Proof 1*: We need to show that $\forall\mathbf x\in\ker(\mathbf A^\top), \forall \mathbf y\in\operatorname{ran}(\mathbf A), \langle \mathbf x, \mathbf y\rangle = 0$. By assumption,
$$
\begin{align*}
\mathbf x\in\ker(\mathbf A^\top) &\implies \mathbf{A^\top x} = 0
\\
\mathbf y\in\operatorname{ran}(\mathbf A) &\implies \exists \mathbf u\in\mathbb R^n, \text{ s.t. } \mathbf y = \mathbf{Au}
\end{align*}
$$
Then, we conclude
$$
\langle \mathbf x, \mathbf y\rangle
= \mathbf x^\top \mathbf y
= \mathbf x^\top \mathbf{Au}
= \mathbf u^\top \underbrace{\mathbf{A^\top x}}_{\mathbf 0}
= 0 \tag*{$\blacksquare$}
$$
*Proof 2*: We only need show that $\ker(\mathbf A^\top \mathbf A) \subseteq \ker(\mathbf A)$ since the inclusion in the opposite direction is trivial.

Consider $\mathbf x\in \ker(\mathbf A^\top \mathbf A)$. By assumption,
$$
\mathbf A^\top \mathbf{Ax} = 0 \implies \mathbf{Ax}\in\ker(\mathbf A^\top)
$$
On the other hand, $\mathbf{Ax}\in\operatorname{ran}(\mathbf A)$. Hence,
$$
\mathbf{Ax}\in\operatorname{ran}(\mathbf A) \cap \ker(\mathbf A^\top)
\implies \mathbf{Ax} = \mathbf 0 \implies \mathbf x\in \ker(\mathbf A)
\tag*{$\blacksquare$}
$$
*Proof 3*: This follows directly from $\ker(\mathbf A^\top \mathbf A) = \ker(\mathbf A)$ as
$$
\mathbf A^\top \mathbf A \text{ invertible }
\iff \ker(\mathbf A^\top \mathbf A) = \{\mathbf 0\}
\iff \ker(\mathbf A) = \{\mathbf 0\}
\iff \mathbf A \text{ has lin-indep. col.}
\tag*{$\blacksquare$}
$$
