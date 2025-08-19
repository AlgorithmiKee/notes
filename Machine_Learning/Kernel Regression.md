---
title: "Kernel Regression"
date: "2024"
author: "Ke Zhang"
---

# Kernel Regression

[toc]

**Preliminary**: standard linear regression, standard ridge regression, kernel method.
$$
\DeclareMathOperator*{\argmin}{arg\,min}
$$

## Recap

* **Given**: training data set
  $$
    D = \{ (x_i, y_i) \}_{i=1}^n \subset \mathbb{R}^d \times \mathbb{R}
  $$

* **Design**: select a feature mapping:
  $$
    \phi: \mathbb{R}^d \to \mathbb{R}^p, x \mapsto \phi(x)
  $$

* **Goal**: Fit the extended linear model
  $$
    \hat y = w^\top \phi(x), \quad w\in\mathbb{R}^p
  $$

### Extended Linear Regression

Cost function:
$$
\begin{align}
C_{\mathrm{LS}}(w) &= \sum_{i=1}^n \left( w^\top \phi(x_i) - y_i \right)^2 \\
&= {\left\Vert \Phi w - y \right\Vert}_2^2
\end{align}
$$

where

$$
\begin{align}
\Phi &\triangleq
  \begin{bmatrix} \phi(x_1)^\top \\ \vdots \\ \phi(x_n)^\top \\ \end{bmatrix}
    =
  \begin{bmatrix}
    \phi_1(x_1) & \cdots & \phi_p(x_1) \\
    \vdots      &        & \vdots      \\
    \phi_1(x_n) & \cdots & \phi_p(x_n) \\
  \end{bmatrix}
  \in\mathbb{R}^{n \times p}
\\
w &= \begin{bmatrix} w_1 \\ \vdots \\ w_p \\ \end{bmatrix} \in\mathbb{R}^{p},
\quad\quad
y = \begin{bmatrix} y_1 \\ \vdots \\ y_n \\ \end{bmatrix} \in\mathbb{R}^{n}
\end{align}
$$

Analytical solution:

$$
\begin{align}
\hat{w}_{\mathrm{LS}} = \left( \Phi^\top \Phi \right)^{-1} \Phi^\top y
\end{align}
$$

Remark:

* The matrix $\Phi$ is called ***design matrix***.
* We call $\hat y = w^\top \phi(x)$ the extended linear model since it requires explicit calculation of $\phi(x)$. We reserve the term kernelized linear model for later discussion.

### Extended Ridge Regression

Cost function:
$$
\begin{align}
C_{\mathrm{Ridge}}(w) &= \sum_{i=1}^n \left( w^\top \phi(x_i) - y_i \right)^2 + \lambda{\Vert w \Vert}_2^2\\
&= {\left\Vert \Phi w - y \right\Vert}_2^2 + \lambda{\Vert w \Vert}_2^2
\end{align}
$$

Analytical solution:

$$
\begin{align}
\hat{w}_{\mathrm{Ridge}} = \left( \Phi^\top \Phi + \lambda I_p \right)^{-1} \Phi^\top y
\end{align}
$$

### Kernel Matrix

The kernel matrix is exactly $\Phi \Phi^\top$  since

$$
\begin{align*}
\Phi \Phi^\top
&=  \begin{bmatrix} \phi(x_1)^\top \\ \vdots \\ \phi(x_n)^\top \\ \end{bmatrix}
    \begin{bmatrix} \phi(x_1),        \cdots,   \phi(x_n)      \\ \end{bmatrix}
\\
&=  \begin{bmatrix}
      \phi(x_1)^\top \phi(x_1) & \cdots & \phi(x_1)^\top \phi(x_n) \\
      \vdots                   & \ddots & \vdots                   \\
      \phi(x_n)^\top \phi(x_1) & \cdots & \phi(x_n)^\top \phi(x_n) \\
    \end{bmatrix}
\\
&=  \underbrace{
    \begin{bmatrix}
      k(x_1, x_1) & \cdots & k(x_1, x_n) \\
      \vdots      & \ddots & \vdots      \\
      k(x_n, x_1) & \cdots & k(x_n, x_n) \\
    \end{bmatrix}
    }_{\text{kernel matrix } K}
\end{align*}
$$

## Representer Theorem

> For the squared loss, there exists a minimizer $\hat w$ in the span of $\phi(x_1), \dots, \phi(x_n)$
> $$
> \begin{align}
>   \hat w = \sum_{j=1}^n \alpha_j \phi(x_j) = \Phi^\top \alpha
>    \end{align}
> $$
>
>For L2-regularised squred loss, the minimizer(s) must lie in the span of $\phi(x_1), \dots, \phi(x_n)$.

Remark:

* The statement is stronger for minimizers of the L2-regularised squred loss.
* Instead of searching in $\mathbb R^p$, it suffices to search in the span of $\phi(x_1), \dots, \phi(x_n)$, which has the dimensionality at most $n \ll p$.
* In SVM, we derived similar results using KKT conditions.
* The theorem can be generalized to $\infty$-dimensional feature space. For that, replace $\mathbb R^p$ with the reproducing kernel Hilbert space. Details omitted here.

*Proof (squared loss part)*: Let $U \subseteq \mathbb R^p$ be the span of  $\phi(x_1), \dots, \phi(x_n)$. By orthogonal decomposition, any vector $w \in\mathbb R^p$ is the sum of a vector $\tilde w\in U$  and another vector in $v \in U^\perp$. Formally,
$$
\begin{align}
w
&= \tilde w + v \\
&= \sum_{j=1}^n \alpha_j \phi(x_j) + v
\end{align}
$$

The extended model becomes

$$
\begin{align}
\hat f(x)
&= \left\langle w,\, \phi(x) \right\rangle \nonumber \\
&= \left\langle \sum_{j=1}^n \alpha_j \phi(x_j) + v,\: \phi(x) \right\rangle \nonumber \\
&= \sum_{j=1}^n \alpha_j \big\langle \phi(x_j), \phi(x) \big\rangle +
\underbrace{\big\langle v, \phi(x)\big\rangle}_{=0 \text{ if } x\in D}
\end{align}
$$

Note: The second term becomes zero if we plug any training sample $x_i$ into $\hat f(\cdot)$.

The squared loss on the training data is independent of $v$ since
$$
\begin{align*}
\sum_{i=1}^n \left( y_i - \hat f(x_i) \right)^2
&= \underbrace{
      \sum_{i=1}^n \left(
        y_i - \sum_{j=1}^n \alpha_j \big\langle \phi(x_j), \phi(x_i) \big\rangle
      \right)^2
    }_{\text{independent of } v} \\
\end{align*}
$$

If $w$ minimizes the squared loss, then the corresponding $\tilde w$, i.e. the orthogonal projection of $w$ onto $U$, is also a minimizer.

*Proof (regularizer part)*: Let $w=\tilde w + v$ be decomposed as in previous part. Using the Pythagorean thoerem, we see
$$
\begin{align*}
\Vert w \Vert_2^2
= \Vert \tilde w \Vert_2^2 + \Vert v \Vert_2^2 \ge \Vert \tilde w \Vert_2^2
\end{align*}
$$

Hence, if $w$ minimizes the L2-regularised squared loss, then the corresponding $v$ must be zero, i.e. $w\in U$. $\quad\square$

## Kernelization

Using the dual representation theorem, we are ready to kernelize the model, cost function, etc.

> How to kernelize?
>
> 1. **Reparameterize** the model: Instead of parametrizing $f(\cdot)$ by $w$, we reparametrize $f(\cdot)$ by $\alpha$.
> 1. Identify inner products of the form $\langle \phi(u), \phi(v) \rangle$ everywhere.
> 1. **Kernelization**: Replace inner products with kernel evaluations of the form $k(u,v)$ everywhere.
>     * Why? Computing $k(u,v)$ is far cheaper than computing $\langle \phi(u), \phi(v) \rangle$.
>

### Kernelized Model

The extended model $f(x)=w^\top \phi(x)$ can be reparametrized (or kernalized) into

> $$
> \begin{align}
>   f(x) &= \sum_{j=1}^n \alpha_j  k(x, x_j)
> \end{align}
> $$

Remark:

* The reparameterized model does not require direct computation of inner products in high-dimensional features. $\to$ computational efficiency.
* The reparametrized model is just a linear combination of kernel evaluations $k(x, x_j)$, which measures the similarity between input $x$ and training sample $x_j$.

*Proof*: Plugging $w=\Phi^\top \alpha$ into the extended linear model, we get
$$
\begin{align*}
  f(x)
  &= w^\top \phi(x) \\
  &= \alpha^\top \Phi \phi(x) \\
  &= \begin{bmatrix} \alpha_1,  \cdots,  \alpha_n \end{bmatrix}
     \begin{bmatrix} \phi(x_1)^\top \\ \vdots \\ \phi(x_n)^\top \\ \end{bmatrix}
     \phi(x) \\
  &= \sum_{j=1}^n \alpha_j \,\underbrace{
       \phi(x_j)^\top \phi(x)
     }_{k(x, x_j)}
\end{align*}
$$

### Kernelized Linear Regression

The cost function $C_\mathrm{LS}(w)$ can be kernelized into

> $$
> \begin{align}
>   \tilde C_\mathrm{LS}(\alpha)
> &= \sum_{i=1}^n \left( y_i - \sum_{j=1}^n \alpha_j  k(x_i, x_j) \right)^2 \\
> &= {\left\Vert K\alpha - y \right\Vert}_2^2
> \end{align}
> $$

Remark:

* Here, the sum of squared loss can be written more compactly as the squared 2-norm of $K\alpha - y$. However, such reformulation is impossible for other loss functions, e.g. hinge loss or logistic loss.
* $\tilde C_\mathrm{LS}(\alpha)$ depends only on kernel matirx $K$, not directly on high-dimensional features.
* $\tilde C_\mathrm{LS}(\alpha)$ is strongly convex in $\alpha$ since $K$ is positive definite.

*Proof*: For each training sample $x_i$, the kernelized model would predict $\hat y_i = \hat f(x_i) = \sum_{j=1}^n \alpha_j  k(x_i, x_j)$. Plug in $\hat y_i$ into $C_\mathrm{LS} = \sum_{i=1}^n \left( y_i - \hat f(x_i) \right)^2$ and we conclude.

*Alt. proof*: Recall that $C_\mathrm{LS}(w) = {\left\Vert \Phi w - y \right\Vert}_2^2$. Plug in $w = \Phi^\top \alpha$. We get

$$
\begin{align*}
C_\mathrm{LS}(w)
&= \big\Vert \underbrace{\Phi \Phi^\top}_{K} \alpha - y \big\Vert_2^2
\end{align*}
$$

The kernelized optimization problem is then

> $$
> \begin{align}
>  \min_{\alpha \in\mathbb R^n} \: {\left\Vert K\alpha - y \right\Vert}_2^2
> \end{align}
> $$

The kernel matrix is often chosen so that the kernel matrix $K$ is invertible. Hence, the kernalized LS has the analytical solution

> $$
> \begin{align}
> \hat \alpha_\mathrm{KLS} = K^{-1} y
> \end{align}
> $$

### Kernelized Ridge Regression

The cost function $C_\mathrm{Ridge}(w)$ can be kernelized into

> $$
> \begin{align}
>   C_\mathrm{KRidge}(w)
> &= \sum_{i=1}^n \left( y_i - \sum_{j=1}^n \alpha_j  k(x_i, x_j) \right)^2 +  \lambda \alpha^\top \Phi \Phi^\top \alpha \\
> &= {\left\Vert K\alpha - y \right\Vert}_2^2 + \lambda \alpha^\top K \alpha
> \end{align}
> $$

The kernelized optimization problem is then

> $$
> \begin{align}
>  \min_{\alpha \in\mathbb R^n} \: {\left\Vert K\alpha - y \right\Vert}_2^2 + \lambda \alpha^\top K \alpha
> \end{align}
> $$

Letting the gradient be zero and using the symmetry of $K$, we get the solution to kernelized ridge regression

> $$
> \begin{align}
> \hat \alpha_\mathrm{KRidge} = \left( K + \lambda I_n \right)^{-1} y
> \end{align}
> $$

## Appendix

### Alt. Proof of the Representer Theorem for LS and Ridge

For extended linear regression, the optimal solution is  $\hat{w}_{\mathrm{LS}} = \left( \Phi^\top \Phi \right)^{-1} \Phi^\top y$.

Multiply the RHS by $I_p = \left( \Phi^\top \Phi \right) \left( \Phi^\top \Phi \right)^{-1}$. We get the dual representation

$$
\hat{w}_{\mathrm{LS}} =
\Phi^\top \underbrace{
  \Phi \left( \Phi^\top \Phi \right)^{-2} \Phi^\top y
}_{\text{some vector }\alpha \in \mathbb R^n}
$$

For extended ridge regression, the optimal solution is $\hat{w}_{\mathrm{Ridge}} = \left( \Phi^\top \Phi + \lambda I_p \right)^{-1} \Phi^\top y$.

Multiply both sides by $\left( \Phi^\top \Phi + \lambda I_p \right)$ and rearrange terms. We get the dual representation

$$
\begin{align*}
\left( \Phi^\top\Phi + \lambda I_p \right) \hat{w}_{\mathrm{Ridge}} &= \Phi^\top y \\
\Phi^\top\Phi \hat{w}_{\mathrm{Ridge}} + \lambda  \hat{w}_{\mathrm{Ridge}} &= \Phi^\top y \\
\hat{w}_{\mathrm{Ridge}} &= \Phi^\top \underbrace{
  \lambda^{-1}\left( y - \Phi^\top \hat{w}_{\mathrm{Ridge}} \right)
}_{\text{some vector }\alpha \in \mathbb R^n}
\end{align*}
$$

Those proofs are quite unintuitive and provide no much insights.
