---
title: "Eigen Theory"
date: "2026"
author: "Ke Zhang"
---

# Eigen Theory

## Eigenvalues and Eigenvectors

Let $A \in\mathbb R^{n \times n}$ be a square matrix. Suppose there exists a **non-zero** vector $v\in\mathbb C^n$ and a scalar $\lambda\in\mathbb C$ s.t.

$$
\begin{align}
Av = \lambda v
\end{align}
$$

Then, we call $\lambda$ an ***eigenvalue*** of $A$ and $v$ an ***eigenvector*** of $A$ corresponding to $\lambda$.

Remarks:

* Both eigenvalue and the eigenvector could be complex-valued even if the matrix $A$ contains only real numbers.
* By definition, the eigenvector is nonzero since $A0 = \lambda 0$ holds for any $\lambda\in\mathbb R$. The eigenvalue could be zero.
* The transformation of eigenvector $v$ under $A$ is a scalar multiple of $v$ itself. Geometrically, the image of an eigenvector is parallel to the eigenvector itself.

Let $\lambda$ be an eigenvalue of matrix $A$. The eigenspace of $\lambda$ is defined as the kernel of $A-\lambda I$:

$$
\begin{align}
E_{\lambda}
&= \ker(A-\lambda I) \\
&= \{x\in\mathbb C^n \mid (A-\lambda I)x = 0\}
\end{align}
$$

In other words, the eigenspace $E_{\lambda}$ is the set of all eigenvectors corresponding to $\lambda$ plus the zero vector.

The set of all eigenvalues of $A \in\mathbb R^{n \times n}$ is called its ***spectrum***, denoted by $\sigma(A)$.

$$
\begin{align}
\sigma(A) = \{ \lambda_1, \dots, \lambda_m \}
\end{align}
$$

> Elementrary properties of eigenvectors:
>
> 1. For a matrix $A$, eigenvectors corresponding to distinct eigenvalues are linearly indepedent.
> 1. Similar matrices share the same spectrum.
> 1. If $\lambda$ is an eigenvalue of $A$, then $\lambda^n$ is an eigenvalue of $A^n$.
> 1. If $\lambda$ is an eigenvalue of an invertible matrix $A$, then $\lambda^{-1}$ is an eigenvalue of $A^{-1}$.

*Proof 1*: Let $\lambda_1, \dots \lambda_m$ be distinct eigenvalues of $A$ with corresponding eigenvectors $v_1, \dots, v_m$. By definition,

$$
Av_k = \lambda_k v_k, \quad k = 1,\dots,m
$$

We need to show that

$$
c_1 v_1 + \dots + c_m v_m = 0 \implies c_1 = \dots = c_m  = 0
$$

Consider the matrix product:

$$
M_1 = \prod_{k=2}^m (A - \lambda_k I)
$$

Left-multiplying $M_1$ on both side of $c_1 v_1 + \dots + c_m v_m = 0$ yeilds

$$
\sum_{k=1}^n c_k M_1v_k = 0
$$

Note that the factors in $M_1$ are communitive. For any $k\ne 1$, we reorder the factors inside $M_1$ to put $(A - \lambda_k I)$ at the end. Hence, most terms in $\sum_{k=1}^n c_k M_1v_k$ vanishes as

$$
\begin{align*}
\forall k \in \{2,\dots,m\}, \quad
M_1 v_k
&= \prod_{j=2}^m (A - \lambda_j I) v_k \\
&= \left[\prod_{\substack{j=2 \\ j \neq k}}^m (A - \lambda_j I)\right] \underbrace{(A - \lambda_k I) v_k}_{0} \\
&= 0
\end{align*}
$$

The equality $\sum_{k=1}^n c_k M_1v_k = 0$ simplifies to:

$$
\begin{align*}
c_1 M_1 v_1 & = 0 \\
c_1 \prod_{k=2}^m (A - \lambda_k I)v_1 &= 0 \\
c_1 \prod_{k=2}^m (\lambda_1 - \lambda_k) v_1 &= 0
\end{align*}
$$

Since all eigenvalues are distinct, $(\lambda_1 - \lambda_k)$ are all non-zero for any $k$. Moreover, $v_1$ is an eigenvector and by definition non-zero. Therefore, $c_1$ must be zero.

Likewise, we can construct for each $i\in\{2,\dots,m\}$ the matrix product:

$$
M_i = \prod_{\substack{k=1, \\ k \ne i}}^m (A - \lambda_k I)
$$

By the same logic, we conclude:

$$
c_2 = \dots = c_m = 0
\tag*{$\blacksquare$}
$$

*Proof 2*: Let $A$ and $\Lambda$ be similar matrices. There exists an invertible matrix $B$ s.t.

$$
\Lambda = B^{-1}AB
$$

Let $\lambda$ be an eigenvalue of $A$ with corresponding eigenvector $v$. We will show that $\lambda$ is also an eigenvalue of $\Lambda$.

Starting from the similarity equation, we have

$$
\begin{align*}
\Lambda B^{-1} &= B^{-1} A B B^{-1} \\
\Lambda B^{-1} &= B^{-1} A \\
\Lambda B^{-1} v &= B^{-1} A v \\
\Lambda \underbrace{B^{-1} v}_{w} &= \lambda \underbrace{B^{-1} v}_{w} \\
\end{align*}
$$

Hence, $\lambda$ is an eigenvalue of $\Lambda$ with corresponding eigenvector $w = B^{-1} v$.

Likewise, we can also show that every eigenvalue of $\Lambda$ is also an eigenvalue of $A$. Therefore, we conclude

$$
\sigma(A) = \sigma(\Lambda)
\tag*{$\blacksquare$}
$$

*Proof 3* and *4*: trivial.

## Computing Eigenvalues and Eigenvectors

### Characteristic Polynomial

### Algebraic and Geometric Multiplicity

## Diagonalization
