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

The system $Av = \lambda v$ can be rewritten as $(A - \lambda I)v = 0$, which has a non-trivial solution if and only if $A - \lambda I$ is singular.

The ***characteristic polynomial*** of $A \in \mathbb{R}^{n \times n}$ is defined as

$$
\begin{align}
\chi_A(\lambda) = \det(\lambda I - A)
\end{align}
$$

> Properties of the characteristic polynomial:
>
> 1. $\lambda$ is an eigenvalue of $A$ if and only if $\chi_A(\lambda) = 0$.
> 1. $\chi_A(\lambda)$ is a monic polynomial of degree $n$.
> 1. $\operatorname{tr}(A) = \sum_{k=1}^n \lambda_k$ and $\det(A) = \prod_{k=1}^n \lambda_k$, where the eigenvalues $\lambda_1, \dots, \lambda_n$ are counted with multiplicity.

*Proof 1*: $\lambda$ is an eigenvalue of $A$ if and only if $(A - \lambda I)v = 0$ has a non-trivial solution, which holds if and only if $A - \lambda I$ is singular:

$$
\lambda \in \sigma(A) \iff \det(A - \lambda I) = 0 \iff \chi_A(\lambda) = 0
\tag*{$\blacksquare$}
$$

*Proof 2*: In the Leibniz expansion of $\det(\lambda I - A)$, the $(i, j)$ entry is $\lambda\delta_{ij} - a_{ij}$, which depends on $\lambda$ only when $i = j$. Every non-identity permutation $\sigma \in S_n$ has at least two non-fixed points, so its corresponding term has degree at most $n - 2$ in $\lambda$. Only the identity permutation contributes degree $n$:

$$
\prod_{i=1}^n (\lambda - a_{ii}) = \lambda^n - \operatorname{tr}(A)\lambda^{n-1} + \cdots
\tag*{$\blacksquare$}
$$

*Proof 3*: By the Fundamental Theorem of Algebra, $\chi_A$ factors over $\mathbb{C}$ as

$$
\begin{align}
\chi_A(\lambda) = \prod_{k=1}^n (\lambda - \lambda_k)
\end{align}
$$

Comparing the coefficient of $\lambda^{n-1}$ on both sides gives $\operatorname{tr}(A) = \sum_{k=1}^n \lambda_k$. Setting $\lambda = 0$ gives $\chi_A(0) = (-1)^n\det(A)$ and $\prod_{k=1}^n(0 - \lambda_k) = (-1)^n\prod_{k=1}^n\lambda_k$, hence

$$
\det(A) = \prod_{k=1}^n \lambda_k
\tag*{$\blacksquare$}
$$

### Algebraic and Geometric Multiplicity

Since $\chi_A(\lambda)$ has degree $n$, it factors completely over $\mathbb{C}$ by the Fundamental Theorem of Algebra. The ***algebraic multiplicity*** of an eigenvalue $\lambda_k$ is its multiplicity as a root of $\chi_A$, denoted $m_a(\lambda_k)$:

$$
\begin{align}
\chi_A(\lambda) = \prod_{k=1}^m (\lambda - \lambda_k)^{m_a(\lambda_k)}, \quad \lambda_1, \dots, \lambda_m \text{ distinct}
\end{align}
$$

The ***geometric multiplicity*** of an eigenvalue $\lambda$ is the dimension of its eigenspace:

$$
\begin{align}
m_g(\lambda) = \dim(E_\lambda) = \dim(\ker(A - \lambda I))
\end{align}
$$

A matrix $A$ is called ***defective*** if $m_g(\lambda) < m_a(\lambda)$ for some eigenvalue $\lambda$.

> Properties of algebraic and geometric multiplicity:
>
> 1. For every eigenvalue $\lambda$ of $A$: $1 \leq m_g(\lambda) \leq m_a(\lambda)$.
> 1. The algebraic multiplicities sum to $n$: $\displaystyle\sum_{k=1}^m m_a(\lambda_k) = n$.

*Proof 1* (lower bound): By definition of eigenvalue, there exists at least one non-zero $v \in E_\lambda$, hence

$$
m_g(\lambda) = \dim(E_\lambda) \geq 1
\tag*{$\blacksquare$}
$$

*Proof 1* (upper bound): Let $r = m_g(\lambda)$ and let $\{u_1, \dots, u_r\}$ be a basis of $E_\lambda$. Extend it to a basis of $\mathbb{C}^n$ and form the change-of-basis matrix $P = [u_1 \mid \cdots \mid u_r \mid w_1 \mid \cdots \mid w_{n-r}]$. Since $Au_k = \lambda u_k$ for all $k \leq r$, the first $r$ columns of $P^{-1}AP$ equal $\lambda e_1, \dots, \lambda e_r$, yielding the block form

$$
P^{-1}AP = \begin{pmatrix} \lambda I_r & B \\ 0 & C \end{pmatrix}
$$

Since similar matrices share the same characteristic polynomial (Property 2 of eigenvectors), by the block-triangular determinant formula:

$$
\chi_A(\mu) = \det\begin{pmatrix} (\mu - \lambda)I_r & -B \\ 0 & \mu I_{n-r} - C \end{pmatrix} = (\mu - \lambda)^r \cdot \chi_C(\mu)
$$

Hence $(\mu - \lambda)^r$ divides $\chi_A(\mu)$, so

$$
m_a(\lambda) \geq r = m_g(\lambda)
\tag*{$\blacksquare$}
$$

*Proof 2*: $\chi_A(\lambda)$ is a monic polynomial of degree $n$ (Property 2 of the characteristic polynomial) and factors as $\prod_{k=1}^m (\lambda - \lambda_k)^{m_a(\lambda_k)}$, so

$$
\sum_{k=1}^m m_a(\lambda_k) = n
\tag*{$\blacksquare$}
$$

## Diagonalization

A matrix $A \in \mathbb{R}^{n \times n}$ is ***diagonalizable*** if there exists an invertible matrix $P \in \mathbb{C}^{n \times n}$ and a diagonal matrix $D = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$ such that

$$
\begin{align}
A = PDP^{-1}
\end{align}
$$

The columns of $P$ are eigenvectors of $A$, and the diagonal entries of $D$ are the corresponding eigenvalues.

Remarks:

* $P$ is the change-of-basis matrix from the eigenbasis $\{p_1, \dots, p_n\}$ to the standard basis. The factorization $A = PDP^{-1}$ decomposes the action of $A$ into three steps: express the input in the eigenbasis via $P^{-1}$, scale each component by the corresponding eigenvalue via $D$, then convert back to the standard basis via $P$.
* The diagonalization is not unique. The matrix $D$ is unique only up to reordering of its diagonal entries, with the columns of $P$ reordered consistently. For a fixed $D$, each column $p_k$ may be replaced by any non-zero scalar multiple $\alpha p_k$, and if $m_g(\lambda_k) > 1$, any basis of $E_{\lambda_k}$ may be used.

> Properties of diagonalizable matrices:
>
> 1. $A$ is diagonalizable if and only if $A$ has $n$ linearly independent eigenvectors.
> 1. $A$ is diagonalizable if and only if $m_g(\lambda) = m_a(\lambda)$ for every eigenvalue $\lambda$ of $A$.
> 1. If $A$ has $n$ distinct eigenvalues, then $A$ is diagonalizable.

*Proof 1* ($\Rightarrow$): Suppose $A = PDP^{-1}$, i.e., $AP = PD$. Comparing the $k$-th column on both sides gives $Ap_k = \lambda_k p_k$, so each column $p_k$ is an eigenvector. Since $P$ is invertible, its columns are linearly independent.

*Proof 1* ($\Leftarrow$): Let $p_1, \dots, p_n$ be $n$ linearly independent eigenvectors with corresponding eigenvalues $\lambda_1, \dots, \lambda_n$. Setting $P = [p_1 \mid \cdots \mid p_n]$ and $D = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$, we have $AP = PD$. Since $P$ is invertible:

$$
A = PDP^{-1}
\tag*{$\blacksquare$}
$$

*Proof 2*: From each eigenspace $E_{\lambda_k}$, one may select at most $m_g(\lambda_k)$ linearly independent eigenvectors. By Property 1 of eigenvectors, eigenvectors from distinct eigenspaces are linearly independent, so the total number of linearly independent eigenvectors across all eigenspaces is $\sum_{k=1}^m m_g(\lambda_k)$. By Property 2 of multiplicities, $\sum_{k=1}^m m_a(\lambda_k) = n$. Therefore:

$$
A \text{ diagonalizable} \iff \sum_{k=1}^m m_g(\lambda_k) = n \iff m_g(\lambda_k) = m_a(\lambda_k) \text{ for all } k
\tag*{$\blacksquare$}
$$

*Proof 3*: $n$ distinct eigenvalues imply $n$ linearly independent eigenvectors by Property 1 of eigenvectors, so diagonalizability follows immediately from Property 1:

$$
|\sigma(A)| = n \implies A \text{ is diagonalizable}
\tag*{$\blacksquare$}
$$
