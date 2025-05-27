---
title: "Singular Value Decomposition"
date: "2024"
author: "Ke Zhang"
---

# Singular Value Decomposition

Any matrix $A\in\mathbb R^{m\times n}$ can be decomposed into (without proof)
$$
\newcommand\norm[1] {\left\lVert#1\right\rVert}
\newcommand\iprod[2]{\left\langle#1, #2\right\rangle}
\DeclareMathOperator{\img}{im}
\DeclareMathOperator{\id}{id}
\DeclareMathOperator*{\argmax}{argmax}
\DeclareMathOperator*{\argmin}{argmin}

A=U\Sigma V^\top
$$
where

* $U\in\mathbb R^{m\times m}$ and $V\in\mathbb R^{n\times n}$ are orthogonal matrices. 
* $\Sigma $ has nonnegative diagonal elements: $\sigma_1,\cdots, \sigma_r$ with $r=\min(m,n)$ and zeros anywhere else.

We call

* The columns of $U$ are left singular vectors of $A$
* The columns of $V$ are right singular vectors of $A$
* The diagonal elements of $\Sigma$ are singular values of $A$ 

Some elementary properties

> 1. The columns of $U$ is eigenvectors of $AA^\top$
>    $$
>    AA^\top u_k = \sigma_k^2 u_k
>    $$
>
> 2. The columns of $V$ are eigenvectors of $A^\top A$ 
>    $$
>    A^\top A v_k = \sigma_k^2 v_k
>    $$
>
> 3. $A$ is sum of rank-1 matrices (note: $\rank A=r$)
>    $$
>    A = \sum_{k=1}^r \sigma_k u_k v_k^\top
>    $$
>
> 4. Singular values are closely related to Frobenius norm
>    $$
>    \norm{A}_F^2 = \norm{\Sigma}^2_F = \sum_{k=1}^r \sigma_k^2
>    $$
>    

*Proof 4* 
$$
\|A\|_F^2=\operatorname{Tr}\left(A^\top A\right)=\operatorname{Tr}\left(V \Sigma^\top U^\top U \Sigma V^\top\right)=\operatorname{Tr}\left(\Sigma^\top \Sigma\right)=\sum_{k=1}^r \sigma_k^2
$$


## Appendix

### Trace as Innerproduct

Consider the vector space $\mathbb R^{m\times n}$ over $\mathbb R$. An inner product can be defined via trace
$$
\iprod{A}{B} = \tr (A^\top B)
$$
The norm induced by trace inner product is exatly Frobenius norm
$$
\norm{A}_F^2 = \tr(A^\top A) = \sum_{i,j} a_{ij}^2
$$
