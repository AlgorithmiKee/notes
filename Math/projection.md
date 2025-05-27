---
title: "Projections"
author: "Ke Zhang"
date: "2023"
fontsize: 12pt
---

# Projections

## Projection and Orthogonal Projection

>**DEF** A linear map $P:V\to V$ is called a **projection** if
>$$
>\newcommand\norm[1] {\left\lVert#1\right\rVert}
>\newcommand\iprod[2]{\left\langle#1, #2\right\rangle}
>\DeclareMathOperator{\img}{im}
>\DeclareMathOperator{\id}{id}
>\DeclareMathOperator*{\argmax}{argmax}
>\DeclareMathOperator*{\argmin}{argmin}
>
>P\circ P=P \label{ff}
>$$

This is a general notion of projections without assuming Hilbert space, which means we are not constrained by the orthogonal interpretation. 

### Properties of Projection

A projection $P$ satisfies...

 >1. $P$ is projection $\implies id-P$ is also a projection.
 >
 >2. $P$ is the identity map on $\img(P)$
 >$$
 >\forall x\in \img(P): Px=x
 >$$
 >
 >3. $\img\left( \id -P\right) = \ker(P)$
 >
 >4. **Unique Decomposition Theorem** 
 >
 >$$
 >\forall x\in V, \exists \text{ unique }  u\in\img(P), v\in\ker(P): x=u+v
 >$$
 >

*Proof  2:* Let $Pu=x$. $\implies Px=P^2u=Pu=x$

*Proof  3  $\subseteq$:* Let $y\in\img(\id-P)$, i.e. $\implies (\id-P)y=y$

​					$\supseteq$: Let $u\in\ker(P)$. $\implies (\id-P)u = u$

### Examples

Consider the vector space $\mathbb R^2$. A  valid projection could be defined by $P(x,y) = (x-y,0)$. Note: this projection has no orthogonal interpretation.

## Orthogonal Projection

The orthogoanl projection requires more preservation properties since Hilber space has extra structures than plain vector space.


> **DEF** If $V$ is a Hilbert space, a projection $P$ becomes **orthogonal projection** if $P$ is symmetric
> $$
> \iprod{Px}{y} = \iprod{x}{Py}
> $$

### Properties of Orthogonal Projection

An orthogoanl projection inherits all properties of projections, plus...

>1. $\id = P_U + P_{U^\perp}$
>2. **Orthogonality Principle:** $\forall x\in V, \forall y\in\img(P): \iprod{y}{x-Px} = 0$
>3. **Minimial Distance:** $Px$ is the best approximation of $x$ in $\img (P)$, i.e.
>
>$$
>\forall x \in V, \: Px=\argmin_{y\in\img{P}} \Vert y-x\Vert
>$$
>

*Proof 1*
$$
\begin{align*}
\forall y\in\img(P), \:
\iprod{Px}{y} = \iprod{x}{Py}  \implies
\iprod{Px}{y}&=\iprod{x}{y}  \\
0 &=\iprod{x-Px}{y}  \\
\end{align*}
$$
*Proof 2* From Pythagoras Theorem 
$$
\begin{align*}
\forall y\in\img(P), 

\Vert x-y \Vert^2 
&= \Vert Px + (x-Px) -y\Vert^2 \\ 
&= \Vert (x-Px) +(Px-y)\Vert^2          
	 & \text{Note: }(Px-y)\in\img(P) \\
&= \Vert (x-Px) \Vert^2 + \Vert(Px-y)\Vert^2  
   & \text{Pythagoras Theorem} \\
\end{align*}
$$


Note: Those properties requires no introduction of basis. The proof is rather elegant.

### Projectino onto 1D subspace

$U$ be a m-dimensional subspace of $\mathbb R^n$ with orthonormal basis $\mathcal U=\{u_1, \cdots, u_m \}$. We consider the orthogonal projection $P_U$ such that $\img(P_U)=U$. 

### Projection onto Subspace

Let $U$ be an m-dimensional subspace of $\mathbb R^n$ with orthonormal basis $\mathcal U=\{u_1, \cdots, u_m \}$. We consider the orthogonal projection $P_U$ such that $\img(P_U)=U$. 

​		$P_U(x)\in U \implies P_U(x)=\alpha_1u_1+\cdots + \alpha_m u_m$ 

​		$\mathcal U$ is orthonormal basis $\implies$ $\alpha_k=\iprod{P_U(x)}{u_k}$

Orthogonality principle:
$$
\begin{align*}
\iprod{x-P_U(x)}{u_k} =0 
&\implies \iprod{x}{u_k}-\iprod{P_U(x)}{u_k} = 0 \\
&\implies \iprod{x}{u_k}-\alpha_k = 0 \\
&\implies \alpha_k = \iprod{x}{u_k} \\
\end{align*}
$$
Hence, $P_U(x)$ is uniquely defined and
$$
P_U(x) = \sum_{k=1}^n \iprod{x}{u_k} u_k
$$
We can also write $P_U(x)$ in matrix form. Let $B=(u_1,\cdots,u_m)\in\mathbb R^{n\times m}$ and $\alpha = (\alpha_1, \cdots, \alpha_m)^\top\in\mathbb R^m$. 

$ \alpha_k = \iprod{x}{u_k}=u_k^\top x \implies \alpha = B^\top x$
$$
\begin{align*}
P_U(x) 
&=\alpha_1u_1+\cdots + \alpha_m u_m \\
&= B\alpha \\
&=BB^\top x
\end{align*}
$$


### Projection onto Affine Space

Let

*  $a\in \mathbb R^n$
* $U$ be a subspace of $\mathbb R^n$ with orthonormal basis $\mathcal U=\{u_1, \cdots, u_m \}$. 
*  $a+U=\{ a+u: u\in U \}$ be the affine subspace. 

Consider the orthogonal projection $P_{a+U}$ such that $\img(P_{a+U})=a+U$. This problem is almost identical to the projection onto subspace up to a translation. Hence (see Appendix),
$$
P_{a+U}(x) = a + P_U(x-a)
$$


## Appendix

Define a translation in vector space $V$: $x'=T(x)=x+a$

Linear map before translation: 
$$
A: V\to V, v\mapsto w=A(v)
$$
Linear map after the translation:
$$
A'=T\circ A\circ T^{-1}: V\to V, w'=A(v'-a)+a
$$
Proof (y'-a)=A(x'-a)
