---
title: "Lagrange Interpolation"
author: "Ke Zhang"
date: "2024"
fontsize: 12pt
---

# Lagrange Interpolation

## Motivation

Given $n$ distinct data points $(x_1, y_1), \dots , (x_n, y_n) \in\mathbb{R}^2$, can you find a polynomial of degree $n-1$ which goes through exactly those points?

* Well, you might assume the target polynomial has the form $p(x) = \displaystyle\sum_{k=0}^{n-1} a_k x^k$. Then, you plug in all data points into $p(x)$. You get a system of $n$ equations with unknowns $a_0,\dots,a_{n-1}$. However, as $n$​ grows, solving this system of equations can be 
  * computationally expensive.
  * error-prone due to numerical issues 

* Is there a faster way to compute $p(x)$?
  * Yes. Lagrage Interpolation does exactly that!


## The Claim

>  Given $n$ distinct data points $(x_1, y_1), \dots , (x_n, y_n) \in\mathbb{R}^2$, there is unique a polynomial $p(x)$ of degree $n-1$ such that
> $$
> \begin{align}
> p(x_k) = y_k, \quad \forall k = 1, \dots , n
> \end{align}
> $$
> and
> $$
> \begin{align}
> p(x) = \sum_{k=1}^n y_k L_k(x), \text{where } 
> L_k(x) \triangleq \prod_{\substack{i=1, \dots n\\ i \ne k}} \frac{x-x_i}{x_k - x_i}
> \end{align}
> $$
> 

Proof: Consider the polynomial $L_k$. We notice that
$$
\forall j=1,\dots, n, \quad 
L_k(x_j) =
\begin{cases}
1 & j=k \\
0 & j \ne k
\end{cases}
$$
In other words, $L_k(x_j) = \delta_{kj}$ where $\delta$ is the [Kronecker delta](https://en.wikipedia.org/wiki/Kronecker_delta).

Part 1: Show that  $L_1, \dots, L_n$ forms a basis of $\mathcal P_{n-1} \triangleq \{\text{Polynomial of deg at most } n-1 \}$.

* Show that $L_1, \dots, L_n$ is linearly independent.  
  Let $\lambda_1 L_1 + \cdots + \lambda_n L_n = \mathbf 0$ where $\mathbf 0$ represents the polynomial with all zero coefficients. In particular, the equality holds when we plug $x_k$ in both sides. Hence,

$$
\begin{align*}
(\lambda_1 L_1 + \cdots + \lambda_n L_n)(x_k) 
&= \lambda_1 L_1(x_k) + \cdots + \lambda_n L_n(x_k) \\
&= \lambda_k L_k(x_k)  && L_i(x_k) = \delta_{ik}\\
&= \lambda_k  && L_i(x_k) = \delta_{ik}\\
& \triangleq 0
\end{align*}
$$

* Since $\dim\mathcal P_{n-1} = n$ and $L_1, \dots, L_n$ are linearly independent, we immediately conclude that $L_1, \dots, L_n$ is a basis of $\mathcal {P}_{n-1}$.

Part 2: Determine the coordinate of the target polynomial $p(x)$ w.r.t. the basis $L_1, \dots, L_n$. From linear algebra, we know that $p(x)$ can be written as a unique linear combination of $L_1, \dots, L_n$. 
$$
p(x) = \lambda_1 L_1(x) + \cdots + \lambda_n L_n(x)
$$
Since $p(x_k) = y_k$ by assumption, we conclude $p(x_k) = \lambda_k = y_k$  $\quad\square$

## Example

Find the 2nd order polynomial passing through
$$
(x_1, y_1)=(1,1), \quad (x_2, y_2)=(2,1) \quad (x_3, y_3)=(3,2)
$$
By Lagrange interpolation, we get
$$
\begin{align*}
L_1(x) 
&= \frac{(x - x_2)(x - x_3)}{(x_1 - x_2)(x_1 - x_3)} 
= \frac{(x - 2)(x - 3)}{(1 - 2)(1 - 3)}
= \frac{x^2}{2}  - \frac{5x}{2} + 3 \\

L_2(x) 
&= \frac{(x - x_1)(x - x_3)}{(x_2 - x_1)(x_2 - x_3)} 
= \frac{(x - 1)(x - 3)}{(2 - 1)(2 - 3)}
= -x^2 + 4x - 3 \\

L_3(x) 
&= \frac{(x - x_1)(x - x_2)}{(x_3 - x_1)(x_3 - x_2)} 
= \frac{(x - 1)(x - 2)}{(3 - 1)(3 - 2)}
= \frac{x^2}{2}  - \frac{3x}{2} + 1 \\
\end{align*}
$$
Hence
$$
\begin{align*}
p(x) 
&= y_1 L_1(x) + y_2 L_2(x) + y_3 L_3(x) \\
&= L_1(x) + L_2(x) + 2 L_3(x) \\
&= \frac{x^2}{2}  - \frac{3x}{2} + 2 \\
\end{align*}
$$


## Relation to Machine Learning

Consider data points $(x_1, y_1), \dots , (x_n, y_n) \in\mathbb{R}^2$ as training data. We would like to discover the underlaying relation between $x$ and $y$ using regression.

Fitting data points to arbitarily complex models leads to overfitting. As Lagrange interpolation suggests, while the Lagrange polynomial achives exactly zero training loss, it ususally generalize poorly.
