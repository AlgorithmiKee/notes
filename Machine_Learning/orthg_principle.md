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

**Subspace Approximation Problem**  
Let $\mathbf x_1, \dots, \mathbf x_n, \mathbf y\in H$. We woule like to find a vector $\hat{\mathbf y}\in\operatorname{span}(\mathbf x_1, \dots, \mathbf x_n)$ s.t. $\hat{\mathbf y}$ is as close to $\mathbf y$ as possible. Formally, we would like to solve

$$
\begin{align}
\min_{\hat{\mathbf y}\in\operatorname{span}(\mathbf x_1, \dots, \mathbf x_n)}
\Vert \hat{\mathbf y} - \mathbf y \Vert^2
\end{align}
$$

or equivalently

$$
\begin{align}
\min_{w_1, \dots, w_n\in\mathbb R}
\left\Vert \sum_{k=1}^n  w_k\mathbf x_k - \mathbf y \right\Vert^2
\end{align}
$$

Remark:

* The equivalent formulation is straight forward as $\hat{\mathbf y}=\sum_{k=1}^n  w_k\mathbf x_k$.
* The spanning vectors $\mathbf x_1, \dots, \mathbf x_n$ are not necessarily linearly independent.

**Orthogonality Principle**  
Let $U=\operatorname{span}(\mathbf x_1, \dots, \mathbf x_n)$. The optimal solution of the subspace approximation problem is the orthogonal projection of $\mathbf y$ to $U$. The approximation error $\hat{\mathbf y} - \mathbf y$ lies in the orthogonal complement of $U$. In particular,

$$
\begin{align}
\langle \hat{\mathbf y} - \mathbf y, \mathbf x_k \rangle = 0, \: \forall k=1,\dots,n
\end{align}
$$

Now, we calculate the optimal weights as follows. Reformulate the orthogonality principle into

$$
TODO
$$

Pluggin in $\hat{\mathbf y}=\sum_{k=1}^n  w_k\mathbf x_k$ yields
