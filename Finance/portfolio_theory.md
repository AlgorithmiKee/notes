---
title: "Portfolio Theory"
author: "Ke Zhang"
date: "2025"
fontsize: 12pt
---

# Portfolio Theory

## Asset

### Single Asset

TODO:

- Define the random variable $R$ representing the return of a single asset.
- Define the expected return: $\mathbb{E}[R]$ (i.e., the mean of $R$).
- Define risk: $\mathrm{Var}(R)$ (i.e., the variance of $R$).

### Multiple Assets

TODO:

- Define the return vector $\mathbf{R} = (R_1, \dots, R_n)$, where each $R_i$ represents the return of asset $i$.
- Expected return: the mean vector $\boldsymbol{\mu} = (\mu_1, \dots, \mu_n)$ where $\mu_i = \mathbb{E}[R_i]$.
- Risk: the covariance matrix $\boldsymbol{\Sigma}$, where $\sigma_{ij} = \mathrm{Cov}(R_i, R_j)$.

## Portfolio

A **portfolio** is a collection of assets. Suppose there are $n$ assets; a portfolio assigns each asset a weight $w_i \in [0,1]$ such that the weights sum to 1. That is, $\sum_{i=1}^n w_i = 1$. Portfolio theory is concerned with how to choose these weights to optimize return and manage risk.

Portfolio theory assumes that investors are rational in the following sense:

- **Profit-seeking**: Given two assets with the same risk, the investor prefers the one with higher expected return.
- **Risk-averse**: Given two assets with the same expected return, the investor prefers the one with lower risk (i.e., lower variance).

### Formal Setup

Let

* $R_i$ denote the return of $i$-th asset.
* $\mu_i$ denote the expected return of $i$-th asset.
* $\sigma_{ij}$ denote the covariance between asset $i$ and asset $j$.

The return of a portfolio is

$$
\begin{align}
R_p = \sum_{i=1}^n w_i R_i
\end{align}
$$

The expected return of the portfolio is then

$$
\begin{align}
\mathbb E[R_p]
&= \sum_{i=1}^n w_i \mathbb E[R_i] \\
&= \sum_{i=1}^n w_i \mu_i
\end{align}
$$

The risk of the portfolio is quantified by

$$
\begin{align}
\mathrm{Var}(R_p)
&= \mathrm{Var} \left[ \sum_{i=1}^n w_i R_i \right] \\
&= \sum_{i=1}^n \sum_{j=1}^n w_i w_j \sigma_{ij}
\end{align}
$$

Let $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$ denote the mean and covariance matrix of $\mathbf{R}$. Moreover, let $\Delta^{n-1}$ denote the simplex embedded in $\mathbb R^n$:

$$
\Delta^{n-1} = \left\{ \mathbf{w} \in \mathbb R^n \:\middle|\:
\sum_{i=1}^n w_i = 1, w_i \ge 0, \forall i = 1,\dots,n \right\}
$$

A portfolio is then represented by a vector $\mathbf{w} \in \Delta^{n-1}$ s.t.

$$
\begin{align}
R_p &= \mathbf{w}^\top \mathbf{R}  \\
\mathbb E[R_p] &= \mathbf{w}^\top \boldsymbol{\mu} \\
\mathrm{Var}[R_p] &= \mathbf{w}^\top \boldsymbol{\Sigma} \ \mathbf{w}
\end{align}
$$

Remarks:

* We write $\Delta^{n-1}$ (instead of $\Delta^{n}$) to highlight the dimension of the simplex rather than the ambient space it is embedded in. This is the standard notation in geometry.

*Proof*: Now we derive the expression of $\mathrm{Var}[R_p] = \mathbf{w}^\top \boldsymbol{\Sigma}$.

$$
\begin{align*}
\mathrm{Var}[R_p]
&= \mathbb E[R_p^2] - \mathbb E[R_p]^2 \\
&= \mathbb E[R_p R_p^\top] - \mathbb E[R_p] \mathbb E[R_p]^\top \\
&= \mathbb E[\mathbf{w}^\top \mathbf{R} \mathbf{R}^\top \mathbf{w}] - \mathbf{w}^\top \boldsymbol{\mu}  \boldsymbol{\mu}^\top \mathbf{w} \\
&= \mathbf{w}^\top \underbrace{
    \left(\mathbb E[\mathbf{R} \mathbf{R}^\top] - \boldsymbol{\mu}  \boldsymbol{\mu}^\top \right)
    }_{\boldsymbol{\Sigma}} \mathbf{w}
\tag*{$\blacksquare$}
\end{align*}
$$

### Mean-Variance Optimization

TODO:

* define the objective mean-variance optimization (Markowitz)
* Remark: convexity

### Other Objectives

TODO:

* max expected return while fixing the variance
* min variance while fixing the expected mean
* max sharpe ratio (non convex)
* connection between each of problems above and Markowitz's problem (if there is any)

## Two-Asset Diversification

TODO:

* optimal solution of Markowitz's problem.
* extrem cases and insights

### Efficient Frontier

## Multi-Asset Diversification

TODO:

* optimal solution of Markowitz's problem.
* extrem cases and insights
