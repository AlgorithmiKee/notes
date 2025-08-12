---
title: "Portfolio Theory"
author: "Ke Zhang"
date: "2025"
fontsize: 12pt
---

[toc]

# Portfolio Theory

## Asset

### Single Asset

A single **asset** is modeled as a random variable $R$ representing its **rate of return** (typically expressed as a decimal or percentage). We define:

* $\mu = \mathbb{E}[R]$: **expected return** of the asset  
* $\sigma^2 = \mathbb{V}(R)$: **variance** of the asset's return  
* $\sigma = \sqrt{\mathbb{V}(R)}$: **volatility** (standard deviation) of the asset's return  

Given an asset, we can take one of three main positions:

* **Long** – Buy the asset in expectation that its price will rise in the future. This is the standard form of investment.  
* **Short** – Borrow the asset and sell it immediately, expecting its price to fall so it can be repurchased later at a lower price and returned to the lender.  
* **Leverage** – Use borrowed capital to buy more of the asset than you could otherwise afford, amplifying both potential gains and losses.  

### Multiple Assets

Multiple assets are modeled as a random vector
$$
\mathbf{R} = (R_1, \dots, R_n)^\top
$$
where each $R_i$ represents the return of asset $i$. Analogously, we define

* expected return vector:
  $$
  \boldsymbol{\mu} = (\mu_1, \dots, \mu_n)^\top
  $$

* covariance matrix:
  $$
  \boldsymbol{\Sigma} =
  \begin{bmatrix}
  \sigma_{11} & \dots & \sigma_{1n} \\
  \vdots & \ddots & \vdots \\
  \sigma_{n1} & \dots & \sigma_{nn} 
  \end{bmatrix},
  \quad
  \sigma_{ij} = \mathrm{Cov}(R_i, R_j)
  $$

For each asset $i$, one can take a long, short, or leveraged position.

## Portfolio

A **portfolio** is a collection of assets. Suppose there are $n$ assets; a portfolio assigns each asset a weight $w_i \in \mathbb R$ such that the weights sum to 1.

$$
\begin{align}
\sum_{i=1}^n w_i = 1
\end{align}
$$

Remarks:

* For now, we do not restict $w_i$ to be in $[0,1]$. This assumption leads to simpler mathematical results and has practical significance.
* $w_i < 0$ means we short-sell asset $i$. i.e. we borrow that asset and immediately sell it, obtaining extra cash.
* $w_i > 1$ means we hold asset $i$ with leverage. i.e. we use borrowed capital to buy more of that asset.

**Example 1** – No short-selling and no leverage:  
You have \$100. You invest \$60 in Apple and \$40 in Tesla:
$$
w_{\text{Apple}} = 0.6, \quad w_{\text{Tesla}} = 0.4
$$

**Example 2** – Short-selling and leverage:  
You have \$100. You are pessimistic about Tesla, and optimistic about Apple. You borrow \$30 worth of Tesla and short-sell it. Now you have \$130 cash, all invested in Apple:
$$
w_{\text{Apple}} = 1.3, \quad w_{\text{Tesla}} = -0.3
$$

### Formal Setup

Let

* $R_i$ denote the return of $i$-th asset.

* $\mathbf{R}$ denote the random vector consisting of $R_1,\dots,R_n$.

* $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$ denote the mean and covariance matrix of $\mathbf{R}$.

A portfolio is then represented by a vector $\mathbf{w}$ s.t.
$$
\begin{align}
\mathbf{1}^\top \mathbf{w} &= 1
\end{align}
$$

The return of the portfolio is defined as
$$
\begin{align}
R_p
&= \mathbf{w}^\top \mathbf{R}
\end{align}
$$
The expected return and the variance of the portfolio is then
$$
\begin{align}
\mu_p &\triangleq \mathbb E[R_p] = \mathbf{w}^\top \boldsymbol{\mu} \\
\sigma_p^2 &\triangleq \mathbb{V}[R_p] = \mathbf{w}^\top \boldsymbol{\Sigma} \ \mathbf{w}
\end{align}
$$
*Proof*: Here, we derive the expression of $\mathbb{V}[R_p] = \mathbf{w}^\top \boldsymbol{\Sigma}$.
$$
\begin{align*}
\mathbb{V}[R_p]
&= \mathbb E[R_p^2] - \mathbb E[R_p]^2 \\
&= \mathbb E[R_p R_p^\top] - \mathbb E[R_p] \mathbb E[R_p]^\top \\
&= \mathbb E[\mathbf{w}^\top \mathbf{R} \mathbf{R}^\top \mathbf{w}] - \mathbf{w}^\top \boldsymbol{\mu}  \boldsymbol{\mu}^\top \mathbf{w} \\
&= \mathbf{w}^\top \underbrace{
    \left(\mathbb E[\mathbf{R} \mathbf{R}^\top] - \boldsymbol{\mu}  \boldsymbol{\mu}^\top \right)
    }_{\boldsymbol{\Sigma}} \mathbf{w}
\tag*{$\blacksquare$}
\end{align*}
$$

Equivalent scalar notation:
$$
\begin{align}
R_p &= \sum_{i=1}^n w_i R_i \\
\mu_p &\triangleq \mathbb E[R_p] = \sum_{i=1}^n w_i \mu_i \\
\sigma_p^2 &\triangleq \mathbb{V}[R_p] = \sum_{i=1}^n \sum_{j=1}^n w_i w_j \sigma_{ij}
\end{align}
$$
**Key Observation**: Portfolio variance is **not** simply the weighted sum of individual variances:
$$
\sigma_p^2 = \sum_{k=1}^n w_k^2 \sigma_k^2 + \sum_{i=1}^n \sum_{j \ne i} w_i w_j \sigma_{ij}
$$

### Mean-Variance Analysis

Portfolio theory assumes that investors are rational in the following sense:

- **Profit-seeking**: Given two assets with the same risk, the investor prefers the one with higher expected return.
- **Risk-averse**: Given two assets with the same expected return, the investor prefers the one with lower variance.

To summarize, a rational investor would like a portfolio located in the top left corner of the mean-variance space.

TODO: plot: A is right above B. C is left of B.

Consider three portfolio A, B, C in the above figure. A rational investor would

* prefer A over B because they have the same variance but A has higher expected return
* prefer C over B because they the same expected return but C has less variance.

Fundamental questions in portfolio theory:

1. Which region in the mean-variance space is feasible?
2. Which part of feasible region represents suboptimal portfolio (and thus should be avoided)?
3. How to specify optimal portfolio?

## Two-Asset Diversification

TODO:

* simplify $\mu_p$ and $\sigma_2$ without applying Lagrangian
* feasible region in mean-variance space
* portfolio with minimum variance
* suboptimal region
* role of correlation

## Multi-Asset Diversification

TODO:

* optimal solution of Markowitz's problem.
* extrem cases and insights

### Mean-Variance Optimization Problem

The mean-variance optimization problem (by Markowitz) is given by

$$
\max_{\mathbf{w} \in \Delta^{n-1}}
\mathbf{w}^\top \boldsymbol{\mu} - \frac{\lambda}{2} \mathbf{w}^\top \boldsymbol{\Sigma} \ \mathbf{w}
$$

Remarks:

* The objective is a concave quadratic function. The feasible region is a convex set.
* $\lambda > 0$ is the ***risk-aversion parameter***. It controls the trade-off between return and risk:
  * A large $\lambda$ prioritizes minimizing the volatility (risk)
  * A small $\lambda$ prioritizes maximizing the expected return

Other common formulations of the portfolio optimization problem are:

1. Maximizing expected return subject to a variance constraint
   $$
   \begin{align}
   \max_{\mathbf{w} \in \Delta^{n-1}} & \quad \mathbf{w}^\top \boldsymbol{\mu} \\
   \text{s.t.} & \quad \mathbf{w}^\top \boldsymbol{\Sigma} \, \mathbf{w} \le \sigma_{\max}^2
   \end{align}
   $$

2. Minimizing variance subject to an expected return constraint
   $$
   \begin{align}
   \min_{\mathbf{w} \in \Delta^{n-1}} & \quad \mathbf{w}^\top \boldsymbol{\Sigma} \, \mathbf{w} \\
   \text{s.t.} & \quad \mathbf{w}^\top \boldsymbol{\mu} \ge \mu_{\min}
   \end{align}
   $$

The mean–variance problem can be seen as the **Lagrangian formulation** of either of these constrained problems:

- Starting from Problem 1, introduce a Lagrange multiplier $\frac{\lambda}{2}$ for the variance constraint; the constrained maximization becomes:
  $$
  \max_{\mathbf{w} \in \Delta^{n-1}}
  \mathbf{w}^\top \boldsymbol{\mu} - \frac{\lambda}{2} \mathbf{w}^\top \boldsymbol{\Sigma} \, \mathbf{w}
  $$
  where $\lambda$ reflects the trade-off between return and variance.

- Starting from Problem 2, introduce a Lagrange multiplier $\gamma$ for the expected return constraint; after rearranging, the same quadratic objective arises, with a different interpretation of the multiplier.

In both cases, the parameter $\lambda$ (or equivalently $\gamma$) governs **risk aversion**: higher values penalize variance more heavily, while lower values place greater emphasis on maximizing return.
