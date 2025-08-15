---
title: "Portfolio Theory"
author: "Ke Zhang"
date: "2025"
fontsize: 12pt
---

# Portfolio Theory

[toc]

## Asset

### Single Asset

A single **asset** is modeled as a random variable $R$ representing its **rate of return** (typically expressed as a decimal or percentage). We define:

* $\mu = \mathbb{E}[R]$: **expected return** of the asset  
* $\sigma^2 = \mathbb{V}(R)$: **variance** of the asset'  
* $\sigma = \sqrt{\mathbb{V}(R)}$: **volatility** (standard deviation) of the asset  

Given an asset, an investor can take three main positions:

* **Long**: buy the asset expecting its price to rise. This is the standard form of investment.  
* **Short**: Borrow the asset and sell it immediately, expecting its price to fall so it can be repurchased later at a lower price and returned to the lender.  
* **Leverage**: Use borrowed capital to buy more of the asset than he could otherwise afford, amplifying both potential gains and losses.  

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

* Correlation coefficient:
  $$
  \rho_{ij} = \frac{\sigma_{ij}}{\sigma_i \sigma_j}
  $$

Remarks:

* For each asset $i$, one can take a long, short, or leveraged position.
* The correlation coefficient describes how correlated two assets are. By Cauchy-Schwarz inequality, $-1 \le \rho_{ij} \le 1$.
  * $\rho_{ij} > 0$: two assets are positively correlated. e.g. S&P 500 and Nasdaq indices.
  * $\rho_{ij} = 0$: two assets are uncorrelated. e.g. Fine art and stocks.
  * $\rho_{ij} < 0$: two assets are negatively correlated. e.g. Stocks and Bonds.
  * $\rho_{ij} = \pm 1$: two assets are perfectly pos/neg correlated. Rare in real life.

## Portfolio

A **portfolio** is a collection of assets. Suppose there are $n$ assets; a portfolio assigns each asset a weight $w_i \in \mathbb R$ such that the weights sum to 1.

$$
\begin{align}
\sum_{i=1}^n w_i = 1
\end{align}
$$

Remarks:

* For now, we do not restrict $w_i$ to be in $[0,1]$. This assumption leads to simpler mathematical results and has practical significance.
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
*Proof*: Here, we derive the expression of $\mathbb{V}[R_p] = \mathbf{w}^\top \boldsymbol{\Sigma}\ \mathbf{w}$.
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

* **Profit-seeking**: Given two assets with the same risk, the investor prefers the one with higher expected return.
* **Risk-averse**: Given two assets with the same expected return, the investor prefers the one with lower volatility.

<img src="./figs/portfolio_comparison.pdf" alt="Portfolio Comparison" style="zoom:100%;" />

Consider three portfolio A, B, C in the above figure. A rational investor would

* prefer A over B because they have the same volatility but A has higher expected return
* prefer C over B because they the same expected return but C has less volatility.

To summarize, a rational investor would like a portfolio located in the top left corner of the mean-volatility space.

Remarks:

* For visualization, we often use mean-volatility space since they have the same unit (in %).
* For mathematical derivation, we apply mean-variance analysis because variance is algebraically cleaner to work with.

Fundamental questions in portfolio theory:

1. Which region in the mean-volatility space is feasible?
2. Which part of feasible region represents suboptimal portfolio (and thus should be avoided)?
3. How to specify optimal portfolio?

## Two-Asset Diversification

We now consider portfolio consisting of two assets. For simplicity, we let
$$
w_1 = w, \quad w_2 = 1-w
$$

By [previous discussion](#formal-setup), we know
$$
\begin{align}
\mu_p &= \mu_1 w + \mu_2 (1-w) \\
\sigma_p^2 &= \sigma_1^2 w^2 + \sigma_2^2(1-w)^2 + 2\sigma_{12} w(1-w)
\end{align}
$$

Hence, the feasible set is a curve parameterized by $w$.

$$
\begin{align}
\begin{bmatrix}
\sigma_p \\
\mu_p
\end{bmatrix}
=
\begin{bmatrix}
\sqrt{\sigma_1^2 w^2 + \sigma_2^2(1-w)^2 + 2\sigma_{12} w(1-w)} \\
\mu_1 w + \mu_2 (1-w)
\end{bmatrix}
\end{align}
$$

Remarks:

* As we change $w$, we move along the curve in the $(\sigma,\mu)$ space. Each $w$ corresponds to a point representing a portfolio. In particular, the feasible set always contains $(\sigma_1, \mu_1)$ and $(\sigma_2, \mu_2)$, meaning that we go all-in asset 1 $(w=1)$ or asset 2 $(w=0)$ respectively.
* The shape of the curve depends on $\mu_1$, $\mu_2$, $\sigma_1$, $\sigma_2$ and $\sigma_{12}$. In nondegenerate case, the feasible set is a hyperbola.

### Shape of Feasible Set

Now, we will determine the shape of the feasible set in $(\sigma, \mu)$ space. 

#### Assets with equal expected return

If $\mu_1 = \mu_2 = \mu$, the feasible set simplifies to a horizontal line

$$
\begin{align}
\begin{bmatrix}
\sigma_p \\
\mu_p
\end{bmatrix}
=
\begin{bmatrix}
\sqrt{\sigma_1^2 w^2 + \sigma_2^2(1-w)^2 + 2\sigma_{12} w(1-w)} \\
\mu
\end{bmatrix}
\end{align}
$$

One interesting property is that we can make $\sigma_p$ lower than $\min\{\sigma_1, \sigma_2\}$ under certain conditions. e.g. Consider two uncorrelated assets with the same expected return $\mu$ and the same volatility $\sigma$. A portfolio with $w-1 = w_2 = \frac{1}{2}$ gives

$$
\sigma_p^2 = \frac{1}{2} \sigma^2 < \sigma^2
$$

Namely, even though two assets have the same expected return and volatility, the portfolio volatility can still be reduced by diversification. The reduction is even greater if these two assets are negatively correlated.

#### Assets with different expected return

Now, assume $\mu_1 \ne \mu_2$. We can eliminate parameter $w$ by
$$
\begin{align*}
\mu_p = \mu_1 w + \mu_2 (1-w)
\implies
w = \frac{\mu_p - \mu_2}{\mu_1 - \mu_2}, \:
1-w = \frac{\mu_1 - \mu_p}{\mu_1 - \mu_2}
\end{align*}
$$
Plugging into the expression of $\sigma_p^2$, we obtain the equation of feasible set:
$$
\begin{align*}
-A\sigma_p^2 + B\mu_p^2 - 2C \mu_p + D = 0
\end{align*}
$$

where

$$
\begin{align*}
A &= (\mu_1 - \mu_2)^2 \\
B &= \sigma_1^2 + \sigma_2^2 - 2\sigma_{12} \\
C &= \sigma_1^2 \mu_2 + \sigma_2^2 \mu_1 - \sigma_{12}(\mu_1 + \mu_2) \\
D &= \sigma_1^2 \mu_2^2 + \sigma_2^2 \mu_1^2 - 2\sigma_{12}\mu_1 \mu_2
\end{align*}
$$

First notice that

$$
B = \sigma_1^2 + \sigma_2^2 - 2\sigma_{12} \ge 0
$$

with the equality holds iff $\sigma_1 = \sigma_2$ and $\rho_{12} = 1$. i.e. Two assets have the same variance and are perfectly linearly correlated. ($\to$ See [Appendix](#non-negativity-of-variance) for proof)

If $B=0$, it can be verified that $C=0$ and $D=\sigma^2A$ where $\sigma_1 = \sigma_2 = \sigma$. The equation becomes

$$
\begin{align*}
-A\sigma_p^2 + \sigma^2A &= 0 \\
\sigma_p^2 &= \sigma^2 \\
\sigma_p &= \sigma
\end{align*}
$$

The resulting feasible set is a vertical line in $(\sigma, \mu)$ space.

If $B>0$, completing the square gives
$$
-A\sigma_p^2 + B\left( \mu_p - \frac{C}{B} \right)^2 + D - \frac{C^2}{B} = 0
$$

It can be verified that $D - \frac{C^2}{B} \ge 0$ with the equality iff $\rho_{12} = \pm 1$, i.e. Two assets are perfectly correlated (with different variances). Hence,

1. If $D - \frac{C^2}{B} = 0$ or equivalently $\rho_{12} = \pm 1$, the feasible set is intersecting lines.
1. If $D - \frac{C^2}{B} > 0$ or equivalently $\rho_{12} \ne \pm 1$, the feasible set is hyperbola.

Summary: The shape of feasible region (assuming $\mu_1 \ne \mu_2$ and $w\in\mathbb R$)

| Shape | $\rho_{12} = 1$ | $\rho_{12} = -1$ | $\rho_{12} \ne \pm 1$ |
| ----- | --------------- | ---------------- | --------------------- |
| $\sigma_1 = \sigma_2$ | vertical line | V-shaped lines | hyperbola |
| $\sigma_1 \ne\sigma_2$ | V-shaped lines| V-shaped lines | hyperbola |

**Case 1**: $\rho_{12} = 1$ and $\sigma_1 = \sigma_2 = \sigma$. Vertical line:

$$
\sigma_p = \sigma
$$

**Case 2**: $\rho_{12} = 1$ and $\sigma_1 \ne\sigma_2$. V-shaped line:

$$
\sigma_p = \left\vert\frac{\sigma_1 - \sigma_2}{\mu_1 - \mu_2}\right\vert \left\vert \mu_p - \frac{\sigma_1\mu_2 - \sigma_2\mu_1}{\sigma_1 - \sigma_2}\right\vert
$$

If we restrict $w\in[0,1]$, then the feasible set becomes a **single** line segment between $(\sigma_1, \mu_1)$ and $(\sigma_2, \mu_2)$.

*Proof*: Consider the turning point

$$
\begin{align*}
\tilde\mu
&\triangleq \frac{\sigma_1\mu_2 - \sigma_2\mu_1}{\sigma_1 - \sigma_2} \\
&= \frac{\sigma_1}{\sigma_1 - \sigma_2} \mu_2 + \frac{-\sigma_2}{\sigma_1 - \sigma_2} \mu_1 \\
&= \lambda \mu_2 + (1-\lambda) \mu_1
\end{align*}
$$

where $\lambda = \frac{\sigma_1}{\sigma_1 - \sigma_2}$. It is easy to verify that $\lambda<0$ or $\lambda>1$. Hence,

$$
\begin{align*}
\tilde\mu < \min\{\mu_1, \mu_2\}
\quad\lor\quad
\tilde\mu > \max\{\mu_1, \mu_2\}
\tag*{$\blacksquare$}
\end{align*}
$$

**Case 3**: $ \rho_{12} = -1$: V-shaped line:

$$
\sigma_p = \left\vert\frac{\sigma_1 + \sigma_2}{\mu_1 - \mu_2}\right\vert \left\vert \mu_p - \frac{\sigma_1\mu_2 + \sigma_2\mu_1}{\sigma_1 + \sigma_2}\right\vert
$$

If we restrict $w\in[0,1]$, then the feasible set becomes a **V-shaped** line segments between $(\sigma_1, \mu_1)$ and $(\sigma_2, \mu_2)$.

*Proof*: Consider the turning point

$$
\begin{align*}
\tilde\mu
&\triangleq \frac{\sigma_1\mu_2 + \sigma_2\mu_1}{\sigma_1 + \sigma_2} \\
&= \frac{\sigma_1}{\sigma_1 + \sigma_2} \mu_2 + \frac{\sigma_2}{\sigma_1 + \sigma_2} \mu_1 \\
&= \lambda \mu_2 + (1-\lambda) \mu_1
\end{align*}
$$

where $\lambda = \frac{\sigma_1}{\sigma_1 + \sigma_2}$. It is easy to verify that $0 < \lambda < 1$. Hence,

$$
\begin{align*}
\min\{\mu_1, \mu_2\} < \tilde\mu < \max\{\mu_1, \mu_2\}
\tag*{$\blacksquare$}
\end{align*}
$$

**Case 4**: $\rho_{12} \ne \pm 1$. Hyperbola:

$$
A\sigma_p^2 - B\left( \mu_p - \frac{C}{B} \right)^2 = D - \frac{C^2}{B}
$$

TODO:

* can we simplify this expression to get more insights?

### Minimum-Risk Portfolio

In this section, we assume

$$
\mu_1 < \mu_2, \: \sigma_1 < \sigma_2
$$

This typically refects the reality that assets with higher expected return are generally more risky.

Which portfolio has the minimum volatility? To answer this question, we let the gradient of the variance equal to zero.
$$
\frac{d\sigma_p^2}{dw} = 2\sigma_1^2 w - 2\sigma_2^2(1-w) + 2\sigma_{12} (1-2w) = 0
$$
Solving this equation, we get
$$
w^* = \frac{\sigma_2^2 - \sigma_{12}}{\sigma_1^2 + \sigma_2^2 - 2\sigma_{12}}
$$
The minimum variance is then
$$
\sigma_p^2(w^*) = \frac{\sigma_1^2 \sigma_2^2 - \sigma_{12}^2}{\sigma_1^2 + \sigma_2^2 - 2\sigma_{12}}
$$
The resulting expected return is
$$
\mu_p(w^*) = \frac{\sigma_2^2 \mu_1 + \sigma_1^2 \mu_2 - \sigma_{12}(\mu_1 + \mu_2)}{\sigma_1^2 + \sigma_2^2 - 2\sigma_{12}}
$$

Consider the case where $\rho_{12} = 0$. Suppose we start by allocating the entire portfolio to asset 1 (the less risky asset). If we shift a small fraction from asset 1 to asset 2 (the riskier asset), we would both increase $\mu_p$ and reduce $\sigma_p$. This is counterintuitive at first glance:

* adding some of the riskier asset actually **reduces** portfolio volatility
* going all-in on the less risky asset is worse in both expected return and volatility.

The intuition is that, with zero correlation, the fluctuations of asset 2 tend to occur independently of asset 1. When combined, the uncorrelated return streams partially offset each other’s variation, lowering the portfolio’s overall standard deviation.

We can continue improving the portfolio by adding more fraction of asset 2 until we reached the minimum risk point $(\sigma_p(w^*), \mu_p(w^*))$. All points on the hyperbola below this point are strictly dominated and thus suboptimal.

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

## Limits of Diversification

## Appendix

### Non-negativity of Variance

Let $\sigma_X^2$ , $\sigma_Y^2$ be the variance of random variables $X$, $Y$ repectively. Let $\sigma_{XY}$ be the covariance between $X$ and $Y$. Then, it holds that

$$
\begin{align}
\sigma_X^2 + \sigma_Y^2 - 2\sigma_{XY}
&\ge 0
\end{align}
$$

where the equality holds iff $Y = X + b$ for some constant $b$.

*Proof*: By AM–GM inequality and Cauchy-Schwarz inequality, we have
$$
\begin{align*}
\sigma_X^2 + \sigma_Y^2 - 2\sigma_{XY}
&\ge 2 \sigma_X \sigma_Y - 2\sigma_{XY}
&& \text{eq.}\iff \sigma_X = \sigma_Y
\\
&\ge 0
&& \text{eq.}\iff \rho_{XY} = 1
\end{align*}
$$

The 1st condition $\rho_{XY} = 1$ is equivalent to that $Y$ and $X$ are perfectly linearly dependent:
$$
Y = aX + b, \quad a > 0, b \in\mathbb R
$$

The 2nd condition says that $a$ can only be 1. $\:\blacksquare$

*Proof (alt.)*: Using the fact that

$$
\begin{align}
\mathrm{Var}(X-Y)
&= \mathrm{Var}(X) + \mathrm{Var}(Y) - 2\mathrm{Cov}(X,Y) \\
&= \sigma_X^2 + \sigma_Y^2 - 2\sigma_{XY}
\\
&\ge 0
\end{align}
$$

The equality holds iff $X-Y = \mathbb E[X-Y]$ almost surely. Namely, the constant $b$ in the previous proof can't simply be an arbitray number. $\:\blacksquare$
