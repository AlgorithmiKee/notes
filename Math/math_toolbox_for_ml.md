---
title: "Math Toolbox for ML"
date: "2024"
author: "Ke Zhang"
---

# Math Toolbox for ML

[toc]

$$
\DeclareMathOperator*{\argmax}{argmax}
\DeclareMathOperator*{\argmin}{argmin}
$$

**Preliminary**: basics of probability, linear algebra, and multivariate calculus.

## Probabilities and Statistics

TODO:

* conditional expectation
* conditional variance
* total variance
* conditional independence

### Notation Caveats

The PDF of a random variable $X$ has multiple notations:

* $p_X(\cdot)$ or $p_X(x)$: the standard notation in probability theory and statistics.
* $p(x)$: short-hand notation when the context is clear

The parameterized PDF of random variable $X$ also has multiple notations:

* $p_X(x ; \theta)$: standard notation in frequentist statistics
* $p(x ; \theta)$: short-hand notation in frequentist statistics
* $p(x \mid \theta)$: standard notation in Bayesian statistics
* $p_\theta(x)$: modern notation, especially in deep learning

Remarks:

* In some literatures, PDFs are denoted by $f$ rather than $p$. e.g. $f_X(\cdot)$ rather than $p_X(\cdot)$.
* Not to be confused: $p_X(x)$ vs. $p_\theta(x)$. The subscript in the former notation indicates that the density is defined for random variable $X$. The subscript in the latter notations denotes the parameter.

### Important inequalities

#### Markov inequality

> Suppose a random variable $X \ge 0$. Then
>
> $$
> \begin{align}
> \forall \epsilon > 0, \quad P(X \ge \epsilon) \le \frac{\mathbb E[X]}{\epsilon}
> \end{align}
> $$

**Core Inituition**: Consider $\mathbb E[X]$ as the center of mass. The inequality tells: If $\mathbb E[X]$ is small, then $X$ is unlikely to be large. e.g. If $\mathbb E[X] = 0.1$, then the probability that $X\ge 10$ is at most $0.01$.

*Proof*: We assume $X$ is continuous and has a PDF. For discrete random variables, replace the integral with summation. By definition of $\mathbb E[X]$ and that $X \ge 0$, we have:

$$
\begin{align*}
\mathbb E[X]
&= \int_{0}^{\infty} xp(x) \,\mathrm{d}x \\
&= \int_{0}^{\epsilon} xp(x) \,\mathrm{d}x + \int_{\epsilon}^{\infty} xp(x) \,\mathrm{d}x \\
&\ge \int_{\epsilon}^{\infty} xp(x) \,\mathrm{d}x \\
&\ge \int_{\epsilon}^{\infty} \epsilon p(x) \,\mathrm{d}x \\
&= \epsilon P(X \ge \epsilon)
\end{align*}
$$

Rearranging the terms, we conclude

$$
\begin{align*}
P(X \ge \epsilon) \le \frac{\mathbb E[X]}{\epsilon}
\tag*{$\blacksquare$}
\end{align*}
$$

#### Chebyshev inequality

> Suppose a random variable $X$ has finite and non zero variance. Then
>
> $$
> \begin{align}
> \forall \epsilon > 0, \quad
> P(\vert X - \mathbb E[X] \vert \ge \epsilon) \le \frac{\mathbb V[X]}{\epsilon^2}
> \end{align}
> $$

**Core Inituition**: Consider $\vert X - \mathbb E[X] \vert$ as the deviation of $X$ from the mean. The inequality tells: If $\mathbb V[X]$ is small, then $X$ is unlikely to deviate far from its mean.

*Proof*: Let $Y = (X - \mathbb E[X])^2$. Then, applying Markov inequality on $Y$ yields Chebyshev inequality:

$$
\begin{align*}
P(Y \ge \epsilon^2) &\le \frac{\mathbb E[Y]}{\epsilon^2} \\
P((X - \mathbb E[X])^2 \ge \epsilon^2) &\le \frac{\mathbb E[(X - \mathbb E[X])^2]}{\epsilon^2} \\
P(\vert X - \mathbb E[X] \vert \ge \epsilon) &\le \frac{\mathbb V[X]}{\epsilon^2}
\tag*{$\blacksquare$}
\end{align*}
$$

#### Jensen Inequality

Let $X\in\mathbb R^d$ be a random vector and $g: \mathbb R^d \to \mathbb R$ be a convex function. Then,

> $$
> \begin{align}
> \mathbb E[g(X)] \ge g(\mathbb E[X])
> \end{align}
> $$

### Law of total probability

> $$
> \begin{align}
> p(x) &= \mathbb E_{\theta\sim p(\theta)} \big[ p(x\mid \theta)\big]
> \\
> p(x\mid z) &= \mathbb E_{\theta\sim p(\theta\mid z)} \big[ p(x\mid z,\theta) \big]
> \end{align}
> $$

Alternative notation:
$$
\begin{align}
p(x) &= \mathbb E_{\Theta} [p(x\mid\Theta)] \\
p(x\mid z) &= \mathbb E_{\Theta \mid z} [p(x\mid z,\Theta)] \\
\end{align}
$$

**Core Inituition**: computing a probability can be seen as

1. introducing an intermediate variable $\theta$,
2. computing the conditional probability,
3. and then averaging the conditional probability over $\theta$.

*Proof*: For the sake of simplicity, we assume all random variables are continuous. Otherwise, replace the integral with sum. By the law of marginal distribution,

$$
\begin{align*}
p(x)
= \int_\theta p(x,\theta) \,\mathrm{d}\theta
= \int_\theta p(\theta)\cdot p(x\mid\theta) \,\mathrm{d}\theta
= \mathbb E_{\theta\sim p(\theta)} \big[ p(x\mid \theta)\big]
\tag*{$\blacksquare$}
\end{align*}
$$

For  $p(x\mid z)$, we have

$$
\begin{align*}
p(x\mid z)
&= \frac{p(x,z)}{p(z)}
&&\text{def. of } p(x\mid z)
\\
&= \frac{1}{p(z)} \int_\theta p(x,z,\theta) \,\mathrm{d}\theta
&&\text{marginalization}
\\
&= \frac{1}{p(z)} \int_\theta p(z) \cdot p(\theta\mid z) \cdot p(x\mid z,\theta) \,\mathrm{d}\theta
&&\text{chain rule}
\\
&= \int_\theta p(\theta\mid z)\cdot p(x\mid z,\theta) \,\mathrm{d}\theta
&&\text{cancel out } p(z)
\\
&= \mathbb E_{\theta\sim p(\theta\mid z)} \big[ p(x\mid z,\theta) \big]
&&\text{def. of } \mathbb E_{\theta\sim p(\theta\mid z)}[\,\cdot\,]
\tag*{$\blacksquare$}
\end{align*}
$$

### Law of total expectation (tower rule)

> $$
> \begin{align}
> \mathbb E_{x\sim p(x)}[g(x)]
> &= \mathbb E_{\theta\sim p(\theta)} \Big[ \mathbb E_{x\sim p(x\mid \theta)}[g(x)] \Big]
> \\
> \mathbb E_{x\sim p(x\mid z)}[g(x)]
> &= \mathbb E_{\theta\sim p(\theta\mid z)} \Big[ \mathbb E_{x\sim p(x\mid z,\theta)}[g(x)] \Big]
> \end{align}
> $$

Alternative notation:

$$
\begin{align}
\mathbb E_X[g(X)] &= \mathbb E_{\Theta} \Big[ \mathbb E_X[g(X)\mid\Theta] \Big] \\
\mathbb E_X[g(X)\mid z] &= \mathbb E_{\Theta \mid z} \Big[ \mathbb E_X[g(X)\mid z,\Theta] \Big]
\end{align}
$$

**Core Inituition**: computing an expectation can be seen as

1. introducing an intermediate variable $\theta$,
2. computing the conditional expectation,
3. and then averaging the conditional expecation over $\theta$.

Special case: $g(\cdot) = \operatorname{id}(\cdot)$

> $$
> \begin{align}
> \mathbb E_{x\sim p(x)}[x]
> &= \mathbb E_{\theta\sim p(\theta)} \Big[ \mathbb E_{x\sim p(x\mid \theta)}[x] \Big]
> \\
> \mathbb E_{x\sim p(x\mid z)}[x]
> &= \mathbb E_{\theta\sim p(\theta\mid z)} \Big[ \mathbb E_{x\sim p(x\mid z,\theta)}[x] \Big]
> \end{align}
> $$

or in alternative notation:

$$
\begin{align}
\mathbb E_X[X] &= \mathbb E_{\Theta} \Big[ \mathbb E_X[X\mid\Theta] \Big] \\
\mathbb E_X[X\mid z] &= \mathbb E_{\Theta} \Big[ \mathbb E_X[X\mid z,\Theta] \Big]
\end{align}
$$

*Proof*: For the sake of simplicity, we assume all random variables are continuous. Otherwise, replace the integral with sum. For $\mathbb E_{x\sim p(x)}[g(x)]$, we have

$$
\begin{align*}
\mathbb E_{x\sim p(x)}[g(x)]
&= \int_x g(x) \cdot p(x) \,\mathrm{d}x
&&\text{def. of } \mathbb E_{x\sim p(x)}[\,\cdot\,]
\\
&= \int_x g(x) \left(\int_\theta p(\theta)\cdot p(x\mid\theta) \,\mathrm{d}\theta\right) \,\mathrm{d}x
&& \text{by } p(x)= \mathbb E_{\theta\sim p(\theta)} \big[ p(x\mid \theta)\big]
\\
&= \int_\theta p(\theta) \int_x g(x) \cdot p(x\mid\theta) \,\mathrm{d}x \,\mathrm{d}\theta
&& \text{switch the order}
\\
&= \int_\theta p(\theta) \, \mathbb E_{x\sim p(x\mid \theta)}[g(x)] \,\mathrm{d}\theta
&&\text{def. of } \mathbb E_{x\sim p(x\mid\theta)}[\,\cdot\,]
\\
&= \mathbb E_{\theta\sim p(\theta)} \big[ \mathbb E_{x\sim p(x\mid \theta)}[g(x)] \big]
&&\text{def. of } \mathbb E_{\theta\sim p(\theta)}[\,\cdot\,]
\tag*{$\blacksquare$}
\end{align*}
$$

For $\mathbb E_{x\sim p(x\mid z)}[g(x)]$, we have

$$
\begin{align*}
\mathbb E_{x\sim p(x\mid z)}[g(x)]
&= \int_x g(x) \cdot p(x\mid z) \,\mathrm{d}x
&&\text{def. of } \mathbb E_{x\sim p(x\mid z)}[\,\cdot\,]
\\
&= \int_x g(x) \left(\int_\theta p(\theta\mid z)\cdot p(x\mid z,\theta) \,\mathrm{d}\theta\right) \,\mathrm{d}x
&& \text{by } p(x\mid z) = \mathbb E_{\theta\sim p(\theta\mid z)} \big[ p(x\mid z,\theta) \big]
\\
&= \int_\theta p(\theta\mid z) \int_x g(x) \cdot p(x\mid z,\theta) \,\mathrm{d}x \,\mathrm{d}\theta
&& \text{switch the order}
\\
&= \int_\theta p(\theta\mid z) \, \mathbb E_{x\sim p(x\mid z,\theta)}[g(x)] \,\mathrm{d}\theta
&&\text{def. of } \mathbb E_{x\sim p(x\mid z,\theta)}[\,\cdot\,]
\\
&= \mathbb E_{\theta\sim p(\theta\mid z)} \big[ \mathbb E_{x\sim p(x\mid z,\theta)}[g(x)] \big]
&&\text{def. of } \mathbb E_{\theta\sim p(\theta\mid z)}[\,\cdot\,]
\tag*{$\blacksquare$}
\end{align*}
$$

#### Joint expectation and conditional expectation

> $$
> \begin{align}
> \mathbb{E}_{(x,y)\sim p(x,y)}[g(x,y)] =
> \mathbb{E}_{x \sim p(x)} \big[ \mathbb{E}_{y \sim p(y \mid x)}[g(x,y)] \big]
> \end{align}
> $$

or in alternative notation:

$$
\begin{align}
\mathbb{E}_{XY}[g(X,Y)] =
\mathbb{E}_{X} \big[ \mathbb{E}_{Y}[g(X,Y) \mid X] \big]
\end{align}
$$

*Proof*: Let $U=(X,Y)$ and $V=X$. Applying the law of total probability yields

$$
\begin{align*}
\mathbb E_U[g(U)] &= \mathbb E_{V} \Big[ \mathbb E_U[g(U) \mid V] \Big] \\
\mathbb E_{X,Y}[g(X,Y)] &= \mathbb E_{X} \Big[ \mathbb E_{X,Y}[g(X,Y) \mid X] \Big] \\
\mathbb E_{X,Y}[g(X,Y)] &= \mathbb E_{X} \Big[ \mathbb E_{Y}[g(X,Y) \mid X] \Big]
\tag*{$\blacksquare$}
\end{align*}
$$

*Proof (alt.)*: The claim can also be directly proved:

$$
\begin{align*}
\mathbb{E}_{(x,y)\sim p(x,y)}[g(x,y)]
&= \int\int g(x,y) \cdot p(x,y) \:\mathrm{d}x \mathrm{d}y \\
&= \int\int g(x,y) \cdot p(x) p(y \mid x) \:\mathrm{d}x \mathrm{d}y \\
&= \int p(x)
     \underbrace{\int g(x,y) \cdot p(y \mid x) \:\mathrm{d}y}_{\mathbb{E}_{y \sim p(y \mid x)}[g(x,y)]}
   \mathrm{d}x
\tag*{$\blacksquare$}
\end{align*}
$$

## Information Theory

### Entropy

Let $p$ be a probability distribution. The ***entropy*** of $p$ is defined as

> $$
> \begin{align}
> H(p)
> &= \mathbb E_{x \sim p(\cdot)} \left[ -\log p(x) \right]
> \end{align}
> $$

For a discrete distribution $p$, the resulting $H(p)$ is called ***Shannon entropy***:

> $$
> \begin{align}
> H(p)
> &= \sum_{x \in \mathcal X} p(x) \log \frac{1}{p(x)}
> \end{align}
> $$

For a continuous distribution $p$, the resulting $H(p)$ is called ***differential entropy***:

> $$
> \begin{align}
> H(p)
> &= \int_{-\infty}^{\infty} p(x) \log \frac{1}{p(x)} \,\mathrm dx
> \end{align}
> $$

Remarks:

* **Core Inituition**: The entropy quantifies the average surprise of $x$ sampled from $p$.
* Without otherwise specified, the term "*entropy*" refers to both Shannon entropy and differential entropy.
* The Shannon entropy is extensively studied in information theory. In machine learning, we also encounter differential entropy, whose properties are different from Shannon entropy.

The univariate Gaussian has the largest differential entropy among all (univariate) distributions with the same mean and the same variance:

> Let $g(x) = \mathcal N(x; \mu, \sigma^2)$ and $p$ be any other distribution on $\mathbb R$ with mean $\mu$ and variance $\sigma^2$. Then,
> $$
> \begin{align}
> H(g) \ge H(p)
> \end{align}
> $$

*Proof*: In section [Cross Entropy](#cross-entropy), we can show that

$$
H(p,g) = H(g)
$$

By the non-negativity of [KL divergence](#kl-divergence), we conclude

$$
\begin{align*}
D_\text{KL}(p \parallel g)
&= H(p,g) - H(p) \\
&= H(g) - H(p) \\
&\ge 0
\tag*{$\blacksquare$}
\end{align*}
$$

### Cross Entropy

Let $p,q$ be two probability distributions. The ***cross entropy*** of $q$ w.r.t $p$ is defined as

> $$
> \begin{align}
> H(p, q)
> &= \mathbb E_{x \sim p(\cdot)} \left[ -\log q(x) \right]
> \end{align}
> $$

Remarks:

* **Core Inituition**: The cross entropy quantifies the average surprise when we assume $x \sim q$ while the true distribution is $p$.
* In machine learning, $p$ often denotes the unknown ground truth distribution which generates the training data. A typical task is to compute a model distribution $q$ as close to $p$ as possible.

For discrete distributions, the cross entropy becomes

> $$
> \begin{align}
> H(p, q)
> &= \sum_{x \in \mathcal X} p(x) \log \frac{1}{q(x)}
> \end{align}
> $$

For continuous distributions, the cross entropy becomes

> $$
> \begin{align}
> H(p, q)
> &= \int_{-\infty}^{\infty} p(x) \log \frac{1}{q(x)} \,\mathrm dx
> \end{align}
> $$

Decomposition of cross entropy:

> $$
> \begin{align}
> H(p, q) = H(p) + D_\text{KL}(p \parallel q)
> \end{align}
> $$

**Core Inituition**: The average surprise when we assume $x \sim q$ while the true distribution is $p$ consits of

* $H(p)$: the uncertainty in the true distribution $p$
* $D_\text{KL}(p \parallel q)$: the additional surprise incurred by assuming $x \sim q$ while the true distribution is $p$

*Proof*:

$$
\begin{align*}
H(p, q)
&= \mathbb E_{x \sim p(\cdot)} \left[ \log\frac{1}{q(x)} \right] \\
&= \mathbb E_{x \sim p(\cdot)} \left[ \log\frac{1}{p(x)} + \log \frac{p(x)}{q(x)} \right] \\
&= \underbrace{\mathbb E_{x \sim p(\cdot)} \left[ \log\frac{1}{p(x)} \right]}_{H(p)} +
   \underbrace{\mathbb E_{x \sim p(\cdot)} \left[ \log \frac{p(x)}{q(x)} \right]}_{D_\text{KL}(p \parallel q)}
\tag*{$\blacksquare$}
\end{align*}
$$

The cross entropy is generally **not** symmetric:

> $$
> \begin{align}
> H(p, q) \ne H(q, p)
> \end{align}
> $$

The cross entropy of a univariate Gaussian is invariant to the true distribution, assuming they have the same 1st and 2nd order moments:

> Let $g(x) = \mathcal N(x; \mu, \sigma^2)$ and $p$ be any other distribution on $\mathbb R$ with mean $\mu$ and variance $\sigma^2$. Then,
> $$
> \begin{align}
> H(p,g) = H(g)
> \end{align}
> $$

*Proof*: It is easy to verify that

$$
-\log g(x) = \log\sqrt{2\pi\sigma^2} + \frac{(x-\mu)^2}{2\sigma^2}
$$

Hence,

$$
\begin{align*}
H(p, g)
&= \mathbb E_{x \sim p(\cdot)} \left[ -\log q(x) \right]
\\
&= \log\sqrt{2\pi\sigma^2} + \frac{1}{2\sigma^2} \underbrace{
   \mathbb E_{x \sim p(\cdot)} \left[ (x-\mu)^2 \right]
   }_{\sigma^2}
\\
&= \log\sqrt{2\pi\sigma^2} + \frac{1}{2}
\\
&= H(g)
\tag*{$\blacksquare$}
\end{align*}
$$

### KL Divergence

Let $p,q$ be two probability distributions. The ***KL Divergence*** (or ***relative entropy***) of $q$ w.r.t $p$ is defined as

> $$
> \begin{align}
> D_\text{KL}(p \parallel q)
> &= \mathbb E_{x \sim p(\cdot)} \left[ \log \frac{p(x)}{q(x)} \right]
> \end{align}
> $$

**Core Inituition**: KL divergence quantifies the additional surprise incurred by assuming $x \sim q$ while the true distribution is $p$. It can also be viewed as a asymmetric distance between $p$ and $q$.

For discrete distributions, the KL divergence becomes

> $$
> \begin{align}
> D_\text{KL}(p \parallel q)
> &= \sum_{x \in \mathcal X} p(x) \log \frac{p(x)}{q(x)}
> \end{align}
> $$

For continuous distributions, the KL divergence becomes

> $$
> \begin{align}
> D_\text{KL}(p \parallel q)
> &= \int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)} \:\mathrm dx
> \end{align}
> $$

KL divergence is generally **asymmetric**:

> $$
> \begin{align}
> D_\text{KL}(p \parallel q) \ne D_\text{KL}(q \parallel p)
> \end{align}
> $$

KL divergence is **ALWAYS** non negative (Gibb's inequality):

> $$
> \begin{align}
> D_\text{KL}(p \parallel q) &\ge 0, \quad \forall p,q \\
> D_\text{KL}(p \parallel q) &= 0 \iff p=q \quad\text{a.e.}
> \end{align}
> $$

Remarks:

* The KL divergence is always non negative, even for continuous distributions. (In contrast, the non negativity of differential entropy is not guaranteed)
* In machine learning, we often compute the model distribution $q$ by minimizing its KL divergence w.r.t. the ground truth distribution $p$.

### Entropy of Multivariate Gaussian

For a multivariate Gaussian $p(x) = \mathcal N(x ; \mu, \Sigma), \, x \in \mathbb R^d$, the differential entropy is:

$$
\begin{align}
H(p)
&= \mathbb E_{x \sim p} \left[ \log \frac{1}{p(x)} \right] \\
&= \frac{1}{2} \log \left[ (2\pi e)^d \det(\Sigma) \right] \\
&= \frac{1}{2} \log \left[ \det(\Sigma) \right] + \frac{d}{2} \log (2\pi e) \\
\end{align}
$$

Let ${L}$ (lower triangular) be the Cholesky factor of the covariance matrix, i.e. $\Sigma = {LL}^\top$. Then,

$$
\begin{align*}
\log \left[ \det(\Sigma) \right]
&= \log \left[ \det(LL^\top) \right] \\
&= \log \left[ \det(L) \cdot \det({L}^\top) \right] \\
&= \log \left[ \det(L)^2 \right] \\
&= 2\log \left[ \det(L) \right] \\
\end{align*}
$$

Therefore, We can express $H(p)$ as

$$
\begin{align}
H(p)
&= \log \left[ \det({L}) \right] + \frac{d}{2} \log (2\pi e) \\
\end{align}
$$
