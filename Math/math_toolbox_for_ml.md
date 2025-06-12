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

*Proof*: Let $Y = (X - \mathbb E[X])^2$. Then, applying Markov inequality on $Y$ yields Chebyshev inequality:

$$
\begin{align*}
P(Y \ge \epsilon^2) &\le \frac{\mathbb E[Y]}{\epsilon^2} \\
P((X - \mathbb E[X])^2 \ge \epsilon^2) &\le \frac{\mathbb E[(X - \mathbb E[X])^2]}{\epsilon^2} \\
P(\vert X - \mathbb E[X] \vert \ge \epsilon) &\le \frac{\mathbb V[X]}{\epsilon^2}
\tag*{$\blacksquare$}
\end{align*}
$$

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

$$
\mathbb{E}_{(x,y)\sim p(x,y)}[g(x,y)] =
\mathbb{E}_{x \sim p(x)} \big[ \mathbb{E}_{y \sim p(y \mid x)}[g(x,y)] \big]
$$

or in alternative notation:

$$
\mathbb{E}_{XY}[g(X,Y)] =
\mathbb{E}_{X} \big[ \mathbb{E}_{Y}[g(X,Y) \mid X] \big]
$$

TODO: proof

### Law of total variance

TODO
