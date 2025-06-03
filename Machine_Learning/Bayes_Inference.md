---
title: "Bayesian Inference"
author: "Ke Zhang"
date: "2025"
fontsize: 12pt
---

# Intro to Bayesian Inference

[toc]

$$
\DeclareMathOperator*{\argmax}{argmax}
\DeclareMathOperator*{\argmin}{argmin}
$$

Preliminary knowledge required:

* conditional, marginal, joint probabilities
* parameter estimation (MLE and MAP)

## Recap: Point Estimate by MAP

Problem set-up of MAP:

* Given: observations $D = \{\mathbf x_1, \dots, \mathbf x_n\} \stackrel{\text{iid}}{\sim} p(\mathbf x \mid \boldsymbol{\theta})$ with prior $p(\boldsymbol{\theta})$
* Goal: estimate $\boldsymbol{\theta}$
* Use: $p(\mathbf x_* \mid \hat{\boldsymbol{\theta}})$

The MAP estimate is the **mode of the posterior**

$$
\begin{align}
\hat{\boldsymbol{\theta}}_\text{MAP}
&= \argmax_{\boldsymbol{\theta}} p(\boldsymbol{\theta} \mid D)
\end{align}
$$

For a new data point $\mathbf x_*$, the ***plug-in predictive distribution*** is fully charactered by the point estimate $\hat{\boldsymbol{\theta}}_\text{MAP}$

$$
\begin{align}
p(\mathbf x_* \mid \hat{\boldsymbol{\theta}}_\text{MAP})
\end{align}
$$

Remark:

* For simplicity, we often call $p(\mathbf x_* \mid \hat{\boldsymbol{\theta}}_\text{MAP})$ ***plug-in predictive***.
* The term "plug-in" hightlights the fact that we plug a single point estimate $\hat{\boldsymbol{\theta}}_\text{MAP}$ into $p(\mathbf x_* \mid \boldsymbol{\theta})$

## Bayesian Inference

Key idea of ***Bayesian inference***:

> Use the **full** posterior distribution $p(\boldsymbol{\theta} \mid D)$ instead of just using its mode (i.e. a point estimate)

Philosophy: Frequentist statistics vs. Bayesian statistics

* Frequentist statistics: $\boldsymbol{\theta}$ is unknown but fixed. It makes no sense to talk about the probability of $\boldsymbol{\theta}$ (either prior or posterior). The true PDF of $\mathbf x$ is thus an unknown but fixed function. The observations are used to estimate the PDF of $\mathbf x$ as accurately as possible.
* Bayesian statistics: $\boldsymbol{\theta}$ is a random variable. There are (infinitely) many possible PDFs of $\mathbf x$, some of which are more likely (or more important) than the others. After we made observations, we update the PDF of $\boldsymbol{\theta}$ and thus updated the importance of each possible PDF of $\mathbf x$.

In full Bayesian approach (or Bayesian inference), we do not seek for a point estimate of $\boldsymbol{\theta}$ but to update our posterior belief on each $\boldsymbol{\theta}$. Even the most "trustworthy" $\boldsymbol{\theta}$ does not monoply all of our belief. In contrast, MAP places all of our belief on a particular parameter vector, namely $\hat{\boldsymbol{\theta}}_\text{MAP}$.

### Prior Predictive Distribution

For a new data point $\mathbf x_*$, each $\boldsymbol{\theta}$ gives $p(\mathbf x_* \mid \boldsymbol{\theta})$. Averaging over all possible $\boldsymbol{\theta}$, we have the ***prior predictive (distribution)***:

> $$
> \begin{align}
> p(\mathbf x_*)
> &= \mathbb E_{\boldsymbol{\theta} \sim p(\boldsymbol{\theta})} \left[ p(\mathbf x_* \mid \boldsymbol{\theta}) \right] \\
> &= \int p(\mathbf x_* \mid \boldsymbol{\theta}) \cdot p(\boldsymbol{\theta}) \:\mathrm{d}\boldsymbol{\theta} \\
> \end{align}
> $$

Remarks:

* The proof is very simple as the RHS is just the marginalization of the joint density $p(\mathbf x_*, \boldsymbol{\theta})$.
* The RHS can also be seen as a weighted average of $p(\mathbf x_* \mid \boldsymbol{\theta})$ w.r.t. the prior $p(\boldsymbol{\theta})$, which measures the importance of each $\boldsymbol{\theta}$. If $p(\boldsymbol{\theta})$ is high, then $p(\mathbf x_* \mid \boldsymbol{\theta})$ has higher contribution to $p(\mathbf x_*)$. Let $\hat{\boldsymbol{\theta}}$ be the mode of the prior. Then, $p(\mathbf x_* \mid \hat{\boldsymbol{\theta}})$ has the highest (but **not the entire!**) contribution to $p(\mathbf x_*)$.
* The formula holds even **before** we made any observations! Bayesian statistics allows prior distribution which may come from our experience.

Not to be confused: prior vs. prior predictive

* prior $p(\boldsymbol{\theta})$: distribution of parameter vector $\boldsymbol{\theta}$ before observing $D$. It quantifies the importance of each $\boldsymbol{\theta}$ before we saw $D$.
* prior predictive $p(\mathbf x_*)$: predictive distribution of unseen observation $\mathbf x_*$. It is the average of the likelihood $p(\mathbf x_* \mid \boldsymbol{\theta})$ according to the prior.

### Posterior Predictive Distribution

After we made observations $D = \{\mathbf x_1, \dots, \mathbf x_n\}$, we update the prior $p(\boldsymbol{\theta})$ to posterior $p(\boldsymbol{\theta} \mid D)$. Now, we can average $p(\mathbf x_* \mid \boldsymbol{\theta})$ w.r.t. the posterior, which leads to ***posterior predictive (distribution)***:

> $$
> \begin{align}
> p(\mathbf x_* \mid D)
> &= \mathbb E_{\boldsymbol{\theta} \sim p(\boldsymbol{\theta} \mid D)} \left[ p(\mathbf x_* \mid \boldsymbol{\theta}) \right] \\
> &= \int p(\mathbf x_* \mid \boldsymbol{\theta}) \cdot p(\boldsymbol{\theta} \mid D) \:\mathrm{d}\boldsymbol{\theta}
> \end{align}
> $$

Remarks:

* The LHS is now conditioned on $D$, i.e. we updated $p(\mathbf x_*)$ to $p(\mathbf x_* \mid D)$ after observing $D$.
* Updating the prior to posterior $\iff$ updating the importance of $p(\mathbf x_* \mid \boldsymbol{\theta})$ for each $\boldsymbol{\theta}$. Let $\hat{\boldsymbol{\theta}}$ be the mode of the posterior. Then, $p(\mathbf x_* \mid \hat{\boldsymbol{\theta}})$ has the highest (but **not the entire!**) contribution to $p(\mathbf x_*)$.
* The integral on the RHS is intractable to compute in general, except for a few special case (which will be discussed later).

*Proof*: This follows from the law of total expectation and the conditional independence between $\mathbf x_*$ and $D$ given $\boldsymbol{\theta}$.

$$
\begin{align*}
p(\mathbf x_* \mid D)
&= \int p(\mathbf x_*, \boldsymbol{\theta} \mid D) \:\mathrm{d}\boldsymbol{\theta}
&& \text{marginalization}
\\
&= \int p(\mathbf x_* \mid \boldsymbol{\theta}, D) \cdot p(\boldsymbol{\theta} \mid D) \:\mathrm{d}\boldsymbol{\theta}
&& \text{factorization}
\\
&= \int p(\mathbf x_* \mid \boldsymbol{\theta}) \cdot p(\boldsymbol{\theta} \mid D) \:\mathrm{d}\boldsymbol{\theta}
&& \mathbf x_* \perp D\mid \boldsymbol{\theta}
\tag*{$\blacksquare$}
\end{align*}
$$

Not to be confused: posterior vs. posterior predictive

* posterior $p(\boldsymbol{\theta} \mid D)$: distribution of parameter vector $\boldsymbol{\theta}$ after observing $D$. It quantifies the importance of each $\boldsymbol{\theta}$ after we saw $D$.
* posterior predictive $p(\mathbf x_* \mid D)$: predictive distribution of unseen observation $\mathbf x_*$. It is the average of the likelihood $p(\mathbf x_* \mid \boldsymbol{\theta})$ according to the posterior.

### Bayesian Inference vs MAP

Both methods

* treat parameters $\boldsymbol{\theta}$ as a random variable
* assume a prior $p(\boldsymbol{\theta})$.
* update the prior to a  posterior $p(\boldsymbol{\theta} \mid D)$ after observing $D$
* use the posterior to compute compute $p(\mathbf x_*)$

Difference: How is $p(\mathbf x_*)$ computed?

| Bayesian Inference | MAP estimate |
| ------------------ | ------------ |
| uses the full posterior distribution | only uses the mode of the posterior |
| preserves uncertainty in the posterior | discards uncertainty in the posterior |
| integration-based (average w.r.t. the posterior) | optimization-based (maximize the posterior) |
| output: posterior predictive | output: plug-in predictive |

> **Connection**: MAP estimation can be seen as a special case of Bayesian inference if we place all probabity mass at on the mode of the posterior $p(\boldsymbol{\theta} \mid D)$. In this case, the posterior predictive simplifies to the plug-in predictive.

Recall that the posterior predictive is

$$
\begin{align*}
p(\mathbf x_* \mid D)
&= \int p(\mathbf x_* \mid \boldsymbol{\theta}) \cdot p(\boldsymbol{\theta} \mid D) \:\mathrm{d}\boldsymbol{\theta}
\end{align*}
$$

The mode of the posterior $p(\boldsymbol{\theta} \mid D)$ is by definition $\hat{\boldsymbol{\theta}}_\text{MAP}$. Replacing the posterior $p(\boldsymbol{\theta} \mid D)$ with the point density $\delta(\boldsymbol{\theta} - \hat{\boldsymbol{\theta}}_\text{MAP})$ in the posterior predictive, we conclude that the posterior predictive becomes plug-in predictive.

$$
\begin{align*}
p(\mathbf x_* \mid D)
&= \int p(\mathbf x_* \mid \boldsymbol{\theta}) \cdot \delta(\boldsymbol{\theta} - \hat{\boldsymbol{\theta}}_\text{MAP}) \:\mathrm{d}\boldsymbol{\theta}
\\
&= p(\mathbf x_* \mid \hat{\boldsymbol{\theta}}_\text{MAP})
\end{align*}
$$

Even though both MAP estimation and Bayesian inference require the posterior, they have the following distinction:

* In MAP estimation, we do not have to compute the posterior exactly since MAP is equivalent to maximize the log of unnormalized posterior (which is generally easier to optimize)

$$
\begin{align*}
\hat{\boldsymbol{\theta}}_\text{MAP}
&= \argmax_{\boldsymbol{\theta}} p(\boldsymbol{\theta} \mid D) \\
&= \argmax_{\boldsymbol{\theta}} \ln p(D \mid \boldsymbol{\theta}) + \ln p(\boldsymbol{\theta})
\end{align*}
$$

* Bayesian inference, however, requires computing the posterior exactly, which is generally has no closed-form solution except for a few special cases. One of those special cases is where the prior $p(\boldsymbol{\theta})$ and the resulting posterior $p(\boldsymbol{\theta} \mid D)$ belong to the same distribution family $\mathcal P$. We call such $p(\boldsymbol{\theta})$ a ***conjugate prior*** **to the likelihood** $p(D \mid \boldsymbol{\theta})$. Examples of conjugate priors will be illustrated in the next sections.

$$
\begin{align*}
\underbrace{p(\boldsymbol{\theta} \mid D)}_{\text{posterior }\in\mathcal P}
\propto p(D \mid \boldsymbol{\theta}) \cdot \underbrace{p(\boldsymbol{\theta})}_{\text{prior }\in\mathcal P}
\end{align*}
$$

## Examples of Bayesian Inference

In general, the posterior and the posterior predictive requires numerical approximation since they have no closed-form solution. Here, we show two exceptions where the prior and the posterior belong to the same parameteric family.

### Learning a Bernoulli Distribution with Beta Prior

Consider a Bernoulli distribution with unknown success rate $\theta\in[0,1]$.

$$
p(x) =
\begin{cases}
\theta   & \text{if } x=1 \\
1-\theta & \text{if } x=0
\end{cases}
$$

Suppose we have iid observations $D=\{1, 1, 0, \dots, 0, 1\}$ containing $n_1$ ones and $n_0$ zeros. Then, the likelihood is

$$
p(D \mid \theta) = \theta^{n_1} (1-\theta)^{n_0}
$$

which is proportional to the PMF of binomial distribution $\operatorname{Bin}(n_1+n_0, \theta)$.

The conjugate prior to binomial distribuion is Beta distribution, i.e. $\theta\sim\operatorname{Beta}(\alpha,\beta)$ with

$$
p(\theta) = \frac{\theta^{\alpha-1} (1-\theta)^{\beta-1}}{B(\alpha,\beta)}
$$

where $B(\alpha,\beta)$ is a normalization constant (not detailed here). Then, the posterior is

$$
\begin{align*}
p(\theta \mid D)
&\propto p(D \mid \theta) \cdot p(\theta) \\
&= \theta^{n_1} (1-\theta)^{n_0} \cdot \theta^{\alpha-1} (1-\theta)^{\beta-1} \\
&= \theta^{n_1+\alpha-1} (1-\theta)^{n_0+\beta-1}
\end{align*}
$$

By the normalization property of Beta distribution (not detailed here), the posterior is in fact also a Beta distribution

$$
\begin{align*}
p(\theta \mid D)
&= \frac{\theta^{n_1+\alpha-1} (1-\theta)^{n_0+\beta-1}}{B(n_1+\alpha,n_0+\beta)} \\
&= \operatorname{Beta}(\theta; n_1+\alpha,n_0+\beta)
\end{align*}
$$

Hence, the posterior predictive $p(x_* \mid D)$ at $x_*=1$ is

$$
\begin{align*}
p(x_*=1 \mid D)
&= \int p(x_*=1 \mid \theta) \cdot p(\theta \mid D) \:\mathrm{d}\theta \\
&= \int \theta \cdot p(\theta \mid D) \:\mathrm{d}\theta \\
&= \mathbb E[\theta \mid D] \\
&= \frac{n_1+\alpha}{n_1+n_0+\alpha+\beta}
\end{align*}
$$

Likewise,

$$
\begin{align*}
p(x_*=0 \mid D)
&= \int p(x_*=0 \mid \theta) \cdot p(\theta \mid D) \:\mathrm{d}\theta \\
&= \int (1-\theta) \cdot p(\theta \mid D) \:\mathrm{d}\theta \\
&= 1 - \mathbb E[\theta \mid D] \\
&= \frac{n_0+\beta}{n_1+n_0+\alpha+\beta}
\end{align*}
$$

### Learning a Gaussian Distribution with Gaussian Prior

Suppose we have iid observations $D=\{\mathbf x_1, \dots, \mathbf x_n\}$ drawn from a multivariate Gaussian with known covariance matirx $\boldsymbol\Sigma$ but unknown mean vector $\boldsymbol\mu$.

$$
\begin{align}
\mathbf x_i \sim \mathcal N(\boldsymbol\mu, \boldsymbol\Sigma),\quad
i=1,\dots,n
\end{align}
$$
Then, the likelihood is
$$
\begin{align*}
p(D \mid \boldsymbol\mu)
&= \prod_{i=1}^n \mathcal N(\mathbf x_i; \boldsymbol\mu, \boldsymbol\Sigma)
\\
&\propto \prod_{i=1}^n \exp\left( -\frac{1}{2}(\mathbf x_i-\boldsymbol\mu)^\top \boldsymbol\Sigma^{-1} (\mathbf x_i-\boldsymbol\mu) \right)
\\
&= \exp\left( -\frac{1}{2} \sum_{i=1}^n (\mathbf x_i-\boldsymbol\mu)^\top \boldsymbol\Sigma^{-1} (\mathbf x_i-\boldsymbol\mu) \right)
\\
&\propto \exp\left( -\frac{1}{2} \sum_{i=1}^n \Big( \boldsymbol\mu^\top \boldsymbol\Sigma^{-1} \boldsymbol\mu -2 \boldsymbol\mu^\top \boldsymbol\Sigma^{-1} \mathbf x_i \Big) \right)
\\
&= \exp\left( -\frac{n}{2}\boldsymbol\mu^\top \boldsymbol\Sigma^{-1} \boldsymbol\mu + \boldsymbol\mu^\top \boldsymbol\Sigma^{-1} \sum_{i=1}^n  \mathbf x_i \right)
\end{align*}
$$

Let $\bar{\mathbf x} \triangleq \frac{1}{n} \sum_{i=1}^n  \mathbf x_i$ denote the sample mean. Then, the likelihood becomes

$$
\begin{align*}
p(D \mid \boldsymbol\mu)
&\propto \exp\left( -\frac{n}{2}\boldsymbol\mu^\top \boldsymbol\Sigma^{-1} \boldsymbol\mu + n\boldsymbol\mu^\top \boldsymbol\Sigma^{-1} \bar{\mathbf x} \right)
\end{align*}
$$

Assume we have a Gaussian prior on $\boldsymbol\mu$:

$$
\begin{align}
\boldsymbol\mu \sim \mathcal N(\boldsymbol\mu_0, \boldsymbol\Sigma_0)
\end{align}
$$

Hence, the posterior is

$$
\begin{align*}
p(\boldsymbol\mu \mid D)
&\propto p(D \mid \boldsymbol\mu) \cdot p(\boldsymbol\mu)
\\
&\propto \exp\left( -\frac{n}{2}\boldsymbol\mu^\top \boldsymbol\Sigma^{-1} \boldsymbol\mu + n\boldsymbol\mu^\top \boldsymbol\Sigma^{-1} \bar{\mathbf x} \right) \cdot
  \exp\left( -\frac{1}{2}\boldsymbol\mu^\top \boldsymbol\Sigma^{-1}_0 \boldsymbol\mu + \boldsymbol\mu^\top \boldsymbol\Sigma^{-1}_0 \boldsymbol\mu_0 \right)
\\
&= \exp\left( -\frac{1}{2} \boldsymbol\mu^\top \left( n\boldsymbol\Sigma^{-1} + \boldsymbol\Sigma^{-1}_0 \right) \boldsymbol\mu +
  \boldsymbol\mu^\top \left(n\boldsymbol\Sigma^{-1}\bar{\mathbf x} + \boldsymbol\Sigma^{-1}_0 \boldsymbol\mu_0 \right) \right)
\end{align*}
$$

Note that the log posterior is a quadratic function of $\boldsymbol{\mu}$. By the normalization property of Gaussian distribution, the posterior is also Gaussian

$$
\begin{align}
\boldsymbol\mu \mid D \sim \mathcal N(\boldsymbol\mu_n, \boldsymbol\Sigma_n)
\end{align}
$$

with posterior mean and posterior variance
$$
\begin{align}
\boldsymbol\Sigma_n^{-1}
&= n\boldsymbol\Sigma^{-1} + \boldsymbol\Sigma^{-1}_0
\\
\boldsymbol\mu_n
&= \boldsymbol\Sigma_n  \left(n\boldsymbol\Sigma^{-1}\bar{\mathbf x} + \boldsymbol\Sigma^{-1}_0 \boldsymbol\mu_0 \right)
\end{align}
$$
The posterior predicitve distribution can be calculated without evaluating any integral by noting that
$$
\begin{align}
\mathbf{x}_* &= \boldsymbol{\mu} + \boldsymbol{\varepsilon}, \quad
\boldsymbol{\varepsilon} \sim \mathcal N(\mathbf 0, \boldsymbol{\Sigma}), \quad
\boldsymbol{\varepsilon} \perp \boldsymbol{\mu}
\end{align}
$$
Recall that sum of independent Gaussian is again Guassian. Hence, we obtain the posterior predictive distribution
$$
\begin{align}
\mathbf x_* \mid D \sim
\mathcal N(\boldsymbol\mu_n, \boldsymbol\Sigma + \boldsymbol\Sigma_n)
\end{align}
$$

Remarks:

* We differentiate three covariance matrices
  * $\boldsymbol\Sigma_0$: prior uncertainty about $\boldsymbol\mu$, i.e. the variance of $\boldsymbol\mu$ before observing any data.
  * $\boldsymbol\Sigma_n$: posterior uncertainty about $\boldsymbol\mu$, i.e. the variance of $\boldsymbol\mu$ after observing $D=\{\mathbf x_1, \dots, \mathbf x_n\}$. It arises due to lask of data and reduces to zero as the number of of observations increases.
  * $\boldsymbol\Sigma$: inherent and irreducible noise in $\mathbf x_*$, independent of $\boldsymbol\mu$, assumed to be known in our set-up.
* The posterior uncertainty (variance) of $\mathbf x_*$ consists of posterior uncertainry about the parameter $\boldsymbol\mu$ plus the inherent noise.

* The mode of $p(\boldsymbol\mu \mid D)$ is exactly $\boldsymbol\mu_n$. The plug-in predictive is thus
  $$
  \mathbf x_* \mid \boldsymbol\mu_n, \boldsymbol\Sigma \sim
  \mathcal N(\boldsymbol\mu_n, \boldsymbol\Sigma)
  $$

* Comparing the plug-in predictive with the posterior predictive, we see that the former discards uncertainty of the posterior $p(\boldsymbol\mu \mid D)$.

To gain more insight about the posterior predictive, we reformulate $\boldsymbol\mu_n$ as

$$
\begin{align*}
\boldsymbol\mu_n
&= \boldsymbol\Sigma_n \left(n\boldsymbol\Sigma^{-1}\bar{\mathbf x} + \boldsymbol\Sigma^{-1}_0 \boldsymbol\mu_0 \right)
\\
&= \boldsymbol\Sigma_n \left(n\boldsymbol\Sigma^{-1}\bar{\mathbf x} - n\boldsymbol\Sigma^{-1} \boldsymbol\mu_0 + n\boldsymbol\Sigma^{-1} \boldsymbol\mu_0 + \boldsymbol\Sigma^{-1}_0 \boldsymbol\mu_0 \right)
\\
&= \boldsymbol\Sigma_n n \boldsymbol\Sigma^{-1} \left( \bar{\mathbf x} - \boldsymbol\mu_0 \right) +
   \boldsymbol\Sigma_n \underbrace{\left( n \boldsymbol\Sigma^{-1} + \boldsymbol\Sigma^{-1}_0 \right)}_{\boldsymbol\Sigma_n^{-1}} \boldsymbol\mu_0
\end{align*}
$$

Therefore,

$$
\begin{align}
\boldsymbol\mu_n
&= \boldsymbol\mu_0 +
   n\boldsymbol\Sigma_n \boldsymbol\Sigma^{-1} \left( \bar{\mathbf x} - \boldsymbol\mu_0 \right)
\end{align}
$$

Remarks:

* The posterior mean $\boldsymbol\mu_n$ is the prior mean $\boldsymbol\mu_0$ plus a correction term toward the sample mean $\bar{\mathbf x}$. The difference $\bar{\mathbf x} - \boldsymbol\mu_0$ quantifies "the new information" brought by observations -- similar idea as innovation in Kalman filter.
* If the noise is low (i.e. $\boldsymbol\Sigma$ is small), the correction towards $\bar{\mathbf x}$ is stronger. i.e. We trust more on the data if they are high-quality.
* The correction towards $\bar{\mathbf x}$ is stronger as $n$ increases, i.e. The more data we have, the more we trust on sample mean.

The 3rd remark follows from the fact that

$$
\lim_{n\to\infty} n\boldsymbol\Sigma_n \boldsymbol\Sigma^{-1} = \mathbf I
$$

*Proof*: Recall that $\boldsymbol\Sigma_n^{-1} \triangleq \left( n\boldsymbol\Sigma^{-1} + \boldsymbol\Sigma^{-1}_0 \right)$. Multiplying both sides with $\boldsymbol\Sigma_n$, we get
$$
\mathbf I
=\boldsymbol\Sigma_n \left( n\boldsymbol\Sigma^{-1} + \boldsymbol\Sigma^{-1}_0 \right)
=n\boldsymbol\Sigma_n \boldsymbol\Sigma^{-1} + \boldsymbol\Sigma_n \boldsymbol\Sigma^{-1}_0
$$

Using the fact that $\lim_{n\to\infty} \boldsymbol\Sigma_n = \mathbf 0$, we conclude

$$
\begin{align*}
\lim_{n\to\infty} n\boldsymbol\Sigma_n \boldsymbol\Sigma^{-1}
&= \lim_{n\to\infty} \mathbf I - \boldsymbol\Sigma_n \boldsymbol\Sigma^{-1}_0 \\
&= \mathbf I - \underbrace{\lim_{n\to\infty} \boldsymbol\Sigma_n \boldsymbol\Sigma^{-1}_0}_{\mathbf 0}
\tag*{$\blacksquare$}
\end{align*}
$$

In 1D speical case, the posterior becomes

$$
\begin{align}
\mu \mid D \sim \mathcal N(\mu_n, \sigma_n^2)
\end{align}
$$

where

$$
\begin{align*}
\frac{1}{\sigma_n^2}
= \left( \frac{n}{\sigma^{2}} + \frac{1}{\sigma^{2}_0} \right)
&\iff
\sigma_n^2
= \frac{\sigma^{2} \sigma^{2}_0}{n\sigma^{2}_0 + \sigma^{2}}
\\
\mu_n
= \sigma_n^2 \left( \frac{n}{\sigma^{2}}\bar{x} + \frac{1}{\sigma^{2}_0}\mu_0 \right)
&\iff
\mu_n
= \mu_0 + \frac{n\sigma^{2}_0}{n\sigma^{2}_0 + \sigma^{2}} \left( \bar{x} - \mu_0 \right)
\end{align*}
$$

The resulting posterior predictive is

$$
\begin{align}
x_* \mid D \sim \mathcal N(\mu_n, \sigma^2 + \sigma_n^2)
\end{align}
$$

If, in addition, the data set happens to be $D=\{x_1\}$ (i.e. $n=1$), then the posterior can be simplified further to
$$
\begin{align*}
\sigma_1^2
&= \frac{1}{\sigma^{-2}_0 + \sigma^{-2}}
\\
\mu_1
&= \mu_0 + \frac{\sigma^{2}_0}{\sigma^{2}_0 + \sigma^{2}} \left( x_1 - \mu_0 \right)
\end{align*}
$$

## Recurisve Bayesian Inference

Key idea: Posterior today $\triangleq$ Prior tomorrow

$$
p(\boldsymbol{\theta})
\xrightarrow{\mathbf x_1}
p(\boldsymbol{\theta} \mid \mathbf x_1)
\xrightarrow{\mathbf x_2}
p(\boldsymbol{\theta} \mid \mathbf x_1, \mathbf x_2)
\xrightarrow{\mathbf x_3}
\cdots
\xrightarrow{\mathbf x_n}
p(\boldsymbol{\theta} \mid \mathbf x_1, \dots, \mathbf x_n)
$$

Motivation: Suppose we collected the data in a sequential manner instead of all at once. Do we have to wait until the complete data collection to perform Inference? The answer is no as we can perform recursive Bayesian inference.

> **RECURSIVE BAYESIAN INFERENCE**  
> init $p(\boldsymbol{\theta})$ before observing any data  
> For $t=1, \dots, n$, do:  
> $\quad$ assume we have $p(\boldsymbol{\theta} \mid \mathbf x_{1:t-1})$  
> $\quad$ **predict**: compute $p(\mathbf x_t \mid \mathbf x_{1:t-1})$ before observing $\mathbf x_t$  
> $\quad$ **update**: compute $p(\boldsymbol{\theta} \mid \mathbf x_{1:t})$ after observing $\mathbf x_t$  

Remarks:

* Here, we use the short-hand notation $\mathbf x_{1:t} \triangleq \{ \mathbf x_1, \dots, \mathbf x_t \}$.
* For $t=1$, we let $\mathbf x_{1:0} \triangleq \varnothing$ and $p(\boldsymbol{\theta} \mid \mathbf x_{1:0}) = p(\boldsymbol{\theta} \mid \varnothing) = p(\boldsymbol{\theta})$.
* $p(\boldsymbol{\theta} \mid \mathbf x_{1:t-1})$ is the posterior at time $t-1$. We use it as the prior at time $t$. (Strictly speaking, the term *prior* is reserved for the distribution before seeing any data, i.e., $p(\boldsymbol{\theta})$. However, in recursive inference, we often use the previous posterior as the new "prior")

Next, we give more details about the prediction step and update step.

### Prediction Step

At time $t$, the **prediction** step computes the posterior **predictive** using past observations $\mathbf x_{1:t-1}$

> $$
> \begin{align}
> p(\mathbf x_{t} \mid \mathbf x_{1:t-1})
> &= \mathbb E_{\boldsymbol{\theta} \sim p(\boldsymbol{\theta} \mid \mathbf x_{1:t-1})}
>    \left[ p(\mathbf x_{t} \mid \boldsymbol{\theta} ) \right]
> \\
> &= \int p(\mathbf x_{t} \mid \boldsymbol{\theta}) \cdot p(\boldsymbol{\theta} \mid \mathbf x_{1:t-1}) \:\mathrm{d} \boldsymbol{\theta}
> \end{align}
> $$

Remarks:

* Here, we use $\mathbf x_{t}$ instead of $\mathbf x_*$ to denote the yet unseen data at time $t$. The posterior predictive gives the forecast of $\mathbf x_{t}$ before we observe it.
* Note that $p(\mathbf x_t \mid \mathbf x_{1:t-1}) \ne p(\mathbf x_t)$, due to assumption $\mathbf x_1,\dots,\mathbf x_t \stackrel{\text{iid}}{\sim} p(\mathbf x \mid \boldsymbol{\theta})$, which translates into conditional indepdence of $\mathbf x_1,\dots,\mathbf x_t$ given $\boldsymbol{\theta}$. Conditional independence $\not\Rightarrow$ independence

*Proof*: The predictive distribution follows from the law of total proability and conditional independence.

$$
\begin{align*}
p(\mathbf x_{t} \mid \mathbf x_{1:t-1})
&= \int p(\mathbf x_{t}, \boldsymbol{\theta} \mid \mathbf x_{1:t-1}) \:\mathrm{d} \boldsymbol{\theta}
&& \text{marginalization}
\\
&= \int p(\mathbf x_{t} \mid \boldsymbol{\theta}, \mathbf x_{1:t-1}) \cdot p(\boldsymbol{\theta} \mid \mathbf x_{1:t-1}) \:\mathrm{d} \boldsymbol{\theta}
&& \text{factorization}
\\
&= \int p(\mathbf x_{t} \mid \boldsymbol{\theta}) \cdot p(\boldsymbol{\theta} \mid \mathbf x_{1:t-1}) \:\mathrm{d} \boldsymbol{\theta}
&& \mathbf x_{1:t-1} \perp \mathbf x_t \mid \boldsymbol{\theta}
\\
&= \mathbb E_{\boldsymbol{\theta} \sim p(\boldsymbol{\theta} \mid \mathbf x_{1:t-1})}
   \left[ p(\mathbf x_{t} \mid \boldsymbol{\theta}) \right]
\tag*{$\blacksquare$}
\end{align*}
$$

### Update Step

At time $t$, the **update** step computes (or updates) the **posterior** using  new observation $\mathbf x_t$.

> $$
> \begin{align}
> \overbrace{
>   p(\boldsymbol{\theta} \mid \mathbf x_{1:t})
> }^{\text{post. at } t}
> &= \frac{
>       \overbrace{p(\mathbf x_t \mid \boldsymbol{\theta})}^{\text{lld. of } \mathbf x_t}
>       \cdot
>       \overbrace{p(\boldsymbol{\theta} \mid \mathbf x_{1:t-1})}^{\text{post. at } t-1}
>     }{
>       \underbrace{p(\mathbf x_t \mid \mathbf x_{1:t-1})}_\text{normalization const}
>     }
> \\[6pt]
> &\propto p(\mathbf x_t \mid \boldsymbol{\theta}) \cdot p(\boldsymbol{\theta} \mid  \mathbf x_{1:t-1})
> \end{align}
> $$

Remarks:

* The normalization constant in the posterior is obtained by plugging the actual observation $\mathbf x_t$ into the predictive distribution computed in prediction step.
* Again, $p(\boldsymbol{\theta} \mid \mathbf x_{1:t})$ can be interpreted both as the posterior at time $t$ and as the "prior" at time $t+1$.
* Intuitively, recursive Bayesian inference allows us to carry forward our belief about the parameters, refining them as we observe new data, rather than starting from scratch.

*Proof*: The key is to use the conditional independence $\mathbf x_{1:t-1} \perp \mathbf x_t \mid \boldsymbol{\theta}$.
$$
\begin{align*}
p(\boldsymbol{\theta} \mid \mathbf x_{1:t})
&= \frac{p(\mathbf x_{1:t} \mid \boldsymbol{\theta}) \cdot p(\boldsymbol{\theta})}{p(\mathbf x_{1:t})}
&& \text{Bayes rule for } p(\boldsymbol{\theta} \mid \mathbf x_{1:t})
\\[6pt]
&= \frac{p(\mathbf x_{t} \mid \boldsymbol{\theta}) \cdot p(\mathbf x_{1:t-1} \mid \boldsymbol{\theta}) \cdot p(\boldsymbol{\theta})}{p(\mathbf x_{1:t})}
&& \mathbf x_{1:t-1} \perp \mathbf x_t \mid \boldsymbol{\theta}
\\[6pt]
&= \frac{p(\mathbf x_{t} \mid \boldsymbol{\theta}) \cdot p(\mathbf x_{1:t-1} \mid \boldsymbol{\theta}) \cdot p(\boldsymbol{\theta})}{p(\mathbf x_t \mid \mathbf x_{1:t-1}) \cdot p(\mathbf x_{1:t-1})}
&& \text{Chain rule on denominator}
\\[6pt]
&= \frac{p(\mathbf x_{t} \mid \boldsymbol{\theta}) \cdot p(\boldsymbol{\theta} \mid \mathbf x_{1:t-1})}{p(\mathbf x_t \mid \mathbf x_{1:t-1})}
&& \text{Bayes rule for } p(\boldsymbol{\theta} \mid \mathbf x_{1:t-1})
\tag*{$\blacksquare$}
\end{align*}
$$
*Proof (alt.)*: The conclusion follows from the property of conditional indepdence:
$$
\begin{align*}
X \perp Y \mid Z \implies
p(z \mid x,y) = \frac{p(y \mid z) \cdot p(z \mid x)}{p(y \mid x)}
\end{align*}
$$

by letting
$$
\begin{align*}
X = \mathbf x_{1:t-1}, \quad
Y = \mathbf x_t, \quad
Z = \boldsymbol{\theta}
\tag*{$\blacksquare$}
\end{align*}
$$
In general, the integrals involved in the prediction and update steps are intractable. However, when both the prior and likelihood are Gaussian, the posterior remains Gaussian, leading to closed-form solutions. e.g. in the Kalman filter.

### Connection to Filtering

So far, we have assumed that the parameter vector $\boldsymbol{\theta}$ is time-invariant, meaning the observations $\mathbf x_{1:n}$ are conditionally independent given $\boldsymbol{\theta}$. A ***Bayesian filtering*** problem generalizes this by allowing the parameter vector (or state) to vary over time. Specifically, we assume

* The state is time-dependent, denoted by $\boldsymbol{\theta}_t$.
* The state evolves according to a Markov process $\boldsymbol{\theta}_t \sim p(\boldsymbol{\theta}_t \mid \boldsymbol{\theta}_{t-1})$
* The observation only depends on current state $\mathbf x_t \sim p(\mathbf x_t \mid \boldsymbol{\theta}_{t})$

Together, the sequence $\{ \boldsymbol{\theta}_{t}, \mathbf x_t \}_{t=1:n}$ forms a hidden Markov model.

> **Note**: In standard Bayesian filtering literature, the state vector is typically denoted by $\mathbf x_t$, and the observation is deonted by $\mathbf y_t$. We're keeping the notation consistent with the rest of our notes for clarity.

Kalman filter is a special case of Bayesian filter where

* Both the state evolution and observation model are described by stochastic linear dynamics.
* All random variables (state, observation and noise) are Gaussian.

### Recursive Inference for Multivariate Gaussian

Recall the example of Multivariate Guassian with unknown mean $\mathcal N(\boldsymbol\mu, \boldsymbol\Sigma)$. Previously, we derived the Bayesian inference based on the data set $D=\{\mathbf x_1, \dots, \mathbf x_n\}$. Now, suppose we collected the data sequentially instead of all at once. Appying recursive Bayesian inference results in closed-from solution due to Gaussian-ness.

Assume we have a Gaussian prior on $\boldsymbol\mu$ before seeing any data.

$$
\begin{align}
\boldsymbol\mu \sim \mathcal N(\boldsymbol\mu_0, \boldsymbol\Sigma_0)
\end{align}
$$

At any $t\ge 1$, we use the posterior from $t-1$ as current prior.
$$
\begin{align}
\boldsymbol\mu \mid \mathbf{x}_{1:t-1} \sim \mathcal N(\boldsymbol\mu_{t-1}, \boldsymbol\Sigma_{t-1})
\end{align}
$$
After observing $\mathbf{x}_t$, we update the posterior to
$$
\begin{align}
\boldsymbol\Sigma_t^{-1}
&= \boldsymbol\Sigma^{-1}_{t-1} + \boldsymbol\Sigma^{-1}
\\
\boldsymbol\mu_t
&= \boldsymbol\Sigma_t \left(\boldsymbol\Sigma^{-1}_{t-1} \boldsymbol\mu_{t-1} + \boldsymbol\Sigma^{-1} \mathbf x_t \right)
\\
&= \boldsymbol\mu_{t-1} +
   \boldsymbol\Sigma_t \boldsymbol\Sigma^{-1} \left(\mathbf x_t - \boldsymbol\mu_{t-1} \right)
\end{align}
$$
*Proof*: This is just a special case of the offline Bayesian inference
$$
\begin{align*}
\boldsymbol\Sigma_n^{-1}
&= n\boldsymbol\Sigma^{-1} + \boldsymbol\Sigma^{-1}_0
\\
\boldsymbol\mu_n
&= \boldsymbol\Sigma_n  \left(n\boldsymbol\Sigma^{-1}\bar{\mathbf x} + \boldsymbol\Sigma^{-1}_0 \boldsymbol\mu_0 \right)
\end{align*}
$$
by assuming
$$
\begin{align*}
&\text{prior: }
\cancel{\boldsymbol\mu_0} \to \boldsymbol\mu_{t-1}, \quad
\cancel{\boldsymbol\Sigma_0} \to \boldsymbol\Sigma_{t-1}
\\
&\text{data: }
D=\{\mathbf x_t\} \implies n=1, \quad \bar{\mathbf x} = \mathbf x_t
\\
&\text{posterior: }
\cancel{\boldsymbol\mu_n} \to \boldsymbol\mu_t, \quad
\cancel{\boldsymbol\Sigma_n} \to \boldsymbol\Sigma_t
\tag*{$\blacksquare$}
\end{align*}
$$
1D special case:
$$
\begin{align}
\sigma_t^{-2}
&= \sigma^{-2}_{t-1} + \sigma^{-2}
\\
\mu_t
&= \mu_{t-1} + \frac{\sigma^{2}_{t-1}}{\sigma^{2}_{t-1} + \sigma^{2}} \left( x_t - \mu_{t-1} \right)
\end{align}
$$

## Appendix

### Conditional Probability

Let $X,Y,Z$ be random variables. Then,

$$
\begin{align}
p(z \mid x,y) = \frac{p(z \mid x) \cdot p(y \mid z,x)}{p(y \mid x)}
\end{align}
$$

If $X$ and $Y$ are independent, then

$$
\begin{align}
p(z \mid x,y) = \frac{p(z \mid x) \cdot p(y \mid z,x)}{p(y)}
\end{align}
$$

If $X$ and $Y$ are conditionally independent given $Z$ , then

$$
\begin{align}
p(z \mid x,y) = \frac{p(z \mid x) \cdot p(y \mid z)}{p(y \mid x)}
\end{align}
$$

If $X$ and $Y$ are both independent and conditionally independent given $Z$ , then

$$
\begin{align}
p(z \mid x,y) = \frac{p(z \mid x) \cdot p(y \mid z)}{p(y)}
\end{align}
$$

### Convolution Rule of Gaussian

$$
\begin{align*}
\int \mathcal N(\mathbf x_* ; \boldsymbol\mu, \boldsymbol\Sigma) \cdot
     \mathcal N(\boldsymbol\mu ; \boldsymbol\mu_n, \boldsymbol\Sigma_n) \:\mathrm{d}\boldsymbol\mu
=
\mathcal N(\mathbf x_* ; \boldsymbol\mu_n, \boldsymbol\Sigma + \boldsymbol\Sigma_n)
\end{align*}
$$