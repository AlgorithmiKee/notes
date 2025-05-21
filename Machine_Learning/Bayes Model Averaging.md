---
title: "Bayesian Inference"
author: "Ke Zhang"
date: "2025"
fontsize: 12pt
---

# Intro to Bayesian Inference

[toc]

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

*Proof*: This follows from the law of total expectation and the independence between $\mathbf x_*$ and $D$.

$$
\begin{align*}
p(\mathbf x_* \mid D)
&= \int p(\mathbf x_*, \boldsymbol{\theta} \mid D) \:\mathrm{d}\boldsymbol{\theta}
\\
&= \int p(\mathbf x_* \mid \boldsymbol{\theta}, D) \cdot p(\boldsymbol{\theta} \mid D) \:\mathrm{d}\boldsymbol{\theta}
\\
&= \int p(\mathbf x_* \mid \boldsymbol{\theta}) \cdot p(\boldsymbol{\theta} \mid D) \:\mathrm{d}\boldsymbol{\theta}
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

In general, the posterior and the posterior predictive requires numerical approximation since they have no closed-form solution. Here, we show two exceptions.

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

Consider a multivariate Gaussian with known covariance matirx $\boldsymbol\Sigma$ but unknown mean vector $\boldsymbol\mu$.

$$
\begin{align}
\mathbf x \sim \mathcal N(\boldsymbol\mu, \boldsymbol\Sigma)
\end{align}
$$

Suppose we have iid observations $D=\{\mathbf x_1, \dots, \mathbf x_n\}$. Then, the likelihood is

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

Identify the posterior mean $\boldsymbol\mu_n$ and posterior covariance matrix $\boldsymbol\Sigma_n$.

$$
\begin{align}
\boldsymbol\Sigma_n^{-1}
&\triangleq \left( n\boldsymbol\Sigma^{-1} + \boldsymbol\Sigma^{-1}_0 \right)
\\
\boldsymbol\mu_n
&\triangleq \boldsymbol\Sigma_n  \left(n\boldsymbol\Sigma^{-1}\bar{\mathbf x} + \boldsymbol\Sigma^{-1}_0 \boldsymbol\mu_0 \right)
\end{align}
$$

The posterior is hence Gaussian

$$
\begin{align}
\boldsymbol\mu \mid D \sim \mathcal N(\boldsymbol\mu_n, \boldsymbol\Sigma_n)
\end{align}
$$

The posterior predictive distribution is

$$
\begin{align*}
p(\mathbf x_* \mid D)
&= \int p(\mathbf x_* \mid \boldsymbol\mu) \cdot p(\boldsymbol\mu \mid D) \:\mathrm{d}\boldsymbol\mu
\\
&= \int \mathcal N(\mathbf x_* ; \boldsymbol\mu, \boldsymbol\Sigma) \cdot \mathcal N(\boldsymbol\mu ; \boldsymbol\mu_n, \boldsymbol\Sigma_n) \:\mathrm{d}\boldsymbol\mu
\\
&= \int \mathcal N(\mathbf x_* - \boldsymbol\mu ; \mathbf 0, \boldsymbol\Sigma) \cdot \mathcal N(\boldsymbol\mu ; \boldsymbol\mu_n, \boldsymbol\Sigma_n) \:\mathrm{d}\boldsymbol\mu
\end{align*}
$$

By the convolution rule of multivariate Gaussian (see Appendix), the posterior predictive is also Gaussian

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

* Comparing the plug-in predictive with the posterior predicitve, we see that the former discards uncertainty of the posterior $p(\boldsymbol\mu \mid D)$.

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
&= \left( \frac{n}{\sigma^{2}} + \frac{1}{\sigma^{2}_0} \right)
\\
\mu_n
&= \sigma_n^2 \left( \frac{n}{\sigma^{2}}\bar{x} + \frac{1}{\sigma^{2}_0}\mu_0 \right)
\end{align*}
%%%%%%%%%
\implies
%%%%%%%%%
\begin{align*}
\sigma_n^2
&= \frac{\sigma^{2} \sigma^{2}_0}{n\sigma^{2}_0 + \sigma^{2}}
\\
\mu_n
&= \mu_0 + \frac{n\sigma^{2}_0}{n\sigma^{2}_0 + \sigma^{2}} \left( \bar{x} - \mu_0 \right)
\end{align*}
$$

The resulting posterior predictive is

$$
\begin{align}
x_* \mid D \sim \mathcal N(\mu_n, \sigma^2 + \sigma_n^2)
\end{align}
$$

## Iterative Bayesian Inference

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

Motivation: Suppose we collected the data in a sequential manner instead of all at once. Do we have to wait until the complete data collection to perform Inference?

Let $p(\boldsymbol{\theta})$ be the prior before we observe anything.  After we see $\mathbf x_1$, we update the prior $p(\boldsymbol{\theta})$ to the posterior

$$
\begin{align}
p(\boldsymbol{\theta} \mid \mathbf x_1)
&= \frac{p(\mathbf x_1 \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})}{p(\mathbf x_1)} \\
&\propto p(\mathbf x_1 \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})
\end{align}
$$

Before we observe $\mathbf x_2$, $p(\boldsymbol{\theta} \mid \mathbf x_1)$ becomes our new prior belief on $\boldsymbol{\theta}$. Namely, there are two ways to interpret $p(\boldsymbol{\theta} \mid \mathbf x_1)$:

* posterior of $\boldsymbol{\theta}$ after we observe $\mathbf x_1$
* prior of $\boldsymbol{\theta}$ before we observe $\mathbf x_2$

After we observe $\mathbf x_2$, we update our belief on $\boldsymbol{\theta}$ again to incooperate the information brought by $\mathbf x_2$.

$$
\begin{align}
\overbrace{
  p(\boldsymbol{\theta} \mid \mathbf x_1, \mathbf x_2)
}^{\text{post at } t=2}
&= \frac{
      \overbrace{p(\mathbf x_2 \mid \boldsymbol{\theta})}^{\text{lld at } t=2}
      \cdot
      \overbrace{p(\boldsymbol{\theta} \mid \mathbf x_1)}^{\text{prio at } t=2}
    }{
      \underbrace{p(\mathbf x_2 \mid \mathbf x_1)}_\text{normalization const}
    }
&& \text{see Appendix}
\\
&\propto p(\mathbf x_2 \mid \boldsymbol{\theta}) \cdot p(\boldsymbol{\theta} \mid \mathbf x_1)
\end{align}
$$

This procedure can continue until we collected the complete data set or until infinity when there is no inherent ending.

Now, we will derive the general framework for iterative Bayesian inference. To make the notation cleaner, we let $D_t$ denote the data collected until (inclusive) time $t$.

$$
\begin{align}
D_t \triangleq \{ \mathbf x_1, \dots, \mathbf x_t \}
\end{align}
$$

At time $t=0$, there is no observations available. We simply write

$$
\begin{align}
D_0 \triangleq \varnothing
\end{align}
$$

First, we initialize $p(\boldsymbol{\theta} \mid D_0) = p(\boldsymbol{\theta})$, i.e. the prior on $\boldsymbol{\theta}$ before we observe any data. For $t=1, \dots, n$:

The **posterior** at time $t$ is

> $$
> \begin{align}
> \overbrace{
>   p(\boldsymbol{\theta} \mid D_{t})
> }^{\text{post. at } t}
> &= \frac{
>       \overbrace{p(\mathbf x_t \mid \boldsymbol{\theta})}^{\text{lld. of } \mathbf x_t}
>       \cdot
>       \overbrace{p(\boldsymbol{\theta} \mid D_{t-1})}^{\text{post. at } t-1}
>     }{
>       \underbrace{p(\mathbf x_t \mid D_{t-1})}_\text{normalization const}
>     }
> \\[24pt]
> &\propto p(\mathbf x_t \mid \boldsymbol{\theta}) \cdot p(\boldsymbol{\theta} \mid  D_{t-1})
> \end{align}
> $$

Remarks:

* The normalization constant in the posterior is $p(\mathbf x_t \mid D_{t-1}) \ne p(\mathbf x_t)$, even though we assume $\mathbf x_1,\dots,\mathbf x_t \stackrel{\text{iid}}{\sim} p(\mathbf x \mid \boldsymbol{\theta})$, which translates into **conditional indepdence** of $\mathbf x_1,\dots,\mathbf x_t$ given $\boldsymbol{\theta}$.
* At time $t$, we treat $p(\boldsymbol{\theta} \mid D_{t-1})$ as the "prior" before observing $\mathbf x_{t}$. Although, strictly speaking, the term *prior* in reserved for $p(\boldsymbol{\theta})$, i.e. the distribution of $\boldsymbol{\theta}$ before seeing any data.
* Again, $p(\boldsymbol{\theta} \mid D_{t})$ can interpreted both as the posterior at time $t$ and as the "prior" at time $t+1$.

*Proof*: By definition of $D_t$, we reformulate the posterior into

$$
p(\boldsymbol{\theta} \mid D_{t}) = p(\boldsymbol{\theta} \mid D_{t-1}, \mathbf x_t)
$$

By assumption $\mathbf x_1,\dots,\mathbf x_t \stackrel{\text{iid}}{\sim} p(\mathbf x \mid \boldsymbol{\theta})$, we have the conditional independence

$$
D_{t-1} \perp \mathbf x_t \mid \boldsymbol{\theta}
$$

Therefore, the conclusion follows from the propoerty of conditional indepdence:

$$
\begin{align*}
X \perp Y \mid Z \implies
p(z \mid x,y) = \frac{p(z \mid x) \cdot p(y \mid z)}{p(y \mid x)}
\tag*{$\blacksquare$}
\end{align*}
$$

The **posterior predictive** at time $t$ is

> $$
> \begin{align}
> p(\mathbf x_{t+1} \mid D_t)
> &= \mathbb E_{\boldsymbol{\theta} \sim p(\boldsymbol{\theta} \mid D_{t})}
>    \left[ p(\mathbf x_{t+1} \mid \boldsymbol{\theta} ) \right]
> \\
> &= \int p(\mathbf x_{t+1} \mid \boldsymbol{\theta}) \cdot p(\boldsymbol{\theta} \mid D_{t}) \:\mathrm{d} \boldsymbol{\theta}
> \end{align}
> $$


Remarks:

* Here, we use $\mathbf x_{t+1}$ instead of $\mathbf x_*$ to denote unseen data. The posterior predicitve gives the distribution of $\mathbf x_{t+1}$ before we observe it. The posterior predictive allows us to forecast $\mathbf x_{t+1}$.
* In practice, the posterior predicitve has no closed-from solution. One exception is when every random variables are Gaussian.

# Bayesian Model Averaging

## Bayesian Linear Regression

**Preliminary**: fixed-desgin linear regression, ordinary least square

In this article, we illustrate *Bayes model averaging* (BMA) for
random-design linear regression problem:

* Given: training dataset $D=\{(\mathbf{x}_i, y_i)\}_{i=1}^n \stackrel{\text{iid}}{\sim} p(\mathbf x, y)$ where $(\mathbf{x}_i, y_i) \in \mathbb R^d \times \mathbb R$.
* Statistical model: $y_i = \mathbf{w}^\top \mathbf{x}_i + \varepsilon_i, \quad \varepsilon_i \stackrel{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2_\text{n})$
* Additional assumption: $\varepsilon_i$ and $\mathbf{x}_i$ are statistically independent
* Goal: Predict the label for a new data point $\mathbf{x}_*$ using the **full** posterior distribution $p(\mathbf w \mid D)$

### Main Results about Point Estimates

* Discriminative model:

$$
\begin{align}
p(y_i \mid \mathbf{x}_i, \mathbf{w})
&= \mathcal{N}(y_i \mid \mathbf{w}^\top \mathbf{x}_i, \sigma^2_\text{n}) \\
&= \frac{1}{\sqrt{2\pi \sigma^2_\text{n}}} \exp\left(-\frac{(y_i - \mathbf{w}^\top \mathbf{x}_i)^2}{2\sigma^2_\text{n}}\right) \nonumber \\
\end{align}
$$

* Log likelihood:

$$
\begin{align}
\ln p(D \mid \mathbf w)
&= \ln \prod_{i=1}^n p(\mathbf{x}_i, y_i \mid\mathbf{w}) + \text{const} \\
&= -\frac{1}{2\sigma^2_\text{n}} (y_i - \mathbf{w}^\top \mathbf{x}_i)^2 + \text{const} 
\end{align}
$$

* MLE $\iff$ least square

$$
\begin{align}
\hat{\mathbf w}_\text{MLE}
&= \argmin_{\mathbf w} \sum_{i=1}^n (y_i - \mathbf{w}^\top \mathbf{x}_i)^2
\end{align}
$$

* Log posterior with Gaussian prior

$$
\begin{align}
\ln p(\mathbf w \mid D)
&= \ln \prod_{i=1}^n p(\mathbf{x}_i, y_i \mid\mathbf{w}) + \ln p(\mathbf{w}) + \text{const} \\
&= -\frac{1}{2\sigma^2_\text{n}} (y_i - \mathbf{w}^\top \mathbf{x}_i)^2 -
    \frac{1}{2\sigma^2_\text{p}} \Vert\boldsymbol{\theta}\Vert^2 +
    \text{const} 
\end{align}
$$

* MAP with Gaussian prior $\iff$ ridge regression

$$
\begin{align}
\hat{\mathbf w}_\text{MAP}
&= \argmin_{\mathbf w} \sum_{i=1}^n (y_i - \mathbf{w}^\top \mathbf{x}_i)^2 +
   \frac{\sigma^2_\text{n}}{\sigma^2_\text{p}} \Vert\boldsymbol{\theta}\Vert^2
\end{align}
$$

## Model Averaging

Both MLE and MAP results in a single estimate of $\mathbf w$, which is used to predict $y_*$ given a test data point $\mathbf x_*$.

Bayes view: For a test data point $\mathbf x_*$, all we are interested in is

$$
\begin{align}
p(y_* \mid \mathbf x_*, D)
&= \int p(y_*, \mathbf w \mid \mathbf x_*, D) \: \mathrm d \mathbf w \\
&= \int p(y_* \mid  \mathbf w, \mathbf x_*, D) p(\mathbf w \mid D) \: \mathrm d \mathbf w
\end{align}
$$

Remark:

* The integral on the RHS is a model averaging. It averages $\mathbf w^\top \mathbf x_*$ for every possible $\mathbf w$.
* In general, this integral aka $p(y_* \mid \mathbf x_*, D)$ has no closed-form solution. However, if everything is Gaussian, we can indeed solve this integral since we only need to solve the mean and variance of $p(y_* \mid \mathbf x_*, D)$.

Key idea: Consider the joint distribution
$$
\begin{align}
p(y_1, \dots, y_n, y_*) \sim \mathcal (\mathbf{0}, \boldsymbol\Sigma)
\end{align}
$$

where $\mathbf{0}\in\mathbb R^{n+1}$ and $\boldsymbol\Sigma\in\mathbb R^{(n+1)\times(n+1)}$.
W.l.o.g. we may assume zero mean.

The covariance matrix $\boldsymbol\Sigma$ has the form

$$
\begin{align}
\boldsymbol\Sigma =
\begin{bmatrix}
\mathbf{K}_\text{train, train} & \mathbf{K}_\text{train, test} \\
\mathbf{K}_\text{train, test}^\top  & \mathbf{K}_\text{test, test}
\end{bmatrix}
\end{align}
$$

where we use kernel matrices as convariance matices.

# Appendix

## Conditional Probability

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

## Multivariant Gaussian

Let $X: \Omega \to \mathbb R^{n+m}$ be a Gaussian random vector.

$$
X \sim \mathcal{N}\left( \boldsymbol\mu, \boldsymbol\Sigma \right)
$$

where

$$
\boldsymbol\mu\in\mathbb{R}^{n+m}, \quad \boldsymbol\Sigma\in\mathbb{R}^{(n+m)\times(n+m)}
$$

We partition $X$ into two subvectors $X_A$ and $X_B$

* $X_A = [X_1, \dots, X_n]^\top$
* $X_B = [X_{n+1}, \dots, X_{n+m}]^\top$

The overall mean and covariance can be partitioned into

$$
\begin{bmatrix}  X_A \\  X_B \end{bmatrix}
\sim\mathcal{N}
\left(
  \begin{bmatrix} \boldsymbol\mu_A \\ \boldsymbol\mu_B \end{bmatrix},
  \begin{bmatrix}
    \boldsymbol\Sigma_{AA} & \boldsymbol\Sigma_{AB} \\
    \boldsymbol\Sigma_{BA} & \boldsymbol\Sigma_{BB}
  \end{bmatrix}
\right)
$$

where

$$
\begin{matrix}
  \boldsymbol\mu_A\in\mathbb{R}^n \\  \boldsymbol\mu_B\in\mathbb{R}^m
\end{matrix}
\quad\text{ and }\quad
\begin{matrix}
  \boldsymbol\Sigma_{AA}\in\mathbb{R}^{n \times n} & \boldsymbol\Sigma_{AB}\in\mathbb{R}^{n \times m} \\
  \boldsymbol\Sigma_{BA}\in\mathbb{R}^{m \times n} & \boldsymbol\Sigma_{BB}\in\mathbb{R}^{m \times m}
\end{matrix}
$$

### Marginalized Gaussian

One can show that the marginal distributions

$$
\begin{align}
p(\mathbf x_A) = \int p(\mathbf x_A, \mathbf x_B ; \boldsymbol\mu, \boldsymbol\Sigma) \:\mathrm{d}\mathbf x_B \\
p(\mathbf x_B) = \int p(\mathbf x_A, \mathbf x_B ; \boldsymbol\mu, \boldsymbol\Sigma) \:\mathrm{d}\mathbf x_A \\
\end{align}
$$

are also Gaussian

$$
\begin{align}
X_A &\sim \mathcal{N}\left( \boldsymbol\mu_A, \boldsymbol\Sigma_{AA}\right) \\
X_B &\sim \mathcal{N}\left( \boldsymbol\mu_B, \boldsymbol\Sigma_{BB}\right)
\end{align}
$$

### Gaussian Conditioning

Conditioned on $X_B = \mathbf x_B$, the distribution of $X_A$
$$
\begin{align}
p(\mathbf x_A \mid \mathbf x_B) 
&= \frac{p(\mathbf x_A, \mathbf x_B ; \boldsymbol\mu, \boldsymbol\Sigma)}{p(\mathbf x_B ; \boldsymbol\mu, \boldsymbol\Sigma)}  \\
&= \frac{p(\mathbf x_A, \mathbf x_B ; \boldsymbol\mu, \boldsymbol\Sigma)}{\int p(\mathbf x_A, \mathbf x_B ; \boldsymbol\mu, \boldsymbol\Sigma) \:\mathrm{d}\mathbf x_A}
\end{align}
$$

is also Gaussian
$$
\begin{align}
X_A \,\mid\, \mathbf x_B &\sim \mathcal{N}(\boldsymbol\mu_{A \mid B}, \boldsymbol\Sigma_{A \mid B}) \\
\boldsymbol\mu_{A \mid B} &= \boldsymbol\mu_{A} + \boldsymbol\Sigma_{AB} \boldsymbol\Sigma_{BB}^{-1} (\mathbf x_B - \boldsymbol\mu_B)\\
\boldsymbol\Sigma_{A \mid B} &= \boldsymbol\Sigma_{AA} - \boldsymbol\Sigma_{AB}\boldsymbol\Sigma_{BB}^{-1} \boldsymbol\Sigma_{BA}
\end{align}
$$

### Convolution Rule

$$
\begin{align*}
\int \mathcal N(\mathbf x_* ; \boldsymbol\mu, \boldsymbol\Sigma) \cdot
     \mathcal N(\boldsymbol\mu ; \boldsymbol\mu_n, \boldsymbol\Sigma_n) \:\mathrm{d}\boldsymbol\mu
=
\mathcal N(\mathbf x_* ; \boldsymbol\mu_n, \boldsymbol\Sigma + \boldsymbol\Sigma_n)
\end{align*}
$$