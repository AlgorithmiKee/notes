---
title: "Probabilistic Modeling for ML"
author: "Ke Zhang"
date: "2025"
fontsize: 12pt
---

# Probabilistic Modeling for ML

[toc]

$$
\DeclareMathOperator*{\argmax}{arg\max}
\DeclareMathOperator*{\argmin}{arg\min}
$$

Notations:

* $\mathbf x\in\mathcal X$: feature vector. $\mathcal X$ could be $\mathbb R^d$ (for continuous data) or a discrete set (for discrete data)
* $y\in\mathcal Y$: label. $\mathcal Y$ could be $\mathbb R$ (for regression) or a discrete set (for classification)

## Overview

Both supervised learning and unsupervised learning can be summarized in the figure below.

<img src="./figs/overview_stat_learning.pdf" alt="overview_stat_learning" style="zoom:67%;" />

Remarks:

* training data $D$: iid from unknown ground truth distribution $p^\star$.
* ML method: compute an estimate $\hat{p}$ from $D$ to estimate $p^\star$ as closely as possible.
* The learned distribution $\hat{p}$ is used on new data for predictions, anormaly detection, etc.

Which probability do we want to learn?

* In unsupervised learning: $p(\mathbf x)$
* In supervised learning: $p(y\mid\mathbf x)$ or $p(\mathbf x, y)$

How to learn the probability?

* Non-parametric methods: kernel density estimation (not detailed here)

* Parametric methods: We restrict to some parametric family $\{p_\theta(\cdot)\}$ (e.g. Gaussian).

  * Point Estimation: Estimate (MLE or MAP) parameters by computing a single best-fit value $\hat{\boldsymbol{\theta}}$.
  * Bayesian Inference (or model averaging):  Instead of a point estimate, compute the full posterior distribution $p(\boldsymbol{\theta} \mid D)$. $\to$ See notes *Bayesian Inference*

The true distribution $p^\star$ may not lie within our assumed family $\{p_\theta(\cdot)\}$ — in which case, the model is said to be **misspecified**.

Examples:

* In unsupervised learning, the ground truth is a mixure model while we assume a single Gaussain.
* In supervised learning, the ground truth is a nonlinear model while we assume linear model.

Assuming correct model specification, we distinguish two major statistical philosophies:

* Frequentist view: The true data-generating distribution $p^\star$ is governed by a true parameter $\boldsymbol{\theta}^\star$, which is unknown but fixed.
* Bayesian view: The parameter is a random variable and follows a true prior distribution. The true data-generating distribution $p^\star$ is an average of $p_\theta$ w.r.t. that true prior.

Throughout this notes:

* We assume that all models are correctly specified, i.e. $p^\star \in \{p_\theta(\cdot)\}$
* We adopt the frequentist perspective, focusing on point estimation. Although MAP uses a prior, we treat it only as a regularizer, not as part of the Bayesian inference.

## Unsuperivsed Learning

Unspervised learning with fully observable data:

* Model: $p(\mathbf x \mid \boldsymbol{\theta})$ with unknown $\boldsymbol{\theta}$.
* Given: training data $D = \{\mathbf x_1, \cdots, \mathbf x_n\} \stackrel{\text{iid}}{\sim} p(\mathbf x \mid \boldsymbol{\theta})$
* Optional: prior distribution $p(\boldsymbol{\theta})$
* Goal: estimate $\boldsymbol{\theta}$

For now, we assume that the data is fully observable. i.e. There is no latent variables. Latent variable models will be briefly discussed later.

### Maximum Likelihood Estimation (MLE)

The likelihood $p(D\mid\boldsymbol{\theta})$ can be factorized as follows by iid assumption

> $$
> \begin{align}
> p(D\mid\boldsymbol{\theta})
> &= p(\mathbf x_1, \cdots, \mathbf x_n \mid \boldsymbol{\theta}) \\
> &= \prod_{i=1}^n p(\mathbf x_i \mid \boldsymbol{\theta})
> \end{align}
> $$

Intuition of MLE: which $\boldsymbol{\theta}$ makes our data most probable? Hence,

> $$
> \begin{align}
> \hat{\boldsymbol{\theta}}_\text{MLE}
> &= \argmax_{\boldsymbol{\theta}} p(D\mid\boldsymbol{\theta}) \\
> &= \argmax_{\boldsymbol{\theta}} \ln p(D\mid\boldsymbol{\theta}) \\
> \end{align}
> $$

Here, we maximize the log of likelihood since

1. the log function converts the product into a sum which is simpler to optimize.
1. the log function is monotonic

The log likelihood is

$$
\begin{align}
\ln p(D\mid\boldsymbol{\theta})
&= \ln \prod_{i=1}^n p(\mathbf x_i \mid \boldsymbol{\theta}) \\
&= \sum_{i=1}^n \ln p(\mathbf x_i \mid \boldsymbol{\theta}) \\
\end{align}
$$

Therefore, the MLE becomes

> $$
> \begin{align}
> \hat{\boldsymbol{\theta}}_\text{MLE}
> &= \argmax_{\boldsymbol{\theta}} \sum_{i=1}^n \ln p(\mathbf x_i \mid \boldsymbol{\theta})
> \end{align}
> $$

If the log likelihood is convex, then $\hat{\boldsymbol{\theta}}_\text{MLE}$ can be computed analytically by solving

$$
\begin{align}
\nabla_{\boldsymbol{\theta}} \ln p(D\mid\boldsymbol{\theta}) &= 0 \\
\sum_{i=1}^n \nabla_{\boldsymbol{\theta}} \ln p(\mathbf x_i \mid \boldsymbol{\theta}) &= 0 \\
\end{align}
$$

If the log likelihood is non-convex,  $\hat{\boldsymbol{\theta}}_\text{MLE}$ can be computed by gradient ascent

$$
\begin{align}
\boldsymbol{\theta}^{(t+1)}
&= \boldsymbol{\theta}^{(t)} + \eta^{(t)} \nabla_{\boldsymbol{\theta}} \ln p(D\mid\boldsymbol{\theta}^{(t)}) \\
&= \boldsymbol{\theta}^{(t)} + \eta^{(t)} \sum_{i=1}^n \nabla_{\boldsymbol{\theta}} \ln p(\mathbf x_i \mid \boldsymbol{\theta}^{(t)}) \\
\end{align}
$$

where $\eta^{(t)}$ is called ***learning rate*** or step size.

If the data set $D$ is large, one can use

* mini-batch of size $m$ to calculate the gradient
    $$
    \begin{align}
    & \text{randomly draw } B^{(t)} \subset D \text{ with } \vert B \vert = m \nonumber
    \\
    & \boldsymbol{\theta}^{(t+1)}
    = \boldsymbol{\theta}^{(t)} + \eta^{(t)} \sum_{\mathbf x\in B^{(t)}} \nabla_{\boldsymbol{\theta}} \ln p(\mathbf x \mid \boldsymbol{\theta}^{(t)})
    \end{align}
    $$

* or stochastic gradient (speical case of mini-batch when $m=1$)
    $$
    \begin{align}
    & \text{randomly draw } \mathbf x^{(t)} \in D \nonumber
    \\
    & \boldsymbol{\theta}^{(t+1)}
    = \boldsymbol{\theta}^{(t)} + \eta^{(t)} \nabla_{\boldsymbol{\theta}} \ln p(\mathbf x^{(t)} \mid \boldsymbol{\theta}^{(t)})
    \end{align}
    $$

### Maximum a Posterior Estimation (MAP)

The posterior probability of $\boldsymbol{\theta}$ is given by Bayes rule:

> $$
> \begin{align}
> p(\boldsymbol{\theta}\mid D)
> &= \frac{p(D\mid \boldsymbol{\theta}) \cdot p(\boldsymbol{\theta})}{p(D)} \\
> &\propto p(D\mid \boldsymbol{\theta}) \cdot p(\boldsymbol{\theta}) \\
> \end{align}
> $$

Remarks:

* $p(\boldsymbol{\theta})$ is the prior probability of $\boldsymbol{\theta}$. It describes the distribution of $\boldsymbol{\theta}$ before we observe any data.
* $p(D)$ is independent of $\boldsymbol{\theta}$ and thus is simply a normalization constant.
* $p(\boldsymbol{\theta}\mid D)$ is proportional to the joint distribution $p(D,\boldsymbol{\theta})$.

Intuition of MAP estimation: which $\boldsymbol{\theta}$ is most probable given the data? Hence,

> $$
> \begin{align}
> \hat{\boldsymbol{\theta}}_\text{MAP}
> &= \argmax_{\boldsymbol{\theta}} p(\boldsymbol{\theta}\mid D) \\
> &= \argmax_{\boldsymbol{\theta}} p(D\mid\boldsymbol{\theta}) \cdot p(\boldsymbol{\theta}) \\
> \end{align}
> $$

Again, for the sake of computation, we take the log on the RHS:

> $$
> \begin{align}
> \hat{\boldsymbol{\theta}}_\text{MAP}
> &= \argmax_{\boldsymbol{\theta}} \underbrace{\ln p(D\mid\boldsymbol{\theta})}_{\text{log likelihood}} + \underbrace{\ln p(\boldsymbol{\theta})}_{\text{log prior}} \\
> &= \argmax_{\boldsymbol{\theta}} \sum_{i=1}^n \ln p(\mathbf x_i \mid \boldsymbol{\theta}) + \ln p(\boldsymbol{\theta})
> \end{align}
> $$

If the objective on the RHS is convex, we can compute $\hat{\boldsymbol{\theta}}_\text{MAP}$ analytically by solving

$$
\begin{align}
\nabla_{\boldsymbol{\theta}}\ln p(D\mid\boldsymbol{\theta}) + \nabla_{\boldsymbol{\theta}}\ln p(\boldsymbol{\theta}) &= 0 \\
\sum_{i=1}^n \nabla_{\boldsymbol{\theta}} \ln p(\mathbf x_i \mid \boldsymbol{\theta}) + \nabla_{\boldsymbol{\theta}}\ln p(\boldsymbol{\theta}) &= 0 \\
\end{align}
$$

Gradient-based optimization:

* Gradient ascent:
    $$
    \begin{align*}
    \boldsymbol{\theta}^{(t+1)}
    &= \boldsymbol{\theta}^{(t)} + \eta^{(t)} \left(\sum_{i=1}^n \nabla_{\boldsymbol{\theta}} \ln p(\mathbf x_i \mid \boldsymbol{\theta}^{(t)}) + \nabla_{\boldsymbol{\theta}}\ln p(\boldsymbol{\theta}^{(t)}) \right)\\
    \end{align*}
    $$

* Mini-bath gradient ascent:
    $$
    \begin{align*}
    & \text{randomly draw } B^{(t)} \subset D \text{ with } \vert B \vert = m
    \\
    & \boldsymbol{\theta}^{(t+1)}
    = \boldsymbol{\theta}^{(t)} + \eta^{(t)} \left(\sum_{\mathbf x\in B^{(t)}} \nabla_{\boldsymbol{\theta}} \ln p(\mathbf x \mid \boldsymbol{\theta}^{(t)}) + \nabla_{\boldsymbol{\theta}}\ln p(\boldsymbol{\theta}^{(t)}) \right)
    \end{align*}
    $$

* Stochastic gradient ascent:
    $$
    \begin{align*}
    & \text{randomly draw } \mathbf x^{(t)} \in D
    \\
    & \boldsymbol{\theta}^{(t+1)}
    = \boldsymbol{\theta}^{(t)} + \eta^{(t)} \left(\nabla_{\boldsymbol{\theta}} \ln p(\mathbf x^{(t)} \mid \boldsymbol{\theta}^{(t)}) + \nabla_{\boldsymbol{\theta}}\ln p(\boldsymbol{\theta}^{(t)}) \right)
    \end{align*}
    $$

In practice, we often impose a Gaussian prior on $\boldsymbol{\theta}$,

$$
\begin{align}
\boldsymbol{\theta} \sim \mathcal N(\mathbf 0, \sigma^2_\mathrm{p}\mathbf I)
\end{align}
$$

which gives the log prior

$$
\begin{align*}
\ln p(\boldsymbol{\theta}) = -\frac{1}{2\sigma^2_\mathrm{p}} \Vert\boldsymbol{\theta}\Vert^2 + \text{const}
\end{align*}
$$

The resulting MAP estimation is then

> $$
> \begin{align}
> \hat{\boldsymbol{\theta}}_\text{MAP}
> &= \argmax_{\boldsymbol{\theta}} \sum_{i=1}^n \ln p(\mathbf x_i \mid \boldsymbol{\theta}) -\frac{1}{2\sigma^2_\mathrm{p}} \Vert\boldsymbol{\theta}\Vert^2
> \end{align}
> $$

The Gaussian prior has regularization effect: It encourages smaller $\boldsymbol{\theta}$ while penalize larger $\boldsymbol{\theta}$ since we maximize the RHS over $\boldsymbol{\theta}$. The smaller $\sigma^2_\mathrm{p}$ is, the stronger is the regularization. This effect is clearer if we look at the gradient ascent

$$
\begin{align}
\boldsymbol{\theta}^{(t+1)}
&= \boldsymbol{\theta}^{(t)} + \eta^{(t)} \left(\sum_{i=1}^n \nabla_{\boldsymbol{\theta}} \ln p(\mathbf x_i \mid \boldsymbol{\theta}^{(t)}) -
   \nabla_{\boldsymbol{\theta}}\frac{1}{2\sigma^2_\mathrm{p}} \Vert\boldsymbol{\theta}^{(t)}\Vert^2 \right) \nonumber \\
&= \boldsymbol{\theta}^{(t)} + \eta^{(t)} \left(\sum_{i=1}^n \nabla_{\boldsymbol{\theta}} \ln p(\mathbf x_i \mid \boldsymbol{\theta}^{(t)}) -
   \frac{1}{\sigma^2_\mathrm{p}} \boldsymbol{\theta}^{(t)} \right) \nonumber \\
&= \boldsymbol{\theta}^{(t)} \left(1 - \frac{\eta^{(t)}}{\sigma^2_\mathrm{p}}\right) + \eta^{(t)} \sum_{i=1}^n \nabla_{\boldsymbol{\theta}} \ln p(\mathbf x_i \mid \boldsymbol{\theta}^{(t)}) \\
\end{align}
$$

### Latent Variable Model

Unspervised learning with latent variables:

* Model: $p(\mathbf x, \mathbf z \mid \boldsymbol{\theta})$ with unknown $\boldsymbol{\theta}$
* Given: training data $D = \{\mathbf x_1, \cdots, \mathbf x_n\} \stackrel{\text{iid}}{\sim} p(\mathbf x \mid \boldsymbol{\theta}) = \displaystyle\int p(\mathbf x, \mathbf z \mid \boldsymbol{\theta}) \,\mathrm d\mathbf z$. The latent variables $\mathbf z_{1:n}$ are missing.
* Optional: prior distribution $p(\boldsymbol{\theta})$

Two predominant goals:

1. For a fixed $\boldsymbol{\theta}$, compute $p(\mathbf z_{1:n} \mid \mathbf x_{1:n}, \boldsymbol{\theta})$. $\to$ variational inference
1. Estimate the parameters $\boldsymbol{\theta}$. $\to$ expectation maximization (EM) algorithm

Remarks:

* Here, we focus on statistical modeling. Variational inference and EM algorithm are big topics and thus not detailed here.
* The latent variables $\mathbf z$ represents some high level description of $\mathbf x$. e.g. In computer vision, $\mathbf z$ may encode the semantic meaning of a "shining doge", and $\mathbf x$ is a 256 by 256 image of dogecoin.
* Learning a latent variable model can be seen as learning a supervised model with missing labels in training data.

The (incomplete-data) likelihood is

$$
\begin{align}
p(D \mid \boldsymbol{\theta})
&= p(\mathbf x_{1:n} \mid \theta) \\
&= \prod_{i=1}^n  p(\mathbf x_{i} \mid \theta) \\
&= \prod_{i=1}^n  \int p(\mathbf x_{i}, \mathbf z_{i} \mid \theta) \,\mathrm d\mathbf z_{i} \\
\end{align}
$$

Taking the log, we obtain the (incomplete-data) log likelihood

$$
\begin{align}
\ln p(D \mid \boldsymbol{\theta})
&= \sum_{i=1}^n  \ln \left( \int p(\mathbf x_{i}, \mathbf z_{i} \mid \theta) \,\mathrm d\mathbf z_{i} \right) \\
\end{align}
$$

The gradient of $\ln p(D \mid \boldsymbol{\theta})$ has very complex form in general.

$$
\begin{align*}
\nabla_{\boldsymbol{\theta}} \ln p(D \mid \boldsymbol{\theta})
&= \sum_{i=1}^n  \nabla_{\boldsymbol{\theta}} \ln p(\mathbf x_{i} \mid \theta) \\
&= \sum_{i=1}^n  \frac{1}{p(\mathbf x_{i} \mid \theta)} \nabla_{\boldsymbol{\theta}} \left( \int p(\mathbf x_{i}, \mathbf z_{i} \mid \theta) \,\mathrm d\mathbf z_{i} \right) \\
&= \sum_{i=1}^n  \frac{1}{p(\mathbf x_{i} \mid \theta)} \int \nabla_{\boldsymbol{\theta}} p(\mathbf x_{i}, \mathbf z_{i} \mid \theta) \,\mathrm d\mathbf z_{i} \\
\end{align*}
$$

In practice, parameters are estimated using EM algorithms instead of directly optimizing the incomplete-data log likelihood.

## Superivsed Learning

Problem formulation:

* Model: $p(\mathbf x, y \mid \boldsymbol{\theta})$ with unknown $\boldsymbol{\theta}$.
* Given: training data $D = \{\mathbf x_1, y_1, \cdots, \mathbf x_n, y_n\} \stackrel{\text{iid}}{\sim} p(\mathbf x, y \mid \boldsymbol{\theta})$.
* Goal: estimate $\boldsymbol{\theta}$.

Additional assumption: The parameterized joint distribution can be factorized either as

$$
\begin{align}
p(\mathbf x, y \mid \boldsymbol{\theta}) &= p(\mathbf x \mid \boldsymbol{\pi}) \cdot p(y \mid \mathbf x,\mathbf{w}) \\
\boldsymbol{\theta} &= (\boldsymbol{\pi}, \mathbf{w})
\end{align}
$$

or as

$$
\begin{align}
p(\mathbf x, y \mid \boldsymbol{\theta}) &= p(y \mid \boldsymbol{\pi}) \cdot p(\mathbf x\mid y,\mathbf{w}) \\
\boldsymbol{\theta} &= (\boldsymbol{\pi}, \mathbf{w})
\end{align}
$$

Remarks:

* The 1st factorization is called ***discriminative modeling***. Philosophically, a discrimiative model thinks $y$ (e.g. price of a house) as an uncertain consequence of $\mathbf x$ (e.g. housing area). To predict the label under discriminative setting, it is acutally sufficient to estimate $\mathbf{w}$ only.
* The 2nd factorization is called ***generative modeling***. Philosophically, a discrimiative model thinks $\mathbf x$ (e.g. size and weight) as an uncertain consequence of $y$ (e.g. animal species). To predict the label under generative setting, we must estimate the whole set of parameters $(\boldsymbol{\pi}, \mathbf{w})$.
* Note that $\boldsymbol{\pi}$ and $\mathbf{w}$ have different meanings under discriminative model and generative models. In discriminative model, $\boldsymbol{\pi}$ parameterizes $p(\mathbf x)$ while $\mathbf{w}$ parameterizes $p(y\mid \mathbf x)$. In generative model, $\boldsymbol{\pi}$ parameterizes $p(y)$ while $\mathbf{w}$ parameterizes $p(\mathbf x\mid y)$.

**Examples**: Discriminative model or generative model?

1. predicting the price of a house given its area $\to$ discriminative model.
1. predicting whether a dish is healthy given its ingradients $\to$ discriminative model.
1. predicting which number given a hand-written digit $\to$ generative model.
1. predicting which species given the weight and size of an animal $\to$ generative model.

### Discriminative Modeling

By iid assumption and that $p(\mathbf x, y \mid \boldsymbol{\theta}) = p(\mathbf x \mid \boldsymbol{\pi}) \, p(y \mid \mathbf x,\mathbf{w})$, the likelihood is

$$
\begin{align}
p(D\mid\boldsymbol{\theta})
&= p(\mathbf x_1, y_1, \cdots, \mathbf x_n, y_n \mid \boldsymbol{\theta}) \\
&= \prod_{i=1}^n p(\mathbf x_i,y_i \mid \boldsymbol{\theta}) \\
&= \prod_{i=1}^n p(\mathbf x_i \mid \boldsymbol{\pi}) \cdot p(y_i \mid \mathbf x_i,\mathbf{w}) \\
&= \left( \prod_{i=1}^n p(\mathbf x_i \mid \boldsymbol{\pi})\right) \cdot \left(\prod_{i=1}^n p(y_i \mid \mathbf x_i,\mathbf{w})\right)
\end{align}
$$

Taking the log, we get the log likelihood

> $$
> \begin{align}
> \ln p(D\mid\boldsymbol{\theta})
> &= \underbrace{\sum_{i=1}^n \ln p(\mathbf x_i \mid \boldsymbol{\pi})}_{J_1(\boldsymbol{\pi})} + \underbrace{\sum_{i=1}^n \ln p(y_i \mid \mathbf x_i,\mathbf{w})}_{J_2(\mathbf{w})}
> \end{align}
> $$

Remarks:

* $J_1(\boldsymbol{\pi})$ is in fact the log likelihood of $\mathbf x_1, \dots, \mathbf x_n$ given $\boldsymbol{\pi}$
    $$
    \begin{align*}
    \ln p(\mathbf x_1, \dots, \mathbf x_n \mid\boldsymbol{\pi})
    = \ln \prod_{i=1}^n p(\mathbf x_i\mid\boldsymbol{\pi})
    = \sum_{i=1}^n \ln p(\mathbf x_i \mid \boldsymbol{\pi})
    \triangleq J_1(\boldsymbol{\pi})
    \end{align*}
    $$

* $J_2(\mathbf{w})$ is the log of conditional likelihood of $y_1, \dots, y_n$ given $\mathbf x_1, \dots, \mathbf x_n$ and $\mathbf{w}$
    $$
    \begin{align*}
    \ln p(y_1, \dots, y_n \mid\mathbf x_1, \dots, \mathbf x_n, \mathbf{w})
    &= \ln \prod_{i=1}^n p(y_i \mid \mathbf x_i, \mathbf{w}) \\
    &= \sum_{i=1}^n \ln p(y_i \mid \mathbf x_i, \mathbf{w})
    \triangleq J_2(\mathbf{w})
    \end{align*}
    $$

Hence, $\boldsymbol{\pi}$ and $\mathbf{w}$ can be estimated separately

> $$
> \begin{align}
> \hat{\boldsymbol{\pi}}_\text{MLE} &= \argmax_{\boldsymbol{\pi}} \sum_{i=1}^n \ln p(\mathbf x_i \mid \boldsymbol{\pi})
> \\
> \hat{\mathbf{w}}_\text{MLE} &= \argmax_{\mathbf{w}} \sum_{i=1}^n \ln p(y_i \mid \mathbf x_i,\mathbf{w})
> \end{align}
> $$

If we have priors on $\boldsymbol{\theta}$ s.t. $p(\boldsymbol{\theta}) = p(\boldsymbol{\pi})\,p(\mathbf{w})$, the posterior distribution of $\boldsymbol{\theta}$ is then

$$
\begin{align}
p(\boldsymbol{\theta} \mid D)
&\propto p(D\mid\boldsymbol{\theta}) \cdot p(\boldsymbol{\theta}) \\
&= p(D\mid\boldsymbol{\theta}) \cdot p(\boldsymbol{\pi}) \, p(\mathbf{w}) \\
&= \left(p(\boldsymbol{\pi}) \, \prod_{i=1}^n p(\mathbf x_i \mid \boldsymbol{\pi})\right) \cdot \left(p(\mathbf{w}) \, \prod_{i=1}^n p(y_i \mid \mathbf x_i,\mathbf{w})\right)
\end{align}
$$

Taking the log, we get the log posterior

> $$
> \begin{align}
> \ln p(\boldsymbol{\theta} \mid D)
> &= \underbrace{\ln p(\boldsymbol{\pi}) + \sum_{i=1}^n \ln p(\mathbf x_i \mid \boldsymbol{\pi})}_{J_1(\boldsymbol{\pi})} +
> \underbrace{\ln p(\mathbf{w}) + \sum_{i=1}^n \ln p(y_i \mid \mathbf x_i,\mathbf{w})}_{J_2(\mathbf{w})} +
> \text{const}
> \end{align}
> $$

Therefore, we get the MAP estimation

> $$
> \begin{align}
> \hat{\boldsymbol{\pi}}_\text{MAP}
> &= \argmax_{\boldsymbol{\pi}} \ln p(\boldsymbol{\pi}) + \sum_{i=1}^n \ln p(\mathbf x_i \mid \boldsymbol{\pi})
> \\
> \hat{\mathbf{w}}_\text{MAP}
> &= \argmax_{\mathbf{w}} \ln p(\mathbf{w}) + \sum_{i=1}^n \ln p(y_i \mid \mathbf x_i,\mathbf{w})
> \end{align}
> $$

Remarks:

* For label prediction, we do not really need to know $p(\mathbf x \mid \boldsymbol{\pi})$. Hence, it is sufficient to estimate $\mathbf{w}$ (either by MLE or MAP) and thus $p(y \mid \mathbf x,\mathbf{w})$.
* In some applications (like signal reconstruction), $\mathbf x$ is deterministic or has no inherent distribution. In such cases, we omit modeling $p(\mathbf x)$ as it is irrelevant for label prediction anyway.

Again, $\hat{\mathbf{w}}_\text{MLE}$ and $\hat{\mathbf{w}}_\text{MAP}$ are computed by gradient-based optimization (For MLE, the prior is simply dropped):

* Gradient ascent:
  $$
  \begin{align*}
  \mathbf{w}^{(t+1)}
  &= \mathbf{w}^{(t)} + \eta^{(t)} \left(\sum_{i=1}^n \nabla_{\mathbf{w}} \ln p(y_i \mid \mathbf x_i, \mathbf{w}^{(t)}) + \nabla_{\mathbf{w}}\ln p(\mathbf{w}^{(t)}) \right)\\
  \end{align*}
  $$

* Mini-bath gradient ascent:
  $$
  \begin{align*}
  & \text{randomly draw } B^{(t)} \subset D \text{ with } \vert B \vert = m
  \\
  & \mathbf{w}^{(t+1)}
  = \mathbf{w}^{(t)} + \eta^{(t)} \left(\sum_{(\mathbf x, y)\in B^{(t)}} \nabla_{\mathbf{w}} \ln p(y \mid \mathbf x, \mathbf{w}^{(t)}) + \nabla_{\mathbf{w}}\ln p(\mathbf{w}^{(t)}) \right)
  \end{align*}
  $$

* Stochastic gradient ascent:
  $$
  \begin{align*}
  & \text{randomly draw } (\mathbf x^{(t)}, y^{(t)}) \in D
  \\
  & \mathbf{w}^{(t+1)}
  = \mathbf{w}^{(t)} + \eta^{(t)} \left(\nabla_{\mathbf{w}} \ln p(y^{(t)} \mid \mathbf x^{(t)}, \mathbf{w}^{(t)}) + \nabla_{\mathbf{w}}\ln p(\mathbf{w}^{(t)}) \right)
  \end{align*}
  $$

How to use a discriminative model for prediction?

By plugging the point estimate $\hat{\mathbf{w}}$ (either MLE or MAP) into the model likelihood $p(y \mid \mathbf{x}, \mathbf{w})$, we obtain  the ***(plug-in) predictive distribution***:
$$
p(y\mid\mathbf x, \hat{\mathbf{w}})
$$

For a new input $\mathbf{x}_*$, we use decision theory to make a prediction based on the predictive distribution (see notes on Bayesian decision theory).

$$
\begin{align}
\hat y
&= \argmin_{a \in \mathcal Y} \mathbb E_y[\ell(y, a) \mid \mathbf x_*, \hat{\mathbf{w}}] \\
&= \argmin_{a \in \mathcal Y} \int \ell(y, a) \cdot p(y \mid \mathbf x_*, \hat{\mathbf{w}}) \:\mathrm dy \\
\end{align}
$$

Main results:

* regression with square loss $\ell(y, a) = (y-a)^2 \implies$ predict the conditional mean

    $$
    \begin{align}
    \hat y &= \mathbb E_y[y\mid\mathbf x_*, \hat{\mathbf{w}}]
    \end{align}
    $$

* classification with 0/1 loss $\ell(y, a) = \mathbb I[y \ne a] \implies$ predict the conditional mode

    $$
    \begin{align}
    \hat y &= \argmax_y p(y\mid\mathbf x_*, \hat{\mathbf{w}})
    \end{align}
    $$

### Generative Modeling

By generative modeling $p(\mathbf x, y \mid \boldsymbol{\theta}) = p(y \mid \boldsymbol{\pi}) \cdot p(\mathbf x\mid y,\mathbf{w})$, the likelihood can be factorized as

$$
\begin{align}
p(D\mid\boldsymbol{\theta})
&= \prod_{i=1}^n p(\mathbf x_i,y_i \mid \boldsymbol{\theta}) \\
&= \left( \prod_{i=1}^n p(y_i \mid \boldsymbol{\pi}) \right) \cdot \left(\prod_{i=1}^n p(\mathbf x_i \mid y_i,\mathbf{w}) \right)
\end{align}
$$

Taking the log, we get the log likelihood

> $$
> \begin{align}
> \ln p(D\mid\boldsymbol{\theta})
> &= \underbrace{\sum_{i=1}^n \ln p(y_i \mid \boldsymbol{\pi})}_{J_1(\boldsymbol{\pi})} + \underbrace{\sum_{i=1}^n \ln p(\mathbf x_i \mid y_i,\mathbf{w})}_{J_2(\mathbf{w})}
> \end{align}
> $$

Again, $\boldsymbol{\pi}$ and $\mathbf{w}$ can be estimated separately

> $$
> \begin{align}
> \hat{\boldsymbol{\pi}}_\text{MLE} &= \argmax_{\boldsymbol{\pi}} \sum_{i=1}^n \ln p(y_i \mid \boldsymbol{\pi})
> \\
> \hat{\mathbf{w}}_\text{MLE} &= \argmax_{\mathbf{w}} \sum_{i=1}^n \ln p(\mathbf x_i \mid y_i,\mathbf{w})
> \end{align}
> $$

Using the prior $p(\boldsymbol{\theta}) = p(\boldsymbol{\pi})\,p(\mathbf{w})$, the posterior of $\boldsymbol{\theta}$ is then

$$
\begin{align}
p(\boldsymbol{\theta} \mid D)
&\propto p(D\mid\boldsymbol{\theta}) \cdot p(\boldsymbol{\theta}) \\
&= p(D\mid\boldsymbol{\theta}) \cdot p(\boldsymbol{\pi}) \, p(\mathbf{w}) \\
&= \left(p(\boldsymbol{\pi}) \, \prod_{i=1}^n p(y_i \mid \boldsymbol{\pi})\right) \cdot \left(p(\mathbf{w}) \, \prod_{i=1}^n p(\mathbf x_i \mid y_i,\mathbf{w})\right)
\end{align}
$$

Taking the log, we get the log posterior

> $$
> \begin{align}
> \ln p(\boldsymbol{\theta} \mid D)
> &= \underbrace{\ln p(\boldsymbol{\pi}) + \sum_{i=1}^n \ln p(y_i \mid \boldsymbol{\pi})}_{J_1(\boldsymbol{\pi})} +
> \underbrace{\ln p(\mathbf{w}) + \sum_{i=1}^n \ln p(\mathbf x_i \mid y_i,\mathbf{w})}_{J_2(\mathbf{w})} +
> \text{const}
> \end{align}
> $$

from which we can compute the MAP estimates

> $$
> \begin{align}
> \hat{\boldsymbol{\pi}}_\text{MAP} &= \argmax_{\boldsymbol{\pi}} \sum_{i=1}^n \ln p(y_i \mid \boldsymbol{\pi}) + \ln p(\boldsymbol{\pi})
> \\
> \hat{\mathbf{w}}_\text{MAP} &= \argmax_{\mathbf{w}} \sum_{i=1}^n \ln p(\mathbf x_i \mid y_i,\mathbf{w}) + \ln p(\mathbf{w})
> \end{align}
> $$

Again, the optimization is often carried out by gradient based methods. Formulas are omitted here as they are similar to those in previous section.

How to use a learned generative model?

Suppose we learned a generative model $p(\mathbf x, y, \hat{\boldsymbol{\theta}})$ where $\hat{\boldsymbol{\theta}} = (\hat{\boldsymbol{\pi}}, \hat{\mathbf{w}})$. We can compute the predictive distribution $p(y \mid \mathbf x, \hat{\boldsymbol{\theta}})$ from it.

$$
\begin{align}
p(y \mid \mathbf x, \hat{\boldsymbol{\theta}})
&= \frac{p(\mathbf x,y \mid \hat{\boldsymbol{\theta}})}{p(\mathbf x \mid \hat{\boldsymbol{\theta}})} \nonumber
\\[8pt]
&= \frac{p(y \mid \hat{\boldsymbol{\pi}}) \cdot p(\mathbf x \mid y, \hat{\mathbf{w}})}{p(\mathbf x \mid \hat{\boldsymbol{\theta}})} \nonumber
\end{align}
$$

where the denominator is computed by

$$
\begin{align}
p(\mathbf x \mid \hat{\boldsymbol{\theta}})
&= \int p(y \mid \hat{\boldsymbol{\pi}}) \cdot p(\mathbf x \mid y, \hat{\mathbf{w}}) \:\mathrm dy
&& \text{for regression}
\\
p(\mathbf x \mid \hat{\boldsymbol{\theta}})
&= \sum_y p(y \mid \hat{\boldsymbol{\pi}}) \cdot p(\mathbf x \mid y, \hat{\mathbf{w}})
&& \text{for classification}
\\
\end{align}
$$

The predictive distribution $p(y \mid \mathbf x, \hat{\boldsymbol{\theta}})$ can be used for label prediction, just like using a discriminative model.

However, generative models (aka the joint distribution) offer more flexibility than discriminative models (aka the conditional distribution). Specifically, generative models provide more statistical information and allow data generation via $p(\mathbf x \mid y, \hat{\mathbf{w}})$.

Think of $y\in\{0,\dots,9\}$ and $\mathbf x$ as a 128x128 image of a hand-written digit. We can generate a hand-written digit of "7" by sampling $\mathbf x$ from $p(\mathbf x \mid y=7, \hat{\mathbf{w}})$.

## Discriminative Regression Model

Problem formulation:

* Given: training data $D = \{\mathbf x_1, y_1, \cdots, \mathbf x_n, y_n\}$ iid from unknown $p(\mathbf x, y)$
* Goal: estimate $p(y \mid \mathbf x)$

We model $y$ as a paramterized function of $\mathbf x$ with additive label noise.

> $$
> \begin{align}
> y = f_{\boldsymbol{\theta}}(\mathbf x) + \epsilon, \quad \epsilon\sim\mathcal N(0, \sigma^2_\text{n})
> \end{align}
> $$

Examples of paramterized function $f_{\boldsymbol{\theta}}: \mathbb R^d \to \mathbb R$:

* linear model: $\boldsymbol{\theta} = \mathbf{w}\in\mathbb R^d$.
    $$
    y = \underbrace{\mathbf{w}^\top \mathbf x}_{f_{\boldsymbol{\theta}}(\mathbf x)} + \epsilon
    $$

* neural net with one hidden layer and activation function $\sigma(\cdot)$.
    $$
    \begin{align*}
    \boldsymbol{\theta} &= \{\mathbf{W}_1\in\mathbb R^{h\times d}, \mathbf{b}_1\in\mathbb R^{h}, \mathbf{w}_2\in\mathbb R^{1\times h}, b_2\in\mathbb R\}
    \\
    y &= \underbrace{\mathbf{w}_2 \,\sigma(\mathbf{W}_1 \mathbf x + \mathbf{b}_1) + b_2}_{f_{\boldsymbol{\theta}}(\mathbf x)} + \epsilon
    \end{align*}
    $$

The conditional distribution $p(y \mid \mathbf x)$ is then paramterized by $\boldsymbol{\theta}$. Hence, we write $p(y \mid \mathbf x, \boldsymbol{\theta})$

> $$
> \begin{align}
> p(y \mid \mathbf x, \boldsymbol{\theta})
> &= \mathcal N(y ; f_{\boldsymbol{\theta}}(\mathbf x), \sigma^2_\text{n}) \\
> &= \frac{1}{\sqrt{2\pi \sigma^2_\text{n}}} \exp\left(
>    -\frac{\left(y - f_{\boldsymbol{\theta}}(\mathbf{x})\right)^2}{2\sigma^2_\text{n}}
>    \right)
> \end{align}
> $$

Remarks:

* The parameter $\boldsymbol{\theta}$ describes the conditional mean
    $$
    f_{\boldsymbol{\theta}}(\mathbf x) = \mathbb E_y[y\mid\mathbf x, \boldsymbol{\theta}]
    $$

* For point estimates (MLE or MAP) of $\boldsymbol{\theta}$, we do not need to know the variance of the noise $\sigma^2_\text{n}$.
* Scenarios requiring knowledge about $\sigma^2_\text{n}$: uncertainty quantification for label prediction, Bayesian inference (model averaging).

The log likelihood
$$
\begin{align}
p(D \mid\boldsymbol{\theta})
&= p(\mathbf x_1, y_1, \cdots, \mathbf x_n, y_n \mid\boldsymbol{\theta}) \nonumber \\
&= \prod_{i=1}^n p(\mathbf x_i, y_i \mid \boldsymbol{\theta}) \nonumber \\
&= \prod_{i=1}^n p(\mathbf x_i) \cdot p(y_i \mid \mathbf x_i, \boldsymbol{\theta}) \nonumber \\
&\propto \prod_{i=1}^n p(y_i \mid \mathbf x_i, \boldsymbol{\theta}) \\
\end{align}
$$

The log likelihood is then

$$
\begin{align}
\ln p(D \mid\boldsymbol{\theta})
&= \ln \prod_{i=1}^n p(y_i \mid \mathbf x_i, \boldsymbol{\theta}) + \text{const} \nonumber \\
&= \sum_{i=1}^n \ln p(y_i \mid \mathbf x_i, \boldsymbol{\theta}) + \text{const} \\
&= \sum_{i=1}^n \ln \mathcal N(y_i ; f_{\boldsymbol{\theta}}(\mathbf x_i), \sigma^2_\text{n}) + \text{const} \nonumber \\
&= -\frac{1}{2\sigma^2_\text{n}} \sum_{i=1}^n \left(y_i - f_{\boldsymbol{\theta}}(\mathbf{x}_i)\right)^2 + \text{const}
\end{align}
$$

Therefore, maximizing the log likelihood $\iff$ minimizing the sum of square loss.

> $$
> \begin{align}
> \hat{\boldsymbol{\theta}}_\text{MLE}
> &= \argmax_{\boldsymbol{\theta}} \sum_{i=1}^n \ln p(y_i \mid \mathbf x_i, \boldsymbol{\theta}) \\
> &= \argmin_{\boldsymbol{\theta}} \sum_{i=1}^n \left(y_i - f_{\boldsymbol{\theta}}(\mathbf{x}_i)\right)^2
> \end{align}
> $$

If we have prior on $\boldsymbol{\theta}$, we can perform the MAP estimation. The posterior is

$$
\begin{align}
p(\boldsymbol{\theta} \mid D)
&= p(\boldsymbol{\theta}) \cdot p(D \mid\boldsymbol{\theta}) \\
&\propto p(\boldsymbol{\theta}) \cdot \prod_{i=1}^n p(y_i \mid \mathbf x_i, \boldsymbol{\theta}) \\
\end{align}
$$

The log posterior is then

$$
\begin{align}
\ln p(\boldsymbol{\theta} \mid D)
&= \ln p(\boldsymbol{\theta}) + \ln \prod_{i=1}^n p(y_i \mid \mathbf x_i, \boldsymbol{\theta}) + \text{const} \\
&= \ln p(\boldsymbol{\theta}) -\frac{1}{2\sigma^2_\text{n}} \sum_{i=1}^n \left(y_i - f_{\boldsymbol{\theta}}(\mathbf{x}_i)\right)^2 + \text{const}
\end{align}
$$

If we use a Gaussian prior

$$
\begin{align}
\boldsymbol{\theta} \sim \mathcal N(\mathbf 0, \sigma^2_\mathrm{p}\mathbf I)
\end{align}
$$

which gives the log prior

$$
\begin{align*}
\ln p(\boldsymbol{\theta}) = -\frac{1}{2\sigma^2_\mathrm{p}} \Vert\boldsymbol{\theta}\Vert^2 + \text{const}
\end{align*}
$$

The log posterior becomes

$$
\begin{align}
\ln p(\boldsymbol{\theta} \mid D)
&= -\frac{1}{2} \left[
    \frac{1}{\sigma^2_\text{n}} \sum_{i=1}^n \left(y_i - f_{\boldsymbol{\theta}}(\mathbf{x}_i)\right)^2 +
    \frac{1}{\sigma^2_\mathrm{p}} \Vert\boldsymbol{\theta}\Vert^2
   \right] + \text{const}
\end{align}
$$

Therefore, MAP with Gaussian prior $\iff$ minimizing L2 regularized the sum of square loss.

> $$
> \begin{align}
> \hat{\boldsymbol{\theta}}_\text{MAP}
> &= \argmax_{\boldsymbol{\theta}}
>    \sum_{i=1}^n \ln p(y_i \mid \mathbf x_i, \boldsymbol{\theta}) + \ln p(\boldsymbol{\theta}) \\
> &= \argmin_{\boldsymbol{\theta}}
>    \sum_{i=1}^n \left(y_i - f_{\boldsymbol{\theta}}(\mathbf{x}_i)\right)^2 +
>    \frac{\sigma^2_\text{n}}{\sigma^2_\mathrm{p}} \Vert\boldsymbol{\theta}\Vert^2
> \end{align}
> $$

Remarks:

* $\lambda \triangleq \sigma^2_\text{n} / \sigma^2_\text{p}$ is the hyper parameter for regularization. The larger $\lambda$ is, the stronger is the regularization -- the heavier are large parameters penalized. The hyper parameter $\lambda$ balances between prior belief and observed data.
* If we have a strong prior belief (i.e. small $\sigma^2_\text{p}$ ) that all parameters should be close to 0, then we have stong regularization (i.e. large $\lambda$ ) and rely more on the prior belief.
* If the label noise is small (i.e. small $\sigma^2_\text{n}$ ), then we have weaker regularization (i.e. small $\lambda$ ) and rely more on the observed data.

Once we computed the point estimate, we obtain the plug-in predictive distribution

$$
p(y\mid\mathbf x, \hat{\boldsymbol{\theta}}) = \mathcal N(y ; f_{\hat{\boldsymbol{\theta}}}(\mathbf x), \sigma^2_\text{n})
$$

For a new input $\mathbf x_*$, assuming square loss is used, the optimal prediction is the mean under the predictive distribution

$$
\begin{align}
\hat y
&= \mathbb E_y [y\mid\mathbf x_*, \hat{\boldsymbol{\theta}}] \nonumber \\
&= \mathbb E_y \left[ \mathcal N(y ; f_{\hat{\boldsymbol{\theta}}}(\mathbf x_*), \sigma^2_\text{n}) \right] \nonumber \\
&= f_{\hat{\boldsymbol{\theta}}}(\mathbf x_*)
\end{align}
$$

|  Algorithmic Perspective | Statistical Perspective |
| ----------- | -------------- |
| $\boldsymbol{\theta}$ parameterizes $f(\mathbf x)$ | $\boldsymbol{\theta}$ parameterizes $p(y \mid \mathbf x)$ |
| choose square loss | label noise is iid Gaussian |
| sum of square loss | negative log likelihood |
| minimize sum of square loss $\to\hat{\boldsymbol{\theta}}$ | MLE $\to\hat{\boldsymbol{\theta}}$ |
| minimize sum of square loss + regularizer $\to\hat{\boldsymbol{\theta}}$ | MAP $\to\hat{\boldsymbol{\theta}}$ |
| L1 regularization | Laplacian prior |
| L2 regularization | Gaussian prior |
| predict: $f_{\hat{\boldsymbol{\theta}}}(\mathbf x_*)$ | predict: $\mathbb E_y[y\mid\mathbf x_*, \hat{\boldsymbol{\theta}}]$ |
| $-$ | uncertainty quantification |
| $-$ | model averaging instead of point estimate |
