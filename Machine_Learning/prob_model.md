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

TODO: create figure: pipeline of statistical learning

Remarks:

* training data $D$: iid from unknown ground truth $\mathbb P$.
* ML method: compute $\mathbb{\widehat P}$ from $D$ to estimate $\mathbb{P}$. We hope that $\mathbb{\widehat P}$ is as close to $\mathbb{P}$ as possible.
* Use $\mathbb{\widehat P}$ on new data for predictions, anormony detection, etc.

Which probability do we want to learn?

* For unsupervised learning: $p(\mathbf x)$
* For supervised learning: $p(y\mid\mathbf x)$ or $p(\mathbf x, y)$

How to learn the probability?

* Non-parametric methods: kernel density estimation

* Parametric methods: The PDF of interest belongs to some parametric family (e.g. normal distribution)

  * parameter ***estimation***: MLE, MAP. Based on optimization. $\to$ point estimate $\hat{\boldsymbol{\theta}}$.
  * Bayesian ***inference*** (or model averaging):  $\to$ use the full posterior distribution $p(\boldsymbol{\theta} \mid D)$.

Our focus: parametric methods. In particular, parameter estimation.

## Unsuperivsed Learning

Problem formulation:

* Given: training data $D = \{\mathbf x_1, \cdots, \mathbf x_n\}$ iid from $p(\mathbf x \mid \boldsymbol{\theta})$ with unknown $\boldsymbol{\theta}$
* Optional: prior distribution $p(\boldsymbol{\theta})$
* Goal: estimate $\boldsymbol{\theta}$

We assume that the data is fully observable, i.e. there is no latent variables. Unsupervised learning with latent variables (e.g. GMM) requires more complex algorithms (e.g. EM algorithm). Not detailed here.

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
\frac{\partial}{\partial\boldsymbol{\theta}} \ln p(D\mid\boldsymbol{\theta}) &= 0 \\
\sum_{i=1}^n \frac{\partial}{\partial\boldsymbol{\theta}} \ln p(\mathbf x_i \mid \boldsymbol{\theta}) &= 0 \\
\end{align}
$$

If the log likelihood is non-convex,  $\hat{\boldsymbol{\theta}}_\text{MLE}$ can be computed by gradient ascent

$$
\begin{align}
\boldsymbol{\theta}^{(t+1)}
&= \boldsymbol{\theta}^{(t)} + \eta^{(t)} \frac{\partial}{\partial\boldsymbol{\theta}} \ln p(D\mid\boldsymbol{\theta}^{(t)}) \\
&= \boldsymbol{\theta}^{(t)} + \eta^{(t)} \sum_{i=1}^n \frac{\partial}{\partial\boldsymbol{\theta}} \ln p(\mathbf x_i \mid \boldsymbol{\theta}^{(t)}) \\
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
    = \boldsymbol{\theta}^{(t)} + \eta^{(t)} \sum_{\mathbf x\in B^{(t)}} \frac{\partial}{\partial\boldsymbol{\theta}} \ln p(\mathbf x \mid \boldsymbol{\theta}^{(t)})
    \end{align}
    $$

* or stochastic gradient (speical case of mini-batch when $m=1$)
    $$
    \begin{align}
    & \text{randomly draw } \mathbf x^{(t)} \in D \nonumber
    \\
    & \boldsymbol{\theta}^{(t+1)}
    = \boldsymbol{\theta}^{(t)} + \eta^{(t)} \frac{\partial}{\partial\boldsymbol{\theta}} \ln p(\mathbf x^{(t)} \mid \boldsymbol{\theta}^{(t)})
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
\frac{\partial}{\partial\boldsymbol{\theta}}\ln p(D\mid\boldsymbol{\theta}) + \frac{\partial}{\partial\boldsymbol{\theta}}\ln p(\boldsymbol{\theta}) &= 0 \\
\sum_{i=1}^n \frac{\partial}{\partial\boldsymbol{\theta}} \ln p(\mathbf x_i \mid \boldsymbol{\theta}) + \frac{\partial}{\partial\boldsymbol{\theta}}\ln p(\boldsymbol{\theta}) &= 0 \\
\end{align}
$$

Gradient-based optimization:

* Gradient ascent:
    $$
    \begin{align}
    \boldsymbol{\theta}^{(t+1)}
    &= \boldsymbol{\theta}^{(t)} + \eta^{(t)} \left(\sum_{i=1}^n \frac{\partial}{\partial\boldsymbol{\theta}} \ln p(\mathbf x_i \mid \boldsymbol{\theta}^{(t)}) + \frac{\partial}{\partial\boldsymbol{\theta}}\ln p(\boldsymbol{\theta}^{(t)}) \right)\\
    \end{align}
    $$

* Mini-bath gradient ascent:
    $$
    \begin{align}
    & \text{randomly draw } B^{(t)} \subset D \text{ with } \vert B \vert = m \nonumber
    \\
    & \boldsymbol{\theta}^{(t+1)}
    = \boldsymbol{\theta}^{(t)} + \eta^{(t)} \left(\sum_{\mathbf x\in B^{(t)}} \frac{\partial}{\partial\boldsymbol{\theta}} \ln p(\mathbf x \mid \boldsymbol{\theta}^{(t)}) + \frac{\partial}{\partial\boldsymbol{\theta}}\ln p(\boldsymbol{\theta}^{(t)}) \right)
    \end{align}
    $$

* Stochastic gradient ascent:
    $$
    \begin{align}
    & \text{randomly draw } \mathbf x^{(t)} \in D \nonumber
    \\
    & \boldsymbol{\theta}^{(t+1)}
    = \boldsymbol{\theta}^{(t)} + \eta^{(t)} \left(\frac{\partial}{\partial\boldsymbol{\theta}} \ln p(\mathbf x^{(t)} \mid \boldsymbol{\theta}^{(t)}) + \frac{\partial}{\partial\boldsymbol{\theta}}\ln p(\boldsymbol{\theta}^{(t)}) \right)
    \end{align}
    $$

In practice, we often impose a Gaussian prior on $\boldsymbol{\theta}$,

$$
\begin{align}
\boldsymbol{\theta} \sim \mathcal N(\mathbf 0, \sigma^2_\mathrm{p}\mathbf I)
\end{align}
$$

which gives the log prior

$$
\begin{align}
\ln p(\boldsymbol{\theta}) = -\frac{1}{2\sigma^2_\mathrm{p}} \Vert\boldsymbol{\theta}\Vert^2 + \text{const}
\end{align}
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
&= \boldsymbol{\theta}^{(t)} + \eta^{(t)} \left(\sum_{i=1}^n \frac{\partial}{\partial\boldsymbol{\theta}} \ln p(\mathbf x_i \mid \boldsymbol{\theta}^{(t)}) -
   \frac{\partial}{\partial\boldsymbol{\theta}}\frac{1}{2\sigma^2_\mathrm{p}} \Vert\boldsymbol{\theta}^{(t)}\Vert^2 \right) \nonumber \\
&= \boldsymbol{\theta}^{(t)} + \eta^{(t)} \left(\sum_{i=1}^n \frac{\partial}{\partial\boldsymbol{\theta}} \ln p(\mathbf x_i \mid \boldsymbol{\theta}^{(t)}) -
   \frac{1}{\sigma^2_\mathrm{p}} \boldsymbol{\theta}^{(t)} \right) \nonumber \\
&= \boldsymbol{\theta}^{(t)} \left(1 - \frac{\eta^{(t)}}{\sigma^2_\mathrm{p}}\right) + \eta^{(t)} \sum_{i=1}^n \frac{\partial}{\partial\boldsymbol{\theta}} \ln p(\mathbf x_i \mid \boldsymbol{\theta}^{(t)}) \\
\end{align}
$$

## Superivsed Learning

Problem formulation:

* Given: training data $D = \{\mathbf x_1, y_1, \cdots, \mathbf x_n, y_n\}$ iid from $p(\mathbf x, y \mid \boldsymbol{\theta})$ with unknown $\boldsymbol{\theta}$
* Goal: estimate $\boldsymbol{\theta}$

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

### Parameter Estimation in Discriminative Model

By iid assumption and that $p(\mathbf x, y \mid \boldsymbol{\theta}) = p(\mathbf x \mid \boldsymbol{\pi}) \, p(y \mid \mathbf x,\mathbf{w})$, the likelihood is

$$
\begin{align}
p(D\mid\boldsymbol{\theta})
&= p(\mathbf x_1, y_1, \cdots, \mathbf x_n, y_n \mid \boldsymbol{\theta}) \\
&= \prod_{i=1}^n p(\mathbf x_i,y_i \mid \boldsymbol{\theta}) \\
&= \prod_{i=1}^n p(\mathbf x_i \mid \boldsymbol{\pi}) \cdot p(y_i \mid \mathbf x_i,\mathbf{w}) \\
&= \left(\prod_{i=1}^n p(\mathbf x_i \mid \boldsymbol{\pi})\right) \cdot \left(\prod_{i=1}^n p(y_i \mid \mathbf x_i,\mathbf{w})\right)
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

**Example** (predicting house price): c.f. Appendix.

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

* For label prediction, we do not really need to know $p(\mathbf x \mid \boldsymbol{\pi})$. Hence, it is sufficient to estimate $\mathbf{w}$ and thus $p(y \mid \mathbf x,\mathbf{w})$. Once we obtained a point estimate (MLE or MAP) of $\mathbf{w}$, we can predict the label on new data points by decision theory. c.f. notes on Bayesian decision theory.
* In some applications (like signal reconstruction), $\mathbf x$ is deterministic or has no inherent distribution. In such cases, we omit modeling $p(\mathbf x)$ as it is irrelevant for label prediction anyway.

### Regression as Discriminative Model

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
    \boldsymbol{\theta} &= \{\mathbf{W}_1\in\mathbb R^{h\times d}, \mathbf{b}_1\in\mathbb R^{h}, \mathbf{W}_2\in\mathbb R^{1\times h}, b_2\in\mathbb R\}
    \\
    y &= \underbrace{\mathbf{W}_2 \,\sigma(\mathbf{W}_1 \mathbf x + \mathbf{b}_1) + b_2}_{f_{\boldsymbol{\theta}}(\mathbf x)} + \epsilon
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
&= p(\mathbf x_1, y_1, \cdots, \mathbf x_n, y_n \mid\boldsymbol{\theta}) \\
&= \prod_{i=1}^n p(\mathbf x_i, y_i \mid \boldsymbol{\theta}) \\
&= \prod_{i=1}^n p(\mathbf x_i) \cdot p(y_i \mid \mathbf x_i, \boldsymbol{\theta}) \\
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
\begin{align}
\ln p(\boldsymbol{\theta}) = -\frac{1}{2\sigma^2_\mathrm{p}} \Vert\boldsymbol{\theta}\Vert^2 + \text{const}
\end{align}
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

|  Algorithmic Perspective | Statistical Perspective |
| ----------- | -------------- |
| $\boldsymbol{\theta}$ parameterizes $f(\mathbf x)$ | $\boldsymbol{\theta}$ parameterizes $p(y \mid \mathbf x)$ |
| choose square loss | assume label noise is iid Gaussian |
| sum of square loss | negative log likelihood |
| minimize sum of square loss $\to\hat{\boldsymbol{\theta}}$ | MLE $\to\hat{\boldsymbol{\theta}}$ |
| minimize sum of square loss + regularizer $\to\hat{\boldsymbol{\theta}}$ | MAP $\to\hat{\boldsymbol{\theta}}$ |
| L1 regularization | Laplacian prior |
| L2 regularization | Gaussian prior |
| predict: $f_{\hat{\boldsymbol{\theta}}}(\mathbf x_\text{new})$ | predict: $\mathbb E_y[y\mid\mathbf x_\text{new}, \hat{\boldsymbol{\theta}}]$ |
| $-$ | uncertainty quantification |
| $-$ | model averaging instead of point estimate |

### Parameter Estimation in Generative Setting

TODO

## Appendix

### Example: predicting house prices

We model the area as a Gaussian $x\sim\mathcal N(\mu_x, \sigma_x^2)$ and the price $y$ as an affine function of $x$ with independent Gaussian noise.
$$
y = wx+b + \epsilon, \quad \epsilon\sim\mathcal N(0, \sigma^2)
$$

Here, the paramters are

$$
\boldsymbol{\pi} = (\mu_x, \sigma_x^2), \quad \mathbf{w} = (w, b)
$$

Given the training data $D= \{x_1, y_1, \cdots, x_n, y_n\}$ about the area and price of $n$ houses, how to estimate $\boldsymbol{\pi}$ and $\mathbf{w}$ with MLE?

For $\boldsymbol{\pi}$: This is standard MLE in unspervised setting.

$$
\hat\mu_x = \frac{1}{n} \sum_{i=1}^n x_i ,
\quad
\hat\sigma_x^2 = \frac{1}{n} \sum_{i=1}^n ( x_i-\hat\mu_x )^2
$$

For $\mathbf{w}$: We need to compute $p(y_i \mid \mathbf x_i,\mathbf{w})$. By assumption,

$$
\begin{align*}
p(y_i \mid \mathbf x_i,\mathbf{w})
&= p_\epsilon(y_i - (wx_i+b)) \\
&= \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(y_i - (wx_i+b))^2}{2\sigma^2} \right) \\
\end{align*}
$$

Hence, the MLE for $\mathbf{w}$ becomes
$$
\begin{align*}
\hat{\mathbf{w}}_\text{MLE}
&= \argmax_{\mathbf{w}} \sum_{i=1}^n \ln p(y_i \mid \mathbf x_i,\mathbf{w}) \\
&= \argmax_{\mathbf{w}} \sum_{i=1}^n \left[\ln\left(\frac{1}{\sqrt{2\pi}\sigma}\right) - \frac{(y_i - (wx_i+b))^2}{2\sigma^2}\right]\\
&= \argmax_{\mathbf{w}} C - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - (wx_i+b))^2 \\
&= \argmin_{\mathbf{w}} \underbrace{\sum_{i=1}^n (y_i - (wx_i+b))^2}_\text{square loss} \\
\end{align*}
$$

Letting the gradient equal to zero, we get

$$
\begin{align*}
\frac{\partial}{\partial w} \sum_{i=1}^n (y_i - (wx_i+b))^2 &= 0
\iff
\sum_{i=1}^n (y_i - (wx_i+b))x_i = 0
\\
\frac{\partial}{\partial b} \sum_{i=1}^n (y_i - (wx_i+b))^2 &= 0
\iff
\sum_{i=1}^n (y_i - (wx_i+b)) = 0
\end{align*}
$$

Solving the linear sytem of equations with unknowns $w$ and $b$, we get

$$
\begin{align*}
\hat w &= \frac{\sum_i(x_i - \hat\mu_x)(y_i - \hat\mu_y)}{\sum_i(x_i - \hat\mu_x)^2} \\
\hat b &= \hat\mu_y - \hat w \hat\mu_x
\end{align*}
$$

where $\hat\mu_y = \frac{1}{n}\sum_i y_i$

Remark:

* $\hat{\boldsymbol{\pi}}_\text{MLE} = (\hat\mu_x, \hat\sigma_x^2)$ only estimates the distribution of house area. Suppose a new house is $120 \text{ m}^2$. Then, $\hat{\boldsymbol{\pi}}_\text{MLE}$ is useful to compare the size of this house to the market average, but not useful for predicting its price.
* $\hat{\mathbf{w}}_\text{MLE} = (\hat w, \hat b)$ estimates the distribution of the price given the housing area. For a new $120 \text{ m}^2$ house, we can predict that its price is Gaussian with mean $120\hat w+\hat b$.
