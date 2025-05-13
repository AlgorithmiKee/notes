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
$$

Notations:

* $\mathbf x\in\mathcal X$: feature vector. $\mathcal X$ could be $\mathbb R^d$ (for continuous data) or a discrete set (for discrete data)
* $y\in\mathcal Y$: label. $\mathcal Y$ could be $\mathbb R$ (for regression) or a discrete set (for discrete data)

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

  * parameter estimation: MLE, MAP. Based on optimization. $\to$ point estimate of the parameters.
  * Bayesian model averaging:  $\to$ distribution of the parameters.

Our focus: parametric methods. In particular, parameter estimation.

## Unsuperivsed Learning

Problem formulation:

* Given: training data $D = \{\mathbf x_1, \cdots, \mathbf x_n\}$ iid from $p(\mathbf x \mid \theta)$ with unknown $\theta$
* Goal: estimate $\theta$

We assume that the data is fully observable, i.e. there is no latent variables. Unsupervised learning with latent variables (e.g. GMM) requires more complex algorithms (e.g. EM algorithm). Not detailed here.

### Maximum Likelihood Estimation (MLE)

The likelihood $p(D\mid\theta)$ can be factorized as follows by iid assumption

$$
\begin{align}
p(D\mid\theta)
&= p(\mathbf x_1, \cdots, \mathbf x_n \mid \theta) \\
&= \prod_{i=1}^n p(\mathbf x_i \mid \theta)
\end{align}
$$

Intuition of MLE: which $\theta$ makes our data most probable? Hence,

$$
\begin{align}
\hat\theta_\text{MLE}
&= \argmax_{\theta} p(D\mid\theta) \\
&= \argmax_{\theta} \ln p(D\mid\theta) \\
\end{align}
$$

Here, we maximize the log of likelihood since

1. the log function converts the product into a sum which is simpler to optimize.
1. the log function is monotonic

The log likelihood is

$$
\begin{align}
\ln p(D\mid\theta)
&= \ln \prod_{i=1}^n p(\mathbf x_i \mid \theta) \\
&= \sum_{i=1}^n \ln p(\mathbf x_i \mid \theta) \\
\end{align}
$$

Therefore, the MLE becomes

$$
\begin{align}
\hat\theta_\text{MLE}
&= \argmax_{\theta} \sum_{i=1}^n \ln p(\mathbf x_i \mid \theta)
\end{align}
$$

If the log likelihood is convex, then $\hat\theta_\text{MLE}$ can be computed analytically by solving

$$
\begin{align}
\frac{\partial}{\partial\theta} \ln p(D\mid\theta) &= 0 \\
\sum_{i=1}^n \frac{\partial}{\partial\theta} \ln p(\mathbf x_i \mid \theta) &= 0 \\
\end{align}
$$

If the log likelihood is non-convex,  $\hat\theta_\text{MLE}$ can be computed by gradient ascent

$$
\begin{align}
\theta^{(t+1)}
&= \theta^{(t)} + \eta^{(t)} \frac{\partial}{\partial\theta} \ln p(D\mid\theta^{(t)}) \\
&= \theta^{(t)} + \eta^{(t)} \sum_{i=1}^n \frac{\partial}{\partial\theta} \ln p(\mathbf x_i \mid \theta^{(t)}) \\
\end{align}
$$

where $\eta^{(t)}$ is called learning rate or step size.

If the data set $D$ is large, one can use

* mini-batch to calculate the gradient
    $$
    \begin{align}
    & \text{randomly draw } B^{(t)} \subset D \text{ with } \vert B \vert = m
    \\
    & \theta^{(t+1)}
    = \theta^{(t)} + \eta^{(t)} \sum_{\mathbf x\in B^{(t)}} \frac{\partial}{\partial\theta} \ln p(\mathbf x \mid \theta^{(t)})
    \end{align}
    $$

* or stochastic gradient (speical case of mini-batch when $m=1$)
    $$
    \begin{align}
    & \text{randomly draw } \mathbf x^{(t)} \in D
    \\
    & \theta^{(t+1)}
    = \theta^{(t)} + \eta^{(t)} \frac{\partial}{\partial\theta} \ln p(\mathbf x^{(t)} \mid \theta^{(t)})
    \end{align}
    $$

### Maximum a Posterior Estimation (MAP)

The posterior probability of $\theta$ is given by Bayes rule:

$$
\begin{align}
p(\theta\mid D)
&= \frac{p(D\mid \theta) \cdot p(\theta)}{p(D)} \\
&\propto p(D\mid \theta) \cdot p(\theta) \\
\end{align}
$$

Remarks:

* $p(\theta)$ is the prior probability of $\theta$. It describes the distribution of $\theta$ before we observe any data.
* $p(D)$ is independent of $\theta$ and thus is simply a normalization constant.
* $p(\theta\mid D)$ is proportional to the joint distribution $p(D,\theta)$.

Intuition of MAP estimation: which $\theta$ is most probable given the data? Hence,

$$
\begin{align}
\hat\theta_\text{MAP}
&= \argmax_{\theta} p(\theta\mid D) \\
&= \argmax_{\theta} p(D\mid\theta) \cdot p(\theta) \\
\end{align}
$$

Again, for the sake of computation, we take the log on the RHS:

$$
\begin{align}
\hat\theta_\text{MAP}
&= \argmax_{\theta} \underbrace{\ln p(D\mid\theta)}_{\text{log likelihood}} + \underbrace{\ln p(\theta)}_{\text{log prior}} \\
&= \argmax_{\theta} \sum_{i=1}^n \ln p(\mathbf x_i \mid \theta) + \ln p(\theta)
\end{align}
$$

If the objective on the RHS is convex, we can compute $\hat\theta_\text{MAP}$ analytically by solving

$$
\begin{align}
\frac{\partial}{\partial\theta}\ln p(D\mid\theta) + \frac{\partial}{\partial\theta}\ln p(\theta) &= 0 \\
\sum_{i=1}^n \frac{\partial}{\partial\theta} \ln p(\mathbf x_i \mid \theta) + \frac{\partial}{\partial\theta}\ln p(\theta) &= 0 \\
\end{align}
$$

Gradient-based optimization:

* Gradient ascent:
    $$
    \begin{align}
    \theta^{(t+1)}
    &= \theta^{(t)} + \eta^{(t)} \left(\sum_{i=1}^n \frac{\partial}{\partial\theta} \ln p(\mathbf x_i \mid \theta^{(t)}) + \frac{\partial}{\partial\theta}\ln p(\theta^{(t)}) \right)\\
    \end{align}
    $$

* Mini-bath gradient ascent:
    $$
    \begin{align}
    & \text{randomly draw } B^{(t)} \subset D \text{ with } \vert B \vert = m
    \\
    & \theta^{(t+1)}
    = \theta^{(t)} + \eta^{(t)} \left(\sum_{\mathbf x\in B^{(t)}} \frac{\partial}{\partial\theta} \ln p(\mathbf x \mid \theta^{(t)}) + \frac{\partial}{\partial\theta}\ln p(\theta^{(t)}) \right)
    \end{align}
    $$

* Stochastic gradient ascent:
    $$
    \begin{align}
    & \text{randomly draw } \mathbf x^{(t)} \in D
    \\
    & \theta^{(t+1)}
    = \theta^{(t)} + \eta^{(t)} \left(\frac{\partial}{\partial\theta} \ln p(\mathbf x^{(t)} \mid \theta^{(t)}) + \frac{\partial}{\partial\theta}\ln p(\theta^{(t)}) \right)
    \end{align}
    $$

### Example

## Superivsed Learning

Problem formulation:

* Given: training data $D = \{\mathbf x_1, y_1, \cdots, \mathbf x_n, y_n\}$ iid from $p(\mathbf x, y \mid \theta)$ with unknown $\theta$
* Goal: estimate $\theta$

Additional assumption: The parameterized joint distribution can be factorized either as

$$
\begin{align}
p(\mathbf x, y \mid \theta) &= p(\mathbf x \mid \theta_{\mathbf x}) \cdot p(y \mid \mathbf x,\theta_y) \\
\theta &= (\theta_{\mathbf x}, \theta_y)
\end{align}
$$

or as

$$
\begin{align}
p(\mathbf x, y \mid \theta) &= p(y \mid \theta_y) \cdot p(\mathbf x\mid y,\theta_{\mathbf x}) \\
\theta &= (\theta_{\mathbf x}, \theta_y)
\end{align}
$$

Discriminative model or generative model?

Use discriminative model if

* we focus on predicting labels
* there is no inherent distribution $p(\mathbf x\mid y)$

Use generative model if

* we focus on modeling how data is generated

Examples:

1. predicting the price of a house given its area $\to$ discriminative model.
1. predicting whether a dish is healthy given its ingradients $\to$ discriminative model.
1. predicting which number given a hand-written digit $\to$ generative model.
1. predicting which species given the weight and size of an animal $\to$ generative model.

### MLE for Joint Distribution

The likelihood $p(D\mid\theta)$ is by iid assumption

$$
\begin{align}
p(D\mid\theta)
&= p(\mathbf x_1, y_1, \cdots, \mathbf x_n, y_n \mid \theta) \\
&= \prod_{i=1}^n p(\mathbf x_i,y_i \mid \theta)
\end{align}
$$

By assumption $p(\mathbf x, y \mid \theta) = p(\mathbf x \mid \theta_{\mathbf x}) \cdot p(y \mid \mathbf x,\theta_y)$, the likelihood becomes

$$
\begin{align}
p(D\mid\theta)
&= \prod_{i=1}^n p(\mathbf x_i \mid \theta_{\mathbf x}) \cdot p(y_i \mid \mathbf x_i,\theta_y) \\
&= \left(\prod_{i=1}^n p(\mathbf x_i \mid \theta_{\mathbf x})\right) \cdot \left(\prod_{i=1}^n p(y_i \mid \mathbf x_i,\theta_y)\right)
\end{align}
$$

Taking the log, we get the log likelihood of $\theta$

$$
\begin{align}
\ln p(D\mid\theta)
&= \underbrace{\sum_{i=1}^n \ln p(\mathbf x_i \mid \theta_{\mathbf x})}_{J_1(\theta_{\mathbf x})} + \underbrace{\sum_{i=1}^n \ln p(y_i \mid \mathbf x_i,\theta_y)}_{J_2(\theta_y)}
\end{align}
$$

Remarks:

* $J_1(\theta_{\mathbf x})$ is in fact the log likelihood of $\mathbf x_1, \dots, \mathbf x_n$ given $\theta_{\mathbf x}$ as
    $$
    \begin{align}
    \ln p(\mathbf x_1, \dots, \mathbf x_n \mid\theta_{\mathbf x})
    &= \ln \prod_{i=1}^n p(\mathbf x_i\mid\theta_{\mathbf x}) \\
    &= \sum_{i=1}^n \ln p(\mathbf x_i \mid \theta_{\mathbf x}) \triangleq J_1(\theta_{\mathbf x})
    \end{align}
    $$

* $J_2(\theta_y)$ is the log of conditional likelihood of $y_1, \dots, y_n$ given $\mathbf x_1, \dots, \mathbf x_n$ and $\theta_y$ as
    $$
    \begin{align}
    \ln p(y_1, \dots, y_n \mid\mathbf x_1, \dots, \mathbf x_n, \theta_y)
    &= \ln \prod_{i=1}^n p(y_i \mid \mathbf x_i, \theta_y) \\
    &= \sum_{i=1}^n \ln p(y_i \mid \mathbf x_i, \theta_y) \\
    &\triangleq J_2(\theta_y)
    \end{align}
    $$

Hence, $\theta_{\mathbf x}$ and $\theta_y$ can be estimated separately

$$
\begin{align}
\hat \theta_{\mathbf x} &= \argmax_{\theta_{\mathbf x}} \sum_{i=1}^n \ln p(\mathbf x_i \mid \theta_{\mathbf x})
\\
\hat \theta_{y} &= \argmax_{\theta_y} \sum_{i=1}^n \ln p(y_i \mid \mathbf x_i,\theta_y)
\end{align}
$$

For regression problem, we are often interested in estimating $\theta_y$ only because $\theta_{\mathbf x}$ is irrelevant to predicting labels. i.e. We do not care how $\mathbf x$ is distributed. In fixed-design regression problems, $\mathbf x$ is even deterministic.

Example: predicting house prices. We model the area as a Gaussian $x\sim\mathcal N(\mu_x, \sigma_x^2)$ and the price $y$ as an affine function of $x$ with independent Gaussian noise.

$$
y = wx+b + \epsilon, \quad \epsilon\sim\mathcal N(0, \sigma^2)
$$

Here, the paramters are

$$
\theta_x = (\mu_x, \sigma_x^2), \quad \theta_y = (w, b)
$$

Given the training data $D= \{x_1, y_1, \cdots, x_n, y_n\}$ about the area and price of $n$ houses, how to estimate $\theta_x$ and $\theta_y$ with MLE?

For $\theta_x$: This is standard MLE in unspervised setting.

$$
\hat\mu_x = \frac{1}{n} \sum_{i=1}^n x_i ,
\quad
\hat\sigma_x^2 = \frac{1}{n} \sum_{i=1}^n ( x_i-\hat\mu_x )^2
$$

For $\theta_y$: We need to compute $p(y_i \mid \mathbf x_i,\theta_y)$. By assumption,

$$
\begin{align*}
p(y_i \mid \mathbf x_i,\theta_y)
&= p_\epsilon(y_i - (wx_i+b)) \\
&= \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(y_i - (wx_i+b))^2}{2\sigma^2} \right) \\
\end{align*}
$$

Hence, the log likelihood