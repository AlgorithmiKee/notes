---
title: "GMM"
date: "2024"
author: "Ke Zhang"
---

# Gaussian Mixture Model

[toc]

As motivation, consider the following practical example:

* There are 3 types of fish in a lake: bream, salmon, carp
* Each fish has the features: length (in cm) and weight (in gram)
* Among all fish in the lake, 30% of them are bream, 20% of them are salmon, 50% of them are carp.
* For each fish species, the length and weight are jointly Gaussian distributed. Obviously, different fish species have different average length and average weight.

Q1: Joint distribution of [length, weight] for an arbitrary fish in the lake?  
&nbsp;&nbsp;&nbsp;&nbsp; $\rightsquigarrow$ **Gaussian mixture Model (GMM)**

Q2: Given a fish whose feature is [length=26cm, weight=740g]. To which species is this fish most likely to belong?  
&nbsp;&nbsp;&nbsp;&nbsp; $\leadsto$ **Prediction using a GMM**

Suppose we do not know the model parameters (i.e. the percentage of each fish species and Gaussian parameters). Instead, we will learn these parameters from data. The challenge is that the training data are unlabeled.

Q3: Can we still estimate model parameters from unlabeled data?  
&nbsp;&nbsp;&nbsp;&nbsp; $\leadsto$ **Learning a GMM using EM algorithm**

## Probabilistic Model

Let $X \in\mathbb R^d$ be distributed according to GMM. The PDF of $X$ is
> $$
> \begin{align}
> p_X(x) &= \sum_{k=1}^K \pi_k \cdot \mathcal{N}(x \vert \mu_k, \Sigma_k) \\
> \text{where }& \pi_k \in[0, 1],\quad \sum_{k=1}^K \pi_k = 1
> \end{align}
> $$

We call


* $\mathcal{N}(x \vert \mu_k, \Sigma_k), \: k=1,\dots,K$  mixture components
* $\pi_k, \: k=1,\dots,K$  mixture weights

The GMM is fully characterized by its parameters
$$
\begin{align}
\theta =  \{ (\pi_k, \mu_k, \Sigma_k) \in(\mathbb R \times\mathbb R^d \times\mathbb R^{d\times d}) : k=1,\dots,K \}
\end{align}
$$

Practical interpretation of GMM parameters:

* $\pi_k$: weight of cluster (or class) $k$. e.g. The percentage of fish species k in the lake.
* $\mu_k$: center of cluster (or class) $k$. e.g. Mean[length, weight] of fish species k.
* $\Sigma_k$: spread of cluster (or class) $k$. e.g. Cov(length, weight) of fish species k.

In this section, we assume that all GMM parameters are known.

### Generative Modeling Perspective

Introduce a latent random variable $Z \in\{1,\dots, K \}$ with categorical distribution, parameterized by $\pi_k$

$$
\begin{align}
p_Z(k) = \pi_k,\quad k=1,\dots, K
\end{align}
$$

Each cluster $k$ generates $X$ according to the Gaussian distribution, parameterized by $\mu_k, \Sigma_k$
$$
\begin{align}
p_{X \vert Z}(x\vert k) = \mathcal{N}(x \vert \mu_k, \Sigma_k)
\end{align}
$$

The joint distribution is obtained simply from

$$
\begin{align}
p_{XZ}(x,z) = p_Z(k) \cdot p_{X \vert Z}(x\vert k) = \pi_k \cdot \mathcal{N}(x \vert \mu_k, \Sigma_k)
\end{align}
$$

The GMM is interpreted as the marginal distribution of $X$
$$
\begin{align}
p_X(x)
&= \sum_{k=1}^K p_Z(k) \cdot p_{X \vert Z}(x\vert k)
\end{align}
$$

Intuition:

* View $Z$ as a class variable. $Z=k$ means that the feature $X$ is generated from class $k$.
* The component weight $\pi_k$ represents the prior probability $p_Z(k)$.
* The component mixture $\mathcal{N}(x \vert \mu_k, \Sigma_k)$ represents the likelihood $p_{X\vert Z}(x\vert k)$

Given $x\in\mathbb R^d$, the posterior probability of class $k$ is derived from Bayes rule
$$
\begin{align}
p_{Z\vert X}(k\vert x)
&=  \frac{p_Z(k) \cdot p_{X \vert Z}(x\vert k)}{p_X(x)} = \frac{\pi_k \cdot \mathcal{N}(x \vert \mu_k, \Sigma_k) }{\sum_{j=1}^K \pi_j \cdot \mathcal{N}(x \vert \mu_j, \Sigma_j)}
\end{align}
$$

Clearly, for any given $x\in\mathbb R^d$, the posterior probability sums to one.
$$
\begin{align}
\sum_{k=1}^K p_{Z\vert X}(k\vert x) = 1
\end{align}
$$

The posterior probability is useful for clustering. Detailed later.

### Mixture Mean and Mixture Variance

Given a fully characterized GMM,

* The mixture mean = Weighted average of component means
  $$
  \begin{align}
    \mu_X = \sum_{k=1}^K \pi_k \mu_k
  \end{align}
  $$

* The mixture variance = Weighted average of component variance + Weighted average of squared distances of the component means from the mixture mean.
  $$
  \begin{align}
    \Sigma_X = \sum_{k=1}^K \pi_k \Sigma_k +  \sum_{k=1}^K \pi_k (\mu_k - \mu_X)(\mu_k - \mu_X)^\top
  \end{align}
  $$

*Proof*: This follows directly from the law of total expectation.

For the mixture mean, the law of total mean holds
$$
\begin{align*}
\mathbb E_X[X]
&= \mathbb E_Z [\mathbb E_X[X \vert Z]] \\
&= \sum_{k=1}^K \underbrace{p_Z(k)}_{\pi_k} \cdot \underbrace{\mathbb E_X[X \vert k]}_{\mu_k} \\
\end{align*}
$$

For the mixture variance, the law of total variance holds
$$
\begin{align*}
\mathbb V_X[X]
= \mathbb E_Z[ \mathbb V_X[X \vert Z]] + \mathbb V_Z[\mathbb E_X[ X \vert Z]]
\end{align*}
$$

where we consider $\mathbb V_X[ X \vert Z]$ and $\mathbb E_X[ X \vert Z]$ as random var dependent of $Z$. Hence,

$$
\begin{align*}
\mathbb E_Z[ \mathbb V_X[X \vert Z]]
&=\sum_{k=1}^K \underbrace{p_Z(k)}_{\pi_k} \cdot \underbrace{\mathbb V_X[X \vert k]}_{\Sigma_k}
\\
\mathbb V_Z[\mathbb E_X[ X \vert Z]]
&= \mathbb E_Z \!\left[
  \Bigl( \mathbb E_X[ X \vert Z] - \underbrace{\mathbb E_Z[\mathbb E_X[ X \vert Z]]}_{\mu_X}  \Bigr)
  \Bigl( \mathbb E_X[ X \vert Z] - \underbrace{\mathbb E_Z[\mathbb E_X[ X \vert Z]]}_{\mu_X}  \Bigl)^\top
  \right]
\\
&= \sum_{k=1}^K \underbrace{p_Z(k)}_{\pi_k}
\left( \underbrace{\mathbb E_X[X \vert k]}_{\mu_k} - \mu_X \right)
\left( \underbrace{\mathbb E_X[X \vert k]}_{\mu_k} - \mu_X \right)^\top
\end{align*}
$$

## Using a GMM

Suppose we trained a GMM. What can we do with it?

### Clustering

Given a new data point $x^\text{new}$, we can predict which cluster it belongs to by maximizing the posterior probability $p_{Z\vert X}(k\vert x^\text{new}) $.

$$
\DeclareMathOperator*{\argmax}{arg\,max\:}
\begin{align}
\hat k^\text{new} = \argmax_k
\frac{\pi_k \cdot \mathcal{N}(x^\text{new} \vert \mu_k, \Sigma_k) }
  {\sum_{j=1}^K \pi_j \cdot \mathcal{N}(x^\text{new} \vert \mu_j, \Sigma_j)}
\end{align}
$$

Note that the denominator does not depend on $k$. Hence, it suffices to maximize the joint probability
$$
\begin{align}
\hat k^\text{new} = \argmax_k \: \pi_k \cdot \mathcal{N}(x^\text{new} \vert \mu_k, \Sigma_k) \\
\end{align}
$$

## Learning a GMM

Now, assume the parameters of the GMM are unknown. Our aim is to estimate the parameters of the GMM using maximum likelihood estimation.

Learning a GMM can be formalized as follows

* Given: dataset $x_1,\dots, x_N \in \mathbb R^d$
* Model: $p(x\vert\theta) = \sum_{k=1}^K \pi_k \cdot \mathcal{N}(x \vert \mu_k, \Sigma_k)$ with $\theta =  \{ (\pi_k, \mu_k, \Sigma_k): k=1,\dots,K \}$
* Assume:  $x_1,\dots, x_N \sim \text{i.i.d. } p(x\vert\theta)$
* Goal:  Estimate $\theta$.  

> Key Insight: The following estimation problem are equivalent
>
> * Estimate $\theta$ 
> * Estimate the weight, center, and spread of each cluster
> * Estimate ==the joint distribution== $p_{XZ}(x,k) = \pi_k \cdot \mathcal{N}(x \vert \mu_k, \Sigma_k)$

### Log-Likelihood

Due to the iid assumption, the likelihood is
$$
\begin{align}
p(x_1,\dots, x_N\vert \theta)
&= \prod_{n=1}^N p(x_n \vert\theta) \\
\text{where}\quad p(x_n \vert\theta)
&=  \sum_{k=1}^K \pi_k \cdot \mathcal{N}(x_n \vert \mu_k, \Sigma_k)
\end{align}
$$

The Log-likelihood (LLH) is thus
$$
\begin{align}
\ln p(x_1,\dots, x_N\vert \theta)
&= \sum_{n=1}^N \ln p(x_n \vert\theta)  \\
&= \sum_{n=1}^N \ln \left( \sum_{k=1}^K \pi_k \cdot \mathcal{N}(x_n \vert \mu_k, \Sigma_k) \right)  \\
\end{align}
$$

### Optimization Problem

We would like to maximize the log-likelihood subject to the constraint that all mixture weights sums up to one.

> $$
> \begin{align}
> \max_{\pi_{1:K},\, \mu_{1:K},\, \Sigma_{1:K}}\quad
> & \sum_{n=1}^N \ln \left( \sum_{k=1}^K \pi_k \cdot \mathcal{N}(x_n \vert \mu_k, \Sigma_k) \right) \\
> \text{s.t. }\quad  & \pi_1+ \cdots + \pi_K = 1 \\
>             \quad  & \pi_1, \dots ,\pi_K \in\mathbb [0,1] \nonumber \\
>             \quad  & \mu_1, \dots ,\mu_K \in\mathbb R^{d} \nonumber \\
>             \quad  & \Sigma_1, \dots ,\Sigma_K \in\mathbb R^{d\times d} \nonumber \\
> \end{align}
> $$

This is constrained optimization problem. We need the Lagrangian

$$
\begin{align}
L &\triangleq \sum_{n=1}^N \ln p(x_n \vert\theta)  + \lambda \left( \sum_{k=1}^K \pi_k -1 \right)
\end{align}
$$

As a necessary condition, the optimal parameters should make gradients of $L$ to zero
$$
\begin{align}
\frac{\partial}{\partial\mu_k} L
&= \sum_{n=1}^N  \frac{\partial}{\partial\mu_k} \ln p(x_n \vert\theta)
= \sum_{n=1}^N \frac{1}{p(x_n \vert\theta) } \cdot \frac{\partial}{\partial\mu_k} p(x_n \vert\theta) = 0 \\
\frac{\partial}{\partial\Sigma_k} L
&= \sum_{n=1}^N  \frac{\partial}{\partial\Sigma_k} \ln p(x_n \vert\theta)
= \sum_{n=1}^N \frac{1}{p(x_n \vert\theta) } \cdot \frac{\partial}{\partial\Sigma_k} p(x_n \vert\theta) = 0 \\
\frac{\partial}{\partial\pi_k} L
&= \sum_{n=1}^N  \frac{\partial}{\partial\pi_k} \ln p(x_n \vert\theta) +\lambda
= \sum_{n=1}^N \frac{1}{p(x_n \vert\theta) } \cdot \frac{\partial}{\partial\pi_k} p(x_n \vert\theta) +\lambda = 0 \\
\end{align}
$$

However, we will see that it is impossible to derive an analytical solution to above equations due to the complexity of GMM. Instead, we derive iterative update rules for $\mu_k$,  $\Sigma_k$ and $\pi_k$​. The central idea behind the iterative update is application of fixed-point iteration.

### Responsibility

During the derivation, we often see the term

> $$
> \begin{align}
> p(Z_n = k \vert x_n, \theta)
> = \frac{\pi_k \cdot \mathcal{N}(x_n \vert \mu_k, \Sigma_k)}{p(x_n \vert\theta)}
> = \frac{\pi_k \cdot \mathcal{N}(x_n \vert \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \cdot\mathcal{N}(x_n \vert \mu_j, \Sigma_j)} \\
> \end{align}
> $$

also known as the *responsibility* of class $k$ for data point $x_n$. It represents the posterior probability of class $k$ for given data point $x_n$. Later, we will use the short-hand notation $p(k \vert x_n, \theta)$.

### Update Rule for Component Means

The gradient of $L$ w.r.t. component mean $\mu_k$ is
$$
\begin{align*}
\frac{\partial}{\partial\mu_k} L
&= \sum_{n=1}^N \frac{1}{p(x_n \vert\theta) } \cdot \frac{\partial}{\partial\mu_k} p(x_n \vert\theta)
\end{align*}
$$

where  
$$
\begin{align*}
\frac{\partial}{\partial\mu_k} p(x_n \vert\theta)
&= \frac{\partial}{\partial\mu_k} \sum_{j=1}^K \pi_j  \mathcal{N}(x_n \vert \mu_j, \Sigma_j) \\
&= \frac{\partial}{\partial\mu_k} \pi_k  \mathcal{N}(x_n \vert \mu_k, \Sigma_k) \\
&=  \pi_k \mathcal{N}(x_n \vert \mu_k, \Sigma_k) \cdot (x_n - \mu_k)^\top \Sigma_k^{-1}
\end{align*}
$$

Hence,  
$$
\begin{align*}
\frac{\partial}{\partial\mu_k} L
&= \sum_{n=1}^N \frac{1}{p(x_n \vert\theta) } \cdot \left[ \pi_k \mathcal{N}(x_n \vert \mu_k, \Sigma_k) \cdot (x_n - \mu_k)^\top \Sigma_k^{-1} \right] \\
&= \sum_{n=1}^N \underbrace{\frac{ \pi_k\mathcal{N}(x_n \vert \mu_k, \Sigma_k)}{p(x_n \vert\theta)}}_{p(k \vert x_n, \theta)} \cdot (x_n - \mu_k)^\top \Sigma_k^{-1} \\
&= \sum_{n=1}^N p(k \vert x_n, \theta) \cdot (x_n - \mu_k)^\top \cdot \Sigma_k^{-1} \\
\end{align*}
$$

Letting $\frac{\partial L}{\partial\mu_k} = 0$, we get (recall: The covariance matrix $\Sigma_k$ is invertible)
$$
 \sum_{n=1}^N p(k \vert x_n, \theta) \cdot  \mu_k =\sum_{n=1}^N p(k \vert x_n, \theta) \cdot x_n
$$

Note that the RHS also depends on $\mu_k$, even in a very complex way. Therefore, we can't simply derive $\mu_k^*$ which leads to zero gradient. However, we can easily reform the above equation in the form $\mu_k=h(\mu_k)$. Finding the solution to the original equation is now equivalent to finding the fixed point of $h(\cdot)$. Applying fixed point iteration, we get

$$
\begin{align}
\mu_k^\text{new} = \frac{\sum_{n=1}^N\: p(k \vert x_n, \theta^\text{old}) \cdot x_n}{\sum_{n=1}^N\: p(k \vert x_n, \theta^\text{old})}
\end{align}
$$

Remark:

* The RHS is just a weighted sum of the training data $x_1,\dots, x_N$
* The posterior probability $p(k \vert x_n, \theta)$ describes the contribution of $x_n$ to $\mu_k$.
* If $p(k \vert x_n, \theta)$ is large, $\mu_k$ will be pulled stronger towards $x_n$.

### Update Rule for Component Variances

The gradient of $L$ w.r.t. component variance $\Sigma_k$ is
$$
\begin{align*}
\frac{\partial}{\partial\Sigma_k} L  
= \sum_{n=1}^N \frac{1}{p(x_n \vert\theta) } \cdot \frac{\partial}{\partial\Sigma_k} p(x_n \vert\theta)
\end{align*}
$$

where  
$$
\begin{align*}
\frac{\partial}{\partial\Sigma_k} p(x_n \vert\theta)
&= \frac{\partial}{\partial\Sigma_k} \sum_{j=1}^K \pi_j  \mathcal{N}(x_n \vert \mu_j, \Sigma_j) \\
&= \frac{\partial}{\partial\Sigma_k} \pi_k \mathcal{N}(x_n \vert \mu_k, \Sigma_k) \\
&= \text{some lengthy derivation...} \\
&= -\frac{1}{2}\pi_k \mathcal{N}(x_n \vert \mu_k, \Sigma_k) \cdot \Sigma_k^{-1} \left[ \Sigma_k - \underbrace{(x_n-\mu_k)(x_n-\mu_k)^\top}_{\text{In short: } V(x_n \vert \mu_k)} \right] \Sigma_k^{-1}  \\

\end{align*}
$$

Hence,  
$$
\begin{align*}
\frac{\partial}{\partial\Sigma_k} L  
&= \sum_{n=1}^N \frac{1}{p(x_n \vert\theta) } \cdot \left(-\frac{1}{2} \pi_k \mathcal{N}(x_n \vert \mu_k, \Sigma_k) \cdot \Sigma_k^{-1} \left[\Sigma_k - V(x_n \vert \mu_k)\right] \Sigma_k^{-1} \right) \\
&= -\frac{1}{2} \sum_{n=1}^N \frac{\pi_k \mathcal{N}(x_n \vert \mu_k, \Sigma_k)}{p(x_n \vert\theta) }  \left(   \Sigma_k^{-1} \left[\Sigma_k - V(x_n \vert \mu_k)\right] \Sigma_k^{-1} \right) \\
&= -\frac{1}{2} \sum_{n=1}^N p(k \vert x_n, \theta)  \left(   \Sigma_k^{-1} \left[\Sigma_k - V(x_n \vert \mu_k)\right] \Sigma_k^{-1} \right) \\
&= -\frac{1}{2} \sum_{n=1}^N p(k \vert x_n, \theta) \Sigma_k^{-1}  + \frac{1}{2} \Sigma_k^{-1} \left[\sum_{n=1}^N  p(k \vert x_n, \theta) V(x_n \vert \mu_k) \right] \Sigma_k^{-1}  \\
\end{align*}
$$

Let $\frac{\partial L}{\partial\Sigma_k} = 0$ and use $V(x_n \vert \mu_k)=(x_n-\mu_k)(x_n-\mu_k)^\top$. We get
$$
\begin{align*}
\sum_{n=1}^N p(k \vert x_n, \theta) \Sigma_k^{-1}
&= \Sigma_k^{-1} \left[\sum_{n=1}^N p(k \vert x_n, \theta) V(x_n \vert \mu_k) \right] \Sigma_k^{-1}  
\\
\sum_{n=1}^N p(k \vert x_n, \theta) \Sigma_k
&=  \sum_{n=1}^N p(k \vert x_n, \theta) V(x_n \vert \mu_k)
\\
\sum_{n=1}^N p(k \vert x_n, \theta) \Sigma_k
&=  \sum_{n=1}^N p(k \vert x_n, \theta) (x_n-\mu_k)(x_n-\mu_k)^\top
\\
\end{align*}
$$

Once again, we can't completely factor out $\Sigma_k$ since the responsibility also depends on $\Sigma_k$ (in a complex way). Hence, we view $\theta$ in responsibilities as the old parameter estimate, which is used to update the estimate for $\Sigma_k$.
$$
\begin{align}
 \Sigma_k^\text{new}
=  \frac{\sum_{n=1}^N p(k \vert x_n, \theta^\text{old}) (x_n-\mu_k^\text{new})(x_n-\mu_k^\text{new})^\top}{\sum_{n=1}^N p(k \vert x_n, \theta^\text{old})}
\end{align}
$$

Remark:

* The $\mu_k$ on the RHS are updated version since we have already updated them.
* The matrix $(x_n-\mu_k)(x_n-\mu_k)^\top$ describes the spread (variance) between $x_n$ and $\mu_k$.
* The RHS is again a weighted average of the spread between $x_n$ and $\mu_k$.
* If $p(k \vert x_n, \theta)$ is large, $\Sigma_k$ will be pulled stronger towards $(x_n-\mu_k)(x_n-\mu_k)^\top$.

### Update Rule for Mixture Weights

The gradient of $L$ w.r.t. mixture weight $\pi_k$ is
$$
\begin{align*}
\frac{\partial}{\partial\pi_k} L =
\sum_{n=1}^N \frac{1}{p(x_n \vert\theta) } \cdot \frac{\partial}{\partial\pi_k} p(x_n \vert\theta) + \lambda
\end{align*}
$$

where  
$$
\begin{align*}
\frac{\partial}{\partial\pi_k} p(x_n \vert\theta)
&= \frac{\partial}{\partial\pi_k} \sum_{j=1}^K \pi_j \mathcal{N}(x_n \vert \mu_j, \Sigma_j) \\
&= \mathcal{N}(x_n \vert \mu_k, \Sigma_k) \\
\end{align*}
$$

Hence,  
$$
\begin{align*}
\frac{\partial}{\partial\pi_k} L
&= \sum_{n=1}^N \frac{\mathcal{N}(x_n \vert \mu_k, \Sigma_k)}{p(x_n \vert\theta) } + \lambda \\
&= \sum_{n=1}^N \frac{\pi_k\mathcal{N}(x_n \vert \mu_k, \Sigma_k)}{p(x_n \vert\theta)} \frac{1}{\pi_k}+ \lambda \\
&= \sum_{n=1}^N p(k\vert x_n,\theta) \frac{1}{\pi_k}+ \lambda \\
\end{align*}
$$

Let $\frac{\partial}{\partial\pi_k} L =0$ and $\frac{\partial}{\partial\lambda} L =0$. We get
$$
\begin{align*}
\forall k=1,\dots,K, \quad \pi_k &= -\frac{\sum_{n=1}^N p(k\vert x_n,\theta)}{\lambda}  \\
\pi_1  +\cdots + \pi_K &= 1
\end{align*}
$$

which yields the solution $\lambda = -N$ and update rule
$$
\begin{align}
\pi_k^\text{new} = \frac{\sum_{n=1}^N p(k\vert x_n,\theta^\text{old})}{N}
\end{align}
$$

Remark:

* The numerator describes the total responsibility of class $k$ for the entire data set
* If class $k$ has a large total responsibility, then it will get a large weight $\pi_k$

### Summary: EM for Training GMM

Combining the update rules for cluster weights, cluster center and cluster spreads, we get the EM algorithm

> * Initialize parameters $\pi^{(0)}_1 \dots \pi^{(0)}_K, \mu^{(0)}_1 \dots \mu^{(0)}_K,\Sigma^{(0)}_1 \dots \Sigma^{(0)}_K$
>
> * For $t=0,1,2\dots$ until convergence, do
>
>   1. E-step: calculate the posterior probability (i.e. responsibility) of class $k$ given $x_n$ based on current parameter estimate for $\forall  k=1\dots K$ and $\forall  n=1\dots N$
>   $$
>   \begin{align}
>     w_{kn}^{(t)}
>     & \triangleq p(k \vert x_n, \theta^{(t)})
>     = \frac{\pi_k^{(t)} \mathcal{N}(x_n \vert \mu_k^{(t)}, \Sigma_k^{(t)})}{\sum_{j=1}^K \mathcal{N}(x_n \vert \mu_j^{(t)}, \Sigma_j^{(t)})} \\
>     \end{align}
>   $$
>
>   2. M-step: update the parameter estimate according to $\forall k=1\dots K$
>   $$
>   \begin{align}
>     \pi_k^{(t+1)}  &= \frac{\sum_{n=1}^N w_{kn}^{(t)}}{N}     \\
>     \mu_k^{(t+1)} &= \frac{\sum_{n=1}^N\: w_{kn}^{(t)} \cdot x_n}{\sum_{n=1}^N\: w_{kn}^{(t)}}
>     \\
>     \Sigma_k^{(t+1)}
>     &= \frac{\sum_{n=1}^N w_{kn}^{(t)} \left(x_n-\mu_k^{(t+1)}\right) \! \left(x_n-\mu_k^{(t+1)}\right)^\top}{\sum_{n=1}^N w_{kn}^{(t)}}
>     \end{align}
>   $$
>

# Exercise

1. **Responsibility**:  Let $p(k\vert x_n,\theta)$ denote the responsibility in GMM. Show that
    $$
    \sum_{k=1}^K \sum_{n=1}^N p(k\vert x_n,\theta) = N
    $$

2. **GMM with Partial Information**: We studied the EM algorithm for a generic GMM. Now, we would like to simplify the EM algorithm under certain assumptions.

   * Case I: Simplify the EM algorithm for a GMM whose component means are known, denoted by $\mu_1^*,\dots,\mu_K^*$.

   * Case II: Simplify the EM algorithm for a GMM whose component covariance matrices are diagonal, i.e. $\Sigma_k=\operatorname{diag}(\sigma_{k1}^2, \dots, \sigma_{kd}^2)$. Here, $k$ denotes the cluster number, and $d$ denotes the dimensionality of the feature vector.
     
   * Case III: Simplify the EM algorithm for a GMM whose component covariance matrices all equal to $\sigma^2 I$ with unknown $\sigma^2$. Note: $I$ is the $d\times d$ identity matrix.
   
3. **Symmetric Noisy Channel**: The sender sends a binary symbol $S\in\{0, 1\}$ according to a Bernoulli distribution. The probability of sending a $1$ or a $0$​ is unknown. The symbol goes through a symmetric noisy channel and arrives at the receiver.  
   $$
   X = S+W, \quad W \sim\mathcal{N}(0, \sigma^2)
   $$
   where the noise $W$ is statistically independent of $S$. The variance $\sigma^2$ is also unknown.

   * Formulate the PDF of $X$ in the form of GMM.
   * Identify all parameters of $p_X$.
   * You are given a data set $x_1,\dots,x_N$ which is i.i.d. sampled from $p_X$. Derive the EM algorithm for estimating the parameters of $p_X$​.
   * Suppose you trained your GMM on some data set. After training, you discovered that the sender sends 1s and 0s with probabilities of 0.3 and 0.7 respectively, and that the variance $\sigma^2=0.25$. Calculate the decision boundary for classifying a newly received data point.

# Appendix

## Fixed-Point Iteration

Suppose we would like to solve the equation for a function $f:\mathbb R^d \to \mathbb R^d$
$$
f(x) = 0
$$

In general, an analytical solution is not guaranteed to exist. Instead, we use fixed point iteration to obtain a numerical solution. For that, we reform the equation in the form
$$
g(x) = x
$$

Then,
$$
\text{Finding solutions of } f(x)=0 \iff \text{Finding fixed points of } g(\cdot)
$$
We can apply fixed point iteration over $g(\cdot)$ to solve the original equation.

## Law of Total Expectation

### Law of Total Mean

$$
\begin{align}
\mathbb E_Y[Y] = \mathbb E_X[ \mathbb E_Y[Y \vert X]]
\end{align}
$$

Note:

* $ \mathbb E_Y[Y \vert X]$ is a random variable dependent on $X$.
* For an instance $x$, $ \mathbb E_Y[Y \vert x]$ is just a deterministic number.

*Proof*: This is a direct result from the law of total probability

For fixed instance $x$, the conditional mean is
$$
\begin{align*}
\mathbb E_Y[Y\vert x] = \int y\cdot p(y\vert x) \:dy
\end{align*}
$$

Now, for random $X$, $\mathbb E_Y[Y\vert X]$ is a random variable dependent on $X$.
$$
\begin{align*}
\mathbb E_X[\mathbb E_Y[Y\vert X]]
&= \int \mathbb E_Y[Y\vert x]\cdot p(x) \:dx \\
&= \int \left( \int y\cdot p(y\vert x) \:dy \right)\cdot p(x) \:dx \\
&= \int \int y\cdot p(x, y) \:dx\:dy  &&\text{joint prob}\\
&= \int y  \left(\int  p(x, y) \:dx \right)\:dy \\
&= \int y \cdot p(y)\:dy  &&\text{marginal prob}\\
&= \mathbb E_Y[Y]
\end{align*}
$$

### Law of Total Variance

$$
\begin{align}
\mathbb V_Y[Y] = \mathbb E_X[ \mathbb V_Y[Y \vert X]] + \mathbb V_X[\mathbb E_Y[ Y \vert X]]
\end{align}
$$

Note:

* $ \mathbb V_Y[Y \vert X]$ is a random variable depdent on $X$.
* For an instance $x$, $ \mathbb V_Y[Y \vert x]$ is just a deterministic number.

*Proof*: Express the variance in 2nd order moments and use the law of total mean
$$
\begin{align*}
\mathbb V_Y[Y]
&= \mathbb E_Y[YY^\top] - \mathbb E_Y[Y]\cdot\mathbb E_Y[Y]^\top
\\
&= \mathbb E_X[\mathbb E_Y[YY^\top \vert X]] -  
\mathbb E_X[ \mathbb E_Y[Y \vert X]]\cdot \mathbb E_X[ \mathbb E_Y[Y \vert X]]^\top
\\
&= \mathbb E_X[\mathbb V_Y[Y \vert X] + \mathbb E_Y[Y \vert X]\cdot\mathbb E_Y[Y \vert X]^\top]-
\mathbb E_X[ \mathbb E_Y[Y \vert X]]\cdot \mathbb E_X[ \mathbb E_Y[Y \vert X]]^\top
\\
&= \mathbb E_X[\mathbb V_Y[Y \vert X]] + \underbrace{
\mathbb E_X[\mathbb E_Y[Y \vert X]\cdot\mathbb E_Y[Y \vert X]^\top]-
\mathbb E_X[ \mathbb E_Y[Y \vert X]]\cdot \mathbb E_X[ \mathbb E_Y[Y \vert X]]^\top}_{\mathbb V_X[\mathbb E_Y[Y \vert X]]}
\end{align*}
$$
