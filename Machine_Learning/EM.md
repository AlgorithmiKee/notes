# Expecation Maximization

[toc]

## Problem formulation

* Given: observations $x_1,\dots, x_N \in\mathbb R^d$​

* Model: $p(x,z \vert \theta)$ where $Z$ is either latent variable or missing variable
* Assume: $x_1,\dots, x_N \sim \text{i.i.d. } p(x \vert\theta) = \int p(x,z \vert\theta) \:dz$​
* Goal: Iteratively estimate $\theta$​

Notations:

* Observed data: $\mathbf X = [x_1 \dots x_N] \in\mathbb R^{d \times N}$​
* Latent variable: $\mathbf Z = [z_1 \dots z_N] \in\mathbb R^{l \times N}$​
* Joint denstity of a single data point: $p(x_n, z_n \vert \theta)$​
* Joint denstity of the complete data: $p(\mathbf X, \mathbf Z \vert \theta)$

If we had the label, ...

Challenge of missing data: ...

## EM Algorithm

> Expecation Maximization Algorithm
>
> 1. E-step: Calculate expected complete-data LLH w.r.t. posterior distribution based on current parameter estimate
>    $$
>    \begin{align}
>    Q(\theta \:\vert\: \theta^{(t)}) = \mathbb E_{\mathbf Z} \left[ \ln p(\mathbf X, \mathbf Z \vert \theta) \right], \quad \mathbf Z \sim p(\cdot \vert \mathbf X, \theta^{(t)})
>    \end{align}
>    $$
>
> 2. M-step:
>    $$
>    \begin{align}
>    \DeclareMathOperator*{\argmax}{arg\,max\:}
>    \theta^{(t+1)} = \argmax_{\theta} Q(\theta \:\vert\: \theta^{(t)})
>    \end{align}
>    $$

## Application to GMM

Recall: $Z_n = k \iff$ The data point $X_n$ is generated from the $k$-th mixture component.

The complete data log-likelihood is due to the iid assumption
$$
\begin{align*}
\ln p(x_1\dots x_N,z_1\dots z_N \vert\theta)
&= \ln \prod_{n=1}^N p(x_n, z_n \vert\theta) \\
&= \sum_{n=1}^N \ln p(x_n, z_n \vert\theta) \\

\end{align*}
$$
The posterior distribution based on $\theta^{(t)}$ is
$$
\begin{align*}
p(z_1 \dots z_N \vert x_1 \dots x_N ,\theta^{(t)})
&= \prod_{n=1}^N p(z_n \vert x_n, \theta^{(t)})
\end{align*}
$$
In particular
$$
p(z_n \vert x_1 \dots x_N ,\theta^{(t)}) = p(z_n \vert x_n ,\theta^{(t)})
$$
E-step:
$$
\begin{align*}
Q(\theta \:\vert\: \theta^{(t)})
&= \mathbb E_{Z_1\dots Z_N} \left[ \ln p(x_1\dots x_N,z_1\dots z_N \vert\theta) \right] \\
&= \mathbb E_{Z_1\dots Z_N} \left[ \sum_{n=1}^N \ln p(x_n, z_n \vert\theta)  \right] \\
&= \sum_{n=1}^N \mathbb E_{Z_1\dots Z_N} \left[ \ln p(x_n, z_n \vert\theta)  \right] \\
&= \sum_{n=1}^N \mathbb E_{Z_n} \left[ \ln p(x_n, z_n \vert\theta)  \right] \\
&= \sum_{n=1}^N \sum_{z_n = 1}^K p(z_n \vert x_n,\theta^{(t)}) \ln p(x_n, z_n \vert\theta)  \\
&= \sum_{n=1}^N \sum_{z_n = 1}^K p(z_n \vert x_n,\theta^{(t)}) \ln p(z_n \vert\theta) p(x_n\vert z_n, \theta)  \\
\end{align*}
$$
Instead of letting $z_n$ denote the instance of $Z_n$, we use $k$ since $Z_n \in\{1 \dots K \}$ anyway.
$$
\begin{align}
Q(\theta \:\vert\: \theta^{(t)})
&= \sum_{n=1}^N \sum_{k = 1}^K p(k \vert x_n,\theta^{(t)}) \cdot \ln \underbrace{p(k \vert\theta)}_{\pi_k} \underbrace{p(x_n\vert k, \theta)}_{\mathcal{N}(x_n\vert \mu_k,\Sigma_k,\theta)}  \\
&= \sum_{n=1}^N \sum_{k = 1}^K p(k \vert x_n,\theta^{(t)}) \cdot \ln \pi_k \mathcal{N}(x_n\vert \mu_k,\Sigma_k,\theta)  \\
\end{align}
$$

# Exercise

TBD

# Appendix

TBD
