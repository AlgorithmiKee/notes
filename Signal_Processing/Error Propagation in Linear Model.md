# Error Probagation in Linear Estimator Transform

## Notions

| Symbol                                                       | Meaning                                                      |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| $X:\Omega \to \mathcal X, \omega\mapsto x$                   | Random variable (for $\mathcal X=\mathbb R$)or random vector ($\mathcal X = \mathbb R^d$) with parameterised PDF $p(x;\theta)$ |
| $(X_1, \cdots, X_n ):\Omega\to \mathcal X^{n}, \omega\mapsto (x_1, \cdots, x_n)$ | Random sample                                                |
| $\theta\in\mathbb R^p$                                       | Unknown true parameter vector                                |
| $g: \mathbb R^{d\times n}  \to \mathbb R^p, \: (x_1, \cdots, x_n)\mapsto \hat\theta$ | Estimation function. $\hat\theta$ is called the estimate of $\theta$ |
| $\hat\Theta=g\circ (X_1, \cdots, X_p ) : \Omega \to \mathbb R^n, \omega\mapsto\hat\theta$ | Estimator. Note: it is a **statistic**                       |
| $b, \: b_{\hat\Theta}$ or $\bias(\hat\Theta, \theta)$        | Bias of $\hat\Theta$, defined by $b=\E[\hat\Theta - \theta]$ |
| $C, \: C_{\hat\Theta}$ or $C( \hat\Theta, \theta )$          | Variance of $\hat\Theta$, defined by $C=\E\left[(\hat\Theta-\E\hat\Theta) (\hat\Theta-\E\hat\Theta) ^{\top}\right]$ |
| $\mse, \: \mse_{\hat\Theta}$ or $\mse( \hat\Theta, \theta )$ | MSE of $\hat\Theta$, defined by $\mse=\E\left[ \norm{\hat\Theta-\theta}^2 \right]$ |

Note: The expcetation is taken over the distribution of the random sample

## Error Propagation

Consider the linear transform of parameters
$$
\DeclareMathOperator{\bias} {bias}
\DeclareMathOperator{\var}  {Var}
\DeclareMathOperator{\mse}  {MSE}
\DeclareMathOperator{\E} {\mathbb{E}}

\newcommand\norm[1]{\left\lVert#1\right\rVert}
\newcommand\iprod[2]{\left\langle#1, #2\right\rangle}
\newcommand\argmax[1] {\mathrm{argmax}\left( #1\right)}

\xi = A\theta, \quad \mathrm{where} \: \theta\in\mathbb R^p,\: \xi\in\mathbb R^q
$$

Let

* $\hat\Theta$ be an esitmator of $\theta$  with bias $b$, variance $C$ and mean squared error $\mse$. 
* $\hat\Xi=A\hat\Theta$ be an esitmator of $\xi$ with bias $\tilde b$ , variance $\tilde C$ and mean squared error $\widetilde \mse$. 

Then,

> The error propagation is 
>
> * $\tilde b = Ab$
> * $\tilde C = A\cdot C\cdot A^\top$
> * $\widetilde{\mse} = \tr(A^\top A C) + \norm{Ab}_2^2$ 

Note: There is no explicit equality between $\widetilde \mse$ and $\mse$ in general. However, if $A$ is orthogonal, it is clear that $\widetilde\mse =\mse$

*Proof: Bias Propagation*
$$
\begin{align*}
\tilde b
&= \E[\hat\Xi] -  \xi \\
&= \E[ A\hat\Theta] -  A\theta \\
&=  A\cdot \left( \E[ \hat\Theta] -  \theta \right) \\
&=  Ab
\end{align*}
$$
*Proof: Variance Propagation* 
$$
\begin{align*}
\tilde C
&= \E\left[ ( \hat\Xi -  \xi) ( \hat\Xi -  \xi)^\top \right] \\
&= \E\left[ ( A\hat\Theta -  A\theta) ( A\hat\Theta -  A\theta)^\top \right] \\
&= \E\left[ A( \hat\Theta -  \theta) ( \hat\Theta -  \theta)^\top A^\top \right] \\
&= A\E\left[ ( \hat\Theta -  \theta) ( \hat\Theta -  \theta)^\top  \right]A^\top \\
&= A \cdot C \cdot A^\top \\


\end{align*}
$$


## An Upper Bound for Propogated MSE

Although there is no explicit equality between between $\widetilde \mse$ and $\mse$ in general, we can still upper-bound  $\widetilde\mse$ with $\mse.$ 

>$$
>\widetilde \mse \le \norm{A}_2^2 \mse
>$$
>
>with equality iff all following conditions are true
>
>* $q\ge p$ 
>* All singular values of $A$ are identical.
>* $\hat\Theta$ is unbiased **OR** $b$ is an eigenvector of $A^\top A$ corresponding to its largest eigenvalue

*Proof of the upper bound*

The key is use the SVD of $A$. Suppose $A$ has SVD
$$
A = U\Sigma V^\top
$$
with $K=\min(p, q)$  singular values $\sigma_1 \ge \cdots\ge \sigma_K \ge 0$

Recall the matrix 2-norm is defined as
$$
\norm{A}_2 = \sqrt{\lambda_{\max}(A^\top A)} = \sigma_{\max}(A)
$$

* For the variance part,
  $$
  \DeclareMathOperator\diag{diag}
  \begin{align*}
  \tr(A^\top A C) 
  &= \tr(V\Sigma^\top U^\top \cdot U\Sigma V^\top \cdot C) \\
  &= \tr(V\Sigma^\top\Sigma V^\top C) \\
  &= \tr(\Sigma^\top\Sigma \cdot V^\top CV) & \text{cyclic property of trace} \\
  &= \sum_{i=1}^K \sigma_i^2\left[ V^\top CV \right]_{ii} 
  	&\text
  	{$ 
  			[\Sigma^\top\Sigma]_{1:K,\, 1:K} = \diag(\sigma_1^2,\cdots, \sigma_K^2)
  	$} \\
  &\le \sigma_1^2 \sum_{i=1}^K \left[ V^\top CV \right]_{ii} 
  	&\text{eq $\iff \sigma_1= \cdots =\sigma_K$ } \\
  &\le \sigma_1^2 \tr \left( V^\top CV \right)
  	&\text{eq $\iff p=K=\min(p, q) \iff p\le q$} \\
  &= \norm{A}_2^2 \tr(VV^\top C) \\
  &= \norm{A}_2^2 \tr(C)
  
  \end{align*}
  $$

* For the bias part,
  $$
  \begin{align*}
  \norm{Ab}_2^2 \le \norm{A}_2 \norm{b}_2^2
  \end{align*}
  $$
  Here, equality $\iff$ $b$ is either

  * the eigenvector of $A^\top A$ corresponding to $\sigma_1^2$. 
  * 0 vector i.e. $\hat\Theta$ is unbiased

# Error Propagation in Kalman Filter

Consider the state space model
$$
\begin{align}
X_t &=  AX_{t-1} +U_t,  &U_t \sim \mathcal{N}(0, Q)\\
Y_t &=  AX_t +V_t,  &V_t \sim \mathcal{N}(0, R)
\end{align}
$$
where $$X\in\mathbb R^n$$ and $$ Y\in\mathbb R^m$$ are real-valued Gaussian random vectors. 

Given the estimator $\hat X_{t-1|t-1}$ for $\E[X_{t-1}]$ with bias $e_{t-1|t-1}$ and variance $\Sigma_{t-1|t-1}$. We obtain a prior estimate of $\E[X_t]$ from the model evolution:
$$
\hat X_{t|t-1} = A \hat X_{t-1|t-1}
$$
Hence, the error propagation from $\hat X_{t-1|t-1}$ to  $\hat X_{t|t-1}$ is quantified as
$$
\begin{align}
 e_{t|t-1} &= 
\end{align}
$$




# Appendix

## Trace as Linear Product

The **Frobenius norm** of a matrix $A\in\mathbb R^{m\times n}$ is given by $\norm{A}_F = \sum_{i=1}^m\sum_j^n A_{ij}^2$ . 

The trace of a product defines an inner product over the matrix space. i.e.
$$
\iprod{A}{B}_F = \tr(A^\top B)
$$
The induced norm of $\iprod{\cdot}{\cdot}_F$ is Frobenius norm. i.e.
$$
\iprod{A}{A}_F = \tr(A^\top A) = \norm{A}_F^2 
$$
The Cauchy-Scharz inequality boils down to
$$
\tr(A^TB) \le \norm{A}_F \norm{B}_F
$$

## Matrix 2-Norm

The 2-norm a matrix $A\in\mathbb R^{m\times n}$is defined by
$$
\norm{A}_2 = \sup_{x\in\mathbb R^n, \, x\ne 0} \frac{\norm{Ax}_2}{\norm{x}_2}
$$

> $$
> \norm{A}_2 = \sqrt{\lambda_{\max}(A^\top A)} = \sigma_{\max}(A) 
> $$

*Proof* 



To determine which $x$ achives the matrix norm, we have the theorem

> $$
> x^* = \argmax{ \frac{\norm{Ax}_2}{\norm{x}_2} } \iff x^* \text{is an eigenvector of $A^\top A$ corresponding to $\lambda_{\max}(A^\top A)$}
> $$

*Proof* 













