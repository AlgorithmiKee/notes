# Principle Components Analysis

## Motivation

PCA is about dimension reduction. It maps higher dimentional data to their lower dimensional representaitons. The motivations for dimension reduction are

* Sometimes, there are more features than the number of samples. e.g. The pixels per image often exceeds the number of images in a trainig set. 
* Higher dimensional data are often correlated. An extreme example is a feature set containing both date of birth and age. There is no need to keep both as features.
* Higher dimenional data requires more complex machine learning models (models with more parameters), which suffer from overfitting.

## Basic Idea

To be a bit more formal, given a data set $x_1, \cdots,x_n\in\mathbb R^p$. We would like to find an affine space $a+U$ with $a\in\mathbb R^p$ and $U\subset V, \: \dim U=d<p$ such that 

1. The projections of the data onto $a+U$ are (on average) as close to the original as possible.
2. The projections of the data onto $a+U$ retain as much variance as possible.

Note that the data set is in general not centered. However, we will show that we can assume without loss of generality that the data are centered for further discussion. Let

* $\mu=\frac{1}{n}\sum_{k=1}^n$ be the sample mean
* $x_k' = x_k -\mu, \: \forall k=1,\cdots, n$ be centered data
* $P_W$ be the orthogonal projection onto $W$, which is either a linear subspace or an affine space. 

Claim: The following optimisation problems are equivalent:
$$
\newcommand\norm[1] {\left\lVert#1\right\rVert}
\newcommand\iprod[2]{\left\langle#1, #2\right\rangle}
\DeclareMathOperator{\img}{im}
\DeclareMathOperator{\id}{id}
\DeclareMathOperator*{\argmax}{argmax}
\DeclareMathOperator*{\argmin}{argmin}

\begin{align}
&\min_{a, \, U} \sum_{k=1}^n \norm{x_k - P_{a+U}(x_k)}^2 \\
&\min_{ U} \sum_{k=1}^n \norm{x_k - P_{\mu+U}(x_k)}^2 \\
&\min_{ U} \sum_{k=1}^n \norm{x_k' - P_{U}(x_k')}^2 \\
\end{align}
$$
*Proof* (1)$\iff$(2) : *Spceial thanks to 王家慧 & 曾卓尧*

We notice the following equivalence
$$
\begin{align*}
&\min_{a,U} \sum_{k} \norm{x_k - P_{a+U}(x_k)}^2  \\
\iff &\min_{a,U} \sum_{k} \norm{x_k-a  - P_{U}(x_k-a)}^2  
	& P_{a+U}(x)=a+P_U(x-a) \\
\iff &\min_{a,U} \sum_{k} \norm{x_k -\mu + \mu -a  - P_{U}(x_k-\mu+\mu-a)}^2   
	& \text{coustruct }\mu \\
\iff &\min_{a, U}  \sum_{k} \norm{x_k -\mu + (\mu -a)  - P_{U}(x_k-\mu) - P_{U}(\mu-a)}^2
	& \text{linearity of } P_U \\
\iff &\min_{a, U}  \sum_{k} \norm{x_k - P_{\mu+U}(x_k) + (\mu -a)  - P_{U}(\mu-a)}^2
	& P_{\mu+U}(x)=\mu+P_U(x-\mu)  \\
\iff &\min_{a, U}  \sum_{k} \norm{x_k - P_{\mu+U}(x_k) +  P_{U^\perp}(\mu-a)}^2
	&  \id-P_{U}=P_{U^\perp}  \\
\end{align*}
$$

* Using the fact that $\norm{x+y}^2 = \norm{x}^2 + \norm{y} + 2\iprod{x}{y}$. We get

$$
\norm{x_k - P_{\mu+U}(x_k) +  P_{U^\perp}(\mu-a)}^2 \\
= \norm{x_k - P_{\mu+U}(x_k)}^2 + \norm{P_{U^\perp}(\mu-a)}^2 + 2\iprod{x_k - P_{\mu+U}(x_k)}{P_{U^\perp}(\mu-a)}
$$

* Note that the first argument of the inner product term:

$$
\begin{align*}
\sum_k x_k - P_{\mu+U}(x_k)
&= \sum_k x_k - \mu - P_{U}(x_k-\mu) \\
&= \sum_k \left[ x_k  - P_{U}(x_k) - \mu + P_U(\mu) \right] \\
&= \sum_k \left[ P_{U^\perp}(x_k) - P_{U^\perp}(\mu) \right]\\
&= \sum_k  P_{U^\perp}(x_k - \mu) \\
&= P_{U^\perp}\left( \sum_k x_k - n\mu \right) \\
&= 0
\end{align*}
$$

$$
\begin{align*}
\text{Linearity of inner product}
\implies
\sum_k \iprod{x_k - P_{\mu+U}(x_k)}{P_{U^\perp}(\mu-a)} = 0
\end{align*}
$$

* The original optimisation problem is now equivalent to

$$
\begin{align*}
&\min_{a\in\mathbb R^p, \: U}  \sum_{k} \norm{x_k - P_{\mu+U}(x_k)}^2 + n\norm{P_{U^\perp}(\mu-a)}^2 \\
\iff  &\min_{b\in U^\perp, \: U}  \sum_{k} \norm{x_k - P_{\mu+U}(x_k)}^2 + n\norm{b}^2 \\
\iff  &\: b^*=0 \text{ and } \min_{U}  \sum_{k} \norm{x_k - P_{\mu+U}(x_k)}^2 \\
\end{align*}
$$

* Note: $b^*=0 \iff \mu -a^*\in U \iff a^*\in\mu+U$

*Proof* (2)$\iff$(3) : This is straightforward by directly using $P_{\mu+U}(x)=\mu+P_U(x')$

Hence, without loss of generality, we may assume that the original data are centered. The PCA aims to find a $d$-dimensional linear subspace  $U\in\mathbb R^p$ which 

> 1. Best approximates the original data
>    $$
>    &\min_{ U} \sum_{k=1}^n \norm{x_k - P_{U}(x_k)}^2 \\
>    $$
>
> 2. Preserves the maximal variance (energy)
>    $$
>    \max_{ U}\:  \sum_{k=1}^n \norm{P_U(x_k)}^2
>    $$

In fact, the two interpreations are equivalent

*Proof* This is the direct result of Pythagoras theorem
$$
\norm{x_k}^2 = \norm{P_U(x_k)}^2 + \norm{P_{U^\perp}(x_k)}^2
$$

$$
\begin{align*}
&\min_{ U} \sum_{k=1}^n \norm{x_k - P_{U}(x_k)}^2 \\
\iff &\min_{ U} \sum_{k=1}^n \norm{ P_{U^\perp}(x_k)}^2 \\
\iff &\min_{ U} \sum_{k=1}^n \norm{x_k}^2 - \norm{ P_{U}(x_k)}^2 \\
\iff &\max_{ U} \sum_{k=1}^n \norm{ P_{U}(x_k)}^2 \\
\end{align*}
$$

We note that the discussion so far does not require introduction of basis or matrices. This is not only mathematically elegant but also gives us more generality: By replacing $\mathbb R^p$ by any general inner product space, the results remain valid.

Now, we will introduce basis and matrices to discuss the optimization problem. Suppose we found the optimal subspace $U^*$. We call an orthonormal (ON) basis $u_1,\cdots, u_d$ of $U^*$ the **principle components** of the data set. Let $B=[u_1,\cdots, u_d]\in\mathbb R^{p\times d}$. Then,
$$
\forall k=1,\cdots, n, \: \exists !\hat x_k\in\mathbb R^d: x_k = \sum_{i=1}^d (\beta_{k})_iu_i =B\hat x_k
$$
We call $\hat x_k$ the **lower dimensional representation** of $x_k$. Note that $\hat x_k$ is significantly shorter than $x_k$ in real life application. Since $B$ has ON columns, we conclude
$$
\hat x_k = B^\top x_k
$$


## Calculating PCA with SVD

We will show that the optimal subspace can be found via SVD.

![PCA mindmap](/Users/zhangke/Desktop/PCA mindmap.jpg)

To solve the optimisation problem, we note that
$$
\text{Find the optimal $d$-dimensional subspace $U$}
\iff 
\text{Find the optimal ON basis $u_1,\cdots, u_d$}
$$
Hence, instead of optimizing over subspaces, we aim to find the optimal ON basis which will be our principle components. The optimal subspace is just the span of principle components.

Let $B=[u_1,\cdots, u_d] \in\mathbb R^{p\times d}$ be our optimisation variable such that $B^\top B=I_d$. 

From linear algebra, we know the projection of $x$ onto the column space of $B$.
$$
P_U(x) = BB^\top x
$$
The optimisation problem becomes
$$
\begin{align*}
&\max_{U}\:  \sum_{k=1}^n \norm{P_U(x_k)}^2\\
\iff &\max_{B}\:  \sum_{k=1}^n \norm{BB^\top x_k}^2  &\text{s.t.} \quad B^\top B = I_d \\
\iff &\max_{B}\:  \sum_{k=1}^n \norm{B^\top x_k}^2  &\text{s.t.} \quad B^\top B = I_d \\
\iff &\max_{B}\:  \sum_{k=1}^n x_k^\top BB^\top x_k  &\text{s.t.} \quad B^\top B = I_d
\end{align*}
$$
The objective can be reformulated as 
$$
\begin{align*}
 \sum_{k=1}^n x_k^\top BB^\top x_k
&=  \sum_{k=1}^n \tr(x_k^\top BB^\top x_k) 
	& \tr(a)=a, \forall a\in\mathbb R\\
&=  \sum_{k=1}^n \tr(B^\top x_k x_k^\top B) 
	& \text{cyclic property of }\tr \\
&=   \tr(B^\top \sum_{k=1}^nx_k x_k^\top B) 
	& \text{linearity of } \tr \\
&=   \tr(B^\top XX^\top B) 
	& \text{let }X=[x_1,\cdots, x_n] \in\mathbb R^{p\times n}
\end{align*}
$$
Hence, the original optimization problem is equivalent to
$$
\max_{B} \: \tr(B^\top XX^\top B) \quad \text{s.t.} \quad \quad B^\top B = I_d
$$
The optimal solution is $B^*=[u_1,\cdots, u_d] \in\mathbb R^{p\times d}$ where  $u_1,\cdots, u_d$ are the $d$ leading eigenvectors of $XX^\top$ (see Appendix). Note that the sample variance is $C_{xx} = \frac{1}{n-1}XX^\top = \frac{1}{n-1}\sum_{k=1}^nx_k x_k^\top$. The principle components corresponds to $d$ leading eigenvalues of the sample variance. Intuitively, the PCA selects $d$ most significant (in terms of max. variance) features out of $p$ features. 

Let the SVD of the data matrix $X$ be
$$
X = U\Sigma V^\top \: \text{with} \:  \sigma_1>\cdots>\sigma_r
$$
Recall that columns of $U$ are eigenvectors of $XX^\top$. Hence, the d leading eigenvec of $XX^\top$ is exactly the first $d$ columns of $U$. 

## Summary

Given centered data set $x_1,\cdots, x_n\in\mathbb R^p$. To find the optimal $d$-dimensional representations of the data, do

1. Define data matrix $X= [x_1,\cdots, x_n] \in\mathbb R^{p\times n}$
2. Perform SVD on data matrix: $X=U\Sigma V^\top$ with $\sigma_1>\cdots>\sigma_r$ in $\Sigma$
3. The principle componets are the first d columns of $U$. The optimal subspace is the span of principle components
4. Let $B=U_{:, \, 1:d}$. The lower dimensional representation of $X$ is $\hat X = B^\top X\in\mathbb R^{d\times n}$. 

If the data are not centered,

1. Center the data $X'= X-\mu \mathbf 1^\top = [x_1-\mu,\cdots, x_n-\mu] \in\mathbb R^{p\times n}$ where $\mu$ is the sample mean
2.  Apply PCA to $X'$ 

## Appendix

Let $A\in\mathbb R^{n\times n}$ be symmetric. Then, consider the following optimization problems

1. $$
   \begin{split}
   \max_{x} x^\top A x 
   \quad \text{s.t.} \quad
   & x\in \mathbb R^n, \\ 
   & \Vert x \Vert_2=1
   \end{split}
   $$

2. $$
   \begin{split}
   \max_{X} \tr( X^\top A X)
   \quad \text{s.t.} \quad
   & X\in \mathbb R^{n\times d}, \\ 
   & X^\top X=I_d
   \end{split}
   $$

3. $$
   \begin{split}
   \max_{x_1,\cdots, x_d} \sum_{k=1}^d( x_k^\top A x_k)
   \quad \text{s.t.} \quad
   & x_1,\cdots, x_d\in \mathbb R^{n} \\ 
   & x_1,\cdots, x_d \text{ are ON}
   \end{split}
   $$

   Note: The 1st problem is a special case of the 2nd problem by taking $d=1$. The 3rd problem is equivalent to the 2nd problem. 

Claim:

1. The optimal solution to the 1st problem is the leading eigenvector of $A$. The optimal value of the objective is $\lambda_\mathrm{max}(A)$. 
2. The optimal solution to the 3rd problem is the $d$ leading eigenvectors of $A$. The optimal value of the objective is the sum of the $d$ largest eigenvalues of $A$.
