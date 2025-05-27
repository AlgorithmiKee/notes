---
title: "Fixed-Point Iteration"
author: "Ke Zhang"
date: "2024"
fontsize: 12pt
---

# Fixed-Point Iteration: Theory and Application

[toc]

# Metric Space

In this article, $(X,d)$ represents a metric space. Some typical metric spaces are

* $X=\mathbb R$ with $d(x,y) = \vert x - y \vert$
* $X=\mathbb R^n$ with $d(u,v) = \Vert x - y \Vert$ where $\Vert \cdot \Vert $ can be any norm on $\mathbb R^n$

Later, we will focus a lot on complete metric space. e.g.

* A closed interval in 1D: $X=[a, b]$
* A closed ball in n-D: $X=B_r[c] \triangleq \{ x\in\mathbb R^n: \Vert x-c \Vert\le r \}$

## Fixed Point

> Given a function $f: X \to X$. We call $\bar x$ a **fixed point** (FP) of $f$​ if
> $$
> \begin{equation}
>     \bar x = f(\bar x)
> \end{equation}
> $$

Remarks:

* In general, fixed point of an arbitrary function is **not** unique
* Fixed point of $f$ $\iff$ Equilibrium of the discrete-time dynamic system $x_{n+1} = f(x_n)$
* For $X = \mathbb R$, $(\bar x, f(\bar x))$ is the intersect of $y=f(x)$ and $y=x$.

Example:

* Let $X = \mathbb R$. $f(x) = \sin(x)$ has a unique fixed point $\bar x = 0$
* Let $X = \mathbb R$. $f(x) = x^2 -x$ has two fixed points $\bar x_1 = 0$, $\bar x_2 = 2$
* Let $X = \mathbb R^n$. $\: f(x) = Ax$ where $A \in \mathbb R^{n \times n}$ has at least one fixed point $\bar x = 0$

Question:

1. Which functions have (unique) fixed points?
2. How to find fixed points of a function?

To answer these questions, we need the notion of contractiom mapping.

# Contraction Mapping

> A function $f: X \to X$​ is called a **contraction mapping** if
> $$
> \begin{equation}
> \exists \gamma\in [0,1), \text{ s.t. }
>   \forall u, v \in X, \:
>   d\!\left( f(u), f(v) \right) \le \gamma d\!\left( u, v \right)
> \end{equation}
> $$

Remarks:

* $f$ is a contraction mapping $\iff$ $f$ is Lipschitz continuous with Lipschitz constant $L<1$.
* $f$ is a contraction mapping $\implies$ $f$ is uniformly continuous.
* Intuitively, a contraction mapping brings points closer together. However,
  * The definition $d\!\left( f(u), f(v) \right) \le \gamma d\!\left( u, v \right)$ is stronger than just $d( f(u), f(v) ) < d( u, v )$

Example:

* Let $X = \mathbb R$. $f(x)=kx + b$ is a contraction mapping iff $-1< k <1$
* Let $X = \mathbb [1, \infty)$. $f(x)=x + \dfrac{1}{x}$ also "brings points together" but it is **not** a contraction mapping.

## Criterion for Checking Contraction

Checking whether a function is a contraction mapping by verifying the definition is often less straight-forward. Good news is that we can check the first-order derivative instead.

### First Derivative Test

For univariant functions over closed intervals, we use the fact that

> Let $X = [a,b] \subseteq\mathbb{R}$ and $f: X \to X$ be differentiable. Then,
> $$
> \sup_{x\in X} \vert f'(x)\vert <1 \iff f \text{ is a contraction mapping}
> $$

*Proof $\Rightarrow$:* Apply the mean value theorem (MVT) to arbitrary closed interval $[x, y]\in X$
$$
\forall x, y \in X, \: \exists \xi\in (x, y), \text{ s.t. } f(x) -f(y) = f'(\xi)(x-y)
$$

Let $\gamma = \sup_{x\in X} \vert f'(x)\vert$. By assumption, $\gamma\in [0,1)$.
$$
\begin{align*}
\left\vert f(x) - f(y) \right\vert
&= \left\vert f'(\xi) \right\vert \left\vert x-y \right\vert &&\text{by MVT}
\\
&\le \gamma \left\vert x-y \right\vert &&\gamma = \sup_{x\in X} \vert f'(x)\vert
\end{align*}
$$

Hence, $f$ is a contraction mapping

*Proof $\Leftarrow$:* By assumption:
$
\exists \gamma\in [0,1), \text{ s.t. }
\forall x, y \in X, \:
\left\vert f(x) - f(y) \right\vert
\le \gamma \left\vert x-y \right\vert
$

Choose $y=x+h$ with $h \ne 0$. We get
$$
\left\vert \frac{ f(x+h) - f(x) }{h} \right\vert \le \gamma \\
$$

Taking the limit, we conclude: $\exists \gamma\in [0,1)$, s.t. $\forall x \in X$
$$
\begin{align*}
\left\vert \frac{ f(x+h) - f(x) }{h} \right\vert \le \gamma
&\implies \lim_{h\to 0} \left\vert \frac{ f(x+h) - f(x) }{h} \right\vert \le \gamma \\
&\implies \left\vert \lim_{h\to 0} \frac{ f(x+h) - f(x) }{h} \right\vert \le \gamma \\
&\implies \left\vert f'(x) \right\vert \le \gamma
\end{align*}
$$

Hence, $\sup_{x\in X} \vert f'(x)\vert \le \gamma < 1 \quad\quad\quad \square$

### Jacobian Matrix Test

For multivariante function, the first derivative test is generalised to Jacobian matrix test.

> Let $X \subseteq \mathbb{R}^n$ be **closed** and $f: X \to X$ be differentiable. Then,
> $$
> \sup_{x\in X} \left\Vert J_f(x) \right\Vert < 1
> \iff f \text{ is a contraction mapping}
> $$
> where $J_f(x)$ is the Jacobian matrix of $f$ evaluated at $x$
> and $\left\Vert \cdot \right\Vert$ represents the spectral norm.

Remark:

* The proof is rather similar to the 1D case. Omitted here.
* The spectral norm of a matrix $A$ is the largest sigular value of $A$. Hence, we have the equivalent criterion.
  $$
  \begin{align*}
  \sup_{x\in X} \left\Vert J_f(x) \right\Vert < 1
  &\iff \sup_{x\in X} \sigma_{\max}\left( J_f(x) \right) < 1 \\
  &\iff \sup_{x\in X} \sqrt{\lambda_{\max}\left( J_f(x)^\top J_f(x) \right)} < 1 \\
  &\iff \sup_{x\in X} \sqrt{\lambda_{\max}\left( J_f(x) \right)^2} < 1 \\
  &\iff \sup_{x\in X} \left\vert \lambda_{\max}\left( J_f(x) \right) \right\vert< 1 \\
  \end{align*}
  $$

Corollary:
> Let $X \subseteq \mathbb{R}^n$ be **closed** and $f: X \to X$ be differentiable. Then,
> $$
> \sup_{x\in X} \left\vert \lambda_{\max}\left( J_f(x) \right) \right\vert< 1
> \iff f \text{ is a contraction mapping}
> $$
> where $\lambda_{\max}\left( J_f(x) \right)$ represents the largest eigenvalue of the Jacobian matrix.

## Contraction Mapping Theorem

> Let $(X,d)$ be a non-empty metric space and $f: X \to X$ be a contraction mapping. Then,
>
> 1. $f$ has at most one fixed point
>
> 2. If $(X,d)$ is **complete**, then $f$ has unique fixed point.

Remark:

* Also known as *Banch Contraction Theorem*
* This is a sufficient yet not neccessary condition for exsitence of unique fixed point.
  * e.g. $X=\mathbb R$, $f(x)=\sin(x)$ is not a contraction mapping but it has unique FP.
* To use the theorem, we must make sure that $f$ is contractive. The weaker condtion $d( f(u), f(v) ) < d( u, v )$ does not gurantee the existence of fixed point.
  * e.g. $X = \mathbb [1, \infty)$. $f(x)=x + \dfrac{1}{x}$ satisfies the weaker condition but it has no FP.

*Proof of Part 1*: Suppose $f$ has two distinct fixed points $u$ and $u'$. Then,
$$
\begin{align*}
  d\!\left( u, u' \right)
  &=   d\!\left( f(u), f(u') \right)    &&\text{def. of FP} \\
  &\le \gamma  d\!\left( u, u' \right)  &&\text{$f$ constractive} \\
\end{align*}
$$

Recall: $ \gamma\in[0,1) \implies d\!\left( u, u' \right)  = 0 \iff u = u'$. We get a contradiction. $\quad\quad\quad \square$

*Proof of Part 2*: Breaks down into two steps:

* Construct a Cauchy sequence $(x_n)_{n\in\mathbb N}$ by iterating over $f$.

* Show that the limit of $(x_n)_{n\in\mathbb N}$ is the fixed point

Choose arbitrary $x_0 \in X$. Define the sequence
$$
\begin{equation}
    x_{n+1} = f(x_n)
\end{equation}
$$
The distance between two consecutive terms can be arbitrarily small after sufficiently many iterations since
$$
\begin{align*}
\forall m \ge 0, \:
d\!\left( x_{m+1}, x_{m} \right)
&= d\!\left( f(x_{m}), f(x_{m-1}) \right) && \text{by iteration} \\
&\le \gamma d\!\left( x_{m}, x_{m-1} \right) && \text{$f$ is contractive}\\
& \quad\vdots & \\
&\le \gamma^m d\!\left( x_{1}, x_{0} \right)
\end{align*}
$$

We will use this fact a LOT later.
$$
\begin{equation}
 d\!\left( x_{k+1}, x_{k} \right) \le \gamma^k d\!\left( x_{1}, x_{0} \right)
\end{equation}
$$
To show that $(x_n)_{n\in\mathbb N}$ is a Cauchy sequence, assume w.l.o.g. that $n > m$. Using the above fact, we get
$$
\begin{align*}
d\!\left( x_{m}, x_{n} \right)
&\le d\!\left( x_{m}, x_{m+1} \right) + d\!\left( x_{m+1}, x_{m+2} \right) + \cdots+ d\!\left( x_{n-1}, x_{n} \right)
\\
&\le \gamma^{m} d\!\left( x_{0}, x_{1} \right) + \gamma^{m+1} d\!\left( x_{0}, x_{1} \right) + \cdots + \gamma^{n-1} d\!\left( x_{0}, x_{1} \right)
\\
&\le \left(1 + \gamma + \gamma^2 + \cdots + \gamma^{n-m-1} \right) \gamma^{m} d\!\left( x_{0}, x_{1} \right)
\\
&= \cfrac{1 - \gamma^{n-m} }{1 - \gamma \hphantom{^{n-m}} } \gamma^{m} d\!\left( x_{0}, x_{1} \right)
\\
&\le \cfrac{1}{1 - \gamma} \gamma^{m} d\!\left( x_{0}, x_{1} \right)
\\
&= C\cdot \gamma^{m}
\end{align*}
$$

Note that $C:=\cfrac{d\!\left( x_{0}, x_{1} \right)}{1-\gamma}$ is a constant and $|\gamma| < 1$. Hence, we can make the upper bound of $d\!\left( x_{m}, x_{n} \right)$ arbitrarily small as long as $m$ is sufficiently large.
$$
\forall\varepsilon >0, \exists N = \left\lceil \frac{\ln \epsilon}{\ln \gamma} \right\rceil, \text{ s.t. } \forall n,m>N, d\!\left( x_{m}, x_{n} \right) \le C\gamma^m < C\epsilon
$$
Hence, regardless of $x_0$, $(x_n)_{n\in\mathbb N}$ is always a Cauchy sequence.

Finally, we show that the limit of $(x_n)_{n\in\mathbb N}$ is the fixed point of $f$. With the fact that
$$
\text{coontraction mapping $\implies$ Lipschitz continuity $\implies$ Continuity}
$$

we can switch the order of $\lim$ and $f$. Let $\tilde x = \lim\limits_{n\to\infty} x_n$

$$
\begin{align*}
  f(\tilde x)
  &= f\!\left( \lim_{n\to\infty} x_n \right)    && \\
  &= \lim_{n\to\infty} f\!\left(  x_n \right)   &&\text{conti. of } f \\
  &= \lim_{n\to\infty}  x_{n+1}               &&\text{by iteration} \\
  &= \tilde x                                 &&\text{convergence} \\
\end{align*}
$$

Hence, $\tilde x $ is a FP of $f$. By part 1, we know it is **the** FP of $f$.  $\quad\quad\quad \square$

Summary:

> If $f$ is a contraction mapping on some closed metric space $X$, then
>
> 1. $f$ has unique fixed point in $X$
> 1. The fixed point can be obtained through fixed-point iteration (FPI)
>     $$
>     x_{n+1} = f(x_n)
>     $$
> 1. The initial value can be chosen arbitrarily in $X$

# Numerical Properties

## Error Bounds

Offline error bound:
> $$
> \begin{equation}
>    d\!\left( x_{n}, \bar x \right)
>     \le \frac{\gamma^{n}}{1 - \gamma} d\!\left( x_{0}, x_{1} \right)
> \end{equation}
> $$

Online error bound:
> $$
> \begin{equation}
>     d\!\left( x_{n}, \bar x \right)
>     \le \frac{\gamma}{1 - \gamma} d\!\left( x_{n-1}, x_{n} \right)
> \end{equation}
> $$

Remarks:

* The offline error bound can be computed before starting the PFI. Given some tolerance and a starting point, we can compute how many iterations are needed to achieve the tolerance. (Details: c.f. next section)
* The online error bound estimates how close $x_n$ is to the true FP $\bar x$ given the distance between $x_n$ and $x_{n-1}$.

*Proof:* The error at iteration $n$ satisfies
$$
\begin{align*}
    d\!\left( x_{n}, \bar x \right)
    &= d\!\left( f(x_{n-1}), f(\bar x) \right)
    && \text{by iteration and FP}
    \\
    &\le \gamma\cdot d\!\left( x_{n-1}, \bar x \right)
    && \text{contraction}
    \\
    &\le \gamma\cdot
    \left(
     d\!\left( x_{n-1}, x_{n} \right) + d\!\left( x_{n}, \bar x \right)
    \right)
    && \text{triangle ineq.}
\end{align*}
$$
Rearrange $d\!\left( x_{n}, \bar x \right) $ to the LHS and use the inequality $d\!\left( x_{k+1}, x_{k} \right) \le \gamma^k d\!\left( x_{1}, x_{0} \right)$. We get
$$
\begin{align*}
    d\!\left( x_{n}, \bar x \right)  
    &\le \frac{\gamma}{1 - \gamma} \cdot d\!\left( x_{n-1}, x_{n} \right)
    &&\text{online}
    &
    \\
    &\le \frac{\gamma}{1 - \gamma} \cdot \gamma^{n-1} d\!\left( x_{0}, x_{1} \right)
    & &
    \\
    &= \frac{\gamma^{n}}{1 - \gamma} d\!\left( x_{0}, x_{1} \right)
    &&\text{offline}
    &\quad\quad\square
\end{align*}
$$

## Stopping Rule

> Starting from the initial point $x_0$, the FPI needs
> $$
> N > \frac{\log(\epsilon) + \log(1-\gamma) - \log(d_{01}) }{\log(\gamma)}
> $$
> iterations to achive the tolerance $\epsilon$, where $d_{01} = d\!\left(x_0, f(x_0)\right)$

*Proof:* This is a direct result from the offline error bound. Let
$$
d\!\left( x_{n}, \bar x \right)
\le \frac{\gamma^{n}}{1 - \gamma} d\!\left( x_{0}, x_{1} \right) < \epsilon
\implies
\gamma^{n} < \frac{1-\gamma}{d\!\left( x_{0}, x_{1} \right)} \epsilon
$$

Apply some log algebra and we conclude.

## Rate of Convergence

The rate of convergence is obtained by comparing the errors $d\!\left( x_{n}, \bar x \right)$ and $d\!\left( x_{n-1}, \bar x \right)$.
$$
d\!\left( x_{n}, \bar x \right)
= d\!\left( f(x_{n-1}), f(\bar x) \right)
\le \gamma \cdot d\!\left( x_{n-1}, \bar x \right)
$$
If $\gamma\ne 0$, the fixed-point iteration convergences linearly to the fixed point , i.e.
$$
\lim_{n \to\infty} \frac{d\!\left( x_{n}, \bar x \right) }{d\!\left( x_{n-1}, \bar x \right) } \le \gamma \in(0,1)
$$
If $\gamma=0$, the fixed-point iteration convergences superlinearly.

Special case: For closed $X\subseteq\mathbb R$, we can show that the FPI converges quadratically.

# Application

## Newton's Method

Recall the Newton's method to find the root of a function $g: \mathbb R \to \mathbb R$:
>
> 1. Set some initial $x_0 \in\mathbb R$
> 2. For $n = 0, 1, 2, \cdots$ until hopefully convergence
>
>     $$
>       x_{n+1} = x_n - \frac{g(x_n)}{g'(x_n)}
>     $$

In general, the convergence of Newton's method depends on $x_0$, $g'$ and even  $g''$. A very hard problem!

Here, we will show that if we start sufficiently close to the root of $g$ (plus some extra requirements), Newton's method will converge to the root of $g$.

First, we reformulate Newton's method into a FP problem. Define
$$
  f: \mathbb R \to \mathbb R, x \mapsto x - \frac{g(x)}{g'(x)}
$$

Observation:

* If $\bar x$ is a root of $g$ and $g'(\bar{x}) \ne 0$, then $\bar{x}$ is a fixed point of $f$
* Find the roots of $g$ $\iff$ Find FPs of $f$.
* Newton's method $\iff$ FPI over $f$.

If $f$ happens to be a contraction on $\mathbb R$, the algorithm will definitely converge to the FP of $f$, aka the root of $g$. However, a global contraction property does not hold in general. Nevertheless, we do have a local contration property around the FP. i.e. For sufficiently small $\delta$, $f$ is a contraction in the closed ball $B_\delta[\bar{x}]$.

> Let $g\in C^2(\mathbb R)$ and $\bar{x}$ be a root of $g$. Suppose $g'(\bar{x}) \ne 0$, then
> $$
> \exists \delta >0, \text{ s.t. } f: B_\delta[\bar{x}] \to B_\delta[\bar{x}], x \mapsto x - \frac{g(x)}{g'(x)} \text{ is a contraction}
> $$
> where $B_\delta[\bar{x}] \triangleq \{x\in\mathbb R: \vert x -\bar{x} \vert \le \delta \}$ is the closed ball centered at $\bar{x}$ with radius $\delta$.

Remarks:

* $f$ is a contraction **locally** in $B_\delta[\bar{x}]$
* If the FPI starts from $x_0\in B_\delta[\bar{x}]$, Newton's method will converge to $\bar{x}$

*Proof:* To show $f$ is a contraction in $B_\delta[\bar{x}]$, it is sufficient to show that $\vert f'(x) \vert$ is uniformly bounded by some constant $\gamma\in(0,1)$ in $B_\delta[\bar{x}]$. c.f. First Derivative Test. Formally, we want to show
$$
\exist\delta >0, \exist\gamma\in(0,1),
\text{ s.t. } \forall x\in B_\delta[\bar{x}], \vert f'(x) \vert < \gamma
$$

Some easy calculation yields
$$
\begin{align*}
f'(x)       &= \frac{g(x) \cdot g''(x)}{g'(x)^2} \\
\left.\begin{matrix}
  g(\bar{x}) =0 \\
  g'(\bar{x})\ne 0
\end{matrix}\right\rbrace
&\implies f'(\bar{x}) = 0
\end{align*}
$$

$g$ is twice differentiable $\implies$ $f'$ is continous. In particular,  at $\bar{x}$
$$
\forall \gamma \in(0,1), \exists \delta >0,  \text{ s.t. } \forall x\in B_\delta[\bar{x}],\:
\vert f'(x) - \underbrace{f'(\bar{x})}_{0} \vert < \gamma \\
$$

Hence, $\forall x\in B_\delta[\bar{x}], \vert f'(x) \vert < \gamma \implies $ $f$ is a contraction on $ B_\delta[\bar{x}]$. $\quad\quad\quad\square$

## Existence and Uniqueness Theorem for ODE

We would like to solve the initial value problem
> $$
> \begin{align}
>  \forall t\in I,\:  \dot x(t) = T(x(t), t), \quad x(t_0)=x_0
> \end{align}
> $$

where

* Time interval: $I\subseteq \mathbb{R}$
* State space: $ \mathbb R^n $
* State trajectory:  $ x: I \to \mathbb R^n, t \mapsto v$
* State transition: $ T: \mathbb R^n \times \mathbb R \to V, (v,t) \mapsto w $
* Initial values: $t_0\in I$ and $x_0\in\mathbb R^n$

Note that the dynamic system $T$ is time-varying and even nonlinear in general.

> Suppose that $T$ is continuous in $t$ and Lipschitz continuous in $v$. Then,
> $$
> \exist \varepsilon >0, \forall x_0\in\mathbb R^n,
> \text{ the IVP }
> \begin{cases}
>   \dot x(t) = T(x(t), t), \:\: t \in[t_0,\, t_0+\varepsilon] \\
>   x(t_0)=x_0
> \end{cases}
> \text{ has unique solution}
> $$
>

*Proof*: Basic idea:

1. Reformulate the IVP into a FP problem.
1. Identify the metric space involved in the FP problem
1. Apply the contraction mapping theorem to discuss the uniqueness of solution

Using the fundamental theorem of calculus, we get
$$
\begin{align}
x(t) = x_0 + \int_{t_0}^t T(x(\tau), \tau) \:\mathrm d\tau, \quad t \in[t_0,\, t_0+\varepsilon]
\end{align}
$$

This integral can't be evaluated directly since the RHS depends on $x(\cdot)$ as well. However, this formula has exactly the same structure as FPI, which motivates us to define the mapping

$$
\begin{align}
\mathcal{M}:
C^{???}([t_0,\, t_0+\varepsilon], \mathbb{R}^n) &\to C^{???}([t_0,\, t_0+\varepsilon], \mathbb{R}^n), \\
x(t) &\mapsto \mathcal{M}[x](t) = x_0 + \int_{t_0}^t T(x(\tau), \tau) \:\mathrm d\tau
\end{align}
$$

Hence, $\varphi$ is a solution to the IVP $\iff$ $\varphi$ is a FP of $\mathcal{M}$

The metric space $C^{???}([t_0,\, t_0+\varepsilon], \mathbb{R}^n)$ equipped with the metric defined by infinity norm
$$
\begin{align}
d(x,y) = \Vert x - y \Vert_\infty = \sup_{t \in [t_0,\, t_0+\varepsilon]} \Vert x(t) - y(t) \Vert
\end{align}
$$
is indeed complete.

Now, we show the existence of $\varepsilon$, for which $\mathcal{M}$ is a contraction mapping. More formally, we want to show
$$
\begin{align*}
\exist\varepsilon>0, \exist\gamma\in(0,1), \text{ s.t. }
&\forall x,y \in C^{???}([t_0,\, t_0+\varepsilon], \mathbb{R}^n), \\
&\Vert \mathcal{M}(x)-\mathcal{M}(y) \Vert_\infty < \gamma \Vert x-y \Vert_\infty
\end{align*}
$$

For $\forall x, y \in C^{?}([t_0,\, t_0+\varepsilon], \mathbb{R}^n)$, let $t^* \in[t_0,\, t_0+\varepsilon] $ be such that
$$
\begin{align*}
\Vert \mathcal{M}[x](t^*)-\mathcal{M}[y](t^*) \Vert
&= \Vert \mathcal{M}(x)-\mathcal{M}(y) \Vert_\infty \\
&= \sup_{t \in [t_0,\, t_0+\varepsilon]} \Vert \mathcal{M}[x](t)-\mathcal{M}[y](t) \Vert
\end{align*}
$$
The existence of $t^*$ is guaranteed because a continuous function must attain a maximum on a closed interval.

Then,
$$
\begin{align*}
\Vert \mathcal{M}(x)-\mathcal{M}(y) \Vert_\infty
&= \left\Vert \mathcal{M}[x](t^*)-\mathcal{M}[y](t^*) \right\Vert \\
&= \left\Vert \int_{t_0}^{t^*} T(x(\tau), \tau) \:\mathrm d\tau - \int_{t_0}^{t^*} T(y(\tau), \tau) \:\mathrm d\tau \right\Vert \\
&= \left\Vert \int_{t_0}^{t^*} T(x(\tau), \tau)  -  T(y(\tau), \tau) \:\mathrm d\tau \right\Vert \\
&\le \int_{t_0}^{t^*} \left\Vert  T(x(\tau), \tau)  -  T(y(\tau), \tau) \right\Vert \:\mathrm d\tau\\
&\le \int_{t_0}^{t^*} L\left\Vert  x(\tau)  -  y(\tau) \right\Vert \:\mathrm d\tau\\
&\le \int_{t_0}^{t^*} L\left\Vert  x  -  y \right\Vert_\infty \:\mathrm d\tau\\
&\le \varepsilon L\left\Vert  x  -  y \right\Vert_\infty \\
\end{align*}
$$
Hence, $\mathcal{M}$ is a contraction for $\varepsilon < \frac{1}{L}$. $\implies$ We conclude

* $\mathcal{M}$ has unique fixed point $\varphi\in C^{???}([t_0,\, t_0+\varepsilon], \mathbb{R}^n)$
* $\varphi$ is the unique solution to the IVP. $\quad\quad\quad\square$

### Picard's Iteration

The FPI over $\mathcal{M}$ is called Picard's iteration

> Let $\varphi_0(t) = x_0$  
> For $k=0,1,2,\dots$, do
> $$
> \begin{align}
> \varphi_{k+1}(t) 
> &= \mathcal{M}[\varphi_{k}](t) \\
> &= x_0 + \int_{t_0}^t T(\varphi_{k}(\tau), \tau) \:\mathrm d\tau
> \end{align}
> $$

Example: Consider the linear ODE with initial value
$$
\dot x = x, \quad x(0) = 1
$$
Picard's iteration for this IVP is then
$$
\begin{align*}
\varphi_0 (t) &= 1 \\
\varphi_1 (t) &= 1 + \int_0^t 1 \:\mathrm d\tau = 1+t \\
\varphi_2 (t) &= 1 + \int_0^t (1+t) \:\mathrm d\tau = 1 + t + \frac{1}{2} t^2\\
\varphi_3 (t) &= 1 + \int_0^t (1+t+\frac{1}{2} t^2) \:\mathrm d\tau = 1 + t + \frac{1}{2} t^2 + \frac{1}{6} t^3\\
\end{align*}
$$
We see that $\varphi_k(t)$ represents the $k$-th order Tylor expansion of the exact solution $\bar\varphi(t)=e^t$.

# Exercise

1. Let $u$ be an eigen vector of $A \in \mathbb R^{n \times n}$ associated with eigen value $\lambda$. In addition, $u$ is a fixed point of the linear map defined by $f: \mathbb R^n \to \mathbb R^n, x\mapsto Ax$. Determine the value of $\lambda$.

2. Let $A \in \mathbb R^{n \times n}$ and $f: \mathbb R^n \to \mathbb R^n, x\mapsto Ax $. Show that
$$
    f \text{ has unique fixed point 0} \iff \forall \lambda \in \operatorname{spec}(A), \lambda < 1
$$

3. Let $X=\mathbb R$, $f(x)=\cos(x/2)$
   * Show that $f$ is a contraction mapping
   * We would like to use FPI to estimate the FP of $f$. How many iterations do we need at least to achieve a tolerance less than $10^{-6}$?

4. Let $(X,d)$ be a matrix space and $f: X\to X$ be a contraction mapping with Lipschitz constant $\gamma$.

    (a) Show that

    $$
    \forall x, y \in X, d(x,y) \le \frac{d\!\left( x, f(x) \right) + d\!\left( y, f(y) \right)}{1-\gamma}
    $$

    (b) Simplify the inequality from (a) if $y$ is the FP

# Table of Abbreviations

| Abbreviation | Expansion                      |
| ------------ | ------------------------------ |
| FP           | Fixed Point                    |
| FPI          | Fixed-Point Iteration          |
| MVT          | Mean Value Theorem             |
| ODE          | Ordinary Differential Equation |
