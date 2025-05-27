---
title: "Primal and Dual"
author: "Ke Zhang"
date: "2024"
fontsize: 12pt
---

# Constrained Optimization Problem

Original optimizaiton problem
$$
\boxed{
\begin{aligned}
  \min_{\pmb x}\quad & f(\pmb x) \\
  \text{s.t.}\quad   & g_i(\pmb x)\le 0,  \quad i=1,\cdots, m
\end{aligned}
}
$$

where $f: \mathbb R^n \to \mathbb R$.

In following discussion, we assume that everything is differetiable up to sufficiently high order.

## Reformulation via Indicator Functions

Let $A_i$ denote the feasible region for the i-th constraint
$$
A_i = \{\pmb x\in\mathbb R^n : g_i(\pmb x)\le 0 \}
$$

The original problem is equivalent to the unconstraint optimization problem.
$$
\boxed{\begin{aligned}
  \min_{\pmb x} \quad J(\pmb x) = f(\pmb x) + \sum_{i=1}^m \chi_{A_i}(\pmb x)
\end{aligned}}
$$

where $\chi_{A}$ is the *indicator function* of the set $A$, defined by
$$
\chi_{A}(\pmb x)  =
\begin{cases}
0 & \pmb x\in A \\
+\infty & \pmb x\notin A
\end{cases}
$$

Here, we introduced infinite penalty if the constaint is not satisfied. However, the resulting unconstraint problem is equally hard to solve. Nevertheless, this reformulation gives us insight to the next step.

## Lagrangian Function

The *Lagrangian function* of the original problem is
$$
\boxed{
\begin{aligned}
  L(\pmb x, \pmb\mu)
  &= f(\pmb x) + \sum_{i=1}^m \mu_i g_i(\pmb x) \\
  &= f(\pmb x) + \pmb\mu^\top  \pmb g(\pmb x)
\end{aligned}
}
$$

where $\pmb\mu = \begin{bmatrix} \mu_1, \cdots, \mu_m \end{bmatrix}^\top\!$, $\:\pmb g(\pmb x) = \begin{bmatrix} g_1(\pmb x), \cdots, g_m(\pmb x) \end{bmatrix}^\top$

Remark:

* $\mu_1, \cdots, \mu_m$ are called *Lagrange multipliers*, also known as *dual variables*
* Each constraint is assciated with a Lagrange multiplier.
* For fixed $\pmb x$, the Lagrangian $L(\pmb x, \pmb\mu)$ is affine in $\pmb\mu$, regardless the nature of $f(\cdot)$ and $g_i(\cdot)$

Here, instead of introducing infinite penalty, we introduce linear penalty $\mu_i\ge 0 $ for constraint $g_i(\pmb x) \le 0$. In other words, $L(\pmb x, \pmb\mu)$ is an approximation of $J(\pmb x)$. The resulting problem is easier to handle.

When all $\mu_i$ are non-negative, the Lagrangian $L(\pmb x, \pmb\mu)$ becomes a lower bound of  $J(\pmb x)$
$$
  \forall \pmb\mu \ge \pmb0, \forall \pmb x\in\mathbb R^n, \:
  J(\pmb x) \ge L(\pmb x, \pmb\mu)
$$

*Proof*: Since both $L(\pmb x, \pmb\mu)$ and $J(\pmb x)$ contains the objective, it is sufficient to compare the penalty terms

If $\pmb x$ is feasible, then

$$
\sum_{i=1}^m \chi_{A_i}(\pmb x) = 0 \ge
\sum_{i=1}^m \underbrace{\mu_i}_{\ge 0} \cdot \underbrace{g_i(\pmb x)}_{\le 0}
$$

If $\pmb x$ is not feasible, then

$$
\sum_{i=1}^m \chi_{A_i}(\pmb x) = \infty \ge
\sum_{i=1}^m \underbrace{\mu_i}_{\ge 0} \cdot \underbrace{g_i(\pmb x)}_{\ge 0}
$$

## Reformulation via Lagrangian Functions

The original problem is also equivalent to
$$
\boxed{
  \min_{\pmb x} \max_{\pmb\mu\ge \pmb0} L(\pmb x, \pmb\mu)
}
$$
where the minimization over $\pmb x$ is unconstraint.

To prove this claim, we need the lemma:
$$
\boxed{
  J(\pmb x) = \max_{\pmb\mu\ge \pmb0} L(\pmb x, \pmb\mu)
}
$$

Once we have shown this lemma, we can immediately conclude since
$$
\text{original problem} \iff \min_{\pmb x} J(\pmb x) \iff  \min_{\pmb x}\max_{\pmb\mu\ge \pmb0} L(\pmb x, \pmb\mu)
$$

*Proof of the lemma*: Since we optimize w.r.t. $\pmb\mu$, the primal variable $\pmb x$ is considered as a parameter.

If $\pmb x$ is feasible, then $ g_i(\pmb x)\le 0, i=1,\cdots, m$. $L(\pmb x, \pmb\mu)$ is maximized at $\pmb\mu = \pmb0$

$$
\begin{align*}
\max_{\pmb\mu\ge \pmb0} L(\pmb x, \pmb\mu) 
&= f(\pmb x) + \max_{\pmb\mu\ge \pmb0} \sum_{i=1}^m \mu_i g_i(\pmb x) \\
&= f(\pmb x) + \sum_{i=1}^m 0\cdot g_i(\pmb x)  \\
&= f(\pmb x) = J(\pmb x)
\end{align*}
$$

If $\pmb x$ is not feasible, then $ g_i(\pmb x) > 0, i=1,\cdots, m$. $L(\pmb x, \pmb\mu)$ is maximized at $\pmb\mu = \infty\cdot\pmb1$

$$
\begin{align*}
\max_{\pmb\mu\ge \pmb0} L(\pmb x, \pmb\mu) 
&= f(\pmb x) + \max_{\pmb\mu\ge \pmb0} \sum_{i=1}^m \mu_i g_i(\pmb x) \\
&= f(\pmb x) + \sum_{i=1}^m \infty\cdot g_i(\pmb x)  \\
&= \infty = J(\pmb x)
\end{align*}
$$



Summary: The following optimization problems are equivalent

1. Original constraint problem: $\displaystyle\min_{\pmb x} f(\pmb x)$ s.t.  $g_i(\pmb x)\le 0, \: i=1,\cdots, m$
1. Reformulation with indicator: $\displaystyle\min_{\pmb x} J(\pmb x) $
1. Reformulation with Lagrangian: $\displaystyle\min_{\pmb x} \max_{\pmb\mu\ge \pmb0} L(\pmb x, \pmb\mu) $

# Duality

## Dual Problem

Given the primal problem
$$
\boxed{
\begin{aligned}
  \min_{\pmb x}\quad & f(\pmb x) \\
  \text{s.t.}\quad   & g_i(\pmb x)\le 0,  \: i=1,\cdots, m
\end{aligned}
}
$$

The assoicated dual problem is defined by

$$
\boxed{
\begin{aligned}
  \max_{\pmb\mu}\quad & D(\pmb\mu) \\
  \text{s.t.}\quad   & \pmb\mu \ge \pmb0  \\
\end{aligned}
}
$$
where
$$
\boxed{ 
  D(\pmb\mu) = \min_{\pmb x} L(\pmb x, \pmb\mu)
}
$$
is called the dual function.

Remarks:

* There is a one-to-one correspondence between primal constraint and dual variable
* There is a one-to-one correspondence between dual constraint and primal variable

The 1st remark is obviously true. The 2nd remark is true by considering the construction of $D(\pmb\mu)$. The optimizer 
$$
\hat{\pmb x} = \argmin_{\pmb x} L(\pmb x, \pmb\mu)  \in\mathbb R^n
$$
implicity depends on $\pmb\mu$, which impolses $n$ equality constraints on $\pmb\mu$.

### Why Bother with Dual Problem?

The original problem is constrained and hence hard to solve in general. The dual problem, however, is often easier to solve due to following observation:

* The dual function is defined as $D(\pmb\mu) = \displaystyle\min_{\pmb x} L(\pmb x, \pmb\mu)$ which is an unconstrained optimization problem and potentially easier to calculate.
* Suppose $\hat{\pmb x} = \displaystyle\argmin_{\pmb x} L(\pmb x, \pmb\mu)$. The dual objective
  $$
  D(\pmb\mu) = L(\hat{\pmb x}, \pmb\mu) = f(\hat{\pmb x}) + \pmb\mu^\top \pmb g(\hat{\pmb x})
  $$
  is an affine function in $\pmb\mu$. The dual problem $\displaystyle\max_{\pmb\mu \ge 0} D(\pmb\mu)$ is easy to solve.

## Min-Max Inequality

Before we introduce weak duality, we need to revisit min-max inequality, which is fundamental in proving weak duality theorem.

Let $X, Y $ be two sets and $\varphi: X \times Y \to\mathbb R$. The min-max inequality says
$$
\boxed{
  \max_{\pmb y}\min_{\pmb x} \varphi(\pmb x, \pmb y)
  \le
  \min_{\pmb x}\max_{\pmb y} \varphi(\pmb x, \pmb y)
}
$$

Proof:

* Minimizing over $\pmb x$ yields

$$
\forall \pmb y:\: \forall \pmb x, \:
\min_{\tilde {\pmb x}} \varphi(\tilde{\pmb x}, \pmb y)
\le
\varphi(\pmb x, \pmb y)
$$

* Maximizing over $\pmb y$ yields

$$
\forall \pmb x:\: \forall \pmb y, \:
\varphi(\pmb x, \pmb y)
\le
\max_{\tilde {\pmb y}} \varphi(\pmb x, \tilde{\pmb y})
$$

Hence,
$$
\forall \pmb x, \forall \pmb y, \:
\min_{\tilde {\pmb x}} \varphi(\tilde{\pmb x}, \pmb y)
\le
\max_{\tilde {\pmb y}} \varphi(\pmb x, \tilde{\pmb y})
$$

Note that the LHS depends only on $\pmb y$ and the RHS depends only on $\pmb x$. The inequality holds jointly for $\pmb x$ and $\pmb y$. Hence, taking the maximum of the LHS and the minimum of the RHS, we conclude.

## Weak Duality

Suppose $\pmb x^*$ is an optimal solution of the primal problem and $\pmb\mu^*$ be an optimal solution of the dual problem. Then, the weak duality holds:
$$
\boxed{
  D(\pmb\mu^*) \le f(\pmb x^*)
}
$$

Proof: This is a direct result of the min-max inequality applied to the Lagrangian:
$$
  \max_{\pmb\mu\ge 0} \underbrace{\min_{\pmb x} L(\pmb x, \pmb\mu)}_{D(\pmb\mu)}
  \le
  \min_{\pmb x} \underbrace{\max_{\pmb\mu\ge 0} L(\pmb x, \pmb\mu)}_{J(\pmb x)}
$$

The LHS is exactly the dual problem and the RHS is the primal problem reformulated with $J(\pmb x)$.


# Examples

## Linear Program (LP)

We would like to derive the dual problem of the LP
$$
\boxed{
  \:\:\:
  \begin{aligned}
  \max_{\pmb x \in\mathbb R^n}&  &\pmb{c}^\top\pmb{x}  \\
  \text{s.t.}&  &\pmb{Ax} &\le \pmb{b} \\
             &  &\pmb{x}  &\ge \pmb{0}
  \end{aligned}
  \quad
}
$$
where $\pmb c \in\mathbb R^n$, $\pmb A \in\mathbb R^{m\times n}$, $\pmb b \in\mathbb R^m$

First, we reformulate the primal LP into
$$
\begin{align*}
  \min_{\pmb x \in\mathbb R^n}&  & -\pmb{c}^\top\pmb{x}  \\
  \text{s.t.}&  &\pmb{Ax} - \pmb{b} &\le \pmb{0} \\
             &  & -\pmb{x}          &\le \pmb{0}
\end{align*}
$$

The Lagrangian is
$$
\begin{align*}
L(\pmb x, \pmb\mu, \pmb\lambda) 
&= -\pmb{c}^\top\pmb{x} + \pmb{\mu}^\top(\pmb{Ax} - \pmb{b}) - \pmb{\lambda}^\top\pmb{x} \\
&= (\pmb{A}^\top \pmb{\mu} - \pmb{c} -\pmb{\lambda})^\top \pmb{x} - \pmb{b}^\top \pmb{\mu}
\end{align*}
$$

The dual objective is
$$
D(\pmb\mu, \pmb\lambda) 
= \min_{\pmb x \in\mathbb R^n} L(\pmb x, \pmb\mu, \pmb\lambda) 
= \begin{cases} 
  -\pmb{b}^\top \pmb{\mu}, & \text{if } \pmb{A}^\top \pmb{\mu} - \pmb{c} -\pmb{\lambda} = 0 \\
  -\infty,                 & \text{else} \\
  \end{cases}
$$

To maximize the dual objective, it is sufficient to consider the case where $\pmb{A}^\top \pmb{\mu} - \pmb{c} -\pmb{\lambda} = 0$. The dual problem is therefore
$$
\begin{align*}
\max_{\pmb\mu,\, \pmb\lambda}\quad  &-\pmb{b}^\top \pmb{\mu}\\
\text{s.t.}\quad &\pmb{A}^\top \pmb{\mu} - \pmb{c} -\pmb{\lambda} = \pmb{0} \\
                 &\pmb\mu     \ge \pmb{0} \\
                 &\pmb\lambda \ge \pmb{0}
\end{align*}
$$

Note that
$$
\begin{align*}
\displaystyle\max_{\pmb\mu,\, \pmb\lambda} -\pmb{b}^\top \pmb{\mu}
&\iff \min_{\pmb\mu,\, \pmb\lambda} \pmb{b}^\top \pmb{\mu}
\\
\pmb{A}^\top \pmb{\mu} - \pmb{c} -\pmb{\lambda} = \pmb{0}  \:\land\: \pmb{\lambda}\ge 0
&\iff \pmb{A}^\top \pmb{\mu} - \pmb{c} \ge 0
\end{align*}
$$

Hence, the dual LP can be simplified to 
$$
\boxed{
  \:\:\:
  \begin{aligned}
  \min_{\pmb\mu}&  &\pmb{b}^\top \pmb{\mu}\\
  \text{s.t.}& &\pmb{A}^\top \pmb{\mu} &\ge \pmb{c} \\
            & &\pmb\mu &\ge \pmb{0} \\
  \end{aligned}
  \quad
}
$$