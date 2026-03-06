---
title: "Euler's Method for ODE"
date: "2024"
author: "Ke Zhang"
---
# Euler's Method for ODE

## Problem Formulation

Numerical method to solve 1st order ODE with the initial value problem (IVP):
$$
\begin{align}
x' = f(t, x), \quad x(t_0)=x_0
\end{align}
$$

where $x: \mathbb R \to \mathbb R, t\mapsto x(t)$ is the unknown function.

Analytical solution vs. numerical solution:

* Analytical solution: Obtain $x(\cdot)$ s.t. $x(\cdot)$ exactly satisfies the ODE and the initial condition. However, analytical solution is only easy to obtain for simple ODEs.

* Numerical solution: Obtain $x(t)$ for $t\in\{ t_0, t_1, \cdots, t_n \} \subset\mathbb R$. In practice, we often assume a small constant step size $\Delta t$, i.e.

$$
t_{k+1} = t_k + \Delta t, \quad k= 0,\cdots, n-1
$$

## The Algorithm

Key idea:

1. Starting from $t_0$, the derivative $x'(t_0)$ can be computed from the ODE and the initial condition:

    $$
    x'(t_0) = f(t_0, x_0)
    $$
1. $x(t_1)$ is approximated by linearizing $x(\cdot)$ at $t_0$: 

    $$
    \begin{align}
    x(t_1)
    &= x(t_0) + x'(t_0) (t_1 - t_0) \\
    &= x_0 + f(t_0, x_0) \cdot \Delta t
    \end{align}
    $$

1. We can repeat the above process to compute $x(t_2), x(t_3), \cdots, x(t_n)$.

In following, we use the short-hand notation:

* $t_k = t_0 + k\cdot \Delta t$
* $x_k = x(t_k)$
* $x'_k = x'(t_k)$

---

**Euler's Algorithm**  
Input: ODE: $x' = f(t, x)$,  initial condition: $x(t_0)=x_0$, step size: $\Delta t$.  
Output: $x(t_k)$ for $k\in\{ 0, 1, \cdots, n \}$ where $t_k = t_0 + k\cdot \Delta t$.

For $k = 0, 1, \cdots, n$:  
$\quad$ Compute derivative: $x'_k = f(t_k, x_k)$.  
$\quad$ Linear Approximation: $x_{k+1} = x_k + x'_k\cdot \Delta t$.

---

## Example

Consider the ODE:

$$
x' = -2x, \quad x(0) = 1
$$

The analytical solution is $x(t) = e^{-2t}$. We can apply Euler's method to compute the numerical solution. Let $\Delta t = 0.1$, then we have:

| $t_k$ | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|-------|-----|-----|-----|-----|-----|-----|
| $x_k'$ | -2.0 | -1.6 | -1.28 | -1.024 | -0.8192 | -0.65536 |
| $x_k$ | 1.0 | 0.8 | 0.64 | 0.512 | 0.4096 | 0.32768 |
