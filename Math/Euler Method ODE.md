---
title: "Euler's Method for ODE"
date: "2024"
author: "Ke Zhang"
---
# Euler's Method for ODE

## Problem Formulation

Numerical method to solve 1st order ODE with the initial value problem (IVP):
$$
  x' = f(t, x), \quad x(0)=x_0
$$

where $x: \mathbb R \to \mathbb R, t\mapsto x(t)$ is the unknown function.

Distinguish analytical & numberical solution

* Analytical solution: Obtain $x(t)$ for $\forall t \in\mathbb R$

* Numerical solution: Obtain $x(t)$ for $t\in\{ t_0, t_1, \cdots, t_n \} \subset\mathbb R$. In practice, we often assume constant step size $\Delta t$, i.e.

$$
  t_k = k\cdot \Delta t, \quad k= 0,\cdots, n
$$

## Euler's Method

### Key observation

1. The derivative at time $0$ can be computed from the ODE itself: $x'(t_0) = f(t_0, x_0)$ since the initial values are given
2. $x(t_1)$ is approximated by linearizing $x$ at $t_0$: $x(t_1) = x(t_0) + x'(t_0) (t_1 - t_0)$

In following, we use the short-hand notation:

* $t_k = k\cdot \Delta t$
* $x_k = x(t_k)$
* $x'_k = x'(t_k)$

### The Algorithm

Given IVP: $ x' = f(t, x), \: x(0)=x_0$, solve $x(t_k)$ for $k\in\{ 0, 1, \cdots, n \}$.

> **Euler's Algorithm**
>
> * For $k = 0, 1, \cdots, n$:
>   * Compute derivative: $x'_k = f(t_k, x_k)$
>   * Linear Approximation: $x_{k+1} = x_k + x'_k\cdot (t_{k+1} - t_k)$
