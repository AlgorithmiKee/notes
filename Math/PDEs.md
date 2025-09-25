---
title: "PDEs"
date: "2024"
author: "Ke Zhang"
---

# Partial Differential Equations

[toc]

$$
\DeclareMathOperator*{\argmax}{argmax}
\DeclareMathOperator*{\argmin}{argmin}
$$

## Continuity Equation

The continuity equation is a first-order partial differential equation expressing local mass conservation in a fluid.

Consider a fluid with density $\rho(\mathbf{x}, t)$ and velocity $\mathbf{v}(\mathbf{x}, t)$ at location $\mathbf{x} \in\mathbb R^n$ and time $t$. In physics, $n$ is typically 3.

Let $\Omega \subseteq \mathbb{R}^n$ be a **control volume** with boundary $\partial \Omega$. By conservation of mass, we have for any $t$:

$$
\begin{align}
\frac{\mathrm d}{\mathrm dt} \int_{\Omega} \rho \:\mathrm{d} V
&= - \oint_{\partial\Omega} \rho \mathbf{v} \:\mathrm d \mathbf{A}
\end{align}
$$

Remarks:

* The LHS describes the rate of change of the mass inside the volume $\Omega$.
* The integral on RHS describes the net outflow out of the surface $\partial\Omega$. Positive net outflow reduces the mass inside $\Omega$, hence we need the minus sign.
* The conservation of mass holds at any location $\mathbf{x}$ and at any time $t$.

On the LHS, we can exchanging the order of integral and time derivative.

$$
\frac{\mathrm d}{\mathrm dt} \int_{\Omega} \rho \:\mathrm{d} V
= \int_{\Omega} \frac{\partial}{\partial t}  \rho \:\mathrm{d} V
$$

On the RHS, we apply the Gaussian integral law.

$$
\oint_{\partial\Omega} \rho \mathbf{v} \:\mathrm d \mathbf{A}
= \int_{\Omega}  \nabla \cdot (\rho \mathbf{v}) \:\mathrm d V
$$

Together,

$$
\begin{align}
\int_{\Omega} \frac{\partial}{\partial t}  \rho \:\mathrm{d} V
&= - \int_{\Omega}  \nabla \cdot (\rho \mathbf{v}) \:\mathrm d V
\end{align}
$$

Rearranging the term yields

$$
\begin{align*}
\int_{\Omega} \left[
    \frac{\partial}{\partial t}  \rho + \nabla \cdot (\rho \mathbf{v})
\right]\:\mathrm dV = 0
\end{align*}
$$

Recall this equality must hold everywhere at any time. Hence, the integrand must be zero function. Therefore, we conclude the ***continuity equation*** in fluid mechanics.

$$
\begin{align}
\frac{\partial}{\partial t}  \rho + \nabla \cdot (\rho \mathbf{v}) = 0
\end{align}
$$

Remarks:

* Again, the contiuity equation holds everywhere at any time.
* For incompressible flow ($\rho = \text{const}$), the contiuity equation simplies to
  $$
  \nabla \cdot (\rho \mathbf{v}) = 0
  $$

Using the product rule in multivariate calculus

$$
\nabla \cdot (\rho \mathbf{v}) = \mathbf{v} \cdot \nabla \rho + \rho (\nabla \cdot \mathbf{v})
$$

The contiuity equation can be reformulated as

$$
\begin{align}
\frac{\partial \rho}{\partial t} + \mathbf{v} \cdot \nabla \rho + \rho (\nabla \cdot \mathbf{v}) = 0
\end{align}
$$
