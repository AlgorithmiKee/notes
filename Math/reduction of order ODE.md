---
title: "Reduction of Order"
author: "Ke Zhang"
date: "2026"
---

# Reduction of Order

Consider the second-order linear homogeneous DE:

$$
y'' + P(x) y' + Q(x) y = 0
$$

Suppose $\varphi_1(\cdot)$ is a known nontrivial solution of the DE. We want to find a second linearly independent solution $\varphi_2(\cdot)$.

## Deriving the Second Solution

To do this, we can use the method of reduction of order. We assume that $\varphi_2(x)$ has the form:

$$
\varphi_2(x) = v(x) \varphi_1(x)
$$

where $v(\cdot)$ is an unknown function to be determined. We will substitute $\varphi_2$ into the original DE and solve for $v(\cdot)$.

First, we compute the derivatives of $\varphi_2$:

$$
\begin{aligned}
\varphi_2' &= v' \varphi_1 + v \varphi_1' \\
\varphi_2'' &= v'' \varphi_1 + 2 v' \varphi_1' + v \varphi_1''
\end{aligned}
$$

Next, we substitute $\varphi_2$, $\varphi_2'$, and $\varphi_2''$ into the original DE:

$$
\begin{aligned}
(v'' \varphi_1 + 2 v' \varphi_1' + v \varphi_1'') + P(x) (v' \varphi_1 + v \varphi_1') + Q(x) v \varphi_1 &= 0 \\
\varphi_1 v'' + (2 \varphi_1' + P(x) \varphi_1) v' + (\varphi_1'' + P(x) \varphi_1' + Q(x) \varphi_1) v &= 0
\end{aligned}
$$

Since $\varphi_1$ is a solution of the original DE, we have:

$$
\varphi_1'' + P(x) \varphi_1' + Q(x) \varphi_1 = 0
$$

Hence, $v$ must satisfy:

$$
\varphi_1 v'' + (2 \varphi_1' + P(x) \varphi_1) v' = 0
$$

This is a first-order linear DE in $v'$. We can rewrite it as:

$$
v'' + \left( \frac{2 \varphi_1'}{\varphi_1} + P(x) \right) v' = 0
$$

Let $w = v'$. Then we have:

$$
w' + \left( \frac{2 \varphi_1'}{\varphi_1} + P(x) \right) w = 0
$$

This is a first-order linear homogeneous DE in $w$, which is separable and can be solved by direct integration:

$$
\begin{aligned}
\frac{1}{w} dw &= -\left( \frac{2 \varphi_1'}{\varphi_1} + P(x) \right) dx \\
\ln (w) &= -\int \left( \frac{2 \varphi_1'}{\varphi_1} + P(x) \right) dx + \tilde C \\
\ln (w) &= -2 \ln (\varphi_1) - \int P(x) dx + \tilde C \\
w &= C \frac{e^{-\int P(x) dx}}{\varphi_1^2} \\
\end{aligned}
$$

Note that it is sufficient to find one $\varphi_2$, so we can set $C = 1$ in $w$ without loss of generality:

$$
w = \frac{e^{-\int P(x) dx}}{\varphi_1^2}
$$

Finally, we can integrate $w$ to find $v$:

$$
v = \int w dx = \int \frac{e^{-\int P(x) dx}}{\varphi_1^2} dx
$$

Thus, a second solution $\varphi_2$ is given by:

$$
\begin{aligned}
\varphi_2(x) &= v(x) \varphi_1(x) \\
& = \varphi_1(x) \int \frac{e^{-\int P(x) dx}}{\varphi_1(x)^2} dx
\end{aligned}
$$

This is the commonly used formula for the second solution, often referred to as the reduction of order formula, for a second-order linear homogeneous DE.

## Linear Independence of the Solutions

**Claim**: Given a nontrivial solution $\varphi_1$ of the original differential equation, and a second solution $\varphi_2$ constructed using the reduction of order formula, $\varphi_1$ and $\varphi_2$ are linearly independent.

*Proof*: We can compute the Wronskian of $\varphi_1$ and $\varphi_2$:

$$
\begin{aligned}
W(\varphi_1, \varphi_2)
&= \varphi_1 \varphi_2' - \varphi_1' \varphi_2 \\
&= \varphi_1 (v' \varphi_1 + v \varphi_1') - \varphi_1' (v \varphi_1) \\
&= \varphi_1^2 v' \\
&= \varphi_1^2 \frac{e^{-\int P(x) dx}}{\varphi_1^2} \\
&= e^{-\int P(x) dx}
\end{aligned}
$$

Since $e^{-\int P(x) dx}$ is never zero, the Wronskian is nonzero for all $x$. Therefore, $\varphi_1$ and $\varphi_2$ are linearly independent. $\quad\square$

The second solution $\varphi_2$ can also be expressed in terms of the Wronskian as:

$$
\varphi_2(x) = \varphi_1(x) \int \frac{W(\varphi_1, \varphi_2)}{\varphi_1(x)^2} dx
$$

The general solution to the original DE can then be written as:

$$
\varphi(x) = C_1 \varphi_1(x) + C_2 \varphi_2(x)
$$

where $C_1$ and $C_2$ are arbitrary constants.

## Example: Cauchy–Euler Equation

The Cauchy–Euler equation is a special case of the second-order linear homogeneous DE, given by:

$$
x^2 y'' - a x y' + b y = 0
$$

It is easy to verify that $\varphi_1(x) = x^r$ is a solution to the Cauchy–Euler equation if $r$ satisfies the characteristic equation:

$$
r^2 - (a+1) r + b = 0
$$

Suppose the characteristic equation has a repeated root

$$
\Delta = (a+1)^2 - 4b = 0 \implies r_0 = \frac{a+1}{2}
$$

Then, we can use the reduction of order formula to find the Wronskian:

$$
\begin{aligned}
W(x)
&= e^{-\int P(x) dx} \\
&= e^{-\int \frac{-a}{x} dx} \\
&= e^{a \ln(x)} \\
&= x^a
\end{aligned}
$$

and the second solution:

$$
\begin{aligned}
\varphi_2(x)
&= \varphi_1(x) \int \frac{W(x)}{\varphi_1(x)^2} dx \\
&= x^{r_0} \int \frac{x^a}{x^{2 r_0}} dx \\
&= x^{r_0} \int x^{-1} dx \\
&= x^{r_0} \ln(x)
\end{aligned}
$$

Thus, the general solution to the Cauchy–Euler equation with repeated roots is:

$$
\varphi(x) = C_1 x^{r_0} + C_2 x^{r_0} \ln(x)
$$
