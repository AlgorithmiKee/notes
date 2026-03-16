---
title: "Linear 2nd Order ODEs"
author: "Ke Zhang"
date: 2026
---

# Linear 2nd Order ODEs

[toc]

Consider the following second order linear inhomogeneous ODE:

$$
y'' + p(t)y' + q(t)y = f(t)
$$

where $p(t)$, $q(t)$ and $f(t)$ are continuous functions on an interval $I$.

Remarks:

- The ODE is **linear** because the LHS is a linear function of $y$ and its derivatives. In other words, the ODE can be expressed in the form $L[y] = f$, where $L$ is a linear differential operator.
- The ODE is of **second order** because the highest derivative of $y$ that appears in the equation is the second order derivative $y''$.
- The ODE is called **homogeneous** if $f(t) = 0$ for all $t \in I$. Otherwise, it is called **inhomogeneous**.
- The functions $p(t)$ and $q(t)$ are called the **coefficients** of the ODE. If they are constants, we say the ODE has **constant coefficients**. Otherwise, we say the ODE has **varying coefficients**.

## Physical Interpretation of ODEs

The inhomogeneous ODE can be used to model a wide variety of physical systems, where we consider

- the variable $t$ to represent time
- the variable $y$ to represent the state of a system at time $t$
- the functions $p(t)$ and $q(t)$ to represent the properties of the system. If they are constants, we say the system is **time-invariant**. Otherwise, we say the system is **time-varying**.
- the inhomogeneous term $f(t)$ to represent an external input acting on the system. If $f(t)$ is zero, we say the system is **unforced**. Otherwise, we say the system is **forced**.

The second order is often associated with oscillatory systems, where the state of the system at time $t$ depends on its current state and its rate of change. The inhomogeneous term $f(t)$ can be interpreted as an external force acting on the system, which can cause it to deviate from its natural behavior.

**Example: mechanical vibrations**.
Consider a mass-spring-damper system, where a mass $m$ is attached to a spring with spring constant $k$ and a damper with damping coefficient $c$. The equation of motion for this system can be written as:

$$
m y'' + c y' + k y = F(t)
$$

Here, $y(t)$ represents the displacement of the mass from its equilibrium position at time $t$, and $F(t)$ represents an external force acting on the system. This is a second order linear inhomogeneous ODE, where

- $p(t) = \frac{c}{m}$ (mechanical damping coefficient per unit mass)

- $q(t) = \frac{k}{m}$ (spring constant per unit mass)

- $f(t) = \frac{F(t)}{m}$ (external force per unit mass)

**Example: electrical circuits**.
Consider an RLC circuit, where a resistor with resistance $R$, an inductor with inductance $L$, and a capacitor with capacitance $C$ are connected in series. The voltage across the circuit can be described by the following ODE:

$$
L y'' + R y' + \frac{1}{C} y = V(t)
$$

Here, $y(t)$ represents the charge on the capacitor at time $t$, and $V(t)$ represents an external voltage source. This is also a second order linear inhomogeneous ODE, where

- $p(t) = \frac{R}{L}$ (electrical damping coefficient per unit inductance)

- $q(t) = \frac{1}{LC}$ (inverse of the product of inductance and capacitance)

- $f(t) = \frac{V(t)}{L}$ (external voltage per unit inductance)

## 2nd Order ODE as a System of 1st Order ODEs

The 2nd order ODE $y'' + p(t)y' + q(t)y = f(t)$ can be rewritten as a system of two 1st order ODEs by introducing

$$
y_1 = y, \quad y_2 = y'
$$

Then we have:

$$
\begin{bmatrix}
y_1'(t) \\
y_2'(t)
\end{bmatrix}
=
\begin{bmatrix}
0 & 1 \\
-q(t) & -p(t)
\end{bmatrix}
\begin{bmatrix}
y_1(t) \\
y_2(t)
\end{bmatrix}
+
\begin{bmatrix}
0 \\
f(t)
\end{bmatrix}
$$

Let

$$
\mathbf{y}(t) = \begin{bmatrix} y_1(t) \\ y_2(t) \end{bmatrix}, \quad
\mathbf{A}(t) = \begin{bmatrix} 0 & 1 \\ -q(t) & -p(t) \end{bmatrix}, \quad
\mathbf{f}(t) = \begin{bmatrix} 0 \\ f(t) \end{bmatrix}
$$

Then the system can be compactly written as:

> $$
> \mathbf{y}'(t) = \mathbf{A}(t) \mathbf{y}(t) + \mathbf{f}(t)
> $$

Remarks:

- The system of 1st order ODEs is equivalent to the original 2nd order ODE in the following sense:
  - $\mathbf{y}'(t)$ is an affine transformation of $\mathbf{y}(t)$, which reflects the linearity of the original ODE.
  - $\mathbf{A}(t)$ contains the coefficients of the original ODE.
  - $\mathbf{f}(t)$ represents the inhomogeneous term of the original ODE.
- The equivalence is not just for 2nd order ODEs. Using the same logic, any linear $n$-th order ODE can be rewritten as a system of $n$ 1st order ODEs (but not vice versa). Hence, linear first-order systems theory is more universal.

**Example**: A damped harmonic oscillator can be described by the linear 2nd order ODE:

$$
y'' + 2\zeta\omega y' + \omega^2 y = 0
$$

where $\zeta$ is the damping ratio and $\omega$ is the natural frequency of the oscillator.

Introducing $y_1 = y$ and $y_2 = y'$, we can rewrite this ODE as a system of 1st order ODEs:

$$
\begin{bmatrix}
y_1' \\
y_2'
\end{bmatrix}
=
\begin{bmatrix}
0 & 1 \\
-\omega^2 & -2\zeta\omega
\end{bmatrix}
\begin{bmatrix}
y_1 \\
y_2
\end{bmatrix}
$$

### Existence and Uniqueness Theorem

> **Theorem**: Suppose $\mathbf{A}(t)$ and $\mathbf{f}(t)$ are continuous on an interval $I$ containing $t_0$. Given any initial conditions $y(t_0) = y_0$ and $y'(t_0) = y_1$, there exists a unique solution to the inhomogeneous equation.
>
> $$
> \mathbf{y}'(t) = \mathbf{A}(t) \mathbf{y}(t) + \mathbf{f}(t), \quad \mathbf{y}(t_0) = \begin{bmatrix} y_0 \\ y_1 \end{bmatrix}
> $$

Remarks:

- This theorem is a special case of Picard-Lindelöf theorem, which does not require the system to be linear. The linearity of the system allows us to relax the Lipschitz condition to mere continuity.
- This theorem guarantees the existence and uniqueness of solutions to the original 2nd order ODE, which is useful e.g. for proving the dimension of the solution space to the homogeneous equation.
- The proof requires metric space and fixed point theorem, which are beyond the scope of this note. Omitted here.

If we can solve a general system of 1st order ODEs, we can solve any linear 2nd order ODE. However, the former approach is beyond the scope of this note. We will turn back to the original 2nd order ODE and discuss the structure of the solutions to homogeneous and inhomogeneous equations, as well as methods for finding these solutions.

## Homogeneous Equations

### Structure of the Solutions to Homogeneous Equations

> **Theorem**: The set of all solutions to the homogeneous equation
>
> $$
> y'' + p(t)y' + q(t)y = 0
> $$
>
> forms a 2-dimensional vector space.

Remarks:

- By this theorem, there exist two linearly independent solutions $y_1$ and $y_2$ to the homogeneous equation such that any solution to the homogeneous equation can be expressed as a linear combination of $y_1$ and $y_2$:
    $$
    y(t) = C_1 y_1(t) + C_2 y_2(t)
    $$
- The linearly independent solutions $y_1$ and $y_2$ above are known as **fundamental solutions**. They form a basis for the solution space of the homogeneous equation.
- For a general second-order linear ODE with varying coefficients, the fundamental solutions may not have a simple closed form.

*Proof*: Let $\mathcal{S}$ be the set of all solutions to the homogeneous equation. It is easy to verify that $\mathcal{S}$ is a vector space. To show that $\dim(\mathcal{S}) = 2$, we use the [existence and uniqueness theorem](#existence-and-uniqueness-theorem) for ODEs:

Given any initial conditions $y(t_0) = y_0$ and $y'(t_0) = y_1$, there exists a unique solution $y(t)$ to the homogeneous equation.

Define a mapping

$$
\Phi: \mathcal{S} \to \mathbb{R}^2, \quad \Phi(y) = (y(t_0), y'(t_0))
$$

By existence, the mapping $\Phi$ is surjective. By uniqueness, it is also injective. Hence, $\Phi$ is an isomorphism. Since $\mathbb{R}^2$ has dimension two, the solution space $\mathcal{S}$ must also have dimension two. $\quad\blacksquare$

> **Abel's identity**: The Wronskian determinant of two solutions $y_1$ and $y_2$ to the homogeneous equation satisfies the following identity:
>
> $$
> W(y_1, y_2)(t) = C e^{-\int p(t) dt}
> $$
>
> for all $t$ in the interval $I$, where $C$ is a constant.

Remarks:

- Given two solutions $y_1$ and $y_2$ to the homogeneous equation, their Wronskian determinant $W(y_1, y_2)(t)$ is either zero everywhere or never zero in the interval $I$.
- If $W(y_1, y_2)(t_0) \ne 0$ for some $t_0 \in I$, then $y_1$ and $y_2$ are linearly independent and form a fundamental set of solutions to the homogeneous equation.
- In practice, though, testing the linear dependency between $y_1$ and $y_2$ is straightforward as we only need to check if one is multiple of the other. Abel's identity is a bit overkill for this purpose. The actual power of the Abel's identity lies in its ability to derive a second solution from a known solution.  
  $\blacktriangleright$ See section [reduction of order](#reduction-of-order-for-varying-coefficients).

*Proof*: By definition of Wronskian,

$$
\begin{align*}
W(y_1, y_2)(t)
&= \det
   \begin{bmatrix}
   y_1(t) & y_2(t) \\
   y_1'(t) & y_2'(t)
   \end{bmatrix}
\\
&= y_1(t)y_2'(t) - y_2(t)y_1'(t)
\end{align*}
$$

Taking the derivative, we get

$$
\begin{align*}
W'
&=y_1'y_2' + y_1y_2'' - (y_2'y_1' + y_2y_1'') \\
&= y_1y_2'' - y_2y_1''
\end{align*}
$$

Since $y_1$ and $y_2$ are solutions to the homogeneous equation, we have

$$
y_1'' + py_1' + qy_1 = 0 \implies y_1'' = -py_1' - qy_1
$$

$$
y_2'' + py_2' + qy_2 = 0 \implies y_2'' = -py_2' - qy_2
$$

Hence, $W'$ becomes

$$
\begin{align*}
W'
&= y_1(-py_2' - qy_2) - y_2(-py_1' - qy_1) \\
&= -p(\underbrace{y_1y_2' - y_2y_1'}_{W})
\end{align*}
$$

Hence, $W(y_1, y_2)(t)$ satisfies the 1st order ODE

$$
W'(y_1, y_2)(t) = -p(t)W(y_1, y_2)(t)
$$

which has the solution

$$
W(y_1, y_2)(t) = Ce^{-\int p(t) dt}
\tag*{$\blacksquare$}
$$

### Methods to Solve Homogeneous Equations

Depending on the form of the coefficients $p(t)$ and $q(t)$, there are various methods for finding the fundamental solution, including:

1. **Characteristic Equation**: This method is applicable when the coefficients $p(t)$ and $q(t)$ are constants. We can assume a solution of the form $y = e^{rt}$ and derive a characteristic equation to find the values of $r$.
2. **Reduction of Order**: This method is applicable when we already know one solution to the homogeneous equation. We can use this known solution to reduce the order of the ODE and find a second linearly independent solution.

### Characteristic Equations for Constant Coefficients

Suppose the ODE has constant coefficients:

$$
ay'' + by' + cy = 0
$$

> **Theorem**: $e^{rt}$ is a solution if $r\in\mathbb{C}$ is a root of the characteristic equation
>
> $$
> a r^2 + b r + c = 0
> $$

*Proof*:  Trivial.

Note that the characteristic equation always has two roots on the complex plane, which can be real or complex conjugates.

> **Case 1**: If the roots are real and distinct, the general solution to the homogeneous equation is given by:
>
> $$
> y_h(t) = C_1 e^{r_1 t} + C_2 e^{r_2 t}
> $$
>
> **Case 2**: If the roots are complex conjugates, say $r = \alpha \pm i\beta$, the general solution takes the form:
>
> $$
> y_h(t) = e^{\alpha t}(C_1 \cos(\beta t) + C_2 \sin(\beta t))
> $$
>
> **Case 3**: If the roots are repeated, say $r_1 = r_2$, the general solution takes the form:
>
> $$
> y_h(t) = (C_1 + C_2 t)e^{r_1 t}
> $$

*Proof of case 1*:  Trivial.

*Proof of case 2*:  The roots of characteristic equation gives two fundamental solutions:

$$
y_1(t) = e^{\alpha t + i\beta t}, \quad y_2(t) = e^{\alpha t - i\beta t}
$$

Using Euler's formula, the solution space can be expressed w.r.t. another basis:

$$
\begin{align*}
z_1(t)
&= \frac{y_1(t) + y_2(t)}{2} = e^{\alpha t} \cos(\beta t) \\
z_2(t)
&= \frac{y_1(t) - y_2(t)}{2i} = e^{\alpha t} \sin(\beta t)
\tag*{$\blacksquare$}
\end{align*}
$$

*Proof of case 3*: The repeated root directly gives one fundamental solution

$$
y_1(t) = e^{r_1 t}
$$

where $r_1 = -\frac{b}{2a}$.

To find a second linearly independent solution, we can use the method of reduction of order.

$$
y_2(t) = y_1(t) \int \frac{e^{-\int p(t) dt}}{y_1(t)^2} dt
$$

where $p(t) = \frac{b}{a} = -2r_1$ is a constant.

Hence,

$$
\begin{align*}
y_2(t)
&= y_1(t) \int \frac{e^{2r_1 t}}{y_1(t)^2} dt \\
&= e^{r_1 t} \int \frac{e^{2r_1 t}}{(e^{r_1 t})^2} dt \\
&= e^{r_1 t} \int 1 dt \\
&= e^{r_1 t} t + C
\end{align*}
$$

Since we only need one linearly independent solution, we can set $C = 0$. Hence, we conclude that

$$
y_2(t) = e^{r_1 t} t
\tag*{$\blacksquare$}
$$

### Reduction of Order for Varying Coefficients

In the case of varying coefficients, the methods for finding solutions are generally more complex and may not yield closed-form solutions. However, if we can guess a solution, we can often use the method of reduction of order to find a second linearly independent solution.

Suppose $y_1(t)$ is a known solution to the homogeneous equation

$$
y'' + p(t)y' + q(t)y = 0
$$

Let $y_2(t)$ be any other solution to the homogeneous equation that is linearly independent of $y_1(t)$. The Wronskian determinant of $y_1$ and $y_2$ is given by

$$
W(y_1, y_2)(t) = y_1(t) y_2'(t) - y_2(t) y_1'(t)
$$

By Abel's identity, we have

$$
W(y_1, y_2)(t) = C e^{-\int p(t) dt}
$$

Combining both identities, we find that $y_2$ must satisfy

$$
y_1 y_2' - y_2 y_1' = C e^{-\int p(t) dt}
$$

Since $y_2$ is linearly independent of $y_1$, we must ensure $C \ne 0$. To construct a specific solution $y_2$, it is sufficient to let $C = 1$ and solve the 1st order ODE:

$$
y_1 y_2' - y_2 y_1' = e^{-\int p(t) dt}
$$

Dividing both sides by $y_1^2$, we obtain

$$
\begin{align*}
\frac{y_1 y_2' - y_2 y_1'}{y_1^2} &= \frac{e^{-\int p(t) dt}}{y_1^2}
\\
\left( \frac{y_2}{y_1} \right)' &= \frac{e^{-\int p(t) dt}}{y_1^2}
\\
\frac{y_2}{y_1} &= \int \frac{e^{-\int p(t) dt}}{y_1^2} dt
\end{align*}
$$

Multiplying both sides by $y_1$, we obtain a second solution $y_2(t)$:

> $$
> y_2(t) = y_1(t) \int \frac{e^{-\int p(t) dt}}{y_1(t)^2} dt
> $$

In engineering applications, we often define

$$
v(t) = \int \frac{e^{-\int p(t) dt}}{y_1(t)^2} dt
$$

so that the second solution can be expressed as a product $y_2(t) = v(t) y_1(t)$.

**Example**: Consider the ODE

$$
t^2 y'' - 3ty' + 4y = 0, \quad t > 0
$$

We can verify that $y_1(t) = t^2$ is a solution. By reduction of order, we can form a second solution

$$
y_2(t) = y_1(t) \int \frac{e^{-\int p(t) dt}}{y_1(t)^2} dt
$$

where $p(t) = -\frac{3}{t}$.

Hence,

$$
\begin{align*}
y_2(t)
&= t^2 \int \frac{e^{-\int -\frac{3}{t} dt}}{(t^2)^2} dt \\
&= t^2 \int \frac{e^{3\ln(t)}}{t^4} dt \\
&= t^2 \int \frac{1}{t} dt \\
&= t^2 \ln(t)
\end{align*}
$$

Hence, the general solution is given by:

$$
y(t) = C_1 t^2 + C_2 t^2 \ln(t)
$$

## Inhomogeneous Equations

### Structure of the Solutions to Inhomogeneous Equations

> **Theorem**: The general solution to the inhomogeneous equation
>
> $$
> y'' + p(t)y' + q(t)y = f(t)
> $$
>
> can be expressed as the sum of the general solution to the corresponding homogeneous equation and a particular solution to the inhomogeneous equation:
>
> $$
> y(t) = y_h(t) + y_p(t)
> $$
>
> where
>
> - $y_h$ is the general solution to the homogeneous equation (also known as the **complementary solution**).
> - $y_p$ is a **particular solution** to the inhomogeneous equation.

*Proof*: It is easy to verify that the sum of a complementary solution and a particular solution is indeed a solution to the inhomogeneous equation. It remains to show that any arbitrary solution to the inhomogeneous equation can be expressed in this form.

Let $y(t)$ be any solution to the inhomogeneous equation. Then we have

$$
y'' + p(t)y' + q(t)y = f(t)
$$

Since $y_p$ is a particular solution, it satisfies the inhomogeneous equation:

$$
y_p'' + p(t)y_p' + q(t)y_p = f(t)
$$

Subtracting this equation from the original equation, we get:

$$
(y - y_p)'' + p(t)(y - y_p)' + q(t)(y - y_p) = 0
$$

Hence, their difference $y_h(t) = y(t) - y_p(t)$ satisfies the homogeneous equation:

$$
y_h'' + p(t)y_h' + q(t)y_h = 0
\tag*{$\blacksquare$}
$$

Since the ODE has the order of two, we need two fundamental solutions to the homogeneous equation to form the complementary solution. Let $y_1$ and $y_2$ be two such solutions. Then the complementary solution can be written as:

$$
y_h(t) = C_1 y_1(t) + C_2 y_2(t)
$$

Thus, the general solution to the inhomogeneous equation can be expressed as:

$$
y(t) = C_1 y_1(t) + C_2 y_2(t) + y_p(t)
$$

### Superposition Principle

If the inhomogeneous term $f(t)$ consists of a sum of functions

$$
f(t) = f_1(t) + f_2(t)
$$

and if we can find particular solutions $y_{p1}$ and $y_{p2}$ to the equations

$$
y_{p1}'' + p(t)y_{p1}' + q(t)y_{p1} = f_1(t)
$$

$$
y_{p2}'' + p(t)y_{p2}' + q(t)y_{p2} = f_2(t)
$$

then a particular solution to the original inhomogeneous equation can be expressed as the sum of the particular solutions to the individual equations:

$$
y_p(t) = y_{p1}(t) + y_{p2}(t)
$$

If the inhomogeneous term $f(t)$ consists of a product of a function and a constant

$$
f(t) = k f_0(t)
$$

and if we can find a particular solution $y_{p0}$ to the equation

$$
y_{p0}'' + p(t)y_{p0}' + q(t)y_{p0} = f_0(t)
$$

then a particular solution to the original inhomogeneous equation can be expressed as the product of the particular solution to the individual equation and the constant:

$$
y_p(t) = k y_{p0}(t)
$$

### Methods to Solve Inhomogeneous ODEs

There are several methods for finding a particular solution to an inhomogeneous ODE, including:

1. **Method of Undetermined Coefficients**: This method is applicable when (i) the coefficients $p(t)$ and $q(t)$ are constants, and (ii) the inhomogeneous term $f(t)$ is a simple function such as a polynomial, exponential, sine, or cosine. The idea is to guess a form for $y_p$ based on the form of $f(t)$ and then determine the coefficients by substituting back into the ODE.
2. **Variation of Parameters**: This method is more general and can be used when the method of undetermined coefficients is not applicable. It involves using the solutions to the homogeneous equation to construct a particular solution to the inhomogeneous equation.
3. **Laplace Transform**: This method is particularly useful for solving ODEs with constant coefficients and can be used to find a particular solution by transforming the ODE into an algebraic equation in the Laplace domain.

Here, we will focus on the method of undetermined coefficients and variation of parameters.

### Method of Undetermined Coefficients

Consider an inhomogeneous ODE of the form

$$
a y'' + b y' + c y = f(t)
$$

Suppose the inhomogeneous term $f(t)$ is of a form listed in the table below. Then, we can guess the form of a particular solution $y_p$ based on the form of $f(t)$.

| Form of $f(t)$    | Guess for $y_p$         |
| ------------------------- | ------------------------- |
| $f(t) = ae^{\sigma t}$  | $y_p = Ae^{\sigma t}$  |
| $f(t) = a \cos(\omega t) + b \sin(\omega t)$ | $y_p = A \cos(\omega t) + B \sin(\omega t)$  |
| $f(t) = P_n(t)$ (polynomial of degree $n$)   | $y_p = Q_n(t)$ (polynomial of degree $n$)  |
| $f(t) = e^{\sigma t}P_n(t)$  | $y_p = e^{\sigma t}Q_n(t)$   |
| $f(t) = e^{\sigma t}(a \cos(\omega t) + b \sin(\omega t))$ | $y_p = e^{\sigma t}(A \cos(\omega t) + B \sin(\omega t))$ |

**VERY IMPORTANT**: Having made the guess, we need to check if the guessed form for $y_p$ is already part of the complementary solution. If it is the case, we need to multiply our guess by $t^m$, where $m$ is the smallest positive integer such that the new guess is not part of the complementary solution.

- Mathematically, we say **resonance** occurs when the inhomogeneous term $f(t)$ happens to be a solution to the homogeneous equation.
- Physically, resonance occurs when the external force $f(t)$ happens to match the natural frequency of the system, leading to large oscillations in the system's response.

After checking for resonance, we can substitute our guess for $y_p$ into the original ODE and solve for the coefficients.

**Example: non-resonant system**.  
Consider the following ODE:

$$
y'' + 3y' + 2y = 10\sin(2t)
$$

The complementary solution to the homogeneous equation $y'' + 3y' + 2y = 0$ is given by:

$$
y_h(t) = C_1 e^{-t} + C_2 e^{-2t}
$$

Since the inhomogeneous term $f(t) = 10\sin(2t)$ is not part of the complementary solution, we can guess a particular solution of the form:

$$
y_p(t) = A \cos(2t) + B \sin(2t)
$$

Substituting $y_p$ into the original ODE, we get:

$$
\begin{aligned}
(-4A \cos(2t) - 4B \sin(2t)) + 3(-2A \sin(2t) + 2B \cos(2t)) + 2(A \cos(2t) + B \sin(2t)) &= 10 \sin(2t)
\\
(-2A + 6B) \cos(2t) + (- 6A - 2B ) \sin(2t) &= 10 \sin(2t)
\end{aligned}
$$

Equalizing the coefficients of $\cos(2t)$ and $\sin(2t)$, we obtain the coefficients $A$ and $B$:

$$
\begin{cases}
-2A + 6B = 0 \\
-6A - 2B = 10
\end{cases}
\implies
\begin{cases}
A = -\frac{3}{2} \\
B = -\frac{1}{2}
\end{cases}
$$

Thus, the particular solution is:

$$
y_p(t) = -\frac{3}{2} \cos(2t) - \frac{1}{2} \sin(2t)
$$

The general solution to the inhomogeneous ODE is:

$$
y(t) = C_1 e^{-t} + C_2 e^{-2t} - \frac{3}{2} \cos(2t) - \frac{1}{2} \sin(2t)
$$

**Example: resonant system**.  
Consider the following ODE:

$$
y'' + 2y' + y = 10e^{-t}
$$

The complementary solution to the homogeneous equation $y'' + 2y' + y = 0$ is given by:

$$
y_h(t) = C_1 e^{-t} + C_2 t e^{-t}
$$

Since the inhomogeneous term $f(t) = 10e^{-t}$ is part of the complementary solution, we need to multiply our guess for the particular solution by at least $t^2$ to ensure that it is not part of the complementary solution. Thus, we can guess a particular solution of the form:

$$
y_p(t) = A t^2 e^{-t}
$$

Substituting $y_p$ into the original ODE, we get:

$$
\begin{aligned}
(A(2e^{-t} - 4te^{-t} + t^2e^{-t})) + 2(A(2te^{-t} - t^2e^{-t})) + (At^2e^{-t}) &= 10e^{-t} \\
(2A - 4At + A t^2 + 4At - 2At^2 + A t^2) e^{-t} &= 10e^{-t} \\
2A e^{-t} &= 10e^{-t} \\
A &= 5
\end{aligned}
$$

Hence, the particular solution is:

$$
y_p(t) = 5 t^2 e^{-t}
$$

The general solution to the inhomogeneous ODE is:

$$
y(t) = C_1 e^{-t} + C_2 t e^{-t} + 5 t^2 e^{-t}
$$

## Initial Value Problems

So far, we have discussed general solutions to homogeneous and inhomogeneous equations. From now on, we will focus on solving initial value problems (IVPs) of the form

$$
y'' + p(t)y' + q(t)y = f(t), \quad y(t_0)=y_0,\; y'(t_0)=y_1.
$$

Steps to solve an IVP:

1. Find the complementary solution $y_h$ to the homogeneous equation $y'' + p(t)y' + q(t)y = 0$.
1. Find a particular solution $y_p$ to the inhomogeneous equation $y'' + p(t)y' + q(t)y = f(t)$.
1. Form the general solution to the inhomogeneous equation: $y(t) = y_h(t) + y_p(t)$.
1. Substitute the initial conditions into the general solution to solve for the constants in the complementary solution.

**Example**: Consider the following IVP:

$$
y'' + \omega^2 y = \sin(\omega t), \quad y(0) = y_0, \; y'(0) = 0
$$

The complementary solution is given by:

$$
y_h(t) = C_1 \cos(\omega t) + C_2 \sin(\omega t)
$$

A particular solution can be guessed due to resonance as:

$$
y_p(t) = A t \cos(\omega t) + B t \sin(\omega t)
$$

Applying the method of undetermined coefficients, we can find the coefficients $A$ and $B$:

$$
A = -\frac{1}{2\omega},\quad B = 0
$$

Hence, the particular solution is $y_p(t) = -\frac{1}{2\omega} t \cos(\omega t)$.

The general solution to the inhomogeneous equation is:

$$
y(t) = C_1 \cos(\omega t) + C_2 \sin(\omega t) - \frac{1}{2\omega} t \cos(\omega t)
$$

Substituting the initial conditions, we get:

$$
C_1 = y_0, \quad C_2 = \frac{1}{2\omega^2}
$$

Thus, the solution to the IVP is:

$$
\begin{align*}
y(t)
&= y_0 \cos(\omega t) + \frac{1}{2\omega^2} \sin(\omega t) - \frac{1}{2\omega} t \cos(\omega t) \\
&= \left(y_0 - \frac{t}{2\omega}\right) \cos(\omega t) + \frac{1}{2\omega^2} \sin(\omega t)
\end{align*}
$$

### Zero-input and Zero-state Responses

Define:

- **Zero-input response** $y_{\text{zi}}$ as the solution of
  $$
  y_{\text{zi}}'' + p(t)y_{\text{zi}}' + q(t)y_{\text{zi}} = 0,\quad
  y_{\text{zi}}(t_0)=y_0,\; y_{\text{zi}}'(t_0)=y_1.
  $$

- **Zero-state response** $y_{\text{zs}}$ as the solution of
  $$
  y_{\text{zs}}'' + p(t)y_{\text{zs}}' + q(t)y_{\text{zs}} = f(t),\quad
  y_{\text{zs}}(t_0)=0,\; y_{\text{zs}}'(t_0)=0.
  $$

Then, the solution to the original IVP can be expressed as the sum of the zero-input response and the zero-state response:

> $$
> y(t) = y_{\text{zi}}(t) + y_{\text{zs}}(t)
> $$

Remarks:

- The zero-input response $y_{\text{zi}}$ captures the effect of the initial conditions on the system's behavior. Physically, it represents the system's response due to the initial energy stored in the system, without any external forcing.
- The zero-state response $y_{\text{zs}}$ captures the effect of the external forcing on the system's behavior. Physically, it represents the system's response due to the external input, assuming that the system initially contains no energy.

*Proof*: It is easy to verify that $y_{\text{zi}} + y_{\text{zs}}$ is a solution to the original IVP. By the existence and uniqueness theorem for ODEs, this solution must be the unique solution to the original IVP. $\quad\blacksquare$
