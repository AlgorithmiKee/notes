---
title: Fourier Series
author: Ke Zhang
date: 2026
---

# Fourier Series

[toc]

## Definitions

Fourier series is a powerful mathematical tool that allows us to represent periodic functions as infinite sums of simpler base functions, such as complex exponentials, sines, and cosines.

### Complex Exponential Form of Fourier Series

Let $x: \mathbb{R} \to \mathbb{C}$ be a periodic function with period $T$. The complex Fourier series of $x(t)$ is given by:

$$
\begin{align}
x(t) &= \sum_{n=-\infty}^{\infty} c_n \exp\left( j 2 \pi n \frac{t}{T} \right) \\
\end{align}
$$

where the complex Fourier coefficients $c_n \in \mathbb{C}$ are calculated as:

$$
\begin{align}
c_n &= \frac{1}{T} \int_{T} x(t) \exp\left( -j 2 \pi n \frac{t}{T} \right) dt \\
\end{align}
$$

Remarks:

* We assume that regularity conditions are satisfied so that the Fourier series exists and converges to $x(t)$. Depending on the "goodness" of $x(t)$, the Fourier series may converge pointwise, uniformly, or in the mean square sense. Detailed convergence behavior is not covered in this note.
* To compute $c_n$, we may integrate over any interval of length $T$. For simplicity, we often choose the interval $[0, T]$ or $[-T/2, T/2]$.
* The Fourier series indicates that any periodic function can be represented as a sum of complex sinusoids. Each term in the series corresponds to a specific frequency component, called harmonic, with frequency $n/T$ and amplitude $c_n$.

Let $\omega_0 = 2 \pi / T$ be the fundamental (or base) angular frequency. Then the Fourier series can be rewritten as:

$$
\begin{align}
x(t) &= \sum_{n=-\infty}^{\infty} c_n e^{j n \omega_0 t} \\
c_n &= \frac{1}{T} \int_{T} x(t) e^{-j n \omega_0 t} dt \\
\end{align}
$$

The Fourier coefficients $c_n$ can be written in polar form as

$$
c_n = |c_n| e^{j \arg(c_n)}
$$

where

* $|c_n|$ is the magnitude of the $n$-th Fourier coefficient, which represents the amplitude of the corresponding frequency component.
* $\arg(c_n)$ is the phase of the $n$-th Fourier coefficient, which represents the phase shift of that component.

The original periodic function can be expressed as a sum of sinusoids with different frequencies, amplitudes, and phases:

$$
\begin{align}
x(t) = \sum_{n=-\infty}^{\infty} |c_n| e^{j \left[n \omega_0 t + \arg(c_n)\right]}
\end{align}
$$

**Fact**: If $x: \mathbb{R} \to \mathbb{R}$ is a real-valued function, then its Fourier coefficients satisfy the conjugate symmetry property:

$$
\begin{align}
c_{-n} = c_n^* \quad \forall n \in \mathbb{Z}
\end{align}
$$

*Proof*: By definition of $c_{-n}$, we have:

$$
c_{-n} = \frac{1}{T} \int_{0}^{T} x(t) e^{-j (-n) \omega_0 t} dt = \frac{1}{T} \int_{0}^{T} x(t) e^{j n \omega_0 t} dt
$$

On the other hand, the complex conjugate of $c_n$ is:

$$
\begin{align*}
c_n^*
&= \left[\frac{1}{T} \int_{0}^{T} x(t) e^{-j n \omega_0 t} dt \right]^* \\
&= \frac{1}{T} \int_{0}^{T} \left[ x(t) e^{-j n \omega_0 t} \right]^* dt \\
&= \underbrace{\frac{1}{T} \int_{0}^{T} x(t) e^{j n \omega_0 t} dt}_{c_{-n}}
\tag*{$\blacksquare$}
\end{align*}
$$

**Example**: Consider the periodic rectangular waveform with period $T$ and pulse width $\tau < T$:

$$
x(t) = \sum_{k=-\infty}^{\infty} p(t - kT)
$$

where $p(t)$ is the rectangular pulse defined as:

$$
p(t) = \begin{cases}
1, & |t| < \frac{\tau}{2} \\
0, & \text{else}
\end{cases}
$$

The Fourier coefficients of $x(t)$ can be calculated as:

$$
\begin{align*}
c_n
&= \frac{1}{T} \int_{-T/2}^{T/2} x(t) e^{-j n \omega_0 t} dt
= \frac{1}{T} \int_{-\tau/2}^{\tau/2} 1 \cdot e^{-j n \omega_0 t} dt \\
&= \frac{1}{T} \left[ \frac{e^{-j n \omega_0 t}}{-j n \omega_0} \right]_{-\tau/2}^{\tau/2}
= \frac{1}{T} \cdot \frac{e^{-j n \omega_0 \tau/2} - e^{j n \omega_0 \tau/2}}{-j n \omega_0} \\
&= \frac{1}{T} \cdot \frac{-2j \sin\left( n \omega_0 \frac{\tau}{2} \right)}{-j n \omega_0}
= \frac{\tau}{T} \cdot \operatorname{sinc}\left( \frac{n\omega_0 \tau}{2} \right)
\end{align*}
$$

Remarks:

* $\operatorname{sinc}(x) = \sin(x)/x$ is the sinc function ($\blacktriangleright$ See [Appendix](#appendix) for more details).
* The original rectangular waveform is an even function. Its Fourier coefficients are real-valued and satisfy $c_{-n} = c_n$. Later, we will see that such property is not a coincidence but a general result for all real-valued even functions.

Hence, the Fourier series of the rectangular waveform is:

$$
x(t) = \sum_{n=-\infty}^{\infty} \frac{\tau}{T} \cdot \operatorname{sinc}\left( \frac{n\omega_0 \tau}{2} \right) e^{j n \omega_0 t}
\tag*{$\lozenge$}
$$

### Trigonometric Form of Fourier Series

When work with real-valued functions, it is often more convenient to express the Fourier series in terms of sine and cosine functions. The trigonometric form of the Fourier series is given by:

$$
\begin{align}
x(t) &= a_0 + \sum_{n=1}^{\infty} \left[ a_n \cos\left( 2 \pi n \frac{t}{T} \right) + b_n \sin\left( 2 \pi n \frac{t}{T} \right) \right] \\
\end{align}
$$

where the Fourier coefficients $a_0, a_n, b_n$ are calculated as:

$$
\begin{align}
a_0 &= \frac{1}{T} \int_{T} x(t) dt \\
a_n &= \frac{2}{T} \int_{T} x(t) \cos\left( 2 \pi n \frac{t}{T} \right) dt \\
b_n &= \frac{2}{T} \int_{T} x(t) \sin\left( 2 \pi n \frac{t}{T} \right) dt
\end{align}
$$

Let $\omega_0 = 2 \pi / T$ be the fundamental angular frequency. Then the trigonometric form can be rewritten as:

$$
\begin{align}
x(t) &= a_0 + \sum_{n=1}^{\infty} \left[ a_n \cos\left( n \omega_0 t \right) + b_n \sin\left( n \omega_0 t \right) \right] \\
a_0 &= \frac{1}{T} \int_{T} x(t) dt \\
a_n &= \frac{2}{T} \int_{T} x(t) \cos\left( n \omega_0 t \right) dt \\
b_n &= \frac{2}{T} \int_{T} x(t) \sin\left( n \omega_0 t \right) dt
\end{align}
$$

Remarks:

* We call $a_0$ the DC (direct current) component, which represents the average value of the function over one period.
* We call $a_1 \cos(\omega_0 t)$ and $b_1 \sin(\omega_0 t)$ the fundamental frequency components, or first harmonics, which correspond to the base frequency $\omega_0$.
* We call $a_n \cos(n \omega_0 t)$ and $b_n \sin(n \omega_0 t)$ the $n$-th harmonics, which correspond to the frequency $n \omega_0$.
* The trigonometric form is also well defined for complex-valued functions. In that case, the Fourier coefficients $a_n$ and $b_n$ are complex numbers. However, we prefer to use the complex exponential form for complex-valued functions while the trigonometric form for real-valued functions.

The trigonometric form is equivalent to the complex exponential form. The relationship between the coefficients is given by:

$$
\begin{align*}
a_0 &= c_0 \\
a_n &= 2 \operatorname{Re}(c_n) \quad \forall n \ge 1 \\
b_n &= -2 \operatorname{Im}(c_n) \quad \forall n \ge 1
\end{align*}
$$

or equivalently,

$$
\begin{align*}
c_0 &= a_0 \\
c_n &= \frac{1}{2} (a_n - j b_n) \quad \forall n \ge 1 \\
c_{-n} &= \frac{1}{2} (a_n + j b_n) \quad \forall n \ge 1
\end{align*}
$$

*Proof*: Starting from the complex exponential form, we can combine the terms for $n$ and $-n$ as:

$$
\begin{align*}
c_n e^{j n \omega_0 t} + c_{-n} e^{-j n \omega_0 t}
&= c_n e^{j n \omega_0 t} + c_n^* e^{-j n \omega_0 t}
&& c_{-n} = c_n^* \\
&= 2 \operatorname{Re}(c_n e^{j n \omega_0 t})
&& z + z^* = 2 \operatorname{Re}(z) \\
&= 2 \operatorname{Re}(c_n) \cos(n \omega_0 t) - 2 \operatorname{Im}(c_n) \sin(n \omega_0 t)
&& \text{Euler's formula} \\
&= a_n \cos(n \omega_0 t) + b_n \sin(n \omega_0 t)
\tag*{$\blacksquare$}
\end{align*}
$$

**Example**: Recall the rectangular waveform example.

$$
x(t) = \sum_{k=-\infty}^{\infty} p(t - kT), \: \text{ where} \quad
p(t) =
\begin{cases}
1, & |t| < \frac{\tau}{2} \\
0, & \text{else}
\end{cases}
$$

The trigonometric Fourier coefficients of $x(t)$ can be calculated as:

* DC component:
    $$
    a_0 = \frac{1}{T} \int_{-T/2}^{T/2} x(t) dt = \frac{1}{T} \int_{-\tau/2}^{\tau/2} 1 dt = \frac{\tau}{T}
    $$
* $a_n$ for $n \ge 1$:
    $$
    \begin{align*}
    a_n
    &= \frac{2}{T} \int_{-T/2}^{T/2} x(t) \cos(n \omega_0 t) dt = \frac{2}{T} \int_{-\tau/2}^{\tau/2} 1 \cdot \cos(n \omega_0 t) dt \\
    &=  \frac{4}{T} \int_{0}^{\tau/2} \cos(n \omega_0 t) dt = \frac{4}{T} \left[ \frac{\sin(n \omega_0 t)}{n \omega_0} \right]_{0}^{\tau/2} \\
    &= \frac{2\tau}{T} \cdot \operatorname{sinc}\left( \frac{n\omega_0 \tau}{2} \right)
    \end{align*}
    $$
* $b_n$ for $n \ge 1$:
    $$
    b_n = \frac{2}{T} \int_{-T/2}^{T/2} x(t) \sin(n \omega_0 t) dt = \frac{2}{T} \int_{-\tau/2}^{\tau/2} 1 \cdot \sin(n \omega_0 t) dt = 0
    $$

The original rectangular waveform is an even real-valued function. Its sine coefficients $b_n$ are all zero while its cosine coefficients $a_n$ happen to be $2c_n$, which follows from the fact that $c_n \in \mathbb{R}$ in this example and that
$$
a_n = 2 \operatorname{Re}(c_n) = 2c_n, \quad b_n = -2 \operatorname{Im}(c_n) = 0
$$

Hence, the trigonometric Fourier series of the rectangular waveform is:

$$
x(t) = \frac{\tau}{T} + \sum_{n=1}^{\infty} \frac{2\tau}{T} \cdot \operatorname{sinc}\left( \frac{n\omega_0 \tau}{2} \right) \cos(n \omega_0 t)
\tag*{$\lozenge$}
$$

## Fourier Series as Orthogonal Projection

Let $L^2([0, T], \mathbb{C})$ be the space of square-integrable complex-valued functions on $[0, T]$ with the inner product defined as:

$$
\begin{align}
\langle f, g \rangle = \int_{0}^{T} f(t) \cdot g^*(t) dt
\end{align}
$$

Let $\omega_0 = 2 \pi / T$ be the fundamental angular frequency. Then, the set of functions $\{ e^{j n \omega_0 t} : n \in \mathbb{Z} \}$ forms an orthogonal basis for $L^2([0, T], \mathbb{C})$.

*Proof*: For any $m, n \in \mathbb{Z}$, we have:

$$
\begin{align*}
\langle e^{j m \omega_0 t}, e^{j n \omega_0 t} \rangle
= \int_{0}^{T} e^{j (m-n) \omega_0 t} dt
=
\begin{cases}
T, & m = n \\
0, & m \neq n
\end{cases}
\tag*{$\blacksquare$}
\end{align*}
$$

To make the basis orthonormal, we can normalize each base function by its norm $\sqrt{T}$:

$$
\begin{align}
\left\{ \frac{1}{\sqrt{T}} e^{j n \omega_0 t} : n \in \mathbb{Z} \right\}
\end{align}
$$

Let $x(t)$ be a periodic function with period $T$. Then, $x$ resctricted to $[0, T]$ can be expressed as a linear combination of the orthonormal basis functions:

$$
\begin{align}
x(t) = \sum_{n=-\infty}^{\infty} \lambda_n \cdot \frac{1}{\sqrt{T}} e^{j n \omega_0 t}
\end{align}
$$

where the coefficients $\lambda_n$ are given by the inner product of $x$ with the corresponding basis function:

$$
\begin{align}
\lambda_n
&= \left\langle x, \frac{1}{\sqrt{T}} e^{j n \omega_0 t} \right\rangle \\
&= \frac{1}{\sqrt{T}} \int_{0}^{T} x(t) e^{-j n \omega_0 t} dt
\end{align}
$$

Letting $c_n = \frac{\lambda_n}{\sqrt{T}}$, we recover the complex Fourier coefficients:

$$
c_n = \frac{\lambda_n}{\sqrt{T}} = \frac{1}{T} \int_{0}^{T} x(t) e^{-j n \omega_0 t} dt
$$

Intuitively, the Fourier coefficients $c_n$ are the coordinates of $x$ w.r.t. the unnormalized orthogonal basis $\{ e^{j n \omega_0 t} : n \in \mathbb{Z} \}$.

## Basic Properties of Fourier Series

### Even and Odd Functions

Complex Fourier coefficients of even and odd functions have special properties as summarized in the following table:

| \# |  Periodic function $x(t)$ | Complex Fourier coefficients $c_n$ |
| :---: | :---: | :---: |
| 1 | $x: \mathbb{R} \to \mathbb{C}$ even | $c_n \in \mathbb{C}$ and $c_{-n} = c_n$ |
| 2 | $x: \mathbb{R} \to \mathbb{R}$ even | $c_n \in \mathbb{R}$ and $c_{-n} = c_n$ |
| 3 | $x: \mathbb{R} \to \mathbb{C}$ odd | $c_n \in \mathbb{C}$ and $c_{-n} = -c_n$ |
| 4 | $x: \mathbb{R} \to \mathbb{R}$ odd | $c_n \in j\mathbb{R}$ and $c_{-n} = -c_n$ |

*Proof*: We only show case 1 and 2 since the proof for case 3 and 4 is similar.

To show case 1, we have:

$$
\begin{align*}
c_{-n}
&= \frac{1}{T} \int_{-T/2}^{T/2} x(t) e^{j n \omega_0 t} dt \\
\end{align*}
$$

Let $t' = -t$. Then, we have:

$$
\begin{align*}
c_{-n}
&= \frac{1}{T} \int_{T/2}^{-T/2} x(-t') e^{-j n \omega_0 t'} (-dt') \\
&= \frac{1}{T} \int_{-T/2}^{T/2} x(-t') e^{-j n \omega_0 t'} dt' \\
&= \underbrace{\frac{1}{T} \int_{-T/2}^{T/2} x(t') e^{-j n \omega_0 t'} dt'}_{c_n}
\tag*{$\blacksquare$}
\end{align*}
$$

To show case 2, we use the fact that $c_{-n} = c_n^*$ for real-valued functions. Hence,

$$
c_{-n} = c_n^* = c_n \implies c_n \in \mathbb{R}
\tag*{$\blacksquare$}
$$

The trigonometric Fourier coefficients of real-valued even/odd functions also have special properties as summarized in the following table:

| \# |  Periodic function $x(t)$ | $a_n$ | $b_n$ |
| :---: | :---: | :---: | :---: |
| 1 | $x: \mathbb{R} \to \mathbb{R}$ even | $a_n \in \mathbb{R}$ | $0$ |
| 2 | $x: \mathbb{R} \to \mathbb{R}$ odd | $0$ | $b_n \in \mathbb{R}$ |

*Proof*: In case 1, we know that $c_n \in \mathbb{R}$. Hence,

$$
a_n = 2 \operatorname{Re}(c_n) \in \mathbb{R}, \quad b_n = -2 \operatorname{Im}(c_n) = 0
\tag*{$\blacksquare$}
$$

In case 2, we know that $c_n \in j \mathbb{R}$. Hence,

$$
a_n = 2 \operatorname{Re}(c_n) = 0, \quad b_n = -2 \operatorname{Im}(c_n) \in \mathbb{R}
\tag*{$\blacksquare$}
$$

### Shifting Properties

Illustration of time-shift:

$$
\begin{align*}
x(t) &\xrightarrow{{\text{Fourier coefficients}}} \{ c_n \}_{n\in \mathbb{Z}} \\
x(t - \tau) &\xrightarrow{{\text{Fourier coefficients}}} \{ c_n \cdot e^{-j n \omega_0 \tau} \}_{n\in \mathbb{Z}} \\
\end{align*}
$$

Let $x(t)$ be a periodic function with period $T$ and Fourier coefficients $c_n$. Then, the Fourier coefficients $c_n'$ of the shifted function $x(t - \tau)$ is

$$
\begin{align}
c_n' &= c_n \cdot e^{-j n \omega_0 \tau}
\end{align}
$$

Remarks:

* Time-shift in the time domain $\iff$ Phase-shift in the frequency domain.
* The magnitude of the Fourier coefficients does not change after time-shift. Only the phase changes.

*Proof*: By definition of $c_n'$, we have:

$$
\begin{align*}
c_n' &= \frac{1}{T} \int_{0}^{T} x(t - \tau) e^{-j n \omega_0 t} dt \\
&= \frac{1}{T} \int_{-\tau}^{T - \tau} x(t') e^{-j n \omega_0 (t' + \tau)} dt' \\
&= e^{-j n \omega_0 \tau} \cdot \frac{1}{T} \int_{-\tau}^{T - \tau} x(t') e^{-j n \omega_0 t'} dt' \\
&= e^{-j n \omega_0 \tau} \cdot c_n
\tag*{$\blacksquare$}
\end{align*}
$$

Illustration of frequency-shift (modulation):

$$
\begin{align*}
x(t) &\xrightarrow{{\text{Fourier coefficients}}} \{ c_n \}_{n\in \mathbb{Z}} \\
e^{j m \omega_0 t} x(t) &\xrightarrow{{\text{Fourier coefficients}}} \{ c_{n-m}  \}_{n\in \mathbb{Z}} \\
\end{align*}
$$

Let $x(t)$ be a periodic function with period $T$ and Fourier coefficients $c_n$. Then, the Fourier coefficients $c_n'$ of the modulated function $e^{j m \omega_0 t} x(t)$ is

$$
\begin{align}
c_n' &= c_{n-m}
\end{align}
$$

Remarks:

* $e^{j m \omega_0 t} x(t)$ is called modulation of $x(t)$, commonly used in communication systems.
* Modulation in the time domain $\iff$ Frequency-shift in the frequency domain.
* The shape of the Fourier coefficients does not change after modulation. Only the indices shift.

*Proof*: By definition of $c_n'$, we have:

$$
\begin{align*}
c_n' &= \frac{1}{T} \int_{0}^{T} e^{j m \omega_0 t} x(t) e^{-j n \omega_0 t} dt \\
&= \frac{1}{T} \int_{0}^{T} x(t) e^{-j (n-m) \omega_0 t} dt \\
&= c_{n-m}
\tag*{$\blacksquare$}
\end{align*}
$$

## Appendix

### Tricks for Computing Integrals

Let $a > 0$ be a positive real number. Consider the integral:

$$
\int_{-a}^{a} f(x) \, dx
$$

* If $f$ is an even function, then the integral can be simplified as:
    $$
    \int_{-a}^{a} f(x) \, dx = 2 \int_{0}^{a} f(x) \, dx
    $$
* If $f$ is an odd function, then the integral is zero as:
    $$
    \int_{-a}^{a} f(x) \, dx = 0
    $$

Let $u = -x$. Then, we have:

$$
\int_{-a}^{a} f(x) \, dx = \int_{a}^{-a} f(-u) \, (-du) = \int_{-a}^{a} f(-u) \, du
$$

### Complex Numbers

Euler's formula:

$$
\begin{align}
\forall \theta \in \mathbb{R}, \quad e^{j \theta} = \cos(\theta) + j \sin(\theta)
\end{align}
$$

Real and imaginary parts of a complex number:

$$
\begin{align}
\operatorname{Re}(z) = \frac{z + z^*}{2}, \quad \operatorname{Im}(z) = \frac{z - z^*}{2j}
\end{align}
$$

### Sinc Function

The sinc function is defined as:

$$
\begin{align}
\operatorname{sinc}(x)
= \begin{cases}
\frac{\sin(x)}{x}, & x \neq 0 \\
1, & x = 0
\end{cases}
\end{align}
$$

Remarks:

* The sinc function is continuous at $x=0$ since
    $$
    \lim_{x \to 0} \frac{\sin(x)}{x} = 1
    $$
* In MATLAB, the sinc function is defined slightly differently as
    $$
    \operatorname{sinc}(x) = \begin{cases}
    \frac{\sin(\pi x)}{\pi x}, & x \neq 0 \\
    1, & x = 0
    \end{cases}
    $$
* The sinc function is even, i.e., $\operatorname{sinc}(-x) = \operatorname{sinc}(x)$ for all $x \in \mathbb{R}$.