---
title: "Weakly Stationary Process"
author: "Ke Zhang"
date: "2023"
fontsize: 12pt
---

# Weakly Stationary Process

This note focuses on weakly stationary processes. For the general theory (finite dimensional distributions, moments, stationarity), see [Stochastic Process](stochastic_process.md).

Notation:

* We consider real-valued discrete-time processes $\{ X(t) \}_{t\in \mathbb Z}$. The continuous-time case is analogous: replace sums by integrals and the DTFT by the continuous-time Fourier transform.
* We write $R_X := R_{XX}$ and $C_X := C_{XX}$ for short.

## Review: Weak Stationarity

A stochastic process $X$ is called **weakly stationary** (also known as wide-sense stationary) if

1. $X$ has **finite power**
    $$
        \forall t \in \mathbb Z, \quad \mathbb E[X(t)^2] < \infty
    $$
2. $X$ has **constant mean**
    $$
        \forall t \in \mathbb Z, \quad \mathbb E[X(t)] = \mu_X
    $$
3. The auto-correlation is **shift-invariant**
    $$
        \forall \tau, t_1, t_2 \in \mathbb Z, \quad R_{XX}(t_1+\tau, t_2+\tau) = R_{XX}(t_1, t_2)
    $$
    In particular, setting $t_1 = t_2 = t$ and $\tau = -t$ shows that $X$ has **constant power**
    $$
        \forall t \in \mathbb Z, \quad \mathbb E[X(t)^2] = R_{XX}(0, 0)
    $$

In other words, the mean and auto-correlation are shift-invariant. See [Stochastic Process](stochastic_process.md#weak-stationarity) for how weak stationarity relates to the other notions of stationarity.

## Correlation Analysis

### Auto-Correlation Function

For a weakly stationary signal, the auto-correlation depends only on the time difference, not on the location of time origin. i.e.

$$
\forall t_1, t_2, s_1, s_2 \in\mathbb Z \text{ with }  t_1 - t_2 = s_1 - s_2
\implies
\mathbb E[X(t_1) X(t_2)] = \mathbb E[X(s_1) X(s_2)]
$$

Hence, we define the **auto-correlation function** as a function of a single variable:
$$
    R_X(t) = \mathbb E[X(t+\tau) X(\tau)]
$$

Similarly, we define the **auto-covariance function**:

$$
\begin{split}
    C_X(t)
    &= \mathbb E\left[(X(t+\tau) - \mu_X) (X(\tau) - \mu_X)\right] \\
    &= R_X(t) - \mu_X^2
\end{split}
$$

Note: Both $R_X(t)$ and $C_X(t)$ are independent of $\tau$ in their definitions.

Properties of a weakly stationary signal $X$:

1. The signal power is $R_X(0)$. The variance is $C_X(0)$.

2. $R_X(t)$ is an even function
    $$
        R_X(-t) =  R_X(t)
    $$

    *Proof:* Substitute $\tau' = \tau - t$.
    $$
        R_X(-t)
        = \mathbb E[X(\tau - t) X(\tau)]
        = \mathbb E[X(\tau') X(\tau' + t)]
        = R_X(t)
    $$

3. $\vert R_X(t) \vert $ is bounded by the signal power
    $$
        \forall t\in\mathbb Z, \: \left\vert R_X(t) \right\vert \le  R_X(0)
    $$

    *Proof:* This follows from the Cauchy-Schwarz inequality.

    $$
        \vert R_X(t) \vert
        =   \left\vert \mathbb E[X(t+\tau) X(\tau)] \right\vert
        \le \sqrt{\mathbb E[X(t+\tau)^2] \cdot \mathbb E[X(\tau)^2] }
        =   \sqrt{R_X(0) \cdot R_X(0)}
        =   R_X(0)
    $$

4. $R_X(t)$ is positive semi-definite
    $$
        \forall n\ge 1, \: \forall t_1, \cdots, t_n\in\mathbb Z, \: \forall a_1, \cdots, a_n\in\mathbb R, \quad
        \sum_{i=1}^n \sum_{j=1}^n a_i a_j R_X(t_i - t_j) \ge 0
    $$

    *Proof:*
    $$
        \sum_{i=1}^n \sum_{j=1}^n a_i a_j R_X(t_i - t_j)
        = \mathbb E\left[ \sum_{i=1}^n \sum_{j=1}^n a_i a_j X(t_i) X(t_j)\right]
        = \mathbb E\left[ \left( \sum_{i=1}^n a_i X(t_i) \right)^2 \right]
        \ge 0
    $$

    Equivalently, the auto-correlation matrix $\mathbf R = [R_X(t_i - t_j)]_{i,j}$ is symmetric positive semi-definite. For $t_i = i$, it is in addition a Toeplitz matrix.

**Example:** A white noise $W$ with power $\sigma^2$ has the auto-correlation function
$$
    R_W(t) = \sigma^2 \delta(t)
$$

**Example:** The random phase sinusoid $X(t) = A\cos(\omega_0 t + \Phi)$, with $\Phi$ uniformly distributed on $[0, 2\pi)$ and independent of $A$, is weakly stationary with
$$
    \mu_X = 0, \quad R_X(t) = \frac{\mathbb E[A^2]}{2} \cos(\omega_0 t)
$$
regardless of the distribution of $A$.
Unlike the white noise, its auto-correlation does not decay: samples arbitrarily far apart remain correlated.

### Cross-Correlation Function

Two stochastic processes $\{ X(t) \}_{t\in \mathbb Z}$ and $\{ Y(t) \}_{t\in \mathbb Z}$ are called **jointly weakly stationary** if both

$$
\begin{split}
    1. \quad & \text{Both } X \text{ and } Y \text{ are weakly stationary} \\
    2. \quad & \forall \tau, t_1, t_2 \in \mathbb Z, \quad
    \mathbb E[X(t_1+\tau) Y(t_2+\tau)] = \mathbb E[X(t_1) Y(t_2)].
\end{split}
$$

The second criterion means that the cross-correlation does not depend on time shift.

Hence, we define the **cross-correlation function**:
$$
    R_{XY}(t) = \mathbb E[X(t+\tau) Y(\tau)]
$$

Similarly, we define the **cross-covariance function**:
$$
\begin{split}
    C_{XY}(t)
    &= \mathbb E\left[(X(t+\tau) - \mu_{X}) (Y(\tau) - \mu_{Y})\right] \\
    &= R_{XY}(t) - \mu_X \mu_Y
\end{split}
$$

Properties of $R_{XY}(t)$ of jointly weakly stationary signals:

1. $R_{XY}(t)$ is in general not even, but satisfies
    $$
        R_{XY}(-t) =  R_{YX}(t)
    $$

2. $\vert R_{XY}(t) \vert$ is bounded by the signal powers
    $$
        \forall t\in\mathbb Z, \: \left\vert R_{XY}(t) \right\vert \le \sqrt{R_X(0) R_Y(0)}
    $$

    *Proof:* Cauchy-Schwarz inequality, as for the auto-correlation function.

## Spectral Analysis

Throughout this section, we assume that $R_X$ is absolutely summable, i.e. $\sum_{t} \vert R_X(t) \vert < \infty$, so that its DTFT exists.

### Power Spectral Density

The **power spectral density (PSD)** of a weakly stationary signal $X$ is defined as the discrete-time Fourier transform (DTFT) of $R_X$

$$
\begin{equation}
\fbox{$\displaystyle
    S_X(\omega) = \sum_{t = -\infty}^{\infty} R_X(t) e^{-j\omega t}, \quad \omega \in [-\pi, \pi]
$}
\end{equation}
$$

Conversely, $R_X$ can be recovered by the inverse DTFT

$$
    R_X(t) = \frac{1}{2\pi} \int_{-\pi}^{\pi} S_X(\omega) e^{j\omega t} \, d\omega
$$

Properties of $S_X(\omega)$ of a real weakly stationary signal:

1. $S_X(\omega)$ is real and even
    $$
        S_X(-\omega) = S_X(\omega) \in \mathbb R
    $$

    *Proof:* This follows from $R_X(-t) = R_X(t)$ and $R_X(t)\in\mathbb R$.

2. $S_X(\omega)$ is non-negative
    $$
        \forall \omega \in [-\pi, \pi], \: S_X(\omega) \ge 0
    $$

    *Proof:* Consider the DTFT of a finite segment of $X$, $X_N(\omega) = \sum_{t=-N}^{N} X(t) e^{-j\omega t}$. Then
    $$
    \begin{split}
        \frac{1}{2N+1} \mathbb E\left[ \left\vert X_N(\omega) \right\vert^2 \right]
        &= \frac{1}{2N+1} \sum_{t=-N}^{N} \sum_{s=-N}^{N} R_X(t-s) e^{-j\omega (t-s)} \\
        &= \sum_{k=-2N}^{2N} \left( 1 - \frac{\vert k \vert}{2N+1} \right) R_X(k) e^{-j\omega k}
    \end{split}
    $$
    The left-hand side is non-negative. Since $R_X$ is absolutely summable, the right-hand side converges to $S_X(\omega)$ as $N\to\infty$. Hence $S_X(\omega) \ge 0$.

    This is the essence of the **Wiener-Khinchin theorem**: the PSD equals the limit of the expected normalised periodogram $\frac{1}{2N+1}\mathbb E\left[ \vert X_N(\omega) \vert^2 \right]$.

3. The signal power is the integral of the PSD over all frequencies
    $$
        \mathbb E[X(t)^2] = R_X(0) = \frac{1}{2\pi} \int_{-\pi}^{\pi} S_X(\omega) \, d\omega
    $$

    This justifies the name *power spectral density*: $S_X(\omega)$ describes how the signal power is distributed over frequencies.

**Example:** The white noise $W$ with power $\sigma^2$ has the PSD
$$
    S_W(\omega) = \sum_{t = -\infty}^{\infty} \sigma^2 \delta(t) e^{-j\omega t} = \sigma^2
$$
i.e. all frequencies contribute equally to the signal power. Hence the name *white* noise (analogous to white light).

### Cross Power Spectral Density

For jointly weakly stationary signals $X$ and $Y$, the **cross power spectral density** is defined as
$$
    S_{XY}(\omega) = \sum_{t = -\infty}^{\infty} R_{XY}(t) e^{-j\omega t}
$$

Remarks:

* Since $R_{XY}$ is in general not even, $S_{XY}(\omega)$ is in general complex-valued.
* From $R_{XY}(-t) = R_{YX}(t)$, we get $S_{XY}(\omega) = \overline{S_{YX}(\omega)}$.

## Linear Filtering

Let $\{ X(t)\}_{t\in \mathbb Z}$ be a weakly stationary stochastic signal and $\{ h(t) \}_{t\in \mathbb Z}$ be the impulse response of a discrete-time LTI system. The output signal is another stochastic process $\{ Y(t)\}_{t\in \mathbb Z}$, defined by the convolution

$$
    Y(t) = (X * h)(t) = \sum_{\tau = -\infty}^{\infty} X(t-\tau) h(\tau)
$$

### Time Domain Analysis

Suppose that the impulse response $h$ is real and absolutely summable (i.e. the system is BIBO stable). We can draw the following conclusions

1. $Y$ is weakly stationary with mean
    $$
        \mu_Y = \mu_X \sum_{\tau = -\infty}^{\infty}  h(\tau)
    $$
2. $X$ and $Y$ are jointly weakly stationary with cross-correlation function
    $$
        R_{YX} (t) = (h * R_X)(t)
    $$

    *Proof:*
    $$
        R_{YX}(t)
        = \mathbb E[Y(t+\tau) X(\tau)]
        = \sum_{k = -\infty}^{\infty} h(k) \, \mathbb E[X(t+\tau-k) X(\tau)]
        = \sum_{k = -\infty}^{\infty} h(k) R_X(t-k)
    $$
3. $Y$ has the auto-correlation function
    $$
        R_{Y} (t) = (h^r * R_{YX})(t) = (h^r * h * R_{X})(t)
    $$
    where $h^r(t) = h(-t)$

    *Proof:*
    $$
        R_{Y}(t)
        = \mathbb E[Y(t+\tau) Y(\tau)]
        = \sum_{k = -\infty}^{\infty} h(k) \, \mathbb E[Y(t+\tau) X(\tau-k)]
        = \sum_{k = -\infty}^{\infty} h(k) R_{YX}(t+k)
        = \sum_{k = -\infty}^{\infty} h^r(k) R_{YX}(t-k)
    $$

### Frequency Domain Analysis

Let $H(\omega)$ be the frequency response of the LTI system
$$
    H(\omega) = \sum_{t = -\infty}^{\infty} h(t) e^{-j\omega t}
$$

Applying the DTFT to the results of the time domain analysis, we get

1. The mean of $Y$ is scaled by the DC gain
    $$
        \mu_Y = \mu_X H(0)
    $$
2. The cross PSD of $Y$ and $X$ is
    $$
        S_{YX}(\omega) = H(\omega) S_X(\omega)
    $$
3. The PSD of $Y$ is
    $$
    \begin{equation}
    \fbox{$\displaystyle
        S_Y(\omega) = \left\vert H(\omega) \right\vert^2 S_X(\omega)
    $}
    \end{equation}
    $$

    *Proof:* Convolution in time domain becomes multiplication in frequency domain. Since $h$ is real, the DTFT of $h^r$ is
    $$
        \sum_{t = -\infty}^{\infty} h(-t) e^{-j\omega t}
        = \sum_{t = -\infty}^{\infty} h(t) e^{j\omega t}
        = \overline{H(\omega)}
    $$
    Hence,
    $$
        S_Y(\omega) = \overline{H(\omega)} H(\omega) S_X(\omega) = \left\vert H(\omega) \right\vert^2 S_X(\omega)
    $$

Remarks:

* The output PSD depends only on the magnitude response $\vert H(\omega) \vert$, not on the phase response of the system.
* The output power is
    $$
        \mathbb E[Y(t)^2] = \frac{1}{2\pi} \int_{-\pi}^{\pi} \left\vert H(\omega) \right\vert^2 S_X(\omega) \, d\omega
    $$
* If the input is a white noise with power $\sigma^2$, then $S_Y(\omega) = \sigma^2 \left\vert H(\omega) \right\vert^2$. i.e. The output PSD has the shape of the squared magnitude response. This is a common way to model colored noise.

**Example (AR(1) process):** Let $W$ be a white noise with power $\sigma^2$ and $\vert a \vert < 1$. Consider the auto-regressive process
$$
    X(t) = a X(t-1) + W(t)
$$
Its stationary solution is the filtered white noise $X = h * W$ with
$$
    h(t) =
    \begin{cases}
        a^t & t\ge 0 \\
        0 & t < 0 \\
    \end{cases}
$$
which is absolutely summable since $\vert a \vert < 1$. The frequency response is
$$
    H(\omega) = \sum_{t=0}^{\infty} a^t e^{-j\omega t} = \frac{1}{1 - a e^{-j\omega}}
$$
Hence, $X$ has the PSD
$$
    S_X(\omega) = \frac{\sigma^2}{\left\vert 1 - a e^{-j\omega} \right\vert^2} = \frac{\sigma^2}{1 - 2a\cos\omega + a^2}
$$
and the auto-correlation function
$$
    R_X(t) = \sigma^2 (h^r * h)(t) = \sigma^2 \sum_{k=0}^{\infty} a^{k} a^{k + \vert t \vert} = \frac{\sigma^2 a^{\vert t \vert}}{1 - a^2}
$$
For $a > 0$, the system is a low-pass filter and neighbouring samples are positively correlated. For $a < 0$, it is a high-pass filter and neighbouring samples are negatively correlated.
