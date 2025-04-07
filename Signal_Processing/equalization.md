---
title: "Channel Equalization"
author: "Ke Zhang"
date: "2024"
fontsize: 12pt
---

# Channel Equalization

## Motivation

Consider the digital communication channel modeled by FIR filter, characterized by the dicrete-time complex-valued impulse response $h[\cdot]$:
$$
h: \mathbb Z \to \mathbb C, k \mapsto h[k] 
\begin{cases}
\ne 0 & k = 0,\dots,L \\
=0   & \text{else}
\end{cases}
$$

The TX sends a sequence of complex symbols $x[k]\in\mathbb C$. The RX receives the convoluted signal added with noise $n[\cdot]$.

$$
y[k] = (h*x) [k] = \sum_{i=0}^L h[i] x[k-i] + n[k]
$$

Remarks:

* $h[\cdot]$ models multipath delay. $h[i]$ is the channel coefficient when $x[k]$ is delayed by $i$ tacs.
* $L$ represents the maximal delay of the channel.
  * If $L=0$, there is no ISI. The communication channel is narrow band since the received signal from multiple paths arrives at RX almost at the same time.
  * If $L>0$, there is ISI as multiple $x[k-i], i\in{0,\dots,L}$ make contribution to $y[k]$. The communication is now wide band.


The RX has access to received signal $y[\cdot]$ and channel impluse response $h[\cdot]$. One important question is how to recover $x[\cdot]$ from $y[\cdot]$. This "deconvolution" problem is called channel equalization.

In this article, we consider only linear equalization, i.e. the reconvered signal is obtained by linearly filtering $y[\cdot]$. More concretely, we would like to design a filter $g[\cdot]$, such that
$$
\hat x [k] = (g*y)[k] = \sum_{i=-\infty}^\infty g[i]y[k-i]
$$
 is as close to $x[k]$ as possible.

Moreover, the equalization filter $g[\cdot]$ should be

* **casual** because the RX can only predict $\hat x[k]$ based on $y[k], y[k-1], y[k-2], \dots$
  $$
  g[k]=0, \forall k<0
  $$
  

* **stable** because of the presence of the noise $n[\cdot]$.
  $$
  \sum_{k=0}^\infty \vert g[k] \vert < \infty
  $$
  

## Zero-Forcing

Let $H(z)$ be the z-transform of $h[\cdot]$. One natural approach is to design $g[\cdot]$ s.t. cascade of $H(z)$ and $G(z)$ is delay by $D\in\mathbb N$ tacs.
$$
G(z)H(z) = z^{-D}
$$
The resulting equalization filter is called inverse filter of $H(z)$ up to a delay factor $z^{-D}$.
$$
G(z) = \frac{z^{-D}}{H(z)}
$$


Hence, the received signal is a delayed version of the original signal.
$$
\begin{align}
\hat X(z)
&= G(z) Y(z) \\
&= G(z) \left( H(z)X(z) + N(z) \right) \\
&= G(z)H(z) X(z) + G(z) N(z) \\
&= z^{-D} \left( X(z) + \frac{N(z)}{H(z)} \right)  \\
\end{align}
$$
In general, $G(z)$ represents an IIR filter. Now, we will answer the following questions:

1. Does $G(z)$ represent a casual and stable filter?
2. Is the resulting equalization resilliant to noise?

From signal processing, we know that $g[\cdot]$ is causal and stable $\iff$ All poles of $G(z)$ lies inside of the unit circle. Since $G(z)$ is the inverse of $H(z)$ up to a delay factor. It follows that $g[\cdot]$ is causal and stable $\iff$ All zeros of $H(z)$ lies inside of the unit cirle.

Example: ZF-IIR equalizer. Consider $H(z) = z^{-1} + \frac{1}{2} z^{-2}$. It has one zero at $z=-\frac{1}{2}$. Hence, the inverse filer is stable.  The inverse filter is
$$
\frac{1}{H(z)}
= \frac{1}{z^{-1} + \frac{1}{2} z^{-2}}
= \frac{z}{1 + \frac{1}{2} z^{-1}}
$$


Using the z-transform table, we get

This approach works well if the noise power is low (or equivalently high SNR)

## Matched Filter

## LMMSE Equalizer

## Table of Abbreviations

| Abbr. | Full name                 |
| ----- | ------------------------- |
| ISI   | Inter-symbol interference |
| FIR   | Finite impulse response   |
| SNR   | signal to noise ratio     |

