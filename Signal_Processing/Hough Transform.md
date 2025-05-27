---
title: "Hough Transform"
author: "Ke Zhang"
date: "2024"
fontsize: 12pt
---

# Hough Transform

Road map:

1. Intro to Hough space
2. Voting
3. Line detection

## Hesse Normal Form of 2D Lines

Recall: A line in $\mathbb R^2$ has the **Hesse normal form**
$$
  L=\{(x,y)\in\mathbb R^2 : x\cos\theta + y\sin\theta = r\}
$$
where

* $\theta$: orientation of the normal vector
* $r$: distance from the origin to $L$

Remarks:

* We call $\mathbb R^2$ where the line resides the **image space**
* Each pair of parameters $(\theta, r)\in[0, \pi)\times [0, \infty)$ uniquely specifies a line in image space.
* We call $\mathcal H := [0, \pi)\times [0, \infty)$ the **Hough space** or **parameter space**

## Single Point Voting

Consider all straight lines through a single point $(x_0, y_0)$ in image space. From previous discussion, we know that each line is characterized by its paramters $\theta$ and $r$. Hence, in Hough space, it is equivalent to consider
$$
  V(x_0, y_0) := \{ (\theta, r)\in\mathcal H : r = x_0\cos\theta + y_0\sin\theta \}
$$

Remarks:

* A single point in image space yields a sinusoidal curve in Hough space
* We call $V(x_0, y_0)$ the **voting** of $(x_0, y_0)$ or the Hough-transform of $(x_0, y_0)$.
* Each $(\theta, r)\in V(x_0, y_0)$ get voted once. Others get voted zero times.
* The voting of $(x_0, y_0)$ represents all possible lines through $(x_0, y_0)$ in Hough space

## Double Point Voting

Consider two distinct points $(x_1, y_1)$ and $(x_2, y_2)$ with votings $V(x_1, y_1)$ and $V(x_2, y_2)$ respectively. The intersection of sinusoidal curves in Hough space represents the line through both $(x_1, y_1)$ and $(x_2, y_2)$.
$$
\{(\theta^*, r^*)\} = V(x_1, y_1) \cap V(x_2, y_2) \iff \text{ The line through $(x_1, y_1)$ and $(x_2, y_2)$}
$$

Summary:

* Two distinct points in image space yields two concurrent sinusoidal curves in Hough space
* The true line parameters $(\theta^*, r^*)$ get voted twice.
* The intersection of sinusoidal curves in Hough space represents the line through the two points in image space.

## Collinear Point Voting

Consider $n$ **collinear** points $(x_1, y_1), \cdots, (x_n, y_n)$ such that
$$
   x_k\cos\theta^* + y_k\sin\theta^* = r^*, \quad k=1,\cdots,n
$$
with unknown parameter $(\theta^*, r^*)$.

Let $V(x_k, y_k)$ denote the voting of each  $(x_k, y_k)$. Then,

$$
\{(\theta^*, r^*)\} = \bigcap_{k=1}^n V(x_k, y_k) \iff \text{ The line through $(x_1, y_1), \cdots, (x_2, y_2)$}
$$

Summary:

* $n$ collinear points in image space yields $n$ concurrent sinusoidal curves in Hough space
* The true line parameters $(\theta^*, r^*)$ get the most votings, i.e. $n$ times.
* The intersection of sinusoidal curves in Hough space represents the line through all points in image space.

## Line Detection

The above discussion yields an algorithm for line detection. Suppose we have n points in image space, we let each point vote in Hough space. After the voting, we detect **hot spots** in Hough space, i.e. those $(\theta, r)$ which get exceptionally many votings. Those hot spots in Hough space represents lines in image space.

>**Detect lines in a binary image with bright pixels $(x_1, y_1), \cdots, (x_n, y_n)$.**
>
> * Init a score function: $S(\theta, r) = 0, \: \forall (\theta, r)\in\mathcal H$
> * For $k = 1,\cdots, n$:
>   * Compute Hough transform $V(x_k, y_k)$
>   * For each $(\theta, r)\in V(x_k, y_k)$:
>     * Vote: $S(\theta, r) \gets S(\theta, r) +1$
> * Compute local maxima of $S$.

The practical implementation shares the same principle but has following subtleties:

* Both image space and Hough space are discrete and finite.
  * Image space: $[1,\cdots,H] \times [1,\cdots,W] $ instead of $\mathbb R^2$
  * Hough space: Both $[0,\pi)$ and $[0, r_\text{max})$ are divided into bins.
* The score function becomes a matrix of the size #(bins in $r$) $\times$ #(bins in $\theta$).

## Examples

### Detecting a Single Line

Hot spots in Hough space: a single point $(\theta_0, r_0)$

![Line detection](https://www.desmos.com/calculator/bwmkkjpm6c)

### Detecting Two Intersecting Lines

Hot spots in Hough space: Two point $(\theta_1, r_1), (\theta_2, r_2)$ with $\theta_1 \ne \theta_2$

![Parellel Lines Detection](https://)

### Detecting Two Parellel Lines

Hot spots in Hough space: Two points in vertical $(\theta_0, r_1), (\theta_0, r_2)$

![Parellel Lines Detection](https://)

### Detecting a Rectangle

![Parellel Lines Detection](https://)
Hot spots in Hough space: Four points $(\theta_0, r_1), (\theta_0, r_2), (\theta_0 +\pi/2, r_3), (\theta_0 +\pi/2, r_4)$
