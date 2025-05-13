---
title: "Bayes Decision Theory"
date: "2024"
author: "Ke Zhang"
---

# Intro to Bayes Decision Theory

Decision Theory = Making actions under uncertainty.  
Bayes Decision Theory = Making actions to minimize expected cost

[toc]

$$
\DeclareMathOperator*{\argmax}{argmax}
\DeclareMathOperator*{\argmin}{argmin}
$$

## Pipeline of Statistical Learning

Many statistical learning tasks can be summarised in the pipeline shown below.

<img src="./figs/Statistical Learning.pdf" alt="Statistical Learning" style="zoom:67%;" />

The learning procedure aims to estimate $p(y \mid x)$ from training data. We can either

* learn a discriminative model: learn $p(y \mid x)$ directly. e.g. linear regression, logistic regression, or
* learn a generative model: learn $p(x,y)$. e.g. linear discriminant analysis, naive Bayes classifier.

The technical details about learning $p(y \mid x)$ are not part of this article. Instead, we ask the question: How to use $p(y \mid x)$ to make predictions or decisions?
By assuming that $p(y \mid x)$​ is either known or has been solidly estimated, we can rigoriously define the optimal decision making.

### A Practical Example

Consider a collision avoidance system (CAS) on the airplane. The CAS analyses sensor data to predict obstacles in the air. Based on its belief on the presence of obstacles, it makes one of three actions: Make an evasive move, Warn the pilot, Do nothing.

Given the sensor data, which action should CAS take?  
We define the cost of each action under different circumstances in the table below.

| Actions              | If Obstacle | If No Obstacle |
| -------------------- | ----------- | -------------- |
| Make an evasive move | 0           | 20             |
| Warn the pilot       | 10          | 1              |
| Do Nothing           | 1000        | 0              |

From the table, we can see that the cost of making an evasive move when there is no obstacle is 20, due to extra fuel usage. However, doing nothing when there is an obstacle incurs a very high cost of 1000. Therefore, the CAS should weigh the costs and benefits of each action based on the probability of an obstacle being present.

Suppose that CAS predicts 1% chance of obstacle being present and 99% chance of obstacle not being present after analysing the sensor data. We can calculate the expected cost of each actions based on these probabilities.

$$
\begin{align*}
\text{Make an evasive move:}&\quad 1\%\times 0 + 99\%\times 20 = 19.8 \\
\text{Warn the pilot:}&\quad       1\%\times 10 + 99\%\times 1 = 1.09 \\
\text{Do Nothing:}&\quad           1\%\times 1000 + 99\%\times 0 = 10 \\
\end{align*}
$$

| Actions              | 1% Obstacle  | 99% no Obstacle | Expected Cost |
|----------------------|--------------|-----------------|---------------|
| Make an evasive move | 0            | 20              | 19.8          |
| Warn the pilot       | 10           | 1               | **1.09**      |
| Do Nothing           | 1000         | 0               | 10            |

Hence, given the current sensor reading, the optimal action is to warn the pilot because it has minimal expected cost.

## Defining Bayes Risk

Set up:

* **Feature space** $\mathcal{X}$: typically $x \in\mathbb{R}^d$.
* **Label set** $\mathcal{Y}$: can either be $\mathbb{R}$ (for regression) or a discrete set (for classification)
* **Action set** $\mathcal{A}$: not necessarily equal to $\mathcal{Y}$ in general, as our previous CAS example.
* **Loss function** $\ell: \mathcal{Y} \times \mathcal{A} \to \mathbb{R}, (y, a) \mapsto \ell(y, a)$. Also known as ***cost function***.
* **Decision rule** is a function $f: \mathcal{X} \to \mathcal{A}, x \mapsto a=f(x)$. If the action set and label set coincide, $f$ becomes a label predicting function.
* **Function class** $\mathcal{F}$: the set of all possible decision rules. In training, we often limit the size of $\mathcal{F}$ to prevent overfitting. In decision theory, we do not restrict $\mathcal{F}$.

The ***Bayes risk*** of a decision rule $f(\cdot)$ is the expected loss w.r.t. joint probability
> $$
> \begin{align}
> R(f)
> &\triangleq \mathbb{E}_{XY} \left[ \ell(y, f(x)) \right] \\
> &= \int_\mathcal{Y}\int_\mathcal{X} \ell(y, f(x)) \cdot p(x,y) \:\mathrm{d}x \mathrm{d}y \\
> \end{align}
> $$

The ***conditional risk*** (or ***Bayes expected loss***) of action $a$ for given $x$ is the expected loss w.r.t posterior probability
> $$
> \begin{align}
> R(a \mid x)
> &\triangleq \mathbb{E}_Y[\ell(y, a) \mid X=x] \\
> &= \int_\mathcal{Y} \ell(y, a) \cdot p(y \mid x) \:\mathrm{d}y
> \end{align}
> $$

Remark:

* The Bayes risk is bounded to a decision rule. vs. The conditional risk is bounded to an action (or decision).
* The Bayes risk is the statistical average of $\ell(y, f(x))$ over all possible $x$ and $y$.
* The conditional risk given $x$ is the statistical average of $\ell(y, a)$ over all possible $y$.

Relation between Bayes risk and conditional risk:

> $R(f)$ is the statistical average of $R(f(x) \mid x)$ over $x$. i.e.
> $$
> \begin{align}
> R(f) = \mathbb{E}_X[R(f(x) \mid x)]
> \end{align}
> $$

*Proof*: This follows from the law of total expectation. Applying the chain rule, we get
$$
\begin{align}
R(f)
&= \int_\mathcal{Y}\int_\mathcal{X} \ell(y, f(x)) \cdot p(x) p(y \mid x)  \:\mathrm{d}x \mathrm{d}y \\
&= \int_\mathcal{X} p(x)
  \underbrace{\int_\mathcal{Y} \ell(y, f(x)) \cdot p(y \mid x) \:\mathrm{d}y}_{R(f(x) \mid x)}
\mathrm{d}x
\end{align}
$$

In the CAS example, we have

* Input $x$: sensor data (e.g. images taken from cameras, LiDAR reading)
* Label set $\mathcal{Y} = \{\text{Obstacle},\: \text{No Obstacle} \}$
* Action set $\mathcal{A} = \{\text{Move},\: \text{Warn},\: \text{Do nothing} \}$
* Loss function $\ell(\cdot, \cdot)$: the 1st table
* Posterior probability: $P(\text{Obstacle} \mid x) = 1\%$ and $P(\text{No Obstacle} \mid x) = 99\%$
* Conditional risk of each action: the last column of the 2nd table
* Bayes risk: unknown since it requires $p(x)$, i.e. the distribution of sensor data

## Bayes Risk Minimization

Goal: Find the decision rule  $f(\cdot)$ that minimizes the Bayes risk, i.e.
$$
\hat{f}_\mathrm{BRM} = \argmin_{f \in\mathcal F} R(f)
$$

Key fact:
> Let $\hat{f}\in\mathcal F$. If $\hat f(x)$ minimizes the conditional risk for all $x\in\mathcal{X}$, then it minimizes the Bayes risk. i.e.
> $$
> \hat{f}(x) = \argmin_{a \in\mathcal A} R(a \mid x), \: \forall x \in\mathcal{X}
> \implies
> \hat{f} = \argmin_{f \in\mathcal F} R(f)
> $$

Remark:

* Here, "$\implies$" becomes "$\iff$" if we slightly relax the requirement "$\forall x \in\mathcal X$" to "almost everywhere in $\mathcal X$". c.f. Appendix for a the proof.
* To minimize the Bayes risk, it is sufficient to minimize the conditional risk for each observation $x$​.

*Proof* $\implies$ : By assumption, we have for any other decision rule $f$ that
$$
R(\hat f(x) \mid x) \le R(f(x) \mid x), \: \forall x
$$

Multiplying both sides by $p(x)$​ does not change the direction of the inequality since probability density is non-negative.
$$
p(x) R(\hat f(x) \mid x) \le p(x) R(f(x) \mid x), \: \forall x
$$

Integrate both side w.r.t. $x$. By monotonicity of integral, we get
$$
\begin{align*}
R(\hat f)
= \mathbb{E}_X[R(\hat f(x) \mid x)]
&= \int_\mathcal{X} p(x) R(\hat f(x) \mid x) \mathrm{d}x
\\
&\le \int_\mathcal{X} p(x) R(f(x) \mid x) \mathrm{d}x
=\mathbb{E}_X[R(f(x) \mid x)]
= R(f)
\quad \square
\end{align*}
$$

### Conditional Risk Minimization

The optimal decision rule is obtained by minimizing $R(a \mid x)$ for each $x$

> $$
> \begin{align}
> \hat{f}(x) = \argmin_{a \in\mathcal{A}} \mathbb{E}_{Y} \left[ \ell(y, a) \mid X=x \right]
> \end{align}
> $$

where the expectation $\mathbb{E}_Y [\:\cdot \mid X=x]$ is taken w.r.t. the posterior $p(y \mid x)$.

Remark:

1. For $\mathcal{Y} = \mathbb{R}$ (regression), the decision rule becomes
    > $$
    > \begin{align}
    >  \hat{f}(x) = \argmin_{a \in\mathcal{A}} \int_{-\infty}^{\infty} \ell(y, a) p(y \mid x) \:\mathrm{d}y
    > \end{align}
    > $$
1. For $\mathcal{Y} = \{1, \dots, K \}$ (classification), the decision rule becomes
    > $$
    > \begin{align}
    >  \hat{f}(x) = \argmin_{a \in\mathcal{A}} \sum_{y \in\mathcal Y} \ell(y, a) p(y \mid x)
    > \end{align}
    > $$
1. The optimal decision $\hat{f}(x)$ depends on the observation $x$. In the CAS example, whether to make an evasive move depends on the sensor readings.
1. If $\mathcal{A} = \mathcal{Y}$, the optimal decision $\hat{f}(x)$ represents the optimal label prediction subject to $\ell(\cdot, \cdot)$ given $x$.
1. Minimizing the Bayes risk requires the full statistical information $p(x,y)$. vs. Minimizing the conditional risk only requires partial statistical information $p(y \mid x)$​.

## Label Prediction

Throughout this section, we consider $\mathcal{A} = \mathcal{Y}$. We interpret each $a\in\mathcal{A}$ as label prediction. The loss $\ell(y,a)$ quantifies the cost of predicting $a$ while the true label is $y$. The optimal decision $\hat{f}(x)$ now becomes the optimal label prediction. In following, we will examine some popular choices of loss functions and their resulting optimal label prediction.

### 0/1 Loss & Posterior Mode

Let $\mathcal{A} = \mathcal{Y}$ be any discrete set. The 0/1 loss is defined as
> $$
> \begin{align}
> \ell_{0/1} (y, a) = \mathbb{I}[y \ne a]
> \end{align}
> $$

The resulting optimal prediction is **MAP** (or ***posterior mode***) of $Y$ given $x$, i.e. the most probable $Y$ given $x$.
> $$
> \begin{align}
> \hat{f}(x)
> &= \argmin_{a \in\mathcal{A}} \mathbb{E}_{Y} \left[ \ell_{0/1}(Y, a) \mid X=x  \right] \\
> &= \argmax_{y \in\mathcal{Y}} \: p(y \mid x)
> \end{align}
> $$

*Proof*: Plugging the 0/1 loss into the decision rule yields
$$
\begin{align*}
\hat{f}(x)
&= \argmin_a \sum_y \mathbb{I}[y \ne a] p(y \mid x) \\
&= \argmin_a \sum_y (1 - \mathbb{I}[y = a]) p(y \mid x) \\
&= \argmin_a \underbrace{\sum_y p(y \mid x)}_{1}  -  \underbrace{\sum_y \mathbb{I}[y = a] p(y \mid x)}_{p(a \mid x)} \\
&= \argmax_a\: p(a \mid x) \quad\quad\quad\square\\
\end{align*}
$$

If we assume equal priors $p(y)$, the optimal decision reduces further to MLE of $Y$. (*Proof*: exercise).

$$
\underset{\text{Min. Bayes Risk}}
  {\boxed{\argmin_a \mathbb{E}_{Y} \left[ \ell(y, a) \mid X=x \right]}}
\xrightarrow{\text{use 0/1 loss}}
\underset{\text{MAP}}
  {\boxed{\argmax_y\: p(y \mid x)}}
\xrightarrow{\text{equal prior } p(y)}
\underset{\text{MLE}}
  {\boxed{\argmax_y\: p(x \mid y)}}
$$

The 0/1 loss penalizes all misclassification equally. If we write 0/1 loss in a matrix, we would get zeros on the diagonal, ones elsewhere. e.g. The 0/1 loss for a three-class problem looks like

|   0/1 loss    | $y=1$  | $y=2$ | $y=3$ |
|---------------|--------|-------|-------|
| $\hat{y} = 1$ |   0    |   1   |   1   |
| $\hat{y} = 2$ |   1    |   0   |   1   |
| $\hat{y} = 3$ |   1    |   1   |   0   |

### Square Loss & Posterior Mean

Let $\mathcal{A} = \mathcal{Y} = \mathbb{R}$. The square loss is defined as
> $$
> \begin{align}
> \ell_2 (y, a) = (y-a)^2
> \end{align}
> $$

The resulting optimal prediction is the **posterior mean** of $Y$ given $x$, i.e. the averaged $Y$ given $x$.
> $$
> \begin{align}
> \hat{f}(x)
> &= \argmin_{a \in\mathcal{A}} \mathbb{E}_{Y} \left[ \ell_2(Y, a) \mid X=x  \right] \\
> &= \mathbb{E}_Y [Y \mid X=x]
> \end{align}
> $$

*Proof*: Plugging the square loss into the decision rule yields
$$
\begin{align*}
\hat{f}(x)
&= \argmin_a \mathbb{E}_{Y} \left[ (Y-a)^2 \mid X=x  \right] \\
&= \argmin_a \mathbb{E}_{Y} \left[ Y^2 - 2aY + a^2 \mid X=x  \right] \\
&= \argmin_a \mathbb{E}_{Y} \left[ Y^2 \mid X=x  \right] -2a\cdot\mathbb{E}_{Y} \left[ Y \mid X=x  \right] + a^2 \\
&= \argmin_a\:  a^2 - 2a\cdot\mathbb{E}_Y [Y \mid X=x]
\end{align*}
$$

The objective is convex in $a$.  Hence, letting the derivative be zero yields the optimal decision $\hat{a} = \mathbb{E}_Y [Y \mid X=x]$. $\quad\square$

**Example (Linear Regression)**: In linear regression, we have the model
$$
p(y \mid x) = \mathcal{N}(y \mid w^\top x, \sigma^2)
$$

Suppose we learned the parameter $w$. The optimal prediction of $y$ given a test input $x_\text{test}$ is then
$$
\hat{y} = \mathbb{E}_Y [Y \mid X=x_\text{test}] = w^\top x_\text{test}
$$

### Absolute Loss & Posterior Median

Let $\mathcal{A} = \mathcal{Y} = \mathbb{R}$. The absolute loss is defined as
> $$
> \begin{align}
> \ell_1 (y, a) = \vert y-a\vert
> \end{align}
> $$

The resulting optimal prediction is the **posterior median** of $Y$ given $x$, the value that splits $p(y \mid x)$ evenly.
> $$
> \begin{align}
> \hat{f}(x) &= \argmin_a \mathbb{E}_{Y} \left[ \ell_1(Y, a) \mid X=x  \right] \triangleq \hat y
>  \\
> \text{s.t.} &\int_{-\infty}^{\hat y} p(y \mid x) \:\mathrm{d}y = \int_{\hat y}^{\infty} p(y \mid x) \:\mathrm{d}y = \frac{1}{2}
> \end{align}
> $$

*Proof*: Plugging the absolute loss into the decision rule yields
$$
\begin{align*}
\hat{f}(x)
&= \argmin_a \int_{-\infty}^{\infty} \vert y-a\vert \, p(y \mid x) \:\mathrm{d}y \\
&= \argmin_a
   \underbrace{\int_{-\infty}^{a} (a-y) \, p(y \mid x) \:\mathrm{d}y}_{h_1(a)}
  +\underbrace{\int_{a}^{\infty} (y-a) \, p(y \mid x) \:\mathrm{d}y}_{h_2(a)} \\
\end{align*}
$$

Let $h(a) = h_1(a) + h_2(a)$. We take the derivative of $h(a)$ w.r.t. $a$ using Leibniz's integral rule (c.f. Appendix)
$$
\begin{align*}
\frac{\mathrm d}{\mathrm da} h_1(a)
  &= \bigg[(a-y) \, p(y \mid x)\bigg]_{y=a} + \int_{-\infty}^{a} \frac{\partial}{\partial a} \bigg[(a-y) \, p(y \mid x) \bigg] \:\mathrm{d}y \\
  &= \int_{-\infty}^{a} p(y \mid x) \:\mathrm{d}y \\
\frac{\mathrm d}{\mathrm da} h_2(a)
  &= -\bigg[(y-a) \, p(y \mid x)\bigg]_{y=a} + \int_{a}^{\infty} \frac{\partial}{\partial a} \bigg[(y-a) \, p(y \mid x) \bigg] \:\mathrm{d}y \\
  &= -\int_{a}^{\infty} p(y \mid x) \:\mathrm{d}y \\
\end{align*}
$$

Hence,
$$
\begin{align*}
\frac{\mathrm d}{\mathrm da} h(a)
&= \frac{\mathrm d}{\mathrm da} h_1(a) + \frac{\mathrm d}{\mathrm da} h_2(a) \\
&= \int_{-\infty}^{a} p(y \mid x) \:\mathrm{d}y -\int_{a}^{\infty} p(y \mid x) \:\mathrm{d}y
\end{align*}
$$

Note that $h$ is convex in $a$ since $h_1$ and $h_2$ are so. Setting $\frac{\mathrm d}{\mathrm da} h(a) = 0$, we conclude that
$$
\begin{align*}
\int_{-\infty}^{a} p(y \mid x) \:\mathrm{d}y =
\int_{a}^{\infty} p(y \mid x) \:\mathrm{d}y = \frac{1}{2}
\end{align*}
$$

## Binary Classification

Binary classification is of particular importance and deserves study in greater details. Previously, we examined 0/1 loss and its implications in classification. Minimizing expected 0/1 loss yields MAP of the class label. In some applications, asymmetric loss is preferred. For example, in the case of a fire alarm, we would tolerate false positives rather than false negatives, as the latter could risk lives.

A general loss function for binary classification is defined by four pre-defined numbers

|               |   $y=+$  |   $y=-$ |
|---------------|----------|---------|
| $\hat{y} = +$ | $c_{TP}$ | $c_{FP}$|
| $\hat{y} = -$ | $c_{FN}$ | $c_{TN}$|

Remark:

* The values are chosen such that $c_{FP}>c_{TN}$ and $c_{FN}>c_{TP}$​ since we penalize misclassification more.
* If we set $c_{TP} = c_{TN} =0$ and $c_{FN} = c_{FP} =1$, we get 0/1 loss.

Given the test sample $x$

* The Bayes risk of predicting "$+$" is
    $$
    R(+ \mid x) = p(+ \mid x)\cdot c_{TP} + p(-\mid x)\cdot c_{FP}
    $$
* The Bayes risk of predicting "$-$" is
    $$
    R(-\mid x) = p(+ \mid x)\cdot c_{FN} + p(-\mid x)\cdot c_{TN}
    $$

Hence, predict $\hat{y} = +$ iff $R(+ \mid x) < R(-\mid x)$, which is equivalent to any of followings

1. Posterior ratio test
    $$
    \frac{p(+ \mid x)}{p(-\mid x)} > \frac{c_{FP} - c_{TN}}{c_{FN} - c_{TP}}
    $$
1. Likelihood ratio test
    $$
    \frac{p(x \mid +)}{p(x \mid -)} > \frac{c_{FP} - c_{TN}}{c_{FN} - c_{TP}} \cdot \frac{p(-)}{p(+)}
    $$
1. Log likelihood ratio test
    > $$
    > \ln \frac{p(x \mid +)}{p(x \mid -)} >
    >   \underbrace{\ln\frac{c_{FP} - c_{TN}}{c_{FN} - c_{TP}}}_{T_L} +
    >   \underbrace{\ln\frac{p(-)}{p(+)}}_{T_P}
    > $$

*Proof*: 1 follows from the definition by plugging in Bayes risks. 2 follows from 1 by Bayes rule. 3 follows from 2 by properties of log. Details omited.

Remark:

* In log likelihood ratio test, the threshold $T = T_L + T_P$ where $T_L$ depends on the loss function and $T_P$ depends on the class prior.
* The threshold becomes 0 if the loss function happens to be 0/1 loss AND equal class prior.

The decision boundary is
> $$
> \left\{ x\in\mathbb R^d: \ln \frac{p(x \mid +)}{p(x \mid -)} = T_L + T_P  \right\}
> $$

Binary classification is very similar to binary hypothesis testing. c.f. separate article.

# Exercise

1. **Collision Avoidance System (CAS)**: Back to the CAS problem at the very beginning of this article. Let $p$ be the probability of obstacle being present given the sensor data. The cost function is summarized in the table below. The term *optimal decision* refers to the decision(or action) which minimizes the conditional risk.

    | Actions              | If Obstacle  | If no Obstacle |
    |----------------------|--------------|----------------|
    | Make an evasive move | 0            | 20             |
    | Warn the pilot       | 10           | 1              |
    | Do Nothing           | 1000         | 0              |

    * What is the optimal decision if $p=0.7$?
    * The CAS would decide "Do Nothing" to minimize the expected cost if
      * [ ] $p$ is about 0.5
      * [ ] $p$ is about 1/3
      * [ ] $p$ is about 0
      * [ ] $p$ is about 1
    * Determine the threshold(s) for $p$ such that "Do Nothing" is the optimal decision.
    * Determine the threshold(s) for $p$ such that "Warn the pilot" is the optimal decision.
    * The company asks you to make the airplane safer by redesigning the CAS. The new CAS will decide to make an evasive move as long as $p>0.5$. Adjust the loss function (aka the table entries) to achieve this goal.

1. **Stock**: You are considering whether to buy a stock but you have little knowledge about the stock market. Hence, you predict whether the stock is healthy based on an online rating. Let $X\in\{A, B, C\}$ denote the online rating. Let $Y\in\{\text{Healthy}, \text{Unhealthy} \}$ denote the nature of the stock. Let $a_1$ denote "Buy the stock" and $a_0$ denote "Don't buy the stock". The loss function is summarized in the table
   
    |  Loss | $y=\text{Healthy}$ | $y=\text{Unhealthy}$ |
    | ----- | ----------- | -------------- |
    | $a_1$ | 0           | 20          |
    | $a_0$ | 10       | 0              |
    
    Assume you know the posterior probability as summarized in the table
    |     $p(y \mid x)$    | $x=A$ | $x=B$ | $x=C$ |
    | -------------------- | ----- | ----- | ----- |
    |  $y=\text{Healthy}$  | 0.9   | 0.6   |  0.2  |
    | $y=\text{Unhealthy}$ | 0.1   | 0.4   |  0.8  |
    
    Let's consider two decision rules:
    $$
    \begin{align*}
    f_1 (x) &=
    \begin{cases}
    a_1 &\text{ if } x = A \text{ or } B \\
    a_0 &\text{ if } x = C
    \end{cases}
    \\
    f_2 (x) &=
    \begin{cases}
    a_1 &\text{ if } x = A \\
    a_0 &\text{ if } x = B \text{ or } C
    \end{cases}
    \end{align*}
    $$
    
    * Calculate the conditional risks $R(a_0 \mid A)$ and $R(a_1 \mid A)$​.
    
    * Calculate the conditional risks $R(a_0 \mid B)$ and $R(a_1 \mid B)$.
    
    * Calculate the conditional risks $R(a_0 \mid C)$ and $R(a_1 \mid C)$.
    
    * Let $p_A \triangleq p(X=A), p_B \triangleq p(X=B), p_C \triangleq p(X=C)$. Show that regardless of values of $p_A, p_B, p_C$, the Bayes risks satifsfy 
      $$
      R(f_1) > R(f_2)
      $$

## Appendix

### Indicator Function

The indicator function $\mathbb{I}[\cdot]$ is define as
$$
\mathbb{I}[A] =
\begin{cases}
1 &\text{if assertion } A=\text{true} \\
0 &\text{if assertion } A=\text{false} \\
\end{cases}
$$

> For any assertion $A$, $\lnot A$ denotes the negation of $A$. Then,
> $$
> \mathbb{I}[A] + \mathbb{I}[\lnot A] = 1
> $$

*Proof*: trivial

### Leibniz's Integral Rule

Let $f: \mathbb{R}\times\mathbb{R}: (x,t) \mapsto f(x,t)$. Suppose everything is continuous, differentiable, bounded etc. (if they need to be)

$$
\begin{align}
\frac{\mathrm d}{\mathrm dx} \left(
  \int_{a(x)}^{b(x)} f(x,t) \:\mathrm{d}t
\right)
  &=
  f(x, b(x)) \frac{\mathrm d}{\mathrm dx} b(x)
  -f(x, a(x)) \frac{\mathrm d}{\mathrm dx} a(x)
  +\int_{a(x)}^{b(x)} \frac{\partial}{\partial x} f(x,t) \:\mathrm{d}t
\end{align}
$$

Special cases:
$$
\begin{align}
\frac{\mathrm d}{\mathrm dx} \left(
  \int_{a}^{x} f(x,t) \:\mathrm{d}t
\right)
  &=
  f(x,x) + \int_{a}^{x} \frac{\partial}{\partial x} f(x,t) \:\mathrm{d}t
\\
\frac{\mathrm d}{\mathrm dx} \left(
  \int_{x}^{b} f(x,t) \:\mathrm{d}t
\right)
  &=
  -f(x,x) + \int_{x}^{b} \frac{\partial}{\partial x} f(x,t) \:\mathrm{d}t
\end{align}
$$

### Total Expectation

Let $X$ and $Y$ be two random variables and $g:\mathbb R^2 \to \mathbb R$ be a function.

The ***total expecation*** is defined as the statistical average of $g(x,y)$ w.r.t. $p(x,y)$.
> $$
> \mathbb{E}_{XY}[g(x,y)] = \int\int g(x,y) \cdot p(x,y) \:\mathrm{d}x \mathrm{d}y
> $$

The ***conditional expecation*** for a given $x$ is defined as the statistical average of $g(x,y)$ w.r.t. $p(y \mid x)$.
> $$
> \mathbb{E}_{Y}[g(x,y) \mid x] = \int g(x,y) \cdot p(y \mid x) \:\mathrm{d}y
> $$

The total expectation is the statistical average of conditional expectation w.r.t. $p(x)$
> $$
> \mathbb{E}_{XY}[g(x,y)] =
> \mathbb{E}_{X} \big[ \mathbb{E}_{Y}[g(x,y) \mid x] \big]
> $$

*Proof*: Simply apply the chain rule of joint probability in the total probability
$$
\begin{align*}
\mathbb{E}_{XY}[g(x,y)]
&= \int\int g(x,y) \cdot p(x,y) \:\mathrm{d}x \mathrm{d}y \\
&= \int\int g(x,y) \cdot p(x) p(y \mid x) \:\mathrm{d}x \mathrm{d}y \\
&= \int p(x)
     \underbrace{\int g(x,y) \cdot p(y \mid x) \:\mathrm{d}y}_{\mathbb{E}_{Y}[g(x,y) \mid x]}
   \mathrm{d}x \\
\end{align*}
$$

### Equivalence of Bayes risk minimization and conditional risk minimization

Show that

> $$
> \hat{f} = \argmin_{f \in\mathcal F} R(f)
> \iff
> \hat{f}(x) = \argmin_{a \in\mathcal A} R(a \mid x) \text{ a.e. in } \mathcal{X}
> $$

*Proof*: We will show "$\implies$" direction since the other direction was already shown. Let $\mu$ be the measure defined for the $\sigma$-algebra on $\mathcal X$.
For the sake of contradiction, suppose $\exists \Omega\subset\mathcal X$ with $\mu(\Omega)>0$ s.t.$\forall x \in\Omega$​
$$
\hat{f}(x) \ne \argmin_{a \in\mathcal A} R(a \mid x)
\implies
\min_{a \in\mathcal A} R(a \mid x) < R(\hat{f}(x) \mid x)
$$

Now, define another decision rule
$$
\tilde f (x) =
\begin{cases}\displaystyle
\argmin_{a \in\mathcal A} R(a \mid x), & x \in\Omega \\
\hat f(x),                             & x \in\mathcal{X}\setminus\Omega
\end{cases}
$$

Then, we conclude that $\tilde f $ would achieve a lower Bayes risk, which contradicts with $\displaystyle\hat{f} = \argmin_{f \in\mathcal F} R(f)$.
$$
\begin{align*}
R(\tilde f )
= \int_\mathcal{X} p(x) R(\tilde f (x) \mid x) \mathrm{d}\mu
&= \int_\mathcal{\mathcal{X}\setminus\Omega} p(x) R(\tilde f (x) \mid x) \mathrm{d}\mu + \int_\mathcal{\Omega} p(x) R(\tilde f (x) \mid x) \mathrm{d}\mu
\\
&= \int_\mathcal{\mathcal{X}\setminus\Omega} p(x) R(\hat f(x) \mid x) \mathrm{d}\mu + \int_\mathcal{\Omega} p(x) R(\tilde f (x) \mid x) \mathrm{d}\mu
\\
&< \int_\mathcal{\mathcal{X}\setminus\Omega} p(x) R(\hat f(x) \mid x) \mathrm{d}\mu + \int_\mathcal{\Omega} p(x) R(\hat f(x) \mid x) \mathrm{d}\mu
= R(\hat f)
\quad \square
\end{align*}
$$
