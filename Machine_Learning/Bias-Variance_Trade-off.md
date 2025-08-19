---
title: "Bias-Variance Trade-Off"
author: "Ke Zhang"
date: "2024"
fontsize: 12pt
---

# Bias-Variance Trade-Off

[toc]

## Motivation

Data: quadratic relation. 5 samples.

* Model I: fit a line: underfitting
  * poor performance on test data even trained on infinitely  many training samples.
  * poor performance on test data even trained on noiseless training samples.
  * Key issue: lack of complexity. fails to capture complex relation
  * large bias, low variance
* Model II: fit a polynomial of degree 5: overfitting
  * poor generalization because the model tends to fit to the noise
  * but: good performance on test data if trained on infinitely  many training samples.
  * low bias, large variance

## Problem Formulation

Most supervised learning problem can be formalized as follows

> * **Given**: training data set
>
> $$
> D=\{ (x_i, y_i) \in \mathbb{R}^d \times \mathbb{R} \}_{i=1}^n
> $$
>
> * **Statistical Assumption**: $D$ is generated iid from the statistical model
>
> $$
> \begin{align}
> y_i &= f^\star(\boldsymbol{x}_i) + \mathbf{e}_i \\
> \mathbf{x}_i &\sim p_X \\
> \mathbf{e}_i &\sim \mathcal{N}(0, \sigma^2) \\
> \end{align}
> $$
>
> * **Training**: To estimate the ground truth $f^\star$, we use some ML algorithm $\mathcal{A}: D \mapsto \hat{f}_D$ to obtain a trained model $\hat{f}_D$.
>
> * **Testing**: Apply the trained model $\hat{f}_D$ on test data set. Get the generalization performance of $\hat{f}_D$.

**Question**: Which factors affects the generalization performance?  

...Intuitively, the performance may depends on

* The quality of the training data
* Model complexity, i.e. the complexity of $\hat{f}_D$

## Decomposition of Test Error

Throughout the derivation, we use the square loss as performance metric which reflects regression.
$$
\begin{align}
\ell(\hat{f}_D(x), y) \triangleq (\hat{f}_D(x) - y)^2
\end{align}
$$

### Revisiting Training Process

Both training samples and test samples are drawn from the joint distribution

$$
\begin{align}
p(x,y) 
&= p(x) \cdot p(y \vert x) \\
&= p(x) \cdot p_\mathbf{e}(y - f^\star(x))
\end{align}
$$

Remark:

* In real-life application, the joint distribution $p(x,y)$ is unknown (Otherwise, there is nothing to learn literally). Here, $p(x,y)$ is used for pure theoretical study.
* In signal reconstruction, there is often no randomness in $x$. e.g. In polynomial fitting, the training samples $\{x_t\}_{t=1}^T$ are fixed: $x_t = [ 1, t, t^2, \dots, t^d]^\top$. In such application, the joint distribution simplifies to $p_{XY}(x,y) = p_\mathbf{e}(y - f^\star(x))$. More details: $\to $ *Fixed-Design Linear Regression*.
* Here, we are preserving some abstractness over the ground truth $f^\star$. Indeed, a lot of models such as deep neural networks are parameterized models. i.e. The ground truth $f^\star$ is fully characterized by the ground truth parameters $\theta^\star$. More details: $\to$ *Parameter Estimation*.

Suppose a ML algorithm $\mathcal A$ produces a model $\hat{f}_D$ from $D$. We hope that
$$
\begin{align}
\hat{f}_D = \mathcal{A}(D)
\end{align}
$$
could generalize well.

Remark:

* Example: In classification, the ML algorithm $\mathcal A$ could be SVM, Bayes classifier or decision tree. The model $\hat{f}_D$ is the resulting decision rule which maps a feature vector to a label.
* The trained model $\hat{f}_D$ depends on the training dataset $D$. e.g. If we train two SVMs on two data sets, we will most likely get two slightly different classifiers, even though both training data sets are drawn from the same distribution!

### Defining the Test Error

For a **fixed** $D$, we apply the trained model $\hat{f}_D$ to a test sample drawn from the same distribution $p(x,y)$. We are interested in  
> **Expected Test Error** of a specific model $\hat{f}_D$:
> $$
> \begin{align}
> \mathbb{E}_{XY} \left[ (\hat{f}_D(x) - y)^2 \right]
> = \mathbb{E}_{X} \left[\left(\hat{f}_D(x) - f^\star(x) \right)^2\right] + \sigma^2
> \end{align}
> $$

Remark:

* The LHS represents the MSE of $\hat{f}_D$ on **test sample**
  * $\mathbb{E}_{XY}[\cdot]$ is taken over the joint distribution $p(x,y)$ from which the test sample is generated.
* The RHS shows that the test error of a specific model $\hat{f}_D$ can be decomposed into two parts:
  * $\mathbb{E}_{X} \left[\left(\hat{f}_D(x) - f^\star(x) \right)^2\right]$ represents the goodness of $\hat{f}_D$ compared to the ground truth $f^\star$.
  * $\sigma^2$ represents the goodness of the test data.
* Derivation: see appendix.

For fixed-design regression problems, the expectation $\mathbb{E}_{XY}$ is simplified to $\mathbb{E}_{\mathbf{e}}$. Hence, the formula simplifies to

> $$
> \begin{align}
> \mathbb{E}_{XY} \left[ (\hat{f}_D(x) - y)^2 \right]
> = \left(\hat{f}_D(x) - f^\star(x) \right)^2 + \sigma^2
> \end{align}
> $$

But $\mathbb{E}_{XY} \big[ \ell(\hat{f}_D(x), y) \big]$ only calulates the expected test error of a **specific model** $\hat{f}_D$. Suppose that we run our ML algorithm $\mathcal A$ on different training data sets. As mentioned before, we will get a different model on different training data set. If we pointwise average $\hat{f}_D$ over infinitely  many training sets, we get the **averaged model**

> $$
> \begin{align}
> \bar{f}(x) = \mathbb E_D [\hat{f}_D(x)]
> \end{align}
> $$

Remark:

* $\bar{f}$ is a hypothetical model for theoretical study. It can't be computed in reality since we do not have infinitely  many training sets.
* Later on, we will define
  * the bias as the difference between the average model $\bar{f}$ and the ground truth $f^\star$
  * the variance as the spread of $\hat{f}_D$ w.r.t. $\bar{f}$. i.e. How sensitive is the model $\hat{f}_D$ to the training set $D$​.

Hence, we would like to incorporate two sources of randomness:

1. The random data set $D \sim p_{XY}^n$, on which we obtain the trained model $\hat{f}_D$ throutgh ML algorithm​​.
2. The random test sample $(x,y) \sim p_{XY}$, on which we calculate the expected test error.

and get  
> **Expected Test Error** of the ML algorithm $\mathcal{A}$:
> $$
> \begin{align}
> \mathbb{E}_{XY, D} \left[ (\hat{f}_D(x) - y)^2 \right]
> = \underbrace{\!\!\phantom{\int} \sigma^2 \phantom{\int}\!\!}_\text{Noise}
> + \underbrace{\mathbb E_X \left[\left( \bar{f}(x) - f^\star(x) \right)^2 \right] }_{ 
>   \text{Bias}^2
> }
> + \underbrace{\mathbb E_{X,D} \left[\left( \hat{f}_D(x) - \bar{f}(x) \right)^2 \right]}_{
>   \text{Variance}
> }
> \end{align}
> $$


Remark:

* The LHS represents the MSE of $\hat{f}_D$ on **test sample**, **averaged over all training data sets**.
  * As mentioned before: the training set $D \sim p_{XY}^n$ and the test sample $(x,y) \sim p_{XY}$.
  * The test sample $(x,y)$ is statistically independent of training set $D$ due to iid assumption.
* The RHS is more interesting as the expected test error comprises three parts
  * **Noise**: This reflects the quality of the data. There is nothing we can do about it. Even if we trained a perfect model $\hat{f}_D = f^\star$​, we still can't get rid of this term.
  * **Bias**: This reflects asympototic property of our ML algorithm. We hope that the averaged model converges to the ground truth as we increase the number of training sets.
  * **Variance**: This reflects the sensitivity of the resulting model $\hat{f}_D$ on the training data set $D$. We hope that variance is low so that our ML algorithm produces a consistent model which generalises well.
* Derivation: see appendix.

For fixed-design regression problem, the expected test error simplifies into
> $$
> \begin{align}
> \mathbb{E}_{XY, D} \left[ (\hat{f}_D(x) - y)^2 \right]
> = \underbrace{\!\!\phantom{\int} \sigma^2 \phantom{\int}\!\!}_\text{Noise}
> + \underbrace{\left( \bar{f}(x) - f^\star(x) \right)^2}_{ 
>   \text{Bias}^2
> }
> + \underbrace{\mathbb E_{D} \left[\left( \hat{f}_D(x) - \bar{f}(x) \right)^2 \right]}_{
>   \text{Variance}
> }
> \end{align}
> $$

If the model complexity is too low, the bias remains high no matter how many training sets we have. The resulting model suffers from underfitting because it fails to capture complex relationship between feature and label.

If the model complexity is too high, the variance will be large since the trained model $\hat{f}_D$ becomes highly sensitive to the specific training set $D$. Small changes in the training set can lead to large changes in the model, causing overfitting.

## Connection to Parameter Estimation

In parameter estimation, we assume that the ground truth $f^\star$ is parametrized by $\theta^\star$.

| Supervised Learning                                       | Parameter Estimation                      |
| --------------------------------------------------------- | ----------------------------------------- |
| Try to learn $f^\star$                                    | Try to learn $\theta^\star$               |
| MSE of $\hat{f}_D (x_\text{test})$ w.r.t. $y_\text{test}$ | MSE of $\hat\theta$ w.r.t. $\theta^\star$ |
| Generalization Error = Noise + Bias$^2$ + Variance        | Mean Squared Error = Bias$^2$ + Variance  |

## Appendix

### Deriving the Expected Test Error of $\hat{f}_D$

Show that

$$
\begin{align*}
\mathbb{E}_{XY} \left[ (\hat{f}_D(x) - y)^2 \right]
= \mathbb{E}_{X} \left[\left(\hat{f}_D(x) - f^\star(x) \right)^2\right] + \sigma^2
\end{align*}
$$

*Proof*: Using the fact that $y=f^\star(x) +\varepsilon$, we have

$$
\begin{align*}
\mathbb{E}_{XY}\left[ (\hat{f}_D(x) - y)^2 \right]
&= \mathbb{E}_{X\Epsilon}\left[ (\hat{f}_D(x) - f^\star(x) - \varepsilon)^2 \right]
\\
&= \mathbb{E}_{X}\left[ (\hat{f}_D(x) - f^\star(x))^2 \right] +
   \underbrace{\mathbb{E}_{\Epsilon}(\varepsilon^2)}_{\sigma^2} -
   2\,\underbrace{\mathbb{E}_{X\Epsilon}\left[(\hat{f}_D(x) - f^\star(x)) \cdot\varepsilon\right]}_{0}
\end{align*}
$$

The 3rd term is zero since
$$
\begin{align*}
\mathbb{E}_{X\Epsilon}\left[(\hat{f}_D(x) - f^\star(x)) \cdot\varepsilon\right]
&= \mathbb{E}_{X}\left[\hat{f}_D(x) - f^\star(x)\right] \cdot \underbrace{\mathbb{E}_{\Epsilon}\left[\varepsilon\right]}_{0}
\tag*{$\blacksquare$}
\end{align*}
$$


### Deriving the Expected Test Error of $\bar{f}$

Show that

$$
\mathbb{E}_{D} \left[\mathbb{E}_{XY} \left[ \ell(\hat{f}_D(x), y) \right]\right]
= \sigma^2 
+ \mathbb E_X \left[\left( \bar{f}(x) - f^\star(x) \right)^2 \right] 
+ \mathbb E_{X,D} \left[\left( \hat{f}_D(x) - \bar{f}(x) \right)^2 \right]
$$

*Proof*: Using the result from previous section, we have
$$
\begin{align*}
&\mathbb{E}_D \left[
\mathbb{E}_{X} \left[\left(\hat{f}_D(x) - f^\star(x) \right)^2\right] + \sigma^2
\right] \\
&= \sigma^2 + \mathbb{E}_{X,D} \left[\left(\hat{f}_D(x) - f^\star(x) \right)^2\right] \\
&= \sigma^2 + \mathbb{E}_{X,D} \left[\left(\hat{f}_D(x) - \bar{f}(x) + \bar{f}(x) -f^\star(x) \right)^2\right] \\
&= \sigma^2 + \mathbb{E}_{X,D} \left[\left(\hat{f}_D(x) - \bar{f}(x) \right)^2\right] + \mathbb{E}_{X} \left[\left( \bar{f}(x) - f^\star(x) \right)^2\right] + \mathbb{E}_{X,D} \left[\big(\hat{f}_D(x) - \bar{f}(x)\big) \cdot \big( \bar{f}(x) -f^\star(x) \big)\right]\\
\end{align*}
$$

The 3rd term is zero since
$$
\begin{align*}
\mathbb{E}_{X,D} \left[\big(\hat{f}_D(x) - \bar{f}(x)\big) \big( \bar{f}(x) -f^\star(x) \big)\right]
&= \mathbb{E}_{X} \left[ \mathbb{E}_{D} \left[\big(\hat{f}_D(x) - \bar{f}(x)\big) \cdot \big( \bar{f}(x) -f^\star(x) \big) \right]\right] \\
&= \mathbb{E}_{X} \left[
   \underbrace{\mathbb{E}_{D}\left[\big(\hat{f}_D(x) - \bar{f}(x)\big)\right]}_{0} \cdot \big( \bar{f}(x) -f^\star(x) \big)
   \right]
\tag*{$\blacksquare$}
\end{align*}
$$
