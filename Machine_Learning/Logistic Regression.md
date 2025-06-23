# Logistic Regression

TODO:

* Highlight equations
* Use align environment
* Use mathbf to typeset vectors and matrices

## Logistic Function

The logistic function $\sigma(\cdot)$ is defined as

$$
\sigma: \mathbb R \to \mathbb R, z \mapsto \frac{1}{1+e^{-z}}
$$

For linear decision boundary, parameters are $\theta = (w, b)$.

We model the label noise as

$$
Y \vert x \sim \operatorname{Ber}\left( \sigma(w^\top x + b) \right)
$$

where

$$
\begin{align}
p(Y=+1 \vert x, w, b) 
&= \sigma(w^\top x + b) \\
&= \frac{1}{1+\exp(-w^\top x - b)} \\
p(Y=-1 \vert x, w, b) 
&= \sigma(-(w^\top x + b)) \\
&= \frac{1}{1+\exp(-w^\top x - b)}
\end{align}
$$

Remarks:

* If $x$ is far away from the decision boundary, then the label noise will be very low because
  * $w^\top x + b \gg 0 \implies p(Y=+1 \vert x, \theta) \approx 1$. i.e. $Y$ is very likely $+1$.
  * $w^\top x + b \ll 0 \implies p(Y=+1 \vert x, \theta) \approx 0$. i.e. $Y$ is very likely $-1$.
* If $x$ is close to the boundary, then the label noise will be very high because
  * $w^\top x + b \approx 0 \implies p(Y=+1 \vert x, \theta) \approx 0.5$. i.e. $Y$ is one-half and one-half.
* The formula for $p(Y=+1 \vert x, w, b) $ and $p(Y=-1 \vert x, w, b)$ can be expressed more compactly in
  $$
  \begin{align}
    p(y \vert x, w, b) 
    &= \sigma(y(w^\top x + b)) \\
    &= \frac{1}{1+\exp(-y(w^\top x + b))} \\
  \end{align}  
  $$

## Comparison to Regression

Ground truth
$$
f^\star: \mathbb{R}^d \to \mathbb{R}
$$

Gaussian nosie:

$$
Y \vert x \sim \mathcal{N}\left(f^\star(x), \sigma^2\right)
$$

Ground truth
$$
\operatorname{sign} \circ f^\star: \mathbb{R}^d \to \{+1, -1\}
$$

Beroulli noise:

$$
Y \vert x \sim \operatorname{Ber}\left( \sigma(f^\star(x)) \right)
$$

## MLE for Logistic Regression

$$
\begin{align*}
&\max_{w, b} p( x_1, \dots, x_n, y_1, \dots, y_n \vert w, b) \\
&\iff \max_{w, b} p(y_1, \dots, y_n \vert x_1, \dots, x_n, w, b) p(x_1, \dots, x_n) \\
&\iff \max_{w, b} p(y_1, \dots, y_n \vert x_1, \dots, x_n, w, b) \\
&\iff \max_{w, b} \prod_{i=1}^n p(y_i \vert x_i, w, b) \\
&\iff \max_{w, b} \sum_{i=1}^n \log p(y_i \vert x_i, w, b) \\
\end{align*}
$$

By assumption that $Y_i \vert x_i \sim \operatorname{Ber}\left( \sigma(w^\top x_i + b) \right)$ and [properties of logistic function](#properties-of-logistic-function), we have

$$
\begin{align*}
p(Y=+1 \vert x, \theta) &= \sigma(w^\top x + b) \\

p(Y=-1 \vert x, \theta) &= \sigma(-(w^\top x + b)) \\
\end{align*}
$$

## Appendix

### Properties of logistic function

Let $\sigma(\cdot)$ be the logistic function
$$
\sigma: \mathbb R \to \mathbb R, z \mapsto \frac{1}{1+e^{-z}}
$$

Then,

$$
\begin{align*}
&\text{Symmetry:} & \sigma(z) + \sigma(-z) &= 1 \\
&\text{Derivative:} & \sigma'(z) &= \sigma(z) (1-\sigma(z))
\end{align*}
$$

Proofs are omitted since they are trivial.