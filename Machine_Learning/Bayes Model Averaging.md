# Bayes Model Averaging

**Preliminary**: fixed-desgin linear regression, ordinary least square

In this article, we illustrate *Bayes model averaging* (BMA) for
random-design linear regression problem:

* Given: training dataset $D=\{(\mathbf{x}_i, y_i)\}_{i=1}^n \stackrel{\text{iid}}{\sim} p(\mathbf x, y)$ where $(\mathbf{x}_i, y_i) \in \mathbb R^d \times \mathbb R$.
* Statistical model: $y_i = \mathbf{w}^\top \mathbf{x}_i + \varepsilon_i, \quad \varepsilon_i \stackrel{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2)$
* Additional assumption: $\varepsilon_i$ and $\mathbf{x}_i$ are statistically independent

## Classical Approach

The ground-truth parameter $\mathbf w$ can be estimated either by MLE or MAP. We will demonstrate that the optimization problem for random-design linear regression is equivalent to that of fixed-design linear regression. The key idea is that the training samples $\mathbf x_i$ are indepedent of the parameter $\mathbf w$.

### MLE

Aim: maximize log likelihood:

$$
\begin{align}
\ln p(D \vert \mathbf w)
&= \ln \prod_{i=1}^n p(\mathbf{x}_i, y_i \vert\mathbf{w}) \\
&= \ln \prod_{i=1}^n p(y_i \vert \mathbf{x}_i, \mathbf{w}) p(\mathbf{x}_i) \\
&= \underbrace{
    \ln \prod_{i=1}^n p(y_i \vert \mathbf{x}_i, \mathbf{w})
  }_\text{log of cond. likelihood} +
  \underbrace{
    \ln\prod_{i=1}^n p(\mathbf{x}_i)
  }_{\text{indep. of } \mathbf w} \\
\end{align}
$$

where

$$
\begin{align}
p(y_i \vert \mathbf{x}_i, \mathbf{w})
&= \mathcal{N}(y_i \vert \mathbf{w}^\top \mathbf{x}_i, \sigma^2) \\
&= \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(-\frac{(y_i - \mathbf{w}^\top \mathbf{x}_i)^2}{2\sigma^2}\right) \nonumber \\
\implies
\ln p(y_i \vert \mathbf{x}_i, \mathbf{w})
&= -\frac{1}{2\sigma^2} (y_i - \mathbf{w}^\top \mathbf{x}_i)^2 + \text{const.} \nonumber
\end{align}
$$

Hence, MLE is equivalent to ordinary LS as

$$
\begin{align}
\max_{\mathbf w} \ln p(D \vert \mathbf w)
&\iff  \max_{\mathbf w} \ln \prod_{i=1}^n p(y_i \vert \mathbf{x}_i, \mathbf{w}) \\
&\iff  \max_{\mathbf w} \sum_{i=1}^n \ln p(y_i \vert \mathbf{x}_i, \mathbf{w}) \\
&\iff  \max_{\mathbf w} -\frac{1}{2\sigma^2} \sum_{i=1}^n (y_i - \mathbf{w}^\top \mathbf{x}_i)^2 \\
&\iff  \min_{\mathbf w} \sum_{i=1}^n (y_i - \mathbf{w}^\top \mathbf{x}_i)^2 \\
\end{align}
$$

### MAP

Aim: maximize posterior probability:

$$
\begin{align}
\ln p(\mathbf w \vert D)
&\propto \ln p(D \vert \mathbf w) p(\mathbf w) \\
&= \ln p(D \vert \mathbf w) + \ln p(\mathbf w) \\
&= \ln \prod_{i=1}^n p(y_i \vert \mathbf{x}_i, \mathbf{w}) + \underbrace{\ln \prod_{i=1}^n p(\mathbf{x}_i)}_{\text{indep. of } \mathbf w} + \ln p(\mathbf w)
\end{align}
$$

Hence, MAP is equivalent to regularised LS

$$
\begin{align}
\max_{\mathbf w} \ln p(\mathbf w \vert D)
&\iff \max_{\mathbf w} \ln p(D \vert \mathbf w) p(\mathbf w) \\
&\iff \max_{\mathbf w} \ln \prod_{i=1}^n p(y_i \vert \mathbf{x}_i, \mathbf{w}) + \ln p(\mathbf w) \\
&\iff  \max_{\mathbf w} \sum_{i=1}^n \ln p(y_i \vert \mathbf{x}_i, \mathbf{w}) + \ln p(\mathbf w) \\
&\iff  \max_{\mathbf w} -\frac{1}{2\sigma^2} \sum_{i=1}^n (y_i - \mathbf{w}^\top \mathbf{x}_i)^2 + \ln p(\mathbf w) \\
&\iff  \min_{\mathbf w}  \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - \mathbf{w}^\top \mathbf{x}_i)^2 - \ln p(\mathbf w)
\end{align}
$$

For Guassian prior $p(\mathbf w) \sim \mathcal N(0, \lambda I)$, the MAP coincides with L2-regularised LS

$$
\begin{align}
\max_{\mathbf w} \ln p(\mathbf w \vert D)
\xLeftrightarrow[]{p(\mathbf w) \sim \mathcal N(0, \lambda I) \vphantom{\int_l}}
\min_{\mathbf w} \sum_{i=1}^n (y_i - \mathbf{w}^\top \mathbf{x}_i)^2 + \lambda \Vert \mathbf w \Vert^2_2
\end{align}
$$

## Model Averaging

Both MLE and MAP results in a single estimate of $\mathbf w$, which is used to predict $y_\text{test}$ given a test data point $\mathbf x_\text{test}$.

Bayes view: For a test data point $\mathbf x_\text{test}$, all we are interested in is

$$
\begin{align}
p(y_\text{test} \vert \mathbf x_\text{test}, D)
&= \int p(y_\text{test}, \mathbf w \vert \mathbf x_\text{test}, D) \: \mathrm d \mathbf w \\
&= \int p(y_\text{test} \vert  \mathbf w, \mathbf x_\text{test}, D) p(\mathbf w \vert D) \: \mathrm d \mathbf w
\end{align}
$$

Remark:

* The integral on the RHS is a model averaging. It averages $\mathbf w^\top \mathbf x_\text{test}$ for every possible $\mathbf w$.
* In general, this integral aka $p(y_\text{test} \vert \mathbf x_\text{test}, D)$ has no closed-form solution. However, if everything is Gaussian, we can indeed solve this integral since we only need to solve the mean and variance of $p(y_\text{test} \vert \mathbf x_\text{test}, D)$.

Key idea: Consider the joint distribution
$$
\begin{align}
p(y_1, \dots, y_n, y_\text{test}) \sim \mathcal (\mathbf{0}, \boldsymbol\Sigma)
\end{align}
$$

where $\mathbf{0}\in\mathbb R^{n+1}$ and $\boldsymbol\Sigma\in\mathbb R^{(n+1)\times(n+1)}$.
W.l.o.g. we may assume zero mean.

The covariance matrix $\boldsymbol\Sigma$ has the form

$$
\begin{align}
\boldsymbol\Sigma =
\begin{bmatrix}
\mathbf{K}_\text{train, train} & \mathbf{K}_\text{train, test} \\
\mathbf{K}_\text{train, test}^\top  & \mathbf{K}_\text{test, test}
\end{bmatrix}
\end{align}
$$

where we use kernel matrices as convariance matices.

# Appendix

## Multivariant Gaussian

Let $X: \Omega \to \mathbb R^{n+m}$ be a Gaussian random vector.

$$
X \sim \mathcal{N}\left( \boldsymbol\mu, \boldsymbol\Sigma \right)
$$

where

$$
\boldsymbol\mu\in\mathbb{R}^{n+m}, \quad \boldsymbol\Sigma\in\mathbb{R}^{(n+m)\times(n+m)}
$$

We partition $X$ into two subvectors $X_A$ and $X_B$

* $X_A = [X_1, \dots, X_n]^\top$
* $X_B = [X_{n+1}, \dots, X_{n+m}]^\top$

The overall mean and covariance can be partitioned into

$$
\begin{bmatrix}  X_A \\  X_B \end{bmatrix}
\sim\mathcal{N}
\left(
  \begin{bmatrix} \boldsymbol\mu_A \\ \boldsymbol\mu_B \end{bmatrix},
  \begin{bmatrix}
    \boldsymbol\Sigma_{AA} & \boldsymbol\Sigma_{AB} \\
    \boldsymbol\Sigma_{BA} & \boldsymbol\Sigma_{BB}
  \end{bmatrix}
\right)
$$

where

$$
\begin{matrix}
  \boldsymbol\mu_A\in\mathbb{R}^n \\  \boldsymbol\mu_B\in\mathbb{R}^m
\end{matrix}
\quad\text{ and }\quad
\begin{matrix}
  \boldsymbol\Sigma_{AA}\in\mathbb{R}^{n \times n} & \boldsymbol\Sigma_{AB}\in\mathbb{R}^{n \times m} \\
  \boldsymbol\Sigma_{BA}\in\mathbb{R}^{m \times n} & \boldsymbol\Sigma_{BB}\in\mathbb{R}^{m \times m}
\end{matrix}
$$

### Marginalized Gaussian

One can show that the marginal distributions

$$
\begin{align}
p(\mathbf x_A) = \int p(\mathbf x_A, \mathbf x_B ; \boldsymbol\mu, \boldsymbol\Sigma) \:\mathrm{d}\mathbf x_B \\
p(\mathbf x_B) = \int p(\mathbf x_A, \mathbf x_B ; \boldsymbol\mu, \boldsymbol\Sigma) \:\mathrm{d}\mathbf x_A \\
\end{align}
$$

are also Gaussian

$$
\begin{align}
X_A &\sim \mathcal{N}\left( \boldsymbol\mu_A, \boldsymbol\Sigma_{AA}\right) \\
X_B &\sim \mathcal{N}\left( \boldsymbol\mu_B, \boldsymbol\Sigma_{BB}\right)
\end{align}
$$

### Gaussian Conditioning

Conditioned on $X_B = \mathbf x_B$, the distribution of $X_A$
$$
\begin{align}
p(\mathbf x_A \vert \mathbf x_B) 
&= \frac{p(\mathbf x_A, \mathbf x_B ; \boldsymbol\mu, \boldsymbol\Sigma)}{p(\mathbf x_B ; \boldsymbol\mu, \boldsymbol\Sigma)}  \\
&= \frac{p(\mathbf x_A, \mathbf x_B ; \boldsymbol\mu, \boldsymbol\Sigma)}{\int p(\mathbf x_A, \mathbf x_B ; \boldsymbol\mu, \boldsymbol\Sigma) \:\mathrm{d}\mathbf x_A}
\end{align}
$$

is also Gaussian
$$
\begin{align}
X_A \,\vert\, \mathbf x_B &\sim \mathcal{N}(\boldsymbol\mu_{A \vert B}, \boldsymbol\Sigma_{A \vert B}) \\
\boldsymbol\mu_{A \vert B} &= \boldsymbol\mu_{A} + \boldsymbol\Sigma_{AB} \boldsymbol\Sigma_{BB}^{-1} (\mathbf x_B - \boldsymbol\mu_B)\\
\boldsymbol\Sigma_{A \vert B} &= \boldsymbol\Sigma_{AA} - \boldsymbol\Sigma_{AB}\boldsymbol\Sigma_{BB}^{-1} \boldsymbol\Sigma_{BA}
\end{align}
$$