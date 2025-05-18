# Bayes Model Averaging

## Recap: Point Estimate by MAP

Recall: MAP estimation

* Given: observations $D = \{\mathbf x_1, \dots, \mathbf x_n\} \stackrel{\text{iid}}{\sim} p(\mathbf x \mid \boldsymbol{\theta})$ with prior $p(\boldsymbol{\theta})$
* Goal: estimate $\boldsymbol{\theta}$

The MAP estimate is the **mode of the posterior**

$$
\begin{align}
\hat{\boldsymbol{\theta}}_\text{MAP}
&= \argmax_{\boldsymbol{\theta}} p(\boldsymbol{\theta} \mid D)
\end{align}
$$

For a new data point $\mathbf x_*$, the ***plug-in predictive distribution*** is fully charactered by the point estimate $\hat{\boldsymbol{\theta}}_\text{MAP}$

$$
\begin{align}
p(\mathbf x_* \mid \hat{\boldsymbol{\theta}}_\text{MAP})
\end{align}
$$

Remark:

* For simplicity, we often call $p(\mathbf x_* \mid \hat{\boldsymbol{\theta}}_\text{MAP})$ ***plug-in predictive***.
* The term "plug-in" hightlights the fact that we plug a single point estimate $\hat{\boldsymbol{\theta}}_\text{MAP}$ into $p(\mathbf x_* \mid \boldsymbol{\theta})$

## Bayesian Inference

Key idea of ***Bayesian inference***:

> Use the **full** posterior distribution $p(\boldsymbol{\theta} \mid D)$ instead of just using its mode (i.e. a point estimate)

Philosophy: Frequentist statistics vs. Bayesian statistics

* Frequentist statistics: $\boldsymbol{\theta}$ is unknown but fixed. It makes no sense to talk about the probability of $\boldsymbol{\theta}$ (either prior or posterior). The true PDF of $\mathbf x$ is thus an unknown but fixed function. The observations are used to estimate the PDF of $\mathbf x$ as accurately as possible.
* Bayesian statistics: $\boldsymbol{\theta}$ is a random variable. There are (infinitely) many possible PDFs of $\mathbf x$, some of which are more likely (or more important) than the others. After we made observations, we update the PDF of $\boldsymbol{\theta}$ and thus updated the importance of each possible PDF of $\mathbf x$.

For a new data point $\mathbf x_*$, each $\boldsymbol{\theta}$ gives $p(\mathbf x_* \mid \boldsymbol{\theta})$. Averaging over all possible $\boldsymbol{\theta}$, we have the ***prior predictive distribution***:

> $$
> \begin{align}
> p(\mathbf x_*)
> &= \int p(\mathbf x_* \mid \boldsymbol{\theta}) \cdot p(\boldsymbol{\theta}) \:\mathrm{d}\boldsymbol{\theta} \\
> \end{align}
> $$

Remarks:

* The proof is very simple as the RHS is just the marginalization of the joint density $p(\mathbf x_*, \boldsymbol{\theta})$.
* The RHS can also be seen as a weighted average of $p(\mathbf x_* \mid \boldsymbol{\theta})$ w.r.t. the prior $p(\boldsymbol{\theta})$, which measures the importance of each $\boldsymbol{\theta}$. If $p(\boldsymbol{\theta})$ is high, then $p(\mathbf x_* \mid \boldsymbol{\theta})$ has higher contribution to $p(\mathbf x_*)$. Let $\hat{\boldsymbol{\theta}}$ be the mode of the prior. Then, $p(\mathbf x_* \mid \hat{\boldsymbol{\theta}})$ has the highest (but **not the entire!**) contribution to $p(\mathbf x_*)$.
* The formula holds even **before** we made any observations! Bayesian statistics allows prior distribution which may come from our experience.

After we made observations $D = \{\mathbf x_1, \dots, \mathbf x_n\}$, we update the prior $p(\boldsymbol{\theta})$ to posterior $p(\boldsymbol{\theta} \mid D)$. Now, we can average $p(\mathbf x_* \mid \boldsymbol{\theta})$ w.r.t. the posterior, which leads to ***posterior predictive distribution***:

> $$
> \begin{align}
> p(\mathbf x_* \mid D)
> &= \int p(\mathbf x_* \mid \boldsymbol{\theta}) \cdot p(\boldsymbol{\theta} \mid D) \:\mathrm{d}\boldsymbol{\theta}
> \end{align}
> $$

Remarks:

* The LHS is now conditioned on $D$, i.e. we updated $p(\mathbf x_*)$ to $p(\mathbf x_* \mid D)$ after observing $D$.
* Updating the prior to posterior $\iff$ updating the importance of $p(\mathbf x_* \mid \boldsymbol{\theta})$ for each $\boldsymbol{\theta}$. Let $\hat{\boldsymbol{\theta}}$ be the mode of the posterior. Then, $p(\mathbf x_* \mid \hat{\boldsymbol{\theta}})$ has the highest (but **not the entire!**) contribution to $p(\mathbf x_*)$.
* The integral on the RHS is intractable to compute in general, except for a few special case (which will be discussed later).

*Proof*: This follows from the law of total expectation and the independence between $\mathbf x_*$ and $D$.

$$
\begin{align*}
p(\mathbf x_* \mid D)
&= \int p(\mathbf x_*, \boldsymbol{\theta} \mid D) \:\mathrm{d}\boldsymbol{\theta}
\\
&= \int p(\mathbf x_* \mid \boldsymbol{\theta}, D) \cdot p(\boldsymbol{\theta} \mid D) \:\mathrm{d}\boldsymbol{\theta}
\\
&= \int p(\mathbf x_* \mid \boldsymbol{\theta}) \cdot p(\boldsymbol{\theta} \mid D) \:\mathrm{d}\boldsymbol{\theta}
\tag*{$\blacksquare$}
\end{align*}
$$

### Bayesian Inference vs MAP

Both methods

* treat parameters $\boldsymbol{\theta}$ as a random variable and assume a prior $p(\boldsymbol{\theta})$.
* update the prior to the posterior $p(\boldsymbol{\theta} \mid D)$ after observing $D$
* outputs a predictive distribution of $\mathbf x$

Difference: How is the predictive distribution computed?

| Bayesian Inference | MAP estimate |
| ------------------ | ------------ |
| uses the full posterior distribution | only uses the mode of the posterior |
| integration-based (average w.r.t. the posterior) | optimization-based (maximize the posterior) |
| output: posterior distribution | output: plug-in distribution |

### Example: Learning a Bernoulli Distribution with Beta Prior

Consider a Bernoulli distribution with unknown success rate $\theta\in[0,1]$.

$$
p(x) =
\begin{cases}
\theta   & \text{if } x=1 \\
1-\theta & \text{if } x=0
\end{cases}
$$

Suppose we have iid observations $D=\{1, 1, 0, \dots, 0, 1\}$ containing $n_1$ ones and $n_0$ zeros. Then, the likelihood is

$$
p(D \mid \theta) = \theta^{n_1} (1-\theta)^{n_0}
$$

Assume we have a Beta prior on $\theta\sim\operatorname{Beta}(\alpha,\beta)$:

$$
p(\theta) = \frac{\theta^{\alpha-1} (1-\theta)^{\beta-1}}{B(\alpha,\beta)}
$$

where $B(\alpha,\beta)$ is a normalization constant (not detailed here). Then, the posterior is

$$
\begin{align*}
p(\theta \mid D)
&\propto p(D \mid \theta) \cdot p(\theta) \\
&= \theta^{n_1} (1-\theta)^{n_0} \cdot \theta^{\alpha-1} (1-\theta)^{\beta-1} \\
&= \theta^{n_1+\alpha-1} (1-\theta)^{n_0+\beta-1}
\end{align*}
$$

By the normalization property of Beta distribution (not detailed here), the posterior is in fact also a Beta distribution

$$
\begin{align*}
p(\theta \mid D)
&= \frac{\theta^{n_1+\alpha-1} (1-\theta)^{n_0+\beta-1}}{B(n_1+\alpha,n_0+\beta)} \\
&= \operatorname{Beta}(\theta; n_1+\alpha,n_0+\beta)
\end{align*}
$$

Hence, the posterior predictive $p(x_* \mid D)$ at $x_*=1$ is

$$
\begin{align*}
p(x_*=1 \mid D)
&= \int p(x_*=1 \mid \theta) \cdot p(\theta \mid D) \:\mathrm{d}\theta \\
&= \int \theta \cdot p(\theta \mid D) \:\mathrm{d}\theta \\
&= \mathbb E[\theta \mid D] \\
&= \frac{n_1+\alpha}{n_1+n_0+\alpha+\beta}
\end{align*}
$$

Likewise,

$$
\begin{align*}
p(x_*=0 \mid D)
&= \int p(x_*=0 \mid \theta) \cdot p(\theta \mid D) \:\mathrm{d}\theta \\
&= \int (1-\theta) \cdot p(\theta \mid D) \:\mathrm{d}\theta \\
&= 1 - \mathbb E[\theta \mid D] \\
&= \frac{n_0+\beta}{n_1+n_0+\alpha+\beta}
\end{align*}
$$

### Example: Learning a Gaussian Distribution with Gaussian Prior

Consider a multivariate Gaussian with known covariance matirx $\boldsymbol\Sigma$ but unknown mean vector $\boldsymbol\mu$.

$$
\mathbf x \sim \mathcal N(\boldsymbol\mu, \boldsymbol\Sigma)
$$

Suppose we have iid observations $D=\{\mathbf x_1, \dots, \mathbf x_n\}$. Then, the likelihood is

$$
p(D \mid \theta) = \prod_{i=1}^n \mathcal N(\mathbf x_i; \boldsymbol\mu, \boldsymbol\Sigma)
$$

Assume we have a Gaussian prior on $\boldsymbol\mu$:

$$
\boldsymbol\mu \sim \mathcal N(\boldsymbol\mu_\text{p}, \boldsymbol\Sigma_\text{p})
$$

## Point Estimates in Supervised Learning

## Bayesian Linear Regression

**Preliminary**: fixed-desgin linear regression, ordinary least square

In this article, we illustrate *Bayes model averaging* (BMA) for
random-design linear regression problem:

* Given: training dataset $D=\{(\mathbf{x}_i, y_i)\}_{i=1}^n \stackrel{\text{iid}}{\sim} p(\mathbf x, y)$ where $(\mathbf{x}_i, y_i) \in \mathbb R^d \times \mathbb R$.
* Statistical model: $y_i = \mathbf{w}^\top \mathbf{x}_i + \varepsilon_i, \quad \varepsilon_i \stackrel{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2_\text{n})$
* Additional assumption: $\varepsilon_i$ and $\mathbf{x}_i$ are statistically independent
* Goal: Predict the label for a new data point $\mathbf{x}_*$ using the **full** posterior distribution $p(\mathbf w \mid D)$

### Main Results about Point Estimates

* Discriminative model:

$$
\begin{align}
p(y_i \mid \mathbf{x}_i, \mathbf{w})
&= \mathcal{N}(y_i \mid \mathbf{w}^\top \mathbf{x}_i, \sigma^2_\text{n}) \\
&= \frac{1}{\sqrt{2\pi \sigma^2_\text{n}}} \exp\left(-\frac{(y_i - \mathbf{w}^\top \mathbf{x}_i)^2}{2\sigma^2_\text{n}}\right) \nonumber \\
\end{align}
$$

* Log likelihood:

$$
\begin{align}
\ln p(D \mid \mathbf w)
&= \ln \prod_{i=1}^n p(\mathbf{x}_i, y_i \mid\mathbf{w}) + \text{const} \\
&= -\frac{1}{2\sigma^2_\text{n}} (y_i - \mathbf{w}^\top \mathbf{x}_i)^2 + \text{const} 
\end{align}
$$


* MLE $\iff$ least square

$$
\begin{align}
\hat{\mathbf w}_\text{MLE}
&= \argmin_{\mathbf w} \sum_{i=1}^n (y_i - \mathbf{w}^\top \mathbf{x}_i)^2
\end{align}
$$

* Log posterior with Gaussian prior

$$
\begin{align}
\ln p(\mathbf w \mid D)
&= \ln \prod_{i=1}^n p(\mathbf{x}_i, y_i \mid\mathbf{w}) + \ln p(\mathbf{w}) + \text{const} \\
&= -\frac{1}{2\sigma^2_\text{n}} (y_i - \mathbf{w}^\top \mathbf{x}_i)^2 -
    \frac{1}{2\sigma^2_\text{p}} \Vert\boldsymbol{\theta}\Vert^2 +
    \text{const} 
\end{align}
$$

* MAP with Gaussian prior $\iff$ ridge regression

$$
\begin{align}
\hat{\mathbf w}_\text{MAP}
&= \argmin_{\mathbf w} \sum_{i=1}^n (y_i - \mathbf{w}^\top \mathbf{x}_i)^2 +
   \frac{\sigma^2_\text{n}}{\sigma^2_\text{p}} \Vert\boldsymbol{\theta}\Vert^2
\end{align}
$$

## Model Averaging

Both MLE and MAP results in a single estimate of $\mathbf w$, which is used to predict $y_*$ given a test data point $\mathbf x_*$.

Bayes view: For a test data point $\mathbf x_*$, all we are interested in is

$$
\begin{align}
p(y_* \mid \mathbf x_*, D)
&= \int p(y_*, \mathbf w \mid \mathbf x_*, D) \: \mathrm d \mathbf w \\
&= \int p(y_* \mid  \mathbf w, \mathbf x_*, D) p(\mathbf w \mid D) \: \mathrm d \mathbf w
\end{align}
$$

Remark:

* The integral on the RHS is a model averaging. It averages $\mathbf w^\top \mathbf x_*$ for every possible $\mathbf w$.
* In general, this integral aka $p(y_* \mid \mathbf x_*, D)$ has no closed-form solution. However, if everything is Gaussian, we can indeed solve this integral since we only need to solve the mean and variance of $p(y_* \mid \mathbf x_*, D)$.

Key idea: Consider the joint distribution
$$
\begin{align}
p(y_1, \dots, y_n, y_*) \sim \mathcal (\mathbf{0}, \boldsymbol\Sigma)
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
p(\mathbf x_A \mid \mathbf x_B) 
&= \frac{p(\mathbf x_A, \mathbf x_B ; \boldsymbol\mu, \boldsymbol\Sigma)}{p(\mathbf x_B ; \boldsymbol\mu, \boldsymbol\Sigma)}  \\
&= \frac{p(\mathbf x_A, \mathbf x_B ; \boldsymbol\mu, \boldsymbol\Sigma)}{\int p(\mathbf x_A, \mathbf x_B ; \boldsymbol\mu, \boldsymbol\Sigma) \:\mathrm{d}\mathbf x_A}
\end{align}
$$

is also Gaussian
$$
\begin{align}
X_A \,\mid\, \mathbf x_B &\sim \mathcal{N}(\boldsymbol\mu_{A \mid B}, \boldsymbol\Sigma_{A \mid B}) \\
\boldsymbol\mu_{A \mid B} &= \boldsymbol\mu_{A} + \boldsymbol\Sigma_{AB} \boldsymbol\Sigma_{BB}^{-1} (\mathbf x_B - \boldsymbol\mu_B)\\
\boldsymbol\Sigma_{A \mid B} &= \boldsymbol\Sigma_{AA} - \boldsymbol\Sigma_{AB}\boldsymbol\Sigma_{BB}^{-1} \boldsymbol\Sigma_{BA}
\end{align}
$$