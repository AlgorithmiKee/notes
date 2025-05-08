# L2 Regularized Hinge Loss for Soft-Margin SVM

We will show that maximizing the margin of a soft-margin SVM is equivalent to minimizing the L2 regurlarized hinge loss.

## Slack Variables

Special note: The slack variables in the hand-written notes are denoted by $s_i$. Here, we use $\xi_i$ instead to align with standard notation.

**Hard SVM**: max margin subject to **hard** margin constraints

$$
\left.\begin{align*}
\boldsymbol{w}^\top \boldsymbol{x}_{i} + b &\ge 1  &&\text{if}\quad y_i = 1 \\
\boldsymbol{w}^\top \boldsymbol{x}_{i} + b &\le -1 &&\text{if}\quad y_i = -1 \\
\end{align*}\right\}
\iff
y_i \left( \boldsymbol{w}^\top \boldsymbol{x}_{i} + b \right) \ge 1
$$

**Soft SVM**: max margin subject to **soft** margin constraints

$$
\left.\begin{align*}
\boldsymbol{w}^\top \boldsymbol{x}_{i} + b &\ge 1 - \xi_i  &&\text{if}\quad y_i = 1 \\
\boldsymbol{w}^\top \boldsymbol{x}_{i} + b &\le -1 + \xi_i &&\text{if}\quad y_i = -1 \\
\xi_i &\ge 0, \quad \xi_i\in\mathbb{R}
\end{align*}\right\}

\iff

y_i \left( \boldsymbol{w}^\top \boldsymbol{x}_{i} + b \right) \ge 1 - \xi_i, \quad
\xi_i \ge 0
$$

Remarks:

* For each training sample $\boldsymbol{x}_i$, we introduce a nonnegative slack variable $\xi_i$, which quantifies **margin violation**. A more detailed interpretation is summarized in the table
  | $\xi_i = 0$ | $0 < \xi_i < 1$ | $\xi_i > 1$ |
  |---------------|-------------------|---------------|
  | $\boldsymbol{x}_i$ correctly classified | $\boldsymbol{x}_i$ correctly classified | $\boldsymbol{x}_i$ wrongly classified |
  | $\boldsymbol{x}_i$ satisfies the margin | $\boldsymbol{x}_i$ violates the margin   | $\boldsymbol{x}_i$ violates the margin   |
  | No panelty   | Panelty: margin violation   | Panelty: margin violation |
  * Note: The panelty takes place as long as $\boldsymbol{x}_i$ violates the margin, even though it might be correclty classified.
* During optimization, we would like to make each $\xi_i$ as small as possible.
* Graphically, for a positive sample $\boldsymbol{x}_{+}$, the corresponding slack variable quantifies how much $\boldsymbol{x}_+$ lies away from the hyperplane $\{ \boldsymbol{x}: \boldsymbol{w}^\top\boldsymbol{x} + b = 1\}$. The graphical interpretation for negative samples are analogous.

## Max Margin Formulation

The optimization problem of soft SVM can be formulated as

> $$
> \begin{align}
> \min_{\boldsymbol{w}, b} \quad
> \frac{1}{2} \boldsymbol{w}^\top\boldsymbol{w} &+ C \sum_{i=1}^n \xi_i
> \\
> \text{s.t.}  \quad y_i \left( \boldsymbol{w}^\top \boldsymbol{x}_{i} + b \right) &\ge 1 - \xi_i, &\forall i=1,\dots, n
> \\
> \xi_i &\ge 0, &\forall i=1,\dots, n
> \end{align}
> $$

Remark:

* Minimizing the term $\displaystyle \frac{1}{2} \boldsymbol{w}^\top\boldsymbol{w}$ in the objective represents maximizing margin.
* Minimizing the term $\displaystyle C \sum_{i=1}^n \xi_i$ in the objective represents minimizing margin violation (and potentially mis-classification).

## Hinge Loss Formulation

The soft constraints can be reformulated into

$$
\begin{align*}
\xi_i &\ge 1 - y_i \left( \boldsymbol{w}^\top \boldsymbol{x}_{i} + b \right)
\\
\xi_i &\ge 0
\end{align*}
\quad \forall i=1,\dots, n
$$

or more compactly

$$
\begin{align}
\xi_i \ge \max\left\{0,\: 1 - y_i \left( \boldsymbol{w}^\top \boldsymbol{x}_{i} + b \right) \right\} \quad \forall i=1,\dots, n
\end{align}
$$

Hence, maximizing margin subject to soft margin constraints is equivalent to
> $$
> \begin{align}
>   \min_{\boldsymbol{w}, b} \quad
>   \frac{1}{2} \boldsymbol{w}^\top\boldsymbol{w} +
>   C \sum_{i=1}^n \max\left\{0,\: 1 - y_i \left( \boldsymbol{w}^\top \boldsymbol{x}_{i} + b \right) \right\}
> \end{align}
> $$

or

> $$
> \begin{align}
> \min_{\boldsymbol{w}, b} \quad
> \underbrace{
>   \frac{1}{2} \left\Vert\boldsymbol{w}\right\Vert_2^2 \vphantom{\sum_i^n}
> }_\text{L2 Regularizer} + \:\: C
> \underbrace{
>   \sum_{i=1}^n \ell_\text{hinge}\left( \boldsymbol{w}^\top \boldsymbol{x}_{i} + b,\: y_i \right)
> }_\text{Empirical Loss}
> \end{align}
> $$

where the hinge loss is defined as

> $$
> \begin{align}
> \ell_\text{hinge}\left( f(\boldsymbol{x}), y \right)
> &= \max\left\{0, 1-yf(\boldsymbol{x})\right\} \\
> &=\begin{cases}
>     1-yf(\boldsymbol{x}), &\text{if}\quad yf(\boldsymbol{x}) < 1 \\
>     0,       &\text{if}\quad yf(\boldsymbol{x}) \ge 1
>   \end{cases}
> \end{align}
> $$

Remarks:

* The hinge loss formulation is unconstrained and thus can be solved on the primal side.
* The overall cost function comprises empirical loss and regularizer, just as cost functions in other ML problems.
* Soft margin constraints $\iff$ hinge loss.
* Maximizing margin $\iff$ L2 regularization
