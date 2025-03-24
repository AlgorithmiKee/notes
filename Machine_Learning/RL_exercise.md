# Exercise Reinfocement Learning

## Basic Understanding

> Each question has exactly **one** correct solution

1. Consider an MDP where both state space $\mathcal S$ and action space $\mathcal A$ are finite. What is the total number of deteriministic policies? (Assume time-independent policies only)

    $$
    \begin{array}{lll}
    \square\; 1 \quad &
    \square\; \vert\mathcal S\vert + {\vert\mathcal A\vert} \quad &
    \square\; \vert\mathcal S\vert \cdot {\vert\mathcal A\vert} \quad
    \\
    \square\; \vert\mathcal S\vert^{\vert\mathcal A\vert} \quad &
    \square\; \vert\mathcal A\vert^{\vert\mathcal S\vert} \quad &
    \square\; \infty
    \end{array}
    $$

1. Let $\pi$ and $\pi'$ be two distinct policies. Which of the following quantity is equal to $q_{\pi}(s,\pi'(s))$?

    $$
    \begin{array}{llll}
    \square\; v_{\pi}(s) &
    \square\; v_{\pi'}(s) &
    \square\; r(s, \pi'(s)) &
    \square\; q_{\pi'}(s, \pi(s))
    \\
    \square\; \mathcal B_{\pi'} v_{\pi}(s) &
    \square\; \mathcal B_{\pi} v_{\pi'}(s) &
    \square\; \mathcal B_{*} v_{\pi}(s) &
    \square\; \mathcal B_{*} v_{\pi'}(s) &
    \end{array}
    $$

1. Recall the row-stochastic matrix $\mathbf P_{\pi}\in\mathbb R^{n\times n}$ introduced in finite state space MDP, where

    $$
    \begin{align*}
    (\mathbf P_{\pi})_{ij}
    &= p(S' = \varsigma_j \mid S =\varsigma_i,\pi(\varsigma_i)), \forall i,j=1,\dots,n \\
    \mathcal S &= \{\varsigma_1,\dots, \varsigma_n\}
    \end{align*}
    $$

    Which statement about $\mathbf P_{\pi}$ is **wrong**?

   - [ ] $\mathbf P_{\pi}$ is not always symmetric.
   - [ ] $\mathbf P_{\pi}$ is always positive definite.
   - [ ] The all-one vector $\mathbf 1$ is always an eigenvector of $\mathbf P_{\pi}$ with eigenvalue 1.
   - [ ] Let $\boldsymbol{\rho},\boldsymbol{\rho}' \in\mathbb R^{1\times n}$ be such that $(\boldsymbol{\rho})_i = p(S=\varsigma_i)$ and $(\boldsymbol{\rho}')_i = p(S'=\varsigma_i)$. Then, $\boldsymbol{\rho}' = \boldsymbol{\rho}\mathbf P_{\pi}$.
   - [ ] Let $\mathbf{v}_{\pi}\in\mathbb R^{n\times 1}$ be such that $(\mathbf{v}_{\pi})_i = v_{\pi}(\varsigma_i)$. Then, $\mathbb E[v_{\pi}(S')\mid S=\varsigma_k] = \sum_{j=1}^n (\mathbf P_{\pi})_{kj}(\mathbf{v}_{\pi})_j$.

1. Recall the optimal Q-function $q^*(s,a)$. Then, the greedy policy w.r.t. $q^*(s,a)$ must be an optimal policy.

    $$
    \begin{array}{ll}
    \square\; \text{true} \qquad &
    \square\; \text{false} &
    \end{array}
    $$

## Analytical Solution of Bellman Equation

Recall the analytical solution of Bellman equation for finite state space $\mathcal S=\{\varsigma_1,\dots, \varsigma_n\}$
$$
\begin{align*}
\mathbf v_{\pi} = (\mathbf{I} - \gamma \mathbf{P}_{\pi})^{-1} \mathbf r_{\pi}
\end{align*}
$$

where
$$
\gamma\in[0,1),\:
\mathbf v_{\pi} =
\begin{bmatrix}
v_{\pi}(\varsigma_1) \\ v_{\pi}(\varsigma_2) \\ \vdots \\ v_{\pi}(\varsigma_n)
\end{bmatrix}
,\:
\mathbf r_{\pi} =
\begin{bmatrix}
r(\varsigma_1, \pi(\varsigma_1)) \\ r(\varsigma_2, \pi(\varsigma_2)) \\ \vdots \\ r(\varsigma_n, \pi(\varsigma_n))
\end{bmatrix}
,\:
\mathbf P_{\pi} =
\begin{bmatrix}
p(\varsigma_1 \mid \varsigma_1, \pi(\varsigma_1)) & \dots & p(\varsigma_n \mid \varsigma_1, \pi(\varsigma_1))  \\
p(\varsigma_1 \mid \varsigma_2, \pi(\varsigma_1)) & \dots & p(\varsigma_n \mid \varsigma_2, \pi(\varsigma_1))  \\
\vdots & \cdots & \vdots \\
p(\varsigma_1 \mid \varsigma_n, \pi(\varsigma_1)) & \dots & p(\varsigma_n \mid \varsigma_n, \pi(\varsigma_1))
\end{bmatrix}
$$

1. Show that if $\lambda$ is an eigenvalue of $\mathbf P_{\pi}$, then $1-\gamma\lambda$ is an eigenvalue of $\mathbf{I} - \gamma \mathbf{P}_{\pi}$.
1. Is $1-\gamma$ an eigenvalue of $\mathbf{I} - \gamma \mathbf{P}_{\pi}$? If so, what is the correspoinding eigenvector? If not, justify your answer.
1. Show that $\mathbf{I} - \gamma \mathbf{P}_{\pi}$ is invertible for $\forall\gamma\in[0,1)$.

## Bellman Optimality Operator

Recall the definition of Bellman optimality operator $\mathcal B_{*}: \mathcal V \to \mathcal V, v\mapsto \mathcal B_{*}v$ where

$$
\mathcal B_{*} v(s) = \max_{a\in\mathcal A} \Big\{
    r(s, a) + \gamma\mathbb E_{s' \sim p(\cdot \mid s, a)} [ v(s') ]
\Big\}
$$

1. Show that for any policy $\pi$, it holds that $\mathcal B_{*}v_{\pi} \ge v_{\pi}$.
1. Does the inequality $\mathcal B_{*}v \ge v$ hold for arbitray bounded value function $v\in\mathcal V$? Justify your answer.

## Transformed Reward Function

Consider an old MDP with discount factor $\gamma$, reward function $r(s,a)$.
A policy $\pi$ results in the value function $v_{\pi}$ in the old MDP.
Now, let the new MDP be the same as the old MDP, expect that the reward function of the new MDP is an affine transformation of the old reward function:

$$
r'(s,a) = m \cdot r(s,a) + c,
\quad m>0, c\in\mathbb R
$$

The same policy $\pi$ results in value function $v'_{\pi}$ in the new MDP.

1. For any $s\in\mathcal S$, express $v'_{\pi}(s)$ in terms of $v_{\pi}(s)$, $\gamma$, $m$ and $c$.
1. Suppose $\pi^*$ is an optimal policy of the old MDP. Is $\pi^*$ also optimal in the new MDP? Justify your answer.

## Model Mismatch

## Policy Evaluation for Stochastic Linear Dynamics

Let the state space $\mathcal S = \mathbb R^n$ and action space $\mathcal A = \mathbb R^m$. Consider the following stochastic linear dymanics with quadratic reward function
$$
\begin{align*}
s_{t+1} &= As_{t} + Ba_{t} + \epsilon, \quad \epsilon\sim\mathcal N(0,\Sigma) \\
r(s_t,a_t) &= -s_{t}^\top Qs_{t} - a_{t}^\top Ra_{t}
\end{align*}
$$

where

- $t\in\mathbb N, s_{t},s_{t+1},\epsilon\in\mathbb R^n, a\in\mathbb R^m, A\in\mathbb R^{n\times n}, B\in\mathbb R^{n\times m}$.
- $Q\in\mathbb R^{n\times n}$ and $R\in\mathbb R^{m\times m}$ are symmetric positive definite matrices.

A linear policy $\pi:\mathbb R^n \to\mathbb R^m$ is given by

$$
\pi(s) = Ks, \quad K\in\mathbb R^{m\times n}
$$

Throughout this problem, assume the system parameters $(A, B, Q, R, \Sigma, \gamma)$, and the controller matrix $K$ are known.

1. For a given state-action pair $(s,a)$, determine the conditional PDF of the future state.
    $$
    p(s'\mid s,a) \triangleq p(s_{t+1}=s' \mid s_{t}=s, a_{t}=a)
    $$
2. For a given current state $s$, calulate the immediate reward by executing $\pi$.
    $$
    r(s,\pi(s)) \triangleq r(s_{t}=s, a_t=\pi(s))
    $$

Let $v_0$ be the initial guess of the true value function $v_{\pi}$. By applying Bellman update $v_{k+1} = \mathcal B_{\pi}v_{k}$, we obtain $v_1, v_2, \dots$ etc.

3. Show that Bellman update $v_{k+1} = \mathcal B_{\pi}v_{k}$ becomes
    $$
    v_{k+1}(s) =
    -s^{\top}(Q+K^{\top}RK)s + \gamma\mathbb E_{s'\sim\mathcal N((A+BK)s, \Sigma)}[v_k(s')]
    $$
4. We set the initial guess to $v_0(s) = 0, \forall s\in\mathbb R^n$. Show by induction that for all $k\in\mathbb N$, $v_k(s)$ is a quadratic function of the form
    $$
    v_{k}(s) = s^{\top}P_ks + c_k
    $$

    > Hint: For a random vector $\mathbf{x}$ and a symmetric matrix $\mathbf{A}$,
    > $$
    >   \mathbb E_{\mathbf x}[\mathbf{x}^{\top}\mathbf{A}\mathbf{x}] =
    >   \boldsymbol{\mu}^{\top}\mathbf{A}\boldsymbol{\mu} + \operatorname{tr}(\mathbf{A}\boldsymbol{\Sigma})
    > $$
    > where $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$ are the mean and variance of $\mathbf{x}$ respectively.

By the convergence $\displaystyle\lim_{k\to\infty} v_k = v_{\pi}$, we conclude that the true value function is also of the form
$$
v_{\pi}(s) = s^{\top}Ps + c
$$

5. Show that $P$ satisfies the equation
    $$
    P = Q + K^{\top}RK + \gamma(A+BK)^{\top}P(A+BK)
    $$

6. Express $c$ in terms of $P$ and system parameters.
