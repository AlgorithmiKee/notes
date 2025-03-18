# Exercise Reinfocement Learning

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

Show that for any policy $\pi$, it holds that $\mathcal B_{*}v_{\pi} \ge v_{\pi}$.

## Stochastic Linear System

Consider the following stochastic linear dymanics with state space $\mathcal S=\mathbb R^n$ and action space $\mathcal A=\mathbb R^m$.
$$
s' = As + Ba + \epsilon,
\quad \epsilon\sim\mathcal N(0,\Sigma)
$$

where $s,s',\epsilon\in\mathbb R^n, a\in\mathbb R^m$.

The reward function is given by
$$
r(s,a) = -s^\top Qs - a^\top Ra
$$
where $Q$ and $R$ are positive definite matrices.
