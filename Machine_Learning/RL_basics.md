---
title: "Intro to RL"
author: "Ke Zhang"
date: "2025"
fontsize: 12pt
---

# Reinforcement Learning

[toc]

Notation:

* upercase (e.g. $R_t$): random variable

* lowercase (e.g. $s_t$): instance of random variable

* bold straight (e.g. $\mathbf{v}$): determineistic vector
  $$
  \DeclareMathOperator*{\argmax}{arg\max}
  $$

## Markov Decision Process

A Markov decision process (MDP) consists of

* $\mathcal S$: set of states.
* $\mathcal A$: set of actions.
* $p(\cdot\mid s, a)$: state transition probability. i.e. the probability distribution of the new state given the current state $s$ and current action $a$.
* $\gamma\in(0,1)$: discount factor.
* $r: \mathcal S \times \mathcal A \to \mathbb R$: reward function. bounded.

In the following, we consider stationary MDP. i.e. Both the station transition probability and reward function are time-independent.

### Rewards

Given the state $s_t$ at time $t$, the agent takes action $a_t$, which leads to

* the new state $s_{t+1}$, sampled from $p(\cdot \mid s_t, a_t)$
* and the reward $r_t$, determined by the reward function $r_t = r(s_t, a_t)$.

Continue taking actions $a_{t+1}, a_{t+2}, \dots$, we get a state-action-reward trajectory
$$
s_t     \xrightarrow[r_t]    {a_t \:}
s_{t+1} \xrightarrow[r_{t+1}]{a_{t+1} \:}
s_{t+2} \xrightarrow[r_{t+2}]{a_{t+2} \:}
s_{t+3} \cdots
$$

$\to$ Intuitive goal: Maximize the sum of all $r_t$.

In the stochastic setting, all states $S_{t}, S_{t+1}, \dots$ are random. Since the agent takes action based on current state, the action sequence is also random. Similary, the reward sequence is also random. $\implies$ stochastic state-action-reward trajectory:

$$
S_t     \xrightarrow[R_t]    {A_t \:}
S_{t+1} \xrightarrow[R_{t+1}]{A_{t+1} \:}
S_{t+2} \xrightarrow[R_{t+2}]{A_{t+2} \:}
S_{t+3} \cdots
$$

$\to$ Intuitive goal: Maximize $\mathbb E[\text{sum of all } R_t]$.

The total (discounted) reward is defined as the sum of (discounted) rewards starting from $S_t$

$$
\begin{align}
G_t
&= R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \dots \\
&= \sum_{k=0}^\infty \gamma^k R_{t+k}
\end{align}
$$

Remarks:

* $G_t$ is a random quantity as the state transition is stochastic. Executing the same policy from $s_t$ again will yield a different $g_t$.
* The discount factor $\gamma$ serves two purposes:
  1. it ensures that the infinite sum is defined. Recall from math: If $\sum_{n=1}^\infty a_n$ converges absolutely and $\{b_n\}$ is bounded, then $\sum_{n=1}^\infty a_n b_n$ converges absolutely. Here, we have a geometric series which converges absolutely and a bounded sequence of rewards.
  2. it puts more weight on short-term rewards over long-term rewards. e.g. In finance, 100 dollar today is worth more than 100 dollar next year.

Recursive structure: The total reward $G_t$ comprises immediate reward $R_t$ plus a discounted future reward $G_{t+1}$.

$$
\begin{align}
G_t = R_t + \gamma G_{t+1}
\end{align}
$$

### Policy

In MDP, the agent performs actions determined by current state, according to a ***policy***
$$
\pi: \mathcal S \to \mathcal A, s\mapsto a=\pi(s)
$$

Remark: The policy defined here is **time-invariant** and **deterministic**, i.e. For a certain state $s$, the agent takes the **same** action **whenever** he arrives at $s$.

### State Value Function

> For a certain policy $\pi$, the corresponding ***state value function*** is mapping $v_{\pi}:\mathcal S\to\mathbb R, s\mapsto v_{\pi}(s)$.  
> $v_{\pi}(s)$ is called ***state value***, defined as the expected total reward by executing policy $\pi$ starting from state $s$.
>
> $$
> \begin{align}
> v_{\pi}(s) = \mathbb E
> \left[ G_t \:\middle|\: S_t =s \right],
> \quad \forall s \in \mathcal S
> \end{align}
> $$

Remarks:

* The expectation is take over $\{S_k\}_{k \ge t+1}$. The random variable $G_t$ depends implicity on policy $\pi$ as it represents the total reward by executing $\pi$.

* State value $v_{\pi}(s)$ is defined for **ALL** possible states in space $\mathcal S$.

* For a fixed policy $\pi$, $v_{\pi}(s)$ quantifies the goodness of state $s$.

* For a fixed initial state $s$, $v_{\pi}(s)$ quantifies the goodness of policy $\pi$.

* For a stationanry MDP, $v_{\pi}(s)$ is independent of $t$. i.e. The state value of $s$ remains the same regardless of when the agents arrives at $s$. Hence, we can assume without loss of generality that the agent arrived at $s$ at $t=0$. The state value then becomes

  $$
  v_{\pi}(s) = \mathbb E
  \left[
    G_0 \:\middle|\: S_0 =s
  \right]
  $$

### Q-function (State-Action Value)

The ***Q-function*** (or ***state action value***) for a certain policy $\pi$ as follows
> $$
> \begin{align}
> q_{\pi}(s,a)
> &= \mathbb E\left[ G_t \:\middle|\: S_t = s, A_t = a \right],
> \quad \forall s\in\mathcal S, \forall a\in\mathcal A
> \end{align}
> $$

Relation between Q-function and state value for the same $\pi$:

* Interpretation: $q_{\pi}(s,a)$ represents the total reward of taking action $a$ at initial state $s$ and then following a policy $\pi$. vs. $v_{\pi}(s)$ represents the total reward of following $\pi$ from $s$ onward.
* Compute $v_{\pi}(s)$ from $q_{\pi}(s,a)$: simply let $a=\pi(s)$, i.e.
  $$
  \begin{align}
  v_{\pi}(s) = q_{\pi}(s,a) \Big|_{a=\pi(s)}
  \end{align}
  $$

* Compute $q_{\pi}(s,a)$ from $v_{\pi}(s)$: use Bellman equation (will be proved later)
  $$
  \begin{align}
  q_{\pi}(s,a)
  &= r(s,a) + \gamma \mathbb E_{s' \sim p(\cdot \mid s, a)} [ v_{\pi}(s') ] \\[4pt]
  &= r(s,a) + \gamma \sum_{s'\in\mathcal S} p(s' \mid s, a) v_{\pi}(s') & \text{if } \mathcal S \text{ is finite}\\
  \end{align}
  $$

## Bellman Equations

### Bellman Equations for State Value

> Bellman Equations: The state values have the recursive structure
> $$
> \begin{align}
> v_{\pi}(s) = r(s, \pi(s)) + \gamma \mathbb E_{s' \sim p(\cdot \mid s, \pi(s))} [ v_{\pi}(s') ], \quad \forall s\in\mathcal S
> \end{align}
> $$

Remarks:

* The expected total reward is the sum of immediate reward plus the (discounted) expected future reward.
  * $r(s, \pi(s))$: immediate reward at current state $s$ by executing the policy $\pi$
  * $v_{\pi}(s')$: future reward from the next state $s'$ by executing the policy $\pi$. $\mathbb E_{s' \sim p(\cdot \mid s, \pi(s))} [ v_{\pi}(s') ]$ is the average future reward over all possible $s'$.
* The current action $\pi(s)$ influences the immediate reward and the probability distribution of the next state $s'$.
* Bellman equations hold for all $s\in\mathcal S$. If $\mathcal S$ is finite, there are $\vert \mathcal S \vert$ Bellman equations.

*Proof*: Without loss of generality, assume $t=0$. Using $G_0 = R_0 + \gamma G_1$, we get
$$
\begin{align*}
v_{\pi}(s)
&= \mathbb E_{S_1, S_2, \dots}
  \left[
    G_0 \:\middle|\: S_0 =s
  \right] \\
&= \mathbb E_{S_1, S_2, \dots}
  \left[
    R_0+ \gamma G_1 \:\middle|\: S_0 =s
  \right] \\
&= \mathbb E_{S_1, S_2, \dots} \left[R_0 \:\middle|\: S_0 =s \right] +
   \gamma \mathbb E_{S_1, S_2, \dots} \left[G_1 \:\middle|\: S_0 =s \right] \\
&= r(s, \pi(s)) + \gamma \mathbb E_{S_1, S_2, \dots}
  \left[
    G_1 \:\middle|\: S_0 =s
  \right] \\
\end{align*}
$$

Using the law of total expectation and the markov properties (c.f. Appendix), we get

$$
\begin{align*}
v_{\pi}(s)
&=  r(s, \pi(s)) + \gamma \mathbb E_{S_1 \sim p(\cdot \mid s, \pi(s))}
\Bigg[
  \underbrace{\mathbb E_{S_2, S_3, \dots}
  \big[
      G_1 \mid S_1 =s_1
  \big]}_{v_{\pi}(s_1)}
\Bigg] \\
\end{align*}
$$

The underbraced term is the expected total reward by executing policy $\pi$ starting from state $s_1$ which is by definition exactly the state value of $s_1s_1$. Hence, we conclude. $\quad\square$

For finite state space, the expected future reward in Bellman equation can be expressed in a sum. The Bellman equations become

> $$
> \begin{align}
> v_{\pi}(s)
> = r(s, \pi(s)) + \gamma \sum_{s'} p(s' \mid s, \pi(s)) \cdot v_{\pi}(s'),
> \quad \forall s\in\mathcal S
> \end{align}
> $$

*Example*: We model the state of a stock market as $\mathcal S = \{\text{bull}, \text{bear}, \text{flat}\}$. The agent uses some investment policy $\pi$ (not detailed here) which yields the following state transition probability (Note that each row sums to 1).

| $p(s' \mid s, \pi(s))$ | $s'=$ bull | $s'=$ bear | $s'=$ flat |
| ---------------------- | ---------- | ---------- | ---------- |
| $s=$ bull              | 0.8        | 0.1        | 0.1        |
| $s=$ bear              | 0.1        | 0.7        | 0.2        |
| $s=$ flat              | 0          | 0.1        | 0.9        |

Suppose that immediate rewards under this policy are
$$
r(\text{bull}, \pi(\text{bull})) = 8, \quad
r(\text{bear}, \pi(\text{bear})) = -9, \quad
r(\text{flat}, \pi(\text{flat})) = 2,
$$

Then, the three Bellman equations are

$$
\begin{align*}
v_{\pi}(\text{bull})
&= 8 + \gamma
   \left[
     0.8 v_{\pi}(\text{bull}) + 0.1 v_{\pi}(\text{bear}) + 0.1 v_{\pi}(\text{flat})
   \right]
\\
v_{\pi}(\text{bear})
&= -9 + \gamma
   \left[
     0.1 v_{\pi}(\text{bull}) + 0.7 v_{\pi}(\text{bear}) + 0.2 v_{\pi}(\text{flat})
   \right]
\\
v_{\pi}(\text{flat})
&= 2 + \gamma
   \left[
     0 \cdot v_{\pi}(\text{bull}) + 0.1 v_{\pi}(\text{bear}) + 0.9 v_{\pi}(\text{flat})
   \right]
\\
\end{align*}
$$

Reformulation in vector form:

$$
\begin{bmatrix}
v_{\pi}(\text{bull}) \\ v_{\pi}(\text{bear})  \\ v_{\pi}(\text{flat})
\end {bmatrix}
=
\begin{bmatrix}
8 \\ -9 \\ 2
\end {bmatrix}
+
\gamma
\begin{bmatrix}
0.8 & 0.1 & 0.1 \\
0.1 & 0.7 & 0.2 \\
0   & 0.1 & 0.9
\end {bmatrix}

\begin{bmatrix}
v_{\pi}(\text{bull}) \\ v_{\pi}(\text{bear})  \\ v_{\pi}(\text{flat})
\end {bmatrix}
$$

In general, let $\mathcal S = \{ \varsigma_1, \dots, \varsigma_n \}$. (To avoid confusion, we do not use $\{s_1, \dots, s_n\}$ to denote $\mathcal S$ because the indices of $s$ represent time.) Then, we can write $n$ Bellman equations into the vector form
$$
\underbrace{
  \begin{bmatrix}
    v_{\pi}(\varsigma_1) \\ v_{\pi}(\varsigma_2) \\ \vdots \\ v_{\pi}(\varsigma_n)
  \end{bmatrix}
}_{\mathbf v_{\pi}}
=
\underbrace{
  \begin{bmatrix}
    r(\varsigma_1, \pi(\varsigma_1)) \\ r(\varsigma_2, \pi(\varsigma_2)) \\ \vdots \\ r(\varsigma_n, \pi(\varsigma_n))
  \end{bmatrix}
}_{\mathbf r_{\pi}}
+
\gamma
\underbrace{
  \begin{bmatrix}
    p(\varsigma_1 \mid \varsigma_1, \pi(\varsigma_1)) & \dots & p(\varsigma_n \mid \varsigma_1, \pi(\varsigma_1))  \\
    p(\varsigma_1 \mid \varsigma_2, \pi(\varsigma_1)) & \dots & p(\varsigma_n \mid \varsigma_2, \pi(\varsigma_1))  \\
    \vdots & \cdots & \vdots \\
    p(\varsigma_1 \mid \varsigma_n, \pi(\varsigma_1)) & \dots & p(\varsigma_n \mid \varsigma_n, \pi(\varsigma_1))
  \end{bmatrix}
}_{\mathbf P_{\pi}}
\cdot
\underbrace{
  \begin{bmatrix}
    v_{\pi}(\varsigma_1) \\ v_{\pi}(\varsigma_2) \\ \vdots \\ v_{\pi}(\varsigma_n)
  \end{bmatrix}
}_{\mathbf v_{\pi}}
$$

> Bellman equation (vector form)
> $$
> \begin{align}
> \mathbf v_{\pi} = \mathbf r_{\pi} + \gamma \mathbf P_{\pi}\mathbf v_{\pi}
> \end{align}
> $$

Remarks:

* $\mathbf v_{\pi}$ comprises state values for all $s\in\mathcal S$ under policy $\pi$
* $\mathbf r_{\pi}$ comprises immediate rewards arriving at $s\in\mathcal S$ under policy $\pi$
* $\mathbf P_{\pi}$ comprises all state transition probabilities under policy $\pi$
* All state values, immediate rewards and state transition probabilities depend on policy $\pi$.

Now, given the Bellman equations (either in element-wise or vector form), we ask two questions

1. Given the policy $\pi$, how to compute the state values? This is called ***policy evaluation***  
    $\to$ analytical solution or fixed point iteration
1. Is there a policy which maximizes the state values? If so, how to find it?  
    $\to$ dynamic programming

### Bellman Equation for Q-function

Similarly, Q-function also has recursive structure
$$
q_{\pi}(s,a) = r(s,a) + \gamma \mathbb E_{s' \sim p(\cdot \mid s, a)} [ q_{\pi}(s', \pi(s')) ]
$$

## Policy Evaluation

Given a policy $\pi$, computing its value function $v_{\pi}(\cdot)$ is called ***policy evaluation***. Effectively, we would like to evaluate how good $\pi$ is for each state $s$. This is equivelent to solving Bellman equations.

### Solution in Finite State Space

If $\mathcal S$ is fininte, policy evaluation boils down to solving a system of linear equations $\mathbf v_{\pi} = \mathbf r_{\pi} + \gamma \mathbf P_{\pi}\mathbf v_{\pi}$. It is easy to verify that the analytical solution to the Bellman equations is
$$
\begin{align}
\mathbf v_{\pi} = (\mathbf{I} - \gamma \mathbf{P}_{\pi})^{-1} \mathbf r_{\pi}
\end{align}
$$

where $\mathbf{I}$ is the $\vert \mathcal S \vert \times \vert \mathcal S \vert$ identity matrix.

Drawback of analytical solution: involves matrix inversion. High computational complexity when $\vert \mathcal S \vert$ is large.

Numerical solution (fixed point iteration): The state values can be obtained from the following algorithm
> Initialize $\mathbf v^{(0)}$ arbitrarily.  
> For $i=0,1,\dots$, run until convergence:
>
> $$
> \begin{align}
> \mathbf v^{(i+1)} = \mathbf r_{\pi} + \gamma \mathbf P_{\pi} \mathbf v^{(i)}
> \end{align}
> $$

Remarks:

* During iteration, $\mathbf v^{(i)}$ is not a true state value vector since $\mathbf v^{(i)}$ itself does not satisfy Bellman equation. There is no policy associated with $\mathbf v^{(i)}$.
* The sequence $\mathbf v^{(0)}, \mathbf v^{(1)}, \dots$ obtained from above iteration converges to $\mathbf v_{\pi}$, i.e.
  $$
  \begin{align}
  \lim_{i\to\infty} \mathbf v^{(i)}
  = \mathbf v_{\pi}
  = (\mathbf{I} - \gamma \mathbf{P}_{\pi})^{-1} \mathbf r_{\pi}
  \end{align}
  $$

*Proof of convergence*: Let $n=\vert \mathcal S \vert$. From Bellman equation, we know that $\mathbf v_{\pi}$ is a fixed point of the affine function
$$
f: \mathbb R^n \to \mathbb R^n,
\mathbf v \mapsto
f(\mathbf v) = \mathbf r_{\pi} + \gamma \mathbf P_{\pi} \mathbf v
$$
We show that $f(\cdot)$ is a contractive mapping under infinity norm.
$$
\begin{align*}
\Vert f(\mathbf u) - f(\mathbf v) \Vert_{\infty}
&= \left\Vert (\mathbf r_{\pi} + \gamma \mathbf P_{\pi} \mathbf u) - (\mathbf r_{\pi} + \gamma \mathbf P_{\pi} \mathbf v) \right\Vert_{\infty}
\\
&= \gamma \left\Vert  \mathbf P_{\pi} (\mathbf u - \mathbf v) \right\Vert_{\infty}
\\
&\le \gamma \left\Vert  (\mathbf u - \mathbf v) \right\Vert_{\infty},
\end{align*}
$$

The last step follows from the fact that $\Vert \mathbf P_{\pi} \mathbf x \Vert_{\infty} \le \Vert \mathbf x \Vert_{\infty}, \forall x\in\mathbb R^n$, i.e. multiplication with row stochastic matrix does not increase infnity norm. (c.f. Appendix).

By contraction mapping theorem (c.f. separate notes), we conclude that

1. $f(\cdot)$ has a unique fixed point. Since $\mathbf v_{\pi}  = f(\mathbf v_{\pi} )$ by Bellman equation, $\mathbf v_{\pi}$ is the unique fixed point.
2. $\forall \mathbf v^{(0)} \in\mathbb R^n$, the sequence defined by $\mathbf v^{(i+1)} = f(\mathbf v^{(i)})$ converges to $\mathbf v_{\pi} $ in infnity norm

Since all $p$-norms in $\mathbb R^n$ are equivalent, convergence in infnity norm implies convergence in any $p$-norm. $\:\square$

### Bellman Operator

If $\mathcal S$ is a infinite set, there is generally no closed-form solution for $v_{\pi}(s)$ expect for a few special cases (not covered here). Here, we only show a theoretical study based on Bellman operator.

Let $\mathcal V$ be the set of all **bounded** value functions. Then, $\mathcal V$ with the sup norm $\Vert\cdot\Vert_\infty$ is a metric space
$$
\mathcal V =
\left\{
v:\mathcal S\to\mathbb R \:\bigg|\: \Vert v \Vert_\infty = \max_{s\in\mathcal S} \vert v(s) \vert < \infty
\right\}
$$

For a certain policy $\pi$, we define the corresponding Bellman operator $\mathcal B_{\pi}$ which maps a state value function $v(\cdot)$ to another value function $\mathcal B_{\pi}v(\cdot)$.
> $$
> \mathcal B_{\pi}: \mathcal V \to \mathcal V, v(\cdot) \mapsto \mathcal B_{\pi}v(\cdot)
> $$

The resulting value function is
> $$
> \mathcal B_{\pi} v(s) =
> r(s, \pi(s)) + \gamma\mathbb E_{s' \sim p(\cdot \mid s, \pi(s))} [ v(s') ]
> $$

Properties of Bellman operator:

> 1. $\mathcal B_{\pi}$  is monotonic, i.e.
>    $$
>    u(s) \le v(s), \forall s\in\mathcal S \implies
>    \mathcal B_{\pi}u(s) \le \mathcal B_{\pi}v(s), \forall s\in\mathcal S
>    $$
> 1. $\mathcal B_{\pi}$  is a contractive mapping. i.e. 
>    $$
>    \forall u, v: \mathcal S \to \mathbb R,\:
>    \Vert \mathcal B_{\pi}u - \mathcal B_{\pi}v \Vert_\infty \le \Vert u-v\Vert_\infty
>    $$
> 1. $v_{\pi}(\cdot)$ is the unique fixed point of $\mathcal B_{\pi}$, i.e.
>    $$
>    \mathcal B_{\pi} v_{\pi}(s) = v_{\pi}(s), \forall s\in\mathcal S
>    $$

*Proof 1*: By the monotonicity of expectation
$\mathbb E_{s' \sim p(\cdot \mid s, \pi(s))} [ u(s') ] \le \mathbb E_{s' \sim p(\cdot \mid s, \pi(s))} [ v(s') ]$, we conclude. $\quad\square$

*Proof 2*: Recall that the infinity norm of a function $f$ is defined as
$$
\Vert f \Vert_\infty \triangleq \max_x \vert f(x) \vert
$$

Consider $\vert \mathcal B_{\pi}u(s) - \mathcal B_{\pi}v(s) \vert$ for all $s\in\mathcal S$.
$$
\begin{align*}
\vert \mathcal B_{\pi}u(s) - \mathcal B_{\pi}v(s) \vert
&=\Big\vert
    r(s, \pi(s)) + \gamma\mathbb E_{s' \sim p(\cdot \mid s, \pi(s))} [ u(s') ] -
    r(s, \pi(s)) - \gamma\mathbb E_{s' \sim p(\cdot \mid s, \pi(s))} [ v(s') ]
  \Big\vert\\
&=\vert\gamma\vert \cdot
  \Big\vert
    \mathbb E_{s' \sim p(\cdot \mid s, \pi(s))} [ u(s') -v(s') ]
  \Big\vert\\
&\le \gamma \cdot
    \mathbb E_{s' \sim p(\cdot \mid s, \pi(s))} \big[\vert u(s') -v(s') \vert\big]\\
&\le \gamma \cdot
    \mathbb E_{s' \sim p(\cdot \mid s, \pi(s))} \big[\Vert u-v\Vert_\infty\big]\\
&\le \gamma \cdot \Vert u-v\Vert_\infty
\end{align*}
$$

Hence, we conclude
$$
\Vert \mathcal B_{\pi}u - \mathcal B_{\pi}v \Vert_\infty
=\displaystyle\max_{s\in\mathcal S} \vert \mathcal B_{\pi}u(s) - \mathcal B_{\pi}v(s) \vert
\le \gamma \cdot \Vert u-v\Vert_\infty
$$

*Proof 3*: By Bellman equation, we know that $v_{\pi}$ is a fixed point of $\mathcal B_{\pi}$. By contraction mapping theorem, we conclude that $v_{\pi}$ is the unique fixed point.

Followed by contraction mapping theorem, the state value function can be obtained through fixed point iteration

> Starting from any $v^{(0)}(\cdot)\in\mathcal V$  
> For $i=0,1,\dots$, run until $v^{(i)}(\cdot)$ converges  
> $\quad$ For each $s\in\mathcal S$, do  
> $$
> v^{(i+1)}(s) = \mathcal B_{\pi} v(s) =
> r(s, \pi(s)) + \gamma\mathbb E_{s' \sim p(\cdot \mid s, \pi(s))} [ v(s') ]
> $$

Mathematically, the resulting function sequence $\{ v^{(i)} \}_{i\ge0}$ converges to $v_{\pi}$ in the sup norm (and thus converges pointwise)
$$
\lim_{i\to\infty} \big\Vert v^{(i)} - v_{\pi} \big\Vert_\infty = 0
\implies
\lim_{i\to\infty} v^{(i)}(s) = v_{\pi}(s), \forall s\in\mathcal S
$$

However, above algorithm can not be directly implementend since we can not evaluate $v^{(i)}(s)$ for infinitely many $s$. In practice, we use approximation techniques to estimate $v_{\pi}(\cdot)$. (Not detailed here.)

## Bellman Optimality Equations

A policy $\pi$ outperforms another policy $\tilde\pi$ iff the state values $v_{\pi}(s)$ outperforms $v_{\tilde\pi}(s)$ for **ALL** states $s\in\mathcal S$. i.e.

$$
\forall s \in\mathcal S, \, v_{\pi}(s) \ge v_{\tilde\pi}(s)
$$

A policy $\pi^*$ is optimal if it outperforms any other policies, i.e.

$$
\forall \pi, \forall s \in\mathcal S, \, v_{\pi^*}(s) \ge v_{\pi}(s)
$$

The optimal state value $v^*(s)$ is the state value under $\pi^*$

$$
\begin{align}
v^*(s) \triangleq v_{\pi^*}(s) = \max_{\pi} v_{\pi}(s)
\end{align}
$$

We haven't proved the existence of $\pi^*$. For now, let's assume its existence and discover what conditions have to be met for $\pi^*$ and $v^*(s)$. This will lead us to Bellman optimality equations,  from which we will derive an algorithm to find $\pi^*$ (and thus prove its existence).

### Bellman Optimality Criterion

Recall the Bellman equation for $v_{\pi}(s)$ holds for any policy. In particular, Bellman equations also hold for $\pi^*$:

$$
\begin{align*}
v^*(s) = r(s, \pi^*(s)) + \gamma \mathbb E_{s' \sim p(\cdot \mid s, \pi^*(s))} [ v^*(s') ], \quad \forall s\in\mathcal S
\end{align*}
$$

Yet, we are unable to solve the optimal state value since the optimal current action $\pi^*(s)$ is unknown. However, we can reformulate $v^*(s)$ without explicit reference to $\pi^*(s)$.

**Trick**: Note that $\pi^*(s)$ is the best action to take on current state $s$. $\implies$ Taking $\pi^*(s)$ now followed by executing $\pi^*$ yields a state value no less than taking any other $a\in\mathcal A$ followed by exectuting $\pi^*$, as illustrated below.

* Optimal: Executing the optimal policy from now onward.
  $$
  \begin{align*}
  s   \xrightarrow[r(s,\, \pi^*(s) )]{ \pi^*(s) \:}
  S_1 \xrightarrow[R_1]{\pi^*(S_1) \:}
  S_2 \xrightarrow[R_2]{\pi^*(S_2) \:}
  S_3 \cdots
  \end{align*}
  $$

* (Sub)optimal: Taking any other $a \in\mathcal A\setminus \{\pi^*(s)\}$ now, followed by executing $\pi^*$ onward.
  $$
  \begin{align*}
  s   \xrightarrow[r(s,\, a)]{a\:}
  S_1 \xrightarrow[R_1]{\pi^*(S_1) \:}
  S_2 \xrightarrow[R_2]{\pi^*(S_2) \:}
  S_3 \cdots
  \end{align*}
  $$

Formally, this means
$$
\forall s \in\mathcal S,
\forall a \in\mathcal A:
\:
v^*(s) \ge r(s, a) + \gamma \mathbb E_{s' \sim p(\cdot \mid s, a)} [ v^*(s') ]
$$
where the equality holds iff $a=\pi^*(s)$. Namely, the optimal action at state $s$ should maximize the sum of immediate reward and the (discounted) expected future reward
> $$
> \begin{align}
> \pi^*(s) = \argmax_{a\in\mathcal A}
> \left\{
> r(s, a) + \gamma \mathbb E_{s' \sim p(\cdot \mid s, a)} [ v^*(s') ]
>   \right\},
> \quad \forall s\in\mathcal S
> \end{align}
> $$

Remarks:

* This equation holds for all $s\in\mathcal S$.
* Suppose we solved all optimal state values $\{ v^*(s') \mid s'\in\mathcal S \}$, plugging them into this equation yields the optimal policy.

The optimal state value $v^*(s)$ thus satisfies the ***Bellman optimality equations (BOE)***:

> $$
> \begin{align}
> v^*(s) = \max_{a\in\mathcal A}
> \left\{
>   r(s, a) + \gamma \mathbb E_{s' \sim p(\cdot \mid s, a)} [ v^*(s') ]
> \right\},
> \quad \forall s\in\mathcal S
> \end{align}
> $$

Remarks:

* Previously, computing $v^*(s)$ requires knowledge of $r(s, \pi^*(s))$ and $p(\cdot \mid s, \pi^*(s))$, but since $\pi^*(s)$ is unknown, solving $v^*(s)$ is challenging.
* Now with the BOE, we bypass the need to know $\pi^*(s)$ explicitly. Instead, we evaluate  $r(s, a)$ and $p(\cdot \mid s, a)$ (which are provided by MDP) for all $a\in\mathcal A$. Then, $v^*(s)$ can be solved by solving the optimization problem (detailed later).
* Just like Bellman equation, BOE holds for all $s\in\mathcal S$.

For finite state space, the BOE becomes
> $$
> \begin{align}
> v^*(s) = \max_{a\in\mathcal A}
> \left\{
>   r(s, a) + \gamma \sum_{s'} p(s' \mid s, a) \cdot v^*(s')
> \right\},
> \quad \forall s\in\mathcal S
> \end{align}
> $$

Again, let $\mathcal S = \{ \varsigma_1, \dots, \varsigma_n \}$. Then, we can write $n$ BOEs into vector form
$$
\begin{align*}
\begin{bmatrix}
v^*(\varsigma_1) \\ v^*(\varsigma_2) \\ \vdots \\ v^*(\varsigma_n)
\end{bmatrix}
&=
\begin{bmatrix}
 \max_{a\in\mathcal A} \left\{
        r(\varsigma_1, a) + \gamma \left[
      p(\varsigma_1 \mid \varsigma_1, a) v^*(\varsigma_1) + \dots +
      p(\varsigma_n \mid \varsigma_1, a) + v^*(\varsigma_n)
     \right]
 \right\}  \\
 \max_{a\in\mathcal A} \left\{
        r(\varsigma_2, a) + \gamma \left[
      p(\varsigma_1 \mid \varsigma_2, a) v^*(\varsigma_1) + \dots +
      p(\varsigma_n \mid \varsigma_2, a) + v^*(\varsigma_n)
     \right]
 \right\}  \\
\vdots \\
 \max_{a\in\mathcal A} \left\{
        r(\varsigma_n, a) + \gamma \left[
      p(\varsigma_1 \mid \varsigma_n, a) v^*(\varsigma_1) + \dots +
      p(\varsigma_n \mid \varsigma_n, a) + v^*(\varsigma_n)
     \right]
 \right\}
\end{bmatrix}
%%%%%%%%%%%%%%%%%%%%%%%%
\\[12pt]
%%%%%%%%%%%%%%%%%%%%%%%%
\underbrace{
  \begin{bmatrix}
    v^*(\varsigma_1) \\ v^*(\varsigma_2) \\ \vdots \\ v^*(\varsigma_n)
  \end{bmatrix}
}_{\mathbf v^*}
&=
\max_{a\in\mathcal A}
\left\{
    \underbrace{
      \begin{bmatrix}
        r(\varsigma_1, a) \\ r(\varsigma_2, a) \\ \vdots \\ r(\varsigma_n, a)
      \end{bmatrix}
    }_{\mathbf r_a}
    +
    \gamma
    \underbrace{
      \begin{bmatrix}
        p(\varsigma_1 \mid \varsigma_1, a) & \dots & p(\varsigma_n \mid \varsigma_1, a)  \\
        p(\varsigma_1 \mid \varsigma_2, a) & \dots & p(\varsigma_n \mid \varsigma_2, a)  \\
        \vdots & \cdots & \vdots \\
        p(\varsigma_1 \mid \varsigma_n, a) & \dots & p(\varsigma_n \mid \varsigma_n, a)
      \end{bmatrix}
    }_{\mathbf P_a}
    \cdot
    \underbrace{
      \begin{bmatrix}
        v^*(\varsigma_1) \\ v^*(\varsigma_2) \\ \vdots \\ v^*(\varsigma_n)
      \end{bmatrix}
    }_{\mathbf v^*}
\right\}
\end{align*}
$$

> Bellman Optimality Equation (vector form)
> $$
> \begin{align}
> \mathbf v^*
> = \max_{a\in\mathcal A} \left\{\mathbf r_a + \gamma \mathbf P_a\mathbf v^* \right\}
> \end{align}
> $$

where $\max$ acting on a vector is taken element-wise.
$$
\displaystyle\max_{x} \mathbf w(x)
=
\begin{bmatrix}
\max_{x} w_1(x) & \cdots \displaystyle & \max_{x} w_n(x)
\end{bmatrix}^\top
$$

### Solving Optimal State Values

Similar to the algorithm to solve the state values $\mathbf v_{\pi}$ for any given policy, we introduce the following algorithm to solve the optimal state values $\mathbf v^*$.

> Init $\mathbf v^{(0)}$ arbitrarily  
> For $i=0,1,\dots$, run until convergence
> $$
> \begin{align}
> \mathbf v^{(i+1)}
> = \max_{a\in\mathcal A} \left\{\mathbf r_a + \gamma \mathbf P_a\mathbf v^{(i)} \right\}
> \end{align}
> $$

Remarks:

* The sequence $\mathbf v^{(0)}, \mathbf v^{(1)}, \dots$ obtained from above iteration converges to $\mathbf v^*$, i.e.
  $$
  \begin{align}
  \lim_{i\to\infty} \mathbf v^{(i)} = \mathbf v^*
  \end{align}
  $$
* Having computed the optimal state values $\mathbf v^*$, the optimal policy is obtained from
  > $$
  > \begin{align}
  > \pi^*(s) = \argmax_{a\in\mathcal A}
  > \left\{
  >   r(s, a) + \gamma \sum_{s'} p(s' \mid s, a) \cdot v^*(s')
  > \right\},
  > \quad \forall s\in\mathcal S
  > \end{align}
  > $$
* Reminder: The iteration does **not** ensure that the intermediate result $\mathbf v^{(i)}$ satisfy Bellman equation **for any policy**. However, the limit of $\mathbf v^{(i)}$ satisfies BOEs, i.e. the Bellman equation for the optimal policy.

*Proof of convergence*: In the following, all $\max(\cdot)$, $\vert\cdot\vert$ and inequalities are taken element-wise when acting on vectors. Let $\mathcal S = \{ \varsigma_1, \dots, \varsigma_n \}$ and
$$
f: \mathbb R^n \to \mathbb R^n, \mathbf v \mapsto
f(\mathbf v) = \displaystyle\max_{a\in\mathcal A} \left\{\mathbf r_a + \gamma \mathbf P_a\mathbf v \right\}
$$

By BOE, $\mathbf v^*$ is a fixed point of $f(\cdot)$. To prove the convergence, it is sufficient to show that $f(\cdot)$ is contractive.  
For any $\mathbf u, \mathbf v\in\mathbb R^n$, we have two optimization problems w.r.t. $a$. (The maximization is taken element-wise)
$$
\begin{align*}
f(\mathbf u)
&= \displaystyle\max_{a\in\mathcal A} \left\{\mathbf r_a + \gamma \mathbf P_a\mathbf u \right\}
\\
f(\mathbf v)
&= \displaystyle\max_{a\in\mathcal A} \left\{\mathbf r_a + \gamma \mathbf P_a\mathbf v \right\}
\end{align*}
$$
For $f(\mathbf u)$, let $\hat a_k$ be the optimizer at $k$-row of $\mathbf r_a + \gamma \mathbf P_a\mathbf u $. (Note: $\hat a_k$ depends on $\mathbf u$)
$$
\hat a_k = \argmax_{a\in\mathcal A}
\Big\{
r(\varsigma_k, a) + \gamma \sum_{j} p(\varsigma_j \mid \varsigma_k, a) \cdot u_j
\Big\}
$$
Then, we can express $f(\mathbf u)$ in with $\mathbf r_{\hat a}$ and $\mathbf P_{\hat a}$, defined as follows.
$$
\mathbf r_{\hat a} \triangleq
\begin{bmatrix}
  r(\varsigma_1, \hat a_1) \\
  \vdots \\
  r(\varsigma_n, \hat a_n)
\end{bmatrix}
,\:
\mathbf P_{\hat a} \triangleq
\begin{bmatrix}
p(\varsigma_1 \mid \varsigma_1, \hat a_1) & \dots & p(\varsigma_n \mid \varsigma_1, \hat a_1)  \\
\vdots & \ddots & \vdots \\
p(\varsigma_1 \mid \varsigma_n, \hat a_n) & \dots & p(\varsigma_n \mid \varsigma_n, \hat a_n)
\end{bmatrix}
\implies
f(\mathbf u) = \mathbf r_{\hat a} + \gamma \mathbf P_{\hat a}\mathbf u
$$
Likewise, for $f(\mathbf v)$, let $\hat b_k$ be the optimizer at $k$-row of $\mathbf r_a + \gamma \mathbf P_a\mathbf v $. (Note: $\hat b_k$ depends on $\mathbf v$. Hence, $\hat a_k \ne \hat b_k$ in general)
$$
\hat b_k = \argmax_{a\in\mathcal A}
\Big\{
r(\varsigma_k, a) + \gamma \sum_{j} p(\varsigma_j \mid \varsigma_k, a) \cdot v_j
\Big\}
$$
Define $\mathbf r_{\hat b}$ and $\mathbf P_{\hat b}$ in the same way. $\implies f(\mathbf v) = \mathbf r_{\hat b}+ \gamma \mathbf P_{\hat b}\mathbf v$.

By the optimality of $\{\hat a_1, \dots, \hat a_n\}$ and $\{\hat b_1, \dots, \hat b_n\}$,
$$
\begin{align*}
f(\mathbf u)
&= \mathbf r_{\hat a} + \gamma \mathbf P_{\hat a}\mathbf u
\ge \mathbf r_{\hat b} + \gamma \mathbf P_{\hat b}\mathbf u
\\
f(\mathbf v)
&= \mathbf r_{\hat b}+ \gamma \mathbf P_{\hat b}\mathbf v
\ge \mathbf r_{\hat a}+ \gamma \mathbf P_{\hat a}\mathbf v
\\
\end{align*}
$$
Hence, $f(\mathbf u) - f(\mathbf v)$ is element-wise bounded as follows
$$
\begin{align*}
f(\mathbf u) - f(\mathbf v)
&= (\mathbf r_{\hat a} + \gamma \mathbf P_{\hat a}\mathbf u) -
   (\mathbf r_{\hat b}+ \gamma \mathbf P_{\hat b}\mathbf v) \\
&\ge(\mathbf r_{\hat b} + \gamma \mathbf P_{\hat b}\mathbf u) -
    (\mathbf r_{\hat b}+ \gamma \mathbf P_{\hat b}\mathbf v)
= \gamma \mathbf P_{\hat b}(\mathbf u - \mathbf v)
\\[6pt]
f(\mathbf u) - f(\mathbf v)
&= (\mathbf r_{\hat a} + \gamma \mathbf P_{\hat a}\mathbf u) -
   (\mathbf r_{\hat b}+ \gamma \mathbf P_{\hat b}\mathbf v) \\
&\le(\mathbf r_{\hat a} + \gamma \mathbf P_{\hat a}\mathbf u) -
    (\mathbf r_{\hat a}+ \gamma \mathbf P_{\hat a}\mathbf v)
= \gamma \mathbf P_{\hat a}(\mathbf u - \mathbf v)
\\[6pt]
\implies
\gamma \mathbf P_{\hat b}(\mathbf u - \mathbf v)
\le
f(\mathbf u) &- f(\mathbf v)
\le \gamma \mathbf P_{\hat a}(\mathbf u - \mathbf v)
\end{align*}
$$
Taking the absolute values of $f(\mathbf u) - f(\mathbf v)$ element-wise yields
$$
\begin{align*}
\left\vert f(\mathbf u) - f(\mathbf v) \right\vert
\le \max
\left\{
  \gamma \big\vert \mathbf P_{\hat b}(\mathbf u - \mathbf v) \big\vert ,
  \gamma \big\vert \mathbf P_{\hat a}(\mathbf u - \mathbf v) \big\vert
\right\}

&\le \gamma \Vert \mathbf u - \mathbf v \Vert_{\infty} \cdot \mathbf 1
\end{align*}
$$
* ⓘ The last inequality follows from the property of row-stochastic matrix (c.f. Appendix): If $\mathbf P\in\mathbb R^{n\times n}$ is a row-stochastic matrix, then
$$
  \forall \mathbf x\in\mathbb R^n:
  \big\vert(\mathbf{Px})_{i}\big\vert \le \Vert \mathbf x \Vert_{\infty}
  \iff
  \big\vert(\mathbf{Px})\big\vert \le \Vert \mathbf x \Vert_{\infty}\cdot\mathbf 1
$$
Namely, all elements of $\left\vert f(\mathbf u) - f(\mathbf v) \right\vert$ is boudned by $\gamma \Vert \mathbf u - \mathbf v \Vert_{\infty}$. Hence,
$$
\left\Vert f(\mathbf u) - f(\mathbf v) \right\Vert_{\infty}
\le \gamma \Vert \mathbf u - \mathbf v \Vert_{\infty}
\iff
f(\cdot) \text{ is contractive }\quad\square
$$
*Proof of optimal policy*: When deriving BOEs, we showed that
$$
\begin{align*}
\pi^*(s) = \argmax_{a\in\mathcal A}
\left\{
  r(s, a) + \gamma \mathbb E_{s' \sim p(\cdot \mid s, a)} [ v^*(s') ]
\right\},
\quad \forall s\in\mathcal S
\end{align*}
$$
Expanding the expectation into a sum, we conclude. $\quad\square$

### Optimal Q-function

Similary, the optimal Q-function is defined as
$$
q^*(s,a) = q_{\pi^*}(s,a)
$$

Relation between optimal Q-function and optimal state value:

* Compute $v^*(s)$ from $q^*(s,a)$: simply let $a=\pi^*(s)$, i.e.
  $$
  \begin{align}
  v^*(s) = q^*(s,a) \Big|_{a=\pi^*(s)}
  \end{align}
  $$

* Compute $q^*(s,a)$ from $v^*(s)$: use recurisve structure
  $$
  \begin{align}
  q^*(s,a)
  &= r(s,a) + \gamma \mathbb E_{s' \sim p(\cdot \mid s, a)} [ v^*(s') ] \\
  &= r(s,a) + \gamma \sum_{s'\in\mathcal S} p(s' \mid s, a) v^*(s') & \text{if } \mathcal S \text{ is finite}\\
  \end{align}
  $$

The Bellman optimality criterion can also be formulated in terms of Q-function:
$$
\begin{align}
v^*(s) &= \max_{a\in\mathcal A} q^*(s,a) \\
\pi^*(s) &= \argmax_{a\in\mathcal A}\, q^*(s,a) 
\end{align}
$$

## Dynamic Programming

Throughout this section, we assume that both state space and action space are discrete. Given the parameters of an MDP:

* state transition probabilities $p(s' \mid s,a)$ for all $s,s'\in\mathcal S, a\in\mathcal A $
* state-action-rewards $r(s,a)$ for all $s\in\mathcal S, a\in\mathcal A $

From previous sections, we knew that an optimal policy  $\pi^*$ exists. Now, we focus on designing algorithms to compute $\pi^*$.

### Value Iteration

The algorithm introduced earlier to solve BOEs is called value iteration. For the sake of implementation, the algorithm can be unfolded element-wise as follows

> Init $v^{(0)}(s)$ for all $s\in\mathcal S$ by random guessing  
> For $i = 0,1,\dots$, do  
> $\quad$ For each $s\in\mathcal S$, do  
> $\quad\quad$ For each $a\in\mathcal A$, compute Q-function  
> $\quad\quad\qquad q^{(i)}(s,a) = r(s, a) + \gamma \displaystyle\sum_{s'} p(s' \mid s, a) \cdot v^{(i)}(s')$  
> $\quad\quad$ **Policy update**: $\pi^{(i+1)}(s) = \displaystyle\argmax_{a\in\mathcal A}\, q^{(i)}(s,a)$  
> $\quad\quad$ **Value update**: $v^{(i+1)}(s) = \displaystyle\max_{a\in\mathcal A} q^{(i)}(s,a)$  
> until $\big\Vert \mathbf v^{(i+1)} - \mathbf v^{(i)} \big\Vert \le$ some threshold $\epsilon$. (i.e. $\mathbf v^{(i)}$ converges)  
> Return $v^{(i_\text{stop})}(s)$ and $\pi^{(i_\text{stop})}(s)$ for all $s\in\mathcal S$

Remarks:

* In stopping condition, $\mathbf v^{(i)}$ is the vector containing all $v^{(i)}(s), s\in\mathcal S$. The norm can be infinity norm or any other norms.
* Although $v^{(i)}(s)$ converges to $v^{*}(s)$ for all $s\in\mathcal S$, the intermediate values $v^{(i)}(s)$ do **not** generally satisfy Bellman equation for **any** policy. We interpret $v^{(i)}(s)$ as the estimate of $v^*(s)$ at $i$-th iteration rather than the state values under $\pi^{(i)}$ or $\pi^{(i+1)}$, i.e.
  $$
    \begin{align*}
    v^{(i)}(s)
    &\ne r(s, \pi^{(i)}(s)) + \gamma \sum_{s'} p(s'\mid s, \pi^{(i)}(s)) v^{(i)}(s)
    \\
    v^{(i)}(s)
    &\ne r(s, \pi^{(i+1)}(s)) + \gamma \sum_{s'} p(s'\mid s, \pi^{(i+1)}(s)) v^{(i)}(s)
    \end{align*}
  $$
* Likewise, $q^{(i)}(s,a)$ represents the estimate of $q^{*}(s,a)$ at $i$-th iteration.

In policy update step, the new policy $\pi^{(i+1)}(s)$ always picks the action maximizing the current estimtae of Q-function $q^{(i)}(s,a)$. Hence, it is called ***greedy*** policy update.

### Policy Iteration

Policy iteration is another algorithm to compute optimal policy. It starts with abitrary policy and iteratively improves it. Formally:

> Init $\pi^{(0)}$ by random guessing  
> For $i=0,1,2,\dots$, run until convergence  
> $\quad$ **Policy evaluation**: Solve $\mathbf v_{\pi^{(i)}} = \mathbf r_{\pi^{(i)}} + \gamma\mathbf P_{\pi^{(i)}} \mathbf v_{\pi^{(i)}}$ for state values $\mathbf v_{\pi^{(i)}}$  
> $\quad$ **Policy improvement**: Solve $\pi^{(i+1)} = \displaystyle\argmax_{\pi}  \big\{ \mathbf r_{\pi} + \gamma\mathbf P_{\pi} \mathbf v_{\pi^{(i)}} \big\}$ for new policy $\pi^{(i+1)}$

Remark:

* The convergence will be proved later.
* In policy evaluation, the state values can be computed either analytically using
$\mathbf v_{\pi} = (\mathbf{I} - \gamma \mathbf{P}_{\pi})^{-1} \mathbf r_{\pi}$
or numerically using fixed point iteration.
* In policy improvement, the new policy maximizes the immediate reward plus the discounted future reward by following $\pi^{(i)}$. Element-wise, this breaks down to maximizing the Q-functions:
    $$
    \begin{align}
    \pi^{(i+1)}(s)
    &= \argmax_{a\in\mathcal A}  \left\{ r(s,a) + \gamma\sum_{s'\in\mathcal S} p(s'\mid s,a) \, v_{\pi^{(i)}}(s') \right\}, \quad \forall s\in\mathcal S \\
    &= \argmax_{a\in\mathcal A} q_{\pi^{(i)}} (s,a)
    \end{align}
    $$
* Each intermediate $\mathbf v_{\pi^{(i)}}$ satisfies the Bellman equation for policy $\pi^{(i)}$. vs. In value iteration, $\mathbf v^{(i)}$ does not generally satisfy Bellman equation for any policy!

Element-wise formulation of policy iteration:

> Init $\pi^{(0)}(s)$ for all $s\in\mathcal S$ by random guessing  
> For $i=0,1,2,\dots$, do  
> $\quad$ **Policy evaluation**: compute $v_{\pi^{(i)}}(s)$ for all $s\in\mathcal S$ by solving linear equations  
> $\quad$ For each $s\in\mathcal S$, do  
> $\quad\quad\;$ For each $a\in\mathcal A$, compute  
> $\quad\quad\qquad$ Q-function: $q_{\pi^{(i)}} (s,a) = r(s,a) + \gamma\displaystyle\sum_{s'\in\mathcal S} p(s'\mid s,a) \, v_{\pi^{(i)}}(s')$  
> $\quad\quad\;$ **Policy improvement**: $\pi^{(i+1)}(s) = \displaystyle\argmax_{a\in\mathcal A} \: q_{\pi^{(i)}} (s,a)$  
> until $\Vert\mathbf v_{\pi^{(i+1)}} - \mathbf v_{\pi^{(i)}} \Vert < \epsilon$  
> Return $v_{\pi^{(i)}}(s)$ and $\pi^{(i)}(s)$ for all $s\in\mathcal S$

Policy iteratoin works because of following facts

1. Policy improvement theorem: the policy is indeed improved iteratively.
    $$
    \pi^{(i+1)}(s) = \displaystyle\argmax_{a\in\mathcal A} q_{\pi^{(i)}} (s,a)
    \implies
    v_{\pi^{(i+1)}}(s) \ge v_{\pi^{(i)}}(s),\: \forall s\in\mathcal S
    $$
1. The sequence of state values generated by policy iteration indeed converges to the optimal state values.
    $$
    \lim_{i\to\infty} v_{\pi^{(i)}}(s) = v^*(s),\: \forall s\in\mathcal S
    $$

*Proof 1*: For each $s\in\mathcal S$, $\pi^{(i+1)}(s)$ is the maximizer of the Q-function $q_{\pi^{(i)}} (s,a)$. In particular,
$$
q_{\pi^{(i)}} (s, \pi^{(i+1)}(s))
\ge q_{\pi^{(i)}} (s, \pi^{(i)}(s)) = v_{\pi^{(i)}}(s)
\tag{$\star$}
$$

Hence, $\forall s\in\mathcal S$:

$$
\begin{align*}
v_{\pi^{(i+1)}}(s) - v_{\pi^{(i)}}(s)
&\overset{(\star)}{\ge} v_{\pi^{(i+1)}}(s)  - q_{\pi^{(i)}} (s, \pi^{(i+1)}(s))
\\
&= r(s, \pi^{(i+1)}(s)) + \gamma\mathbb E[v_{\pi^{(i+1)}}(S')] -
\left( r(s, \pi^{(i+1)}(s)) + \gamma\mathbb E[v_{\pi^{(i)}}(S')] \right)
\\
&= \gamma\mathbb E\left[ v_{\pi^{(i+1)}}(S') - v_{\pi^{(i)}}(S') \right]
\\
&\ge \gamma \min_{s'\in\mathcal S} \left\{ v_{\pi^{(i+1)}}(s') - v_{\pi^{(i)}}(s') \right\}
\tag{$\star\star$}
\end{align*}
$$

The last step follows by the property of expectation: $\mathbb E[X] \ge x_{\min}$ with $X=v_{\pi^{(i+1)}}(S') - v_{\pi^{(i)}}(S')$.

Taking $\displaystyle\min_{s\in\mathcal S}$ on the LHS, we get
$$
\min_{s\in\mathcal S} \left\{ v_{\pi^{(i+1)}}(s) - v_{\pi^{(i)}}(s) \right\}
\ge \gamma \min_{s'\in\mathcal S} \left\{ v_{\pi^{(i+1)}}(s') - v_{\pi^{(i)}}(s') \right\}
$$

Note that the optimization problem on both sides are the same. Hence,
$$
\begin{align*}
(1-\gamma) \min_{s\in\mathcal S} \left\{ v_{\pi^{(i+1)}}(s) - v_{\pi^{(i)}}(s) \right\} &\ge 0
\\
\min_{s\in\mathcal S} \left\{ v_{\pi^{(i+1)}}(s) - v_{\pi^{(i)}}(s) \right\} &\ge 0
&& \text{since } 0<\gamma<1 \\
\forall s\in\mathcal S, \: v_{\pi^{(i+1)}}(s) - v_{\pi^{(i)}}(s) &\ge 0
&& \text{by } (\star\star)
\end{align*}
$$

We concluded that the policy improvement does not decrease state value. $\qquad \square$

### Generalized Policy Iteration

## Generalization

Stochastic policy and stochastic rewards.

A policy can also be stochastic. i.e. instead of mapping the state to a fixed action, there are multiple possible actions with different probability. A stochastic policy is described by the conditional distribution

$$
\pi(a \mid s), a \in\mathcal A, s \in\mathcal S
$$

Remarks:

* The determinstic policy can be seen as a special of stochastic policy by assigning $\pi(\hat a \mid s)$ to 1 for some $\hat a$.
  $$
  \pi(a \mid s) =
  \begin{cases}
    1 & a=\hat a\\
    0 & \text{else}
  \end{cases}
  $$
* For stochastic policy, it holds that
  $$
  \sum_a \pi(a \mid s) = 1
  $$

* Both deterministic and stochastic policy are time-invariant. i.e. The distribution of $a$ given $s$ is always the same, regardless when we arrived at $s$.

From now on, we stick to deterministic policy and we will answer the following quesitons

1. How to quantify the goodness of a policy?  
    $\to$ state values, Bellman equations
1. Which criterion should the optimal policy satisfy?  
    $\to$ Bellman optimality equations.
1. How to find the optimal policy?  
    $\to$ value iteration, policy iteration

## Appendix

### Cascade of Expectations

Suppose $U-X-Y$ forms a markov chain, then
$$
\begin{align*}
\mathbb E_{XY}[g(x, y) \mid u]
&\triangleq \mathbb E_{XY \sim p(x,y \mid u)}[g(x, y)] \\
&= \mathbb E_{X \sim p(x\mid u)}
\left[
  \mathbb E_{Y \sim p(y \mid x)}[g(x,y)]
\right]
\end{align*}
$$

### Row Stochastic Matrix

A square matrix $\mathbf A \in\mathbb R^{n \times n}$ is called a ***row stochastic matrix*** iff each row of $\mathbf A$ is a probability vector. i.e.

* all its entires are nonnegative: $a_{ij} \ge 0, \forall i,j\in\{1,\dots,n\}$
* and each row sums to 1. $\sum_{j=1}^n a_{ij} = 1, \forall i\in\{1,\dots,n\}$

Let $\mathbf A \in\mathbb R^{n \times n}$ be a state transition matrix. Then,

1. $1$ is always an eigenvalue of $\mathbf A$.
1. Multiplication with $\mathbf A$ does not increase the infinity norm. i.e.
    $$
    \begin{align}
      \Vert\mathbf{Ax}\Vert_{\infty} \le  \Vert\mathbf{x}\Vert_{\infty}, \forall \mathbf{x}\in\mathbb R^n
    \end{align}
    $$
1. The eigenvalues of $\mathbf A$ are at most $1$. i.e.
    $$
    \begin{align}
      \vert \lambda \vert \le 1, \forall\lambda\in\operatorname{spec}(A)
    \end{align}
    $$

*Proof 1*: Let $\mathbf u = [1,\dots,1]^\top \in\mathbb R^n$ be all-one vector. It is easy to verify that $\mathbf{Au} = \mathbf{u}$. Hence, $\mathbf u$ is an eigenvector of $\mathbf A$ with eigen value $1$. $\:\square$

*Proof 2:* Recall the infinity norm is defined by

$$
\Vert \mathbf x \Vert_{\infty} \triangleq \max_{i=1,\dots,n} \vert x_i \vert
$$

Let $\mathbf y = \mathbf{Ax}$. Consider the abs of $i$-th element of $\mathbf y$:

$$
\vert y_i \vert
= \left\vert \sum_{j=1}^n a_{ij} x_j \right\vert
\le  \sum_{j=1}^n \Big\vert  a_{ij} x_j \Big\vert
=  \sum_{j=1}^n a_{ij} \big\vert  x_j \big\vert
\le  \sum_{j=1}^n a_{ij} \Vert \mathbf x \Vert_{\infty}
= \Vert \mathbf x \Vert_{\infty}
$$

$\implies$ All elements of $\mathbf y$ are upper-bounded by $\Vert \mathbf x \Vert_{\infty}$ in abs. Hence,

$$
\Vert \mathbf y \Vert_{\infty}
= \max_{i=1,\dots,n} \vert y_i \vert
\le \Vert \mathbf x \Vert_{\infty}
\quad\quad\quad \square
$$

*Proof 3*: Let $\lambda$ be any eigenvalue of $\mathbf A$ and $\mathbf v$ be the corresponding eigenvector. Using the fact that $\Vert\mathbf{Av}\Vert_{\infty} \le  \Vert\mathbf{v}\Vert_{\infty}$, we conclude

$$
\Vert\mathbf{Av}\Vert_{\infty}
= \Vert\mathbf{\lambda v}\Vert_{\infty}
= \vert\lambda\vert \cdot \Vert\mathbf{v}\Vert_{\infty}
\le \vert\mathbf{v}\Vert_{\infty}
$$

Eigenvector $\mathbf v$ is nonzero $\implies \vert\lambda\vert \le 1$. $\quad\square$
