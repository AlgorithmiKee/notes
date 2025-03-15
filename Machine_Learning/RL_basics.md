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

* $G_t$ is a random quantity as the state transition is stochastic. Taking the same action sequence from $s_t$ again will yield a different total reward.
* The discount factor $\gamma$ serves two purposes:
  1. it ensures that the infinite sum is defined. Recall from math: If $\sum_{n=1}^\infty a_n$ converges absolutely and $\{b_n\}$ is bounded, then $\sum_{n=1}^\infty a_n b_n$ converges absolutely. Here, we have a geometric series which converges absolutely and a bounded sequence of rewards.
  2. it puts more weight on short-term rewards over long-term rewards. e.g. Practical meaning in finance, 100 dollar today is worth more than 100 dollar next year.

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

The ***Q-function*** (or ***state action value***, or simply ***action value***) for a certain policy $\pi$ as follows
> $$
> \begin{align}
> q_{\pi}(s,a)
> &= \mathbb E\left[ G_t \:\middle|\: S_t = s, A_t = a \right],
> \quad \forall s\in\mathcal S, \forall a\in\mathcal A
> \end{align}
> $$

Relation between Q-function and state value for the same $\pi$:

* Interpretation: $q_{\pi}(s,a)$ represents the total reward of taking action $a$ at initial state $s$ and then following a policy $\pi$. vs. $v_{\pi}(s)$ represents the total reward of following $\pi$ from $s$ onward.
* For fixed $\pi$ and $s$, if $q_{\pi}(s,a_1) > q_{\pi}(s,a_2)$, we say that $a_1$ is the better action to take over $a_2$ at state $s$.
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

Depending on the context, the term ***value function*** may refer to state value function or Q-function.

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

The underbraced term is the expected total reward by executing policy $\pi$ starting from state $s_1$ which is by definition exactly the state value of $s_1s_1$. Hence, we conclude. $\quad\blacksquare$

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

### Bellman Equation for Q-function

Similarly, Q-function also has recursive structure
$$
\begin{align}
q_{\pi}(s,a)
&= r(s,a) + \gamma \mathbb E_{s' \sim p(\cdot \mid s, a)} [ v_{\pi}(s') ] \\
&= r(s,a) + \gamma \mathbb E_{s' \sim p(\cdot \mid s, a)} [ q_{\pi}(s', \pi(s')) ] \\
\end{align}
$$

## Policy Evaluation

Given a policy $\pi$, computing its value function $v_{\pi}(\cdot)$ is called ***policy evaluation***. Effectively, we would like to evaluate how good $\pi$ is for each state $s$. This is equivelent to solving Bellman equations. We will see later:

* If $\mathcal S$ is a finite set, solving Bellman equations boils down to solving a system of linear equations.
* If $\mathcal S$ is a infinite set, there is generally no closed-form solution for $v_{\pi}(s)$ expect for a few special cases (not covered here).

### Computing State Values for Finite State Space

#### Analytical Solution

If $\mathcal S$ is fininte, policy evaluation boils down to solving $\mathbf v_{\pi} = \mathbf r_{\pi} + \gamma \mathbf P_{\pi}\mathbf v_{\pi}$ for $\mathbf v_{\pi}$. It is easy to verify that the analytical solution to the Bellman equations is
> $$
> \begin{align}
> \mathbf v_{\pi} = (\mathbf{I} - \gamma \mathbf{P}_{\pi})^{-1} \mathbf r_{\pi}
> \end{align}
> $$

where $\mathbf{I}$ is the $\vert \mathcal S \vert \times \vert \mathcal S \vert$ identity matrix.

Remarks:

* The analytical solution is useful for theoretical study. Only practical when $\vert \mathcal S \vert$ is small.
* Drawback of analytical solution:
  * Requires matrix inversion. High computational complexity (nearly $\mathcal O(\vert \mathcal S \vert^3)$) when $\vert \mathcal S \vert$ is large.
  * No generalization to infinite state space as we can not pack pack all $v(s), s\in\mathcal S$ into a vector

#### Bellman Update

Algorithm to compute state values:

> **BELLMAN UPDATE (vector form)**  
> Initialize $\mathbf v_{0}$ arbitrarily.  
> For $n=0,1,\dots$, run until $\mathbf v_{n}$ converges
>
> $$
> \begin{align}
> \mathbf v_{n+1} = \mathbf r_{\pi} + \gamma \mathbf P_{\pi} \mathbf v_{n}
> \end{align}
> $$

The above algorithm can be reformulated element-wise as follows

> **BELLMAN UPDATE (element-wise form)**  
> Init $v_{0}(s)$ for all $s\in\mathcal S$  
> For $n=0,1,\dots$, run until $v_{n}(s)$ converges  
> $\quad$ For each $s\in\mathcal S$, do  
> $$
> v_{n+1}(s) = r(s, \pi(s)) + \gamma\sum_{s'\in\mathcal S} p(s' \mid s, \pi(s)) [ v_{n}(s') ]
> $$

Remarks:

* During iteration, $\mathbf v_{n}$ itself does not neccessarily satisfy Bellman equation for any policy.
* The sequence $\mathbf v_{0}, \mathbf v_{1}, \dots$ obtained from Bellman update converges to $\mathbf v_{\pi}$, i.e.
  $$
  \begin{align}
  \lim_{n\to\infty} \mathbf v_{n}
  = \mathbf v_{\pi}
  = (\mathbf{I} - \gamma \mathbf{P}_{\pi})^{-1} \mathbf r_{\pi}
  \end{align}
  $$

*Proof of convergence*: Define the affine function
$$
f_{\pi}: \mathbb R^{\vert \mathcal S \vert} \to \mathbb R^{\vert \mathcal S \vert},
\mathbf v \mapsto \mathbf r_{\pi} + \gamma \mathbf P_{\pi} \mathbf v
$$

We show that $f_{\pi}(\cdot)$ is a contractive mapping under infinity norm.
$$
\begin{align*}
\Vert f_{\pi}(\mathbf u) - f_{\pi}(\mathbf v) \Vert_{\infty}
&= \left\Vert (\mathbf r_{\pi} + \gamma \mathbf P_{\pi} \mathbf u) - (\mathbf r_{\pi} + \gamma \mathbf P_{\pi} \mathbf v) \right\Vert_{\infty}
\\
&= \gamma \left\Vert  \mathbf P_{\pi} (\mathbf u - \mathbf v) \right\Vert_{\infty}
\\
&\le \gamma \left\Vert  (\mathbf u - \mathbf v) \right\Vert_{\infty},
\end{align*}
$$

The last step follows from the fact that $\Vert \mathbf P_{\pi} \mathbf x \Vert_{\infty} \le \Vert \mathbf x \Vert_{\infty}, \forall x\in\mathbb R^{\vert \mathcal S \vert}$, i.e. multiplication with row stochastic matrix does not increase infnity norm. (c.f. Appendix).

By contraction mapping theorem (c.f. Appendix), we conclude that

1. $f_{\pi}(\cdot)$ has a unique fixed point. Since $\mathbf v_{\pi}  = f_{\pi}(\mathbf v_{\pi} )$ by Bellman equation, $\mathbf v_{\pi}$ is the unique fixed point.
2. $\forall \mathbf v\in\mathbb R^{\vert \mathcal S \vert}$, the sequence of vectors $f_{\pi}^n (\mathbf v)$ converges to $\mathbf v_{\pi} $ in infnity norm

Since all $p$-norms in $\mathbb R^{\vert \mathcal S \vert}$ are equivalent, convergence in infnity norm implies convergence in any $p$-norm. $\:\blacksquare$

### Bellman Operator for Infinite State Space

Generalization of Bellman update to any $\mathcal S$ (possibly an infinite set). Instead of considering state values $v_{\pi}(s), \forall s\in\mathcal S$ as a vector in $\mathbb R^{\vert \mathcal S \vert}$, we consider the state value function $v_{\pi}(\cdot)$ as a "point" in the following function space.

Let $\mathcal V$ be the set of all **bounded** value functions.
$$
\mathcal V =
\left\{
v:\mathcal S\to\mathbb R \:\Big|\: \Vert v \Vert_\infty  < \infty
\right\}
$$

where the ***sup norm*** is defined as
$$
\Vert v \Vert_\infty \triangleq \max_{s\in\mathcal S} \vert v(s) \vert
$$

The ***sup norm metric*** $d: \mathcal V \times \mathcal V \to \mathbb R, (u,v) \mapsto d(u,v)$ is defined as
$$
d(u,v) = \Vert u-v \Vert_\infty = \max_{s\in\mathcal S} \vert u(s)-v(s) \vert
$$

One can verify that $d(\cdot,\cdot)$ satisfies the metric axioms and that $(\mathcal V, d)$ is a **complete** metric space.

For a certain policy $\pi$, we define the corresponding Bellman operator $\mathcal B_{\pi}$ which maps a state value function $v(\cdot)$ to a new value function $\mathcal B_{\pi}v(\cdot)$.
> $$
> \mathcal B_{\pi}: \mathcal V \to \mathcal V, v \mapsto \mathcal B_{\pi}v
> $$

where the new value function is defined as
> $$
> \mathcal B_{\pi} v(s) =
> r(s, \pi(s)) + \gamma\mathbb E_{s' \sim p(\cdot \mid s, \pi(s))} [ v(s') ]
> $$

Remarks:

* $\mathcal B_{\pi} v$ is well defined even though $v$ itself might not satisfy Bellman equation for any policy. Nevertheless, if $v$ happens to be the value function $v_{\tilde\pi}$ for some policy $\tilde \pi$. Then,
  $$
  \mathcal B_{\pi} v_{\tilde\pi} = q_{\tilde\pi}(s,\pi(s))
  $$

* We will see later that solving Bellman equation is equivalent to search for fixed point of $B_{\pi}$ in function space $\mathcal V$.

Properties of Bellman operator:

> 1. $\mathcal B_{\pi}$  is monotonic, i.e.
>    $$
>    u(s) \le v(s), \forall s\in\mathcal S \implies
>    \mathcal B_{\pi}u(s) \le \mathcal B_{\pi}v(s), \forall s\in\mathcal S
>    $$
> 1. $\mathcal B_{\pi}$  is a contractive mapping. i.e. 
>    $$
>    \forall u, v\in\mathcal V,\:
>    \Vert \mathcal B_{\pi}u - \mathcal B_{\pi}v \Vert_\infty \le \gamma \Vert u-v\Vert_\infty
>    $$
> 1. $v_{\pi}(\cdot)$ is the unique fixed point of $\mathcal B_{\pi}$, i.e.
>    $$
>    \mathcal B_{\pi} v_{\pi}(s) = v_{\pi}(s), \forall s\in\mathcal S
>    $$

*Proof 1*:
By the monotonicity of expectation
$\mathbb E_{s' \sim p(\cdot \mid s, \pi(s))} [ u(s') ] \le \mathbb E_{s' \sim p(\cdot \mid s, \pi(s))} [ v(s') ]$, we conclude. $\:\blacksquare$

*Proof 2*:
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
\tag*{$\blacksquare$}
$$

*Proof 3*:
By Bellman equation, we know that $v_{\pi}$ is a fixed point of $\mathcal B_{\pi}$. The uniqueness follows from contraction mapping theorem and completeness of $\mathcal V$. $\quad\blacksquare$

Hence, starting from any $v\in\mathcal V$ (which does not neccessarily need to satisfy Bellman equation for any policy). Repeatedly applying $\mathcal B_{\pi}$ on $v$ leads to convergence to $v_{\pi}$. Formally,
$$
\forall v \in\mathcal V: \lim_{n\to\infty} \Vert B_{\pi}^n v - v_{\pi}\Vert_{\infty} =0
$$

Using the fact that convergence in sup norm implies point-wise convergence(c.f. Appendix), we get
$$
\forall v \in\mathcal V: \lim_{n\to\infty} B_{\pi}^n v(s) =  v_{\pi}(s), \forall s\in\mathcal S
$$

Let $v_n \triangleq \mathcal B_{\pi}^n v$. Then, the function sequence can be computed recursively
$$
\begin{align*}
v_n(s)
&= \mathcal B_{\pi} v_{n-1}(s) \\
&= r(s, \pi(s)) + \gamma\mathbb E_{s' \sim p(\cdot \mid s, \pi(s))} [ v_{n-1}(s') ] \\
\end{align*}
$$

For finite state space, the above equation reduces to Bellman updates shown in previous section
$$
\begin{align*}
v_n(s)
&= r(s, \pi(s)) + \gamma\sum_{s'\in\mathcal S} p(s' \mid s, \pi(s)) [ v_{n-1}(s') ]
\end{align*}
$$

## Bellman Optimality Equations

A policy $\pi$ outperforms (or improves) another policy $\tilde\pi$ iff the state values $v_{\pi}(s)$ is nonless than $v_{\tilde\pi}(s)$ for **ALL** states $s\in\mathcal S$. i.e.

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

We haven't proved the existence of $\pi^*$. For now, let's assume its existence and discover what conditions have to be met for $\pi^*$ and $v^*(\cdot)$. This will lead us to Bellman optimality equations,  from which we will derive an algorithm to compute $\pi^*$ (and thus prove its existence).

### Optimal State Values

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
* Suppose we solved optimal values functoin $v^*(\cdot)$, plugging them into this equation yields the optimal policy.

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

where $\max$ is taken element-wise.
$$
\max_{a}
\begin{bmatrix}
f_1(a) \\
\vdots \\
f_n(a)
\end{bmatrix}
=
\begin{bmatrix}
\displaystyle\max_{a} f_1(a) \\
\vdots \\
\displaystyle\max_{a} f_n(a)
\end{bmatrix}
=
\begin{bmatrix}
f_1(\hat a_1) \\
\vdots \\
f_n(\hat a_n)
\end{bmatrix}
$$

Hence, we get BOE in vector form
> $$
> \begin{align}
> \mathbf v^*
> = \max_{a\in\mathcal A} \left\{\mathbf r_a + \gamma \mathbf P_a\mathbf v^* \right\}
> \end{align}
> $$

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

Given the parameters of an MDP:

* state transition probabilities $p(s' \mid s,a)$ for all $s,s'\in\mathcal S, a\in\mathcal A $
* state-action-rewards $r(s,a)$ for all $s\in\mathcal S, a\in\mathcal A $

How to compute the optimal policy?  
$\to$ Dynamic programming

### Value Iteration

The recurrent structure in BOE motivates us to define the Bellman optimality operator, which is at the core of value iteration. We will show that iteration over Bellman optimality operator leads to the optimal value function. (Just like iterating over Bellman operator for a certain policy leads to the value function for that policy)

#### Bellman Optimality Operator

Recall the metric space $(\mathcal V, d)$ of bounded value functions equipped with sup norm metirc.

Analogous to Bellman operator for a certain policy, we define the ***Bellman optimality operator*** $\mathcal B_{*}$ as
> $$
> \mathcal B_{*}: \mathcal V \to \mathcal V, v(\cdot) \mapsto \mathcal B_{*}v(\cdot)
> $$

where
> $$
> \mathcal B_{*} v(s) = \max_{a\in\mathcal A} \Big\{
>    r(s, a) + \gamma\mathbb E_{s' \sim p(\cdot \mid s, a)} [ v(s') ]
> \Big\}
> $$

Remarks:

* If $v$ happens to be the value function $v_{\tilde\pi}$ for some policy $\tilde \pi$. Then,
  $$
  \mathcal B_{*} v_{\tilde\pi}(s) = \max_{a\in\mathcal A} q_{\tilde\pi}(s,a)
  $$

* We will see later that solving BOE is the same as searching for fixed point of $\mathcal B_*$ in function space $\mathcal V$.

Properties of Bellman optimality operator:

> 1. $\mathcal B_{*}$  is monotonic, i.e.
>    $$
>    u(s) \le v(s), \forall s\in\mathcal S \implies
>    \mathcal B_{*}u(s) \le \mathcal B_{*}v(s), \forall s\in\mathcal S
>    $$
> 1. $\mathcal B_{*}$  is a contractive mapping. i.e. 
>    $$
>    \forall u, v \in\mathcal V,\:
>    \Vert \mathcal B_{*}u - \mathcal B_{*}v \Vert_\infty \le \gamma\Vert u-v\Vert_\infty
>    $$
> 1. $v_{*}(\cdot)$ is the unique fixed point of $\mathcal B_{*}$, i.e.
>    $$
>    \mathcal B_{*} v_{*}(s) = v_{*}(s), \forall s\in\mathcal S
>    $$

*Proof 1 and 3*: Same as the proof for $\mathcal B_{\pi}$. $\qquad\blacksquare$

*Proof 2*: We will show that $\forall s\in\mathcal S: \vert B_{*}u(s) - \mathcal B_{*}v(s) \vert \le \gamma\Vert u-v\Vert_\infty$ as follows
$$
\begin{align*}
\vert B_{*}u(s) - \mathcal B_{*}v(s) \vert
&= \left\vert
      \max_{a\in\mathcal A} \Big\{r(s, a) + \gamma\mathbb E_{s' \sim p(\cdot \mid s, a)} [u(s')]\Big\} -
      \max_{a\in\mathcal A} \Big\{r(s, a) + \gamma\mathbb E_{s' \sim p(\cdot \mid s, a)} [v(s')]\Big\}
   \right\vert \\
&\le \max_{a\in\mathcal A} \left\vert
      \Big\{r(s, a) + \gamma\mathbb E_{s' \sim p(\cdot \mid s, a)} [u(s')]\Big\} -
      \Big\{r(s, a) + \gamma\mathbb E_{s' \sim p(\cdot \mid s, a)} [v(s')]\Big\}
     \right\vert \\
&= \gamma\max_{a\in\mathcal A} \left\vert
      \mathbb E_{s' \sim p(\cdot \mid s, a)} [u(s')-v(s')]
     \right\vert \\
&\le \gamma\max_{a\in\mathcal A}
      \mathbb E_{s' \sim p(\cdot \mid s, a)} \left[ \vert u(s')-v(s')\vert \right] \\
&\le \gamma\max_{a\in\mathcal A}
      \mathbb E_{s' \sim p(\cdot \mid s, a)} \left[ \Vert u-v\Vert_\infty \right] \\
&= \gamma\Vert u-v\Vert_\infty
\end{align*}
$$

whre the 2nd step follows from the fact (c.f. Appendix) that
$$
\left\vert \max_{x\in\mathcal X} f(x) - \max_{x\in\mathcal X} g(x) \right\vert
\le \max_{x\in\mathcal X} \left\vert f(x) -g(x) \right\vert
\tag*{$\blacksquare$}
$$

Once again, starting from any $v \in\mathcal V$ (which does not neccessarily need to satisfy Bellman equation for any policy), repeatedly applying $\mathcal B_{*}$ leads convergence to the optimal value function $v_{*}$.
$$
\forall v \in\mathcal V: \lim_{n\to\infty} \Vert B_{*}^n v - v_{*}\Vert_{\infty}
\implies  \lim_{n\to\infty} B_{*}^n v(s) =  v_{*}(s)
$$

Let $v_n \triangleq \mathcal B_{*}^n v$. Then, $v_n$ can be iteratively computed as follows (called ***value iteration***)
$$
\begin{align}
v_n(s)
&= \mathcal B_{*} v_{n-1}(s) \\
&= \max_{a\in\mathcal A} \Big\{r(s, a) + \gamma\mathbb E_{s' \sim p(\cdot \mid s, a)} [v_{n-1}(s')]\Big\} \\
&= \max_{a\in\mathcal A} \Big\{ r(s,a) + \gamma\sum_{s'\in\mathcal S} p(s' \mid s,a) [ v_{n-1}(s') ] \Big\} &\text{if $\mathcal S$ is finite}
\end{align}
$$

Having computed the optimal value function $v^*$, the optimal policy is obtained from
$$
\pi^*(s)
= \argmax_{a\in\mathcal A}
\left\{
  r(s, a) + \gamma\mathbb E_{s' \sim p(\cdot \mid s, a)} [v^*(s')]
\right\}
$$

#### The Algorithm

Now, we unfold value iteration algorithm for finite state space.
> **VALUE ITERATION (vector form)**  
> Init $\mathbf v_{0}$ arbitrarily  
> For $n=0,1,\dots$, run until $\mathbf v_{n}$ converges, compute
> $$
>  \begin{align*}
>  \mathbf v_{n+1}
>  &= \displaystyle\max_{a\in\mathcal A} \left\{\mathbf r_a + \gamma \mathbf P_a\mathbf v_n \right\}
>  \\
>  \boldsymbol{\pi}_{n+1}
>  &= \displaystyle\argmax_{a\in\mathcal A} \left\{\mathbf r_a + \gamma \mathbf P_a\mathbf v_n \right\}
>  \end{align*}
>  \quad
>  \text{where } \max_{a\in\mathcal A}(\cdot) \text{ is taken row by row}
> $$
>
> Return $\mathbf v^*$ and $\boldsymbol{\pi}^*$.

Equivalent reformulation in element-wise form
> **VALUE ITERATION (element-wise form)**  
> Init $v_{0}(s)$ for all $s\in\mathcal S$ by random guessing  
> For $n = 0,1,\dots$, do until $v_{0}(s)$ converges for all $s\in\mathcal S$  
> $\quad$ For each $s\in\mathcal S$, do  
> $\quad\quad$ For each $a\in\mathcal A$, compute  
> $\quad\quad\qquad q_n(s,a) = r(s, a) + \gamma \displaystyle\sum_{s'} p(s' \mid s, a) \cdot v_n(s')$  
> $\quad\quad$ **Policy update**: $\pi_{n+1}(s) = \displaystyle\argmax_{a\in\mathcal A}\, q_n(s,a)$  
> $\quad\quad$ **Value update**: $v_{n+1}(s) = \displaystyle\max_{a\in\mathcal A} q_n(s,a)$  
> Return $v^{*}(s)$ and $\pi^{*}(s)$ for all $s\in\mathcal S$

Remarks:

* In vector form, $\boldsymbol{\pi}_n$ consists of $\pi(s)$ for all $s\in\mathcal S$.
* The sequence $v_n(\cdot)$ converges to $v^{*}(\cdot)$ which satisfies BOE. However, $v_n(\cdot)$ itself do **not** generally satisfy Bellman equation for **any** policy. $v_n(s)$ should be interpreted as the estimate of $v^*(s)$ at $n$-th iteration rather than the state values under some policy. In particular,
  $$
    \begin{align*}
    v_n(s)
    &\ne v_{\pi_n} (s)
    \\
    v_n(s)
    &\ne r(s, \pi_n(s)) + \gamma \sum_{s'} p(s'\mid s, \pi_n(s)) v_n(s)
    = \mathcal B_{\pi_{n}} v_n(s)
    \end{align*}
  $$
* Likewise, $q_n(s,a)$ represents the estimate of $q^{*}(s,a)$ at $n$-th iteration rather than the real Q-function for any policy. In particular, $q_n(s,a) \ne q_{\pi_n}(s,a)$

### Policy Iteration

Policy iteration is another algorithm to compute optimal policy. It starts with abitrary policy and iteratively improves it. The algorithm is backed by policy improvement theorem.

#### Policy Improvement Theorem

> Let $\pi$ and $\pi'$ be two policies s.t.
> $$
> \forall s\in\mathcal S: q_{\pi}(s, \pi'(s)) \ge q_{\pi}(s, \pi(s))
> $$
>
> Then, $\pi'$ is an improvement of $\pi$, i.e.
> $$
> \forall s\in\mathcal S, v_{\pi'}(s) \ge v_{\pi}(s)
> $$

*Proof*: Using the definition of Q-functions and Bellman operator, we get
$$
\begin{align*}
q_{\pi}(s, \pi'(s)) &\ge q_{\pi}(s, \pi(s)) \\
r(s, \pi'(s)) + \mathbb E_{s'\sim p(\cdot\mid s, \pi'(s))}[v_{\pi}(s')] &\ge v_{\pi}(s) \\
\mathcal B_{\pi'} v_{\pi}(s) &\ge v_{\pi}(s)
\end{align*}
$$

By induction, we have
$$
\mathcal B_{\pi'}^n v_{\pi}(s) \ge v_{\pi}(s)
\tag{$\star$}
$$
Recall that $\mathcal B_{\pi'}^n$ converges to $v_{\pi'}$ for any $v\in\mathcal V$. Taking the limit in $(\star)$, we get
$$
\lim_{n\to\infty} \mathcal B_{\pi'}^n v_{\pi}(s)
= v_{\pi'}(s)
\ge v_{\pi}(s)
\tag*{$\blacksquare$}
$$

#### Greedy Policy

Given a poliy $\pi$, how to construct a better policy $\pi'$? A natural approach would be for each $s$, pick $a$, such that $q_{\pi}(s,\cdot)$ is maximized. The resulting called greedy policy.

A policy $\pi'$ is called ***greedy*** w.r.t. $q_{\pi}$ if it is constructed as
$$
\begin{align}
\forall s\in\mathcal S, \:
\pi'(s)
&= \argmax_a \: q_{\pi}(s,a) \\
&= \argmax_a \Big\{ r(s, a) + \gamma\mathbb E_{s' \sim p(\cdot \mid s, a)} [v_{\pi}(s')] \Big\} \\
\end{align}
$$

Fact: If $\pi'$ is greedy w.r.t. $q_{\pi}$, then it improves the original policy $\pi$ , i.e.
$$
\begin{align}
\forall s\in\mathcal S, \:
\pi'(s) = \argmax_a \: q_{\pi}(s,a)
\implies v_{\pi'} \ge v_{\pi}
\end{align}
$$

*Proof*: By construction of $\pi'$, we have
$$
\forall s\in\mathcal S, \forall a\in\mathcal A, \:
q_{\pi}(s,\pi'(s)) \ge q_{\pi}(s,a)
$$

In particular,
$$
\forall s\in\mathcal S, \:
q_{\pi}(s,\pi'(s)) \ge q_{\pi}(s,\pi(s))
$$

By policy improvement theorem, we conclude that $v_{\pi'} \ge v_{\pi}$. $\qquad\blacksquare$

Suppose we improved $\pi$ by constructing a greedy policy $\pi'$. We can repeat this process to improve $\pi'$ and get $\pi''$. Keep doing so is called ***policy iteration***. The question is whether policy iteration will lead to an optimal policy. The answer is yes due to the following fact.

> Let $\pi_0$ be any policy. Construct a sequence of greedy policy
> $$
> \pi_{n+1}(s) = \displaystyle\argmax_{a\in\mathcal A} \: q_{\pi_n} (s,a),
> \quad\forall s\in\mathcal S
> $$
>
> Then, the corresponding sequence of value functions $(v_{\pi_n})_{n\in\mathbb N}$ converges to the optimal value function $v^*$.

*Proof*: By construction, $\pi_{n+1}$ is greedy w.r.t. $q_{\pi_n}$ for all $n\in\mathbb N$. By property of greedy policy, we have $v_{n+1} \ge v_{\pi_n}$, i.e. The sequence of functions $v_{\pi_n}$ monotonically increases.

Moreover, all $v_{\pi_n}$ are bounded by $v^*$. By monotonic convergence theorem (c.f. Appendix), $v_{\pi_n}$ converges to some $\bar v\in\mathcal V$ with $\bar v \le v^*$. To show that $\bar v = v^*$, it remains to show that $v^*\le \bar v$.

Note that we can express $q_{\pi_n} (s,\pi_{n+1}(s))$ in two differenet ways:

$$
\begin{align*}
q_{\pi_n} (s,\pi_{n+1}(s))
&= \max_a \Big\{ r(s, a) + \gamma\mathbb E_{s' \sim p(\cdot \mid s, a)} [v_{\pi_n}(s')] \Big\}
= \mathcal B_{*} v_{\pi_n}(s)
\\
&= r(s, \pi_{n+1}(s)) + \gamma\mathbb E_{s' \sim p(\cdot \mid s, \pi_{n+1}(s))} [v_{\pi_n}(s')]
= \mathcal B_{\pi_{n+1}} v_{\pi_n}(s)
\end{align*}
$$

Namely, $\mathcal B_{*} v_{\pi_n}(s) = q_{\pi_n} (s,\pi_{n+1}(s)) = \mathcal B_{\pi_{n+1}} v_{\pi_n}(s), \forall s\in\mathcal S$.

By monotonicity and fixed point property of $\mathcal B_{\pi_{n+1}}$,
$$
\boxed{\mathcal B_{*} v_{\pi_n}}
= \mathcal B_{\pi_{n+1}}  v_{\pi_n}
\le \mathcal B_{\pi_{n+1}}  v_{\pi_{n+1}}
= \boxed{v_{\pi_{n+1}}}
$$

By induction, we get
$$
\mathcal B_{*}^{n+1} v_{\pi_0} \le v_{\pi_{n+1}}
$$

Taking the limit on both sides, we conclude.
$$
\boxed{v^*} =
\lim_{n\to\infty} \mathcal B_{*}^{n+1} v_{\pi_0}
\le \lim_{n\to\infty} v_{\pi_{n+1}}
= \boxed{\bar v}
\tag*{$\blacksquare$}
$$

#### The Algorithm

Now, we unfold the policy iteration for easy implementation. The  greedy policy construction,requires computing $q_{\pi_n}$ which requires knowledge of $v_{\pi_n}$. Hence, we must perform policy evaluation for $v_{\pi_n}$ before improving $\pi_{n}$.

> **POLICY ITERATION(vector form)**  
> Init $\pi_0$ by random guessing  
> For $n=0,1,2,\dots$, run until convergence  
> $\quad$ **Policy evaluation**: Compute $\mathbf v_{\pi_n}$ in $\mathbf v_{\pi_n} = \mathbf r_{\pi_n} + \gamma\mathbf P_{\pi_n} \mathbf v_{\pi_n}$  
> $\quad$ **Policy improvement**: Solve $\pi_{n+1} = \displaystyle\argmax_{\pi}  \big\{ \mathbf r_{\pi} + \gamma\mathbf P_{\pi} \mathbf v_{\pi_n} \big\}$ for new policy $\pi_{n+1}$

Equivalent element-wise formutaiton of policy iteration:
> **POLICY ITERATION(element-wise form)**  
> Init $\pi_{0}(s)$ for all $s\in\mathcal S$ by random guessing  
> For $n=0,1,2,\dots$, do  
> $\quad$ **Policy evaluation**: Compute $v_{\pi_n}(s)$ for all $s\in\mathcal S$ given $\pi_n$.  
> $\quad$ For each $s\in\mathcal S$, do  
> $\quad\quad\;$ For each $a\in\mathcal A$, compute  
> $\quad\quad\qquad$ Q-function: $q_{\pi_n} (s,a) = r(s,a) + \gamma\displaystyle\sum_{s'\in\mathcal S} p(s'\mid s,a) \, v_{\pi_n}(s')$  
> $\quad\quad\;$ **Policy improvement**: compute greedy policy $\pi_{n+1}(s) = \displaystyle\argmax_{a\in\mathcal A} \: q_{\pi_n} (s,a)$  
> until $\Vert\mathbf v_{\pi_{n+1}} - \mathbf v_{\pi_n} \Vert < \epsilon$  
> Return $v_{\pi_n}(s)$ and $\pi_n(s)$ for all $s\in\mathcal S$

Remarks:

* In policy evaluation, either analytical solution (for small state space)or Bellman update (for large state space)is used.

### Comparison of Value Iteration and Policy Iteration

## Generalization

Stochastic policy and stochastic rewards.

A policy can also be stochastic. i.e. instead of mapping the state to a fixed action, there are multiple possible actions with different probability. A stochastic policy is described by the conditional distribution

$$
\pi(a \mid s), a \in\mathcal A, s \in\mathcal S
$$

Remarks:

* By the law of total probability,
  $$
  \sum_a \pi(a \mid s) = 1
  $$
* The determinstic policy can be seen as a special of stochastic policy by assigning $\pi(\hat a \mid s)$ to 1 for some $\hat a$.
  $$
  \pi(a \mid s) =
  \begin{cases}
    1 & a=\hat a\\
    0 & \text{else}
  \end{cases}
  $$
* Both deterministic and stochastic policy are time-invariant. i.e. The distribution of $a$ given $s$ is always the same, regardless when we arrived at $s$.

## Appendix

### Contractive Mapping

Let $(\mathcal X, d)$ be a metric space. We say that $f:\mathcal X \to \mathcal X$ is a ***contractive*** mapping if
$$
\exists \gamma\in[0,1), \text{ s.t. }
\forall x, y\in\mathcal X,
d(f(x), f(y)) \le \gamma d(x,y)
$$

> **Contraction Mapping Theorem**  
> If $(\mathcal X, d)$ is **complete** and $f:\mathcal X \to \mathcal X$ is contractive, then
> 
> 1. $f$ has unique fixed point. i.e. $\exists !\, x^* \in\mathcal X$ s.t. $f(x^*) = x^*$
> 2. Construction of the fixed point: $\forall x\in\mathcal X, \displaystyle\lim_{n\to\infty} f^n(x) = x^*$

*Proof*: c.f. Separete notes.

> **Montonic Contraction**  
> Suppose that $(\mathcal X, d)$ is a complete metric space with a **partial oder** "$\le$". Let $f:\mathcal X \to \mathcal X$ be a **monotonic increasing** contraction mapping w.r.t. "$\le$". Then, 
> $$
> \forall x: x\le x^* \iff x\le f(x)
> $$

*Proof*: The claim is trivially true if $x=x^*$. Hence, we assume $x\ne x^*$ in the following.

$\Longleftarrow$:  
Using the monotonicity of $f$, we get the induction
$$
\boxed{x} \le f(x) \le f^2(x) \le \cdots \le \boxed{f^n(x)},
\quad \forall x\in\mathbb N
$$

Taking the limit, we conclude $x\le \displaystyle\lim_{n\to\infty} f^n(x) = x^*$. $\qquad\blacksquare$

$\implies$:  
For the sake of contradiction, suppose $\exist x: x\le x^*$ but $x> f(x)$. By induction,
$$
x > f^n(x),
\quad \forall x\in\mathbb N
$$

Taking the limit, we get $x \ge \displaystyle\lim_{n\to\infty} f^n(x) = x^*$. Since $x\ne x^*$ by assumption, we get $x>x^*$ which contradicts with $x\le x^*$. $\qquad\blacksquare$

### Dealing with Max and Abs

> Let $f,g: \mathcal X \to \mathbb R$ be any two real-valued functions, where $\mathcal X$ can be any set. Then,
> $$
> \left\vert \max_{x\in\mathcal X} f(x) - \max_{x\in\mathcal X} g(x) \right\vert
> \le \max_{x\in\mathcal X} \left\vert f(x) -g(x) \right\vert
> $$

*Proof*: Let $M_f = \displaystyle\max_{x\in\mathcal X} f(x)$ and $M_g = \displaystyle\max_{x\in\mathcal X} g(x)$. It is sufficient to show that
$$
\exists x\in\mathcal X \text{ s.t. }
\left\vert M_f - M_g \right\vert \le \left\vert f(x) -g(x) \right\vert
$$

Without loss of generality, assume that $M_f\ge M_g$. Then,

$$
\left\vert M_f - M_g \right\vert = M_f - M_g \le M_f - g(x),
\:\forall x\in\mathcal X
$$

Let $x^*$ be the maximizer of $f$, i.e. $M_f=f(x^*)$. We conclude
$$
\left\vert M_f - M_g \right\vert \le f(x^*) - g(x^*) = \vert 
$$

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

> Let $\mathbf A \in\mathbb R^{n \times n}$ be a state transition matrix. Then,
> 
> 1. $1$ is always an eigenvalue of $\mathbf A$.
> 1. Multiplication with $\mathbf A$ does not increase the infinity norm. i.e.
>     $$
>     \begin{align}
>       \Vert\mathbf{Ax}\Vert_{\infty} \le  \Vert\mathbf{x}\Vert_{\infty}, \forall \mathbf{x}\in\mathbb R^n
>     \end{align}
>     $$
> 1. The eigenvalues of $\mathbf A$ are at most $1$. i.e.
>     $$
>     \begin{align}
>       \vert \lambda \vert \le 1, \forall\lambda\in\operatorname{spec}(A)
>     \end{align}
>     $$

*Proof 1*: Let $\mathbf u = [1,\dots,1]^\top \in\mathbb R^n$ be all-one vector. It is easy to verify that $\mathbf{Au} = \mathbf{u}$. Hence, $\mathbf u$ is an eigenvector of $\mathbf A$ with eigen value $1$. $\:\blacksquare$

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
\quad\quad\quad \blacksquare
$$

*Proof 3*: Let $\lambda$ be any eigenvalue of $\mathbf A$ and $\mathbf v$ be the corresponding eigenvector. Using the fact that $\Vert\mathbf{Av}\Vert_{\infty} \le  \Vert\mathbf{v}\Vert_{\infty}$, we conclude

$$
\Vert\mathbf{Av}\Vert_{\infty}
= \Vert\mathbf{\lambda v}\Vert_{\infty}
= \vert\lambda\vert \cdot \Vert\mathbf{v}\Vert_{\infty}
\le \vert\mathbf{v}\Vert_{\infty}
$$

Eigenvector $\mathbf v$ is nonzero $\implies \vert\lambda\vert \le 1$. $\quad\blacksquare$

### Convergence of Sequence of Functions

Let $\mathcal X$ be any set and $(f_n)_{n\in\mathbb N}:\mathcal X \to \mathbb R$ be a sequence of functions. Then, we say that

$(f_n)_{n\in\mathbb N}$ converges to $f$ **point-wise** iff
$$
\begin{align}
\forall x\in\mathcal X, &\lim_{n\to\infty} f_n(x) = f(x) \\
&\qquad \Updownarrow \nonumber \\
\forall x\in\mathcal X, &\forall\epsilon>0, \exists N\in\mathbb N, \text{ s.t. }
\forall n \ge N, \vert f_n(x) - f(x) \vert < \epsilon
\end{align}
$$

$(f_n)_{n\in\mathbb N}$ converges to $f$ **uniformly** iff
$$
\begin{align}
\forall\epsilon>0, \exists N\in\mathbb N, \text{ s.t. }
\forall n \ge N, \forall x\in\mathcal X, \vert f_n(x) - f(x) \vert < \epsilon
\end{align}
$$

$(f_n)_{n\in\mathbb N}$ converges to $f$ **in sup norm** iff
$$
\begin{align}
\lim_{n\to\infty} &\Vert f_n(x) - f(x) \Vert_\infty = 0  \\
&\qquad\Updownarrow \nonumber \\
\forall\epsilon>0, &\exists N\in\mathbb N, \text{ s.t. }
\forall n \ge N, \sup_{x\in\mathcal X}\vert f_n(x) - f(x) \vert < \epsilon
\end{align}
$$

> Relation between different types of convergence:
> $$
> \begin{align*}
> \text{uniform convg.} \iff \text{convg. in sup norm}
> \implies \text{point-wise convg.}
> \end{align*}
> $$

$(f_n)_{n\in\mathbb N}$ is **monotonically increasing** iff for each $x\in\mathcal X$, the sequence $(f_n(x))_{n\in\mathbb N}$ is monotonically increasing, i.e.
$$
\forall x\in\mathcal X, \forall n\in\mathbb N, f_n(x) \le f_{n+1}(x)
$$

$(f_n)_{n\in\mathbb N}$ is **point-wise bounded** iff for each $x\in\mathcal X$, the sequence $(f_n(x))_{n\in\mathbb N}$ is bounded by some $M_x\in\mathbb R$, i.e.
$$
\forall x\in\mathcal X, \exist M_x\in\mathbb R \text{ s.t. }
\forall n\in\mathbb N, \vert f_n(x) \vert \le M_x
$$

$(f_n)_{n\in\mathbb N}$ is **uniformly bounded** iff
$$
\exist M\in\mathbb R \text{ s.t. }
\forall x\in\mathcal X, \forall n\in\mathbb N, \vert f_n(x)\vert \le M
$$

> **Monotonic Convergence Theorem**  
> Let $(f_n)_{n\in\mathbb N}$ be point-wise bounded and monotonically increasing, then $(f_n)_{n\in\mathbb N}$ converges to some $f$.

Remark: The limit function is not necessarily equal to the pointwise bound! i.e. Let $\vert f_n(x)\vert$ be bounded by $M_x$. Then, $f(x)\ne M_x$ in general.
