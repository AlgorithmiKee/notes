---
title: "Dynamic Programming for MDP II"
author: "Ke Zhang"
date: "2025"
fontsize: 12pt
---

# Dynamic Programming for Markov Decision Process II

[toc]

$$
\DeclareMathOperator*{\argmax}{arg\max}
$$

Notation:

* upercase (e.g. $R_t$): random variable

* lowercase (e.g. $s_t$): instance of random variable

* bold straight (e.g. $\mathbf{v}$): determineistic vector

Preliminary: Dynamic Programming for Markov Decision Process I

## Generalization

There are multiple ways to generalize the MDP:

* deterministic policy $\pi(s)\longrightarrow $ stochastic policy $\pi(a\mid s)$.
* deterministic rewards $r(s,a)\longrightarrow $ stochastic rewards $p(r\mid s,a)$.
* infinite time horizon and time invariance $\longrightarrow$ finite time horizon and time dependence.

In the following, we focus on generalization to stochastic policy and stochastic rewards.

### Stochastic Policy

Instead of mapping each state to a fixed action, a stochastic policy samples an action from the conditional distribution $\pi(\cdot\mid s)$.

$$
\begin{align}
A\sim\pi(a \mid s)
\end{align}
$$

Remarks:

* Stochastic policy satisfies the law of total probability.
  $$
  \begin{align}
  \sum_{a\in\mathcal A} \pi(a \mid s) &= 1, \:\forall s\in\mathcal S
  \text{ (for finite $\mathcal A$)}
  \\
  \int_{a\in\mathcal A} \pi(a \mid s) \,\mathrm da &= 1, \:\forall s\in\mathcal S
  \text{ (for continuous $\mathcal A$)}
  \end{align}
  $$

* A determinstic policy $s\mapsto a_s$ can be seen as a special of stochastic policy
  $$
  \begin{align}
  \pi(a \mid s) = \delta(a-a_s), \quad\forall a\in\mathcal A
  \end{align}
  $$
  where $\delta(\cdot)$ is Kronecker delta (for finite $\mathcal A$) or Dirac delta (for continuous $\mathcal A$).

* Under this setting, our stochastic policy is time-invariant (or ***stationary***). i.e. The distribution $\pi(\cdot\mid s)$ does not depend on when we arrived at $s$.

* If both $\mathcal A$ and $\mathcal S$, we can represent a stochastic policy as a $\vert\mathcal S\vert\times\vert\mathcal A\vert$ matrix. Each entry $\pi(a\mid s)$ represents the proability of taking action $a$ at state $s$.

**Example**: An extremely simplifed model of stock trading. $\mathcal S$ denotes the market condition. $\mathcal A$ denotes your trading action.

$$
\begin{align*}
\mathcal S &= \{\text{bullish, bearish}\} \\
\mathcal A &= \{\text{buy, hold, sell}\}
\end{align*}
$$

A trading strategy inspired by the principle *"Be fearful when others are greedy and to be greedy only when others are fearful"* can be modelled as a stochastic policy:

| Market Condition | buy | hold | sell |
|------------------|-----|------|------|
| **bullish**      | 0.2 | 0.4  | 0.4  |
| **bearish**      | 0.6 | 0.3  | 0.1  |

#### Why stochastic policy?

If the agent has perfect knowledge of the environment (state transiton, rewards, etc.), then any deviation from the optimal policy (which is deterministic, as shown later) will indeed yield a lower total reward. In this case, a deterministic policy suffices. The agent only needs to compute the optimal policy -- this is called ***planning problem***. e.g. shortest path problem in a graph with known weights.

In contrast, a ***learning problem*** assumes that the agent does not have the perfect environment model. e.g. shortest path problem in a graph with unknown edge weights. The optimal policy derived from an imperfect environment model is unlikely to be optimal in real environment. In this case, stochastic policy allows the agent to explore the environment by occasionally try different actions, even at a risk of receiving a lower total reward.

#### Exploitation vs. Exploration

* ***Exploitation***: execute the optimal policy based on current understanding of the environment. It yields the highest total reward based on current understanding but misses the oppurtunity for a potentially higher total reward.

* ***Exploration***: deviate from the optimal policy by occasionally trying different actions. It may yield a lower or higher total reward compared to exploitation.

Real-life analogy: Suppose you're looking for good restaurants in your city. You've found that restaurant A is your favorite spot for lunch and restaurant B is your favorite for dinner. Exploitation means you always return to A for lunch and B for dinner — you're likely to be satisfied, but you miss the chance to discover new favorites. Exploration means occasionally trying a new place, say restaurant X. It might not be as good as A, or it could turn out to be even better.

In planning problems, only exploitation is necessary because the environment is fully known. In learning problems, however, it is crucial to balance exploitation and exploration.

### Stochastic Rewards

Given a state action pair $(s,a)$, instead of receving a deterministic reward, the agent receives a stochastic reward sampled from $p(\cdot\mid s,a)$.

$$
R\sim p(r\mid s,a)
$$

Moreover, we let $\mathcal R\subseteq\mathbb R$ denote the reward sets. Then, $p(r\mid s,a)$ is either a PMF (if $\mathcal R$ is finite) or a PDF (if $\mathcal R$ is continuous).

Remarks:

* Stochastic reward satisfies the law of total probability
  $$
  \begin{align}
  \sum_{r\in\mathcal R} p(r\mid s,a) &= 1, \:\forall (s,a)\in\mathcal S\times\mathcal A \\
  \text{or }
  \int_{r\in\mathcal R} p(r\mid s,a) \,\mathrm dr &= 1, \:\forall (s,a)\in\mathcal S\times\mathcal A \\
  \end{align}
  $$

* A determinstic reward $(s,a)\mapsto r_{sa}$ can be seen as a special of stochastic reward
  $$
  \begin{align}
  p(r\mid s,a) = \delta(r - r_{sa}), \quad\forall r\in\mathcal R
  \end{align}
  $$
  where $\delta(\cdot)$ is Kronecker delta (for finite $\mathcal R$) or Dirac delta (for continuous $\mathcal R$).

* The stochastic reward is again stationary. i.e. The distribution of the reward does not depend on when we arrive at $s$ or when we take action $a$.

## Markov Decision Process

A generalized Markov decision process (MDP) consists of

* $\mathcal S$: set of states.
* $\mathcal A$: set of actions.
* $\mathcal R$: set of rewards.
* $p(s'\mid s, a)$: state transition probability.
* $p(r \mid s, a)$: reward probability.
* $\gamma\in[0,1)$: discount factor.

The agent executes a stochastic policy $\pi$. Then, we are interested in

1. policy evaluation
2. computing optimal policy

Consider the execution of $\pi$ from a fixed initial state $s$.

$$
s \xrightarrow[R]{A \:} S' \xrightarrow[R']{A' \:} \cdots
$$

Key observation:

* The immediate reward $R$ is now stochastic. The stochasticity of immediate reward comes from
  1. stoachsticity in action $A$ due to stochastic policy
  1. reward probability
* Starting from the next state, the stochasticity in $R'$ comes from
  1. stoachsticity in state $S'$ due to state transition proability
  1. stoachsticity in action $A'$ due to stochastic policy
  1. reward probability

The total (discounted) reward is defined exactly the same as in MDP I:

$$
\begin{align}
G_t
&= R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \dots \\
&= \sum_{k=0}^\infty \gamma^k R_{t+k}
\end{align}
$$

Remarks:

* For each $\tau=t,t+1,\cdots$, $R_\tau \sim p(r\mid S_\tau, A_\tau)$.
* Recursive structure:

  $$
  \begin{align}
  G_t = R_t + \gamma G_{t+1}
  \end{align}
  $$

### Value functions

State value:

> $$
> \begin{align}
> v_{\pi}(s) = \mathbb E
> \left[ G_t \:\middle|\: S_t =s \right],
> \quad \forall s \in \mathcal S
> \end{align}
> $$

Q-function:

> $$
> \begin{align}
> q_{\pi}(s,a)
> &= \mathbb E\left[ G_t \:\middle|\: S_t = s, A_t = a \right],
> \quad \forall s\in\mathcal S, \forall a\in\mathcal A
> \end{align}
> $$

## Bellman Equations

### Bellman Equations for State Value

Bellman equation for $v_{\pi}$: For each $s\in\mathcal S$, it holds that

> $$
> \begin{align}
> v_{\pi}(s)
> & = \mathbb E_{a\sim\pi(a\mid s)} \Big[\mathbb E_{r\sim p(r\mid s,a)}[r]\Big] + \gamma \mathbb E_{a\sim\pi(a\mid s)} \Big[\mathbb E_{s'\sim p(s'\mid s,a)}[v_{\pi}(s')]\Big]
> \tag{BE-V}
> \end{align}
> $$

Equivalent formulation:

> $$
> \begin{align}
> v_{\pi}(s)
> & = \mathbb E_{a\sim\pi(a\mid s)} \Big[\mathbb E_{r\sim p(r\mid s,a)}[r] + \gamma\mathbb E_{s'\sim p(s'\mid s,a)}[v_{\pi}(s')]\Big] \tag{BE-V1}
> \\[6pt]
> v_{\pi}(s)
> & = \mathbb E_{r\sim p(r\mid s)}[r] + \gamma\mathbb E_{s'\sim p(s'\mid s)}[v_{\pi}(s')] \tag{BE-V2}
> \end{align}
> $$

For finite $\mathcal S$, $\mathcal A$ and $\mathcal R$, Bellman equation becomes

> $$
> \begin{align}
> v_{\pi}(s)
> & = \sum_{a} \pi(a\mid s) \sum_{r} p(r\mid s,a) \cdot r + \gamma \sum_{a} \pi(a\mid s) \sum_{s'} p(s'\mid s,a)\cdot v_{\pi}(s')
> \\
> & = \sum_{a} \pi(a\mid s) \left[\sum_{r} p(r\mid s,a) \cdot r + \gamma \sum_{s'} p(s'\mid s,a)\cdot v_{\pi}(s')\right]
> \end{align}
> $$

Remarks:

* Bellman equation holds for all $s\in\mathcal S$. For finite state space, there are $\vert\mathcal S\vert$ equations in total. There are further equivalent formulation of Bellman equations in the Appendix.
* In $\text{(BE-V)}$: The 1st term represents the expected immediate reward. The 2nd term in $\text{(BE-V)}$ represents the discounted expected future reward.
* In $\text{(BE-V2)}$: The distributions $p(r\mid s)$ and $p(s'\mid s)$ are often denoted by $p_{\pi}(r\mid s)$ and $p_{\pi}(s'\mid s)$ since they both implicitly depend on $\pi$. Specifically, they can be obtained from system parameters by applying the law of total probability (c.f. Appendix)

  > $$
  > \begin{align}
  > p_{\pi}(r \mid s) &= \mathbb E_{a\sim\pi(a\mid s)} \big[p(r \mid s,a)\big] \\
  > p_{\pi}(s'\mid s) &= \mathbb E_{a\sim\pi(a\mid s)} \big[p(s'\mid s,a)\big] \\
  > \end{align}
  > $$

* Illustration:

  $$
  s \xrightarrow[R_t]{A_t \:} S_{t+1} \xrightarrow[R_{t+1}]{A_{t+1} \:} \cdots
  $$

*Proof*: We first show the validity of $\text{(BE-V2)}$. Then, we show $\text{(BE-V)}$ and $\text{(BE-V1)}$ are equivalent to $\text{(BE-V2)}$.

Plugging $G_t = R_t + \gamma G_{t+1}$ into the value function, we get

$$
\begin{align*}
v_{\pi}(s)
&= \mathbb E[ R_t + \gamma G_{t+1} \mid S_t =s ] \\
&= \underbrace{\mathbb E[R_t \mid S_t=s]}_{\mathrm{(I)}} + \gamma \underbrace{\mathbb E[G_{t+1} \mid S_t=s]}_{\mathrm{(II)}} \\
\end{align*}
$$

$\mathrm{(I)} \triangleq$ expected immediate reward:

$$
\begin{align*}
\underbrace{\mathbb E[R_t \mid S_t=s]}_{\mathrm{(I)}}
&= \mathbb E_{r\sim p(r\mid s)} [r]
\end{align*}
$$

$\mathrm{(II)} \triangleq$ expected future reward:

$$
\begin{align*}
\underbrace{\mathbb E[G_{t+1} \mid S_t=s]}_{\mathrm{(II)}}
&= \mathbb E_{s'\sim p(s'\mid s)} \Big[ \mathbb E[G_{t+1} \mid S_t=s, S_{t+1}=s'] \Big] \\
&= \mathbb E_{s'\sim p(s'\mid s)} \Big[ \mathbb E[G_{t+1} \mid S_{t+1}=s'] \Big] \\
&= \mathbb E_{s'\sim p(s'\mid s)} [v_\pi(s')]
\end{align*}
$$

Combining $\text{(I)}$ and $\text{(II)}$, we obtain $\text{(BE-V2)}$:

$$
\begin{align*}
v_{\pi}(s)
&= \underbrace{\mathbb E_{r\sim p(r\mid s)} [r]}_{\mathrm{(I)}}  + \gamma \underbrace{\mathbb E_{s'\sim p(s'\mid s)} [v_\pi(s')]}_{\mathrm{(II)}}
\end{align*}
$$

$\text{(BE-V)}$ and $\text{(BE-V1)}$ follow from $\text{(BE-V2)}$ and the law of total expecation (c.f. Appendix)
as

$$
\begin{align*}
\underbrace{\mathbb E_{r\sim p(r\mid s)} [r]}_{\mathrm{(I)}}
&= \mathbb E_{a\sim\pi(a\mid s)} \Big[ \mathbb E_{r\sim p(r\mid s,a)} [r] \Big]
\\
\underbrace{\mathbb E_{s'\sim p(s'\mid s)} [v_\pi(s')]}_{\mathrm{(II)}}
&= \mathbb E_{a\sim \pi(a\mid s)} \Big[ \mathbb E_{s'\sim p(s'\mid s,a)} [v_\pi(s')] \Big]
\tag*{$\blacksquare$}
\end{align*}
$$

#### Vector-Form Bellman Equation

Recall the Bellman equation for finite $\mathcal S$, $\mathcal A$ and $\mathcal R$:

$$
\begin{align*}
v_{\pi}(s)
& = \sum_{a} \pi(a\mid s) \sum_{r} p(r\mid s,a) \cdot r + \gamma \sum_{a} \pi(a\mid s) \sum_{s'} p(s'\mid s,a)\cdot v_{\pi}(s')
\end{align*}
$$

Switching the orders of sums in both terms, we get

$$
\begin{align*}
v_{\pi}(s)
& = \sum_{r} \underbrace{\sum_{a} \pi(a\mid s) p(r\mid s,a)}_{p_{\pi}(r\mid s)} \cdot r + \gamma  \sum_{s'} \underbrace{\sum_{a} \pi(a\mid s) p(s'\mid s,a)}_{p_{\pi}(s'\mid s)} \cdot v_{\pi}(s')
\\
& = \underbrace{\sum_{r} p_{\pi}(r\mid s) \cdot r}_{r_{\pi}(s)} + \gamma  \sum_{s'} p_{\pi}(s'\mid s) \cdot v_{\pi}(s')
\end{align*}
$$

Assume $\mathcal S=\{\varsigma_1,\dots,\varsigma_{\vert\mathcal S\vert}\}$. Consider $v_{\pi}(\varsigma_i), i=1,\dots,\vert\mathcal S\vert$ as unknowns. For a certain $\pi$, the 1st term $r_{\pi}(s)$ is just a known number. The 2nd term is a linear combination of $\varsigma_1,\dots,\varsigma_{\vert\mathcal S\vert}$. Therefore, we have

$$
\underbrace{
  \begin{bmatrix}
    v_{\pi}(\varsigma_1) \\ v_{\pi}(\varsigma_2) \\ \vdots \\ v_{\pi}(\varsigma_n)
  \end{bmatrix}
}_{\mathbf v_{\pi}}
=
\underbrace{
  \begin{bmatrix}
    r_{\pi}(\varsigma_1) \\ r_{\pi}(\varsigma_2) \\ \vdots \\ r_{\pi}(\varsigma_n)
  \end{bmatrix}
}_{\mathbf r_{\pi}}
+
\gamma
\underbrace{
  \begin{bmatrix}
    p_{\pi}(\varsigma_1 \mid \varsigma_1) & \dots & p_{\pi}(\varsigma_n \mid \varsigma_1)  \\
    p_{\pi}(\varsigma_1 \mid \varsigma_2) & \dots & p_{\pi}(\varsigma_n \mid \varsigma_2)  \\
    \vdots & \cdots & \vdots \\
    p_{\pi}(\varsigma_1 \mid \varsigma_n) & \dots & p_{\pi}(\varsigma_n \mid \varsigma_n)
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

* In finite settings, the Bellman equation has again the same vector form, even though policy and rewards are stochastic. Here, the $\mathbf r_{\pi}$ is a vector of expected immediate rewards w.r.t. $r\sim p_{\pi}(r\mid s)$, which is the statistical average of $p_{\pi}(r\mid s,a)$ over $a\sim\pi(a\mid s)$. The row-stochastic matrix $\mathbf P_{\pi}$ consists of state transition probabilities $p_{\pi}(s'\mid s)$ under $\pi$, which is the statistical average of $p_{\pi}(s'\mid s,a)$ over $a\sim\pi(a\mid s)$.

* The vector-form can again be solve analytically. However, analytical solution is rarely used unless the state space is small. In practice, iterative method is more frequently used (detailed later).

### Bellman Equations for Q-Function

Relation between $q_{\pi}$ and $v_{\pi}$:

> $$
> \begin{align}
> v_{\pi}(s) &= \mathbb E_{a\sim\pi(a\mid s)} [q_{\pi}(s,a)]
> \tag{Q2V}
> \\
> q_{\pi}(s,a) &= \mathbb E_{r\sim p(r\mid s,a)}[r] + \gamma \mathbb E_{s'\sim p(s'\mid s,a)}[v_{\pi}(s')]
> \tag{V2Q}
> \end{align}
> $$

Bellman equation for $q_{\pi}$:

> $$
> \begin{align}
> q_{\pi}(s,a)
> &= \mathbb E_{r\sim p(r\mid s,a)}[r] + \gamma \mathbb E_{s'\sim p(s'\mid s,a)} \Big[
>      \mathbb E_{a'\sim\pi(a'\mid s')} [q_{\pi}(s',a')]
>    \Big]
> \tag{BE-Q}
> \end{align}
> $$

*Proof*: Equation $\text{(Q2V)}$ follows from the law of total expecation:

$$
\begin{align*}
v_{\pi}(s)
&= \mathbb E[ G_t \mid S_t =s ] \\
&= \mathbb E_{a\sim\pi(a\mid s)} \Big[
   \underbrace{\mathbb E[ G_t \mid S_t = s, A_t = a]}_{q_{\pi}(s,a)}
   \Big]
\tag*{$\blacksquare$}
\end{align*}
$$

To show $\text{(V2Q)}$, we use $G_t = R_t + \gamma G_{t+1}$:

$$
\begin{align*}
q_{\pi}(s,a)
&= \mathbb E[ R_t + \gamma G_{t+1} \mid S_t=s, A_t=a]
\\
&= \underbrace{\mathbb E[R_t \mid S_t=s, A_t=a]}_{\mathrm{(I)}} + \gamma \underbrace{\mathbb E[G_{t+1} \mid S_t=s, A_t=a]}_{\mathrm{(II)}}
\\
\end{align*}
$$

The expected immediate reward $\mathrm{(I)}$ is

$$
\begin{align*}
\underbrace{\mathbb E[R_t \mid S_t=s, A_t=a]}_{\mathrm{(I)}} = \mathbb E_{r\sim p(r\mid s,a)}[r]
\end{align*}
$$

The expected future reward $\mathrm{(II)}$ is

$$
\begin{align*}
\underbrace{\mathbb E[G_{t+1} \mid S_t=s, A_t=a]}_{\mathrm{(II)}}
&= \mathbb E_{s'\sim p(s'\mid s,a)} \Big[ \mathbb E[G_{t+1} \mid S_t=s,  A_t=a, S_{t+1}=s'] \Big] \\
&= \mathbb E_{s'\sim p(s'\mid s,a)} \Big[ \mathbb E[G_{t+1} \mid S_{t+1}=s'] \Big] \\
&= \mathbb E_{s'\sim p(s'\mid s,a)} [v_\pi(s')]
\end{align*}
$$

Combining $\text{(I)}$ and $\text{(II)}$, we obtain $\text{(Q2V)}$:

$$
\begin{align*}
v_{\pi}(s)
&= \underbrace{\mathbb E_{r\sim p(r\mid s,a)}[r]}_{\mathrm{(I)}}  + \gamma \underbrace{ \mathbb E_{s'\sim p(s'\mid s,a)} [v_\pi(s')]}_{\mathrm{(II)}}
\tag*{$\blacksquare$}
\end{align*}
$$

Plugging $\text{(Q2V)}$ into $\text{(V2Q)}$ yields $\text{(BE-Q)}$. $\qquad\blacksquare$

## Policy Evaluation

In this section, we assume that $\mathcal S$, $\mathcal A$ and $\mathcal R$ are all finite. Under this setting, Bellman equation can be solved (either analytically or numerically) entirely based on the system model.

In contrast, when $\mathcal S$ is continuous, the Bellman equation becomes an integral equation of $v_{\pi}$. It is challanging to derive model-based algorithms to solve Bellman equation without additional assumptions (e.g. all random variables are Gaussian). This difficulty arises because the expectations in the Bellman equation correspond to high-dimensional integrals, which are generally intractable to compute exactly. To address this difficulty, other techniques like function approximation are used (not detailed here).

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
> v_{n+1}(s) =  \sum_{r} p_{\pi}(r\mid s) \cdot r+ \gamma  \sum_{s'} p_{\pi}(s'\mid s) \cdot v_{\pi}(s')
> $$

## Appendix

### Law of total probability

$$
\begin{align}
p(x)
&= \mathbb E_{\theta\sim p(\theta)} \big[ p(x\mid \theta)\big]
\\
p(x\mid z)
&= \mathbb E_{\theta\sim p(\theta\mid z)} \big[ p(x\mid z,\theta) \big]
\end{align}
$$

**Core Inituition**: computing a probability can be seen as

1. introducing an intermediate variable $\theta$,
2. computing the conditional probability,
3. and then averaging the conditional probability over $\theta$.

### Law of total expectation

$$
\begin{align}
\mathbb E_{x\sim p(x)}[x]
&= \mathbb E_{\theta\sim p(\theta)} \Big[ \mathbb E_{x\sim p(x\mid \theta)}[x] \Big]
\\
\mathbb E_{x\sim p(x\mid z)}[x]
&= \mathbb E_{\theta\sim p(\theta\mid z)} \Big[ \mathbb E_{x\sim p(x\mid z,\theta)}[x] \Big]
\end{align}
$$

**Core Inituition**: computing an expectation can be seen as

1. introducing an intermediate variable $\theta$,
2. computing the conditional expectation,
3. and then averaging the conditional expecation over $\theta$.

### Equivalent Formulation of Bellman equation

In other literatures, the system model is given by $p(r,s'\mid s,a)$ instead of $p(r,s'\mid s,a)$ and $p(r,s'\mid s,a)$. In this setting, Bellman equation becomes

> $$
> \begin{align}
> v_{\pi}(s)
> & = \mathbb E_{a\sim\pi(a\mid s)} \Big[\mathbb E_{r\sim p(r\mid s,a)}[r] + \gamma\mathbb E_{s'\sim p(s'\mid s,a)}[v_{\pi}(s')]\Big]
> \tag{BE-V1} \\
> & = \mathbb E_{a\sim\pi(a\mid s)} \Big[\mathbb E_{r,s'\sim p(r,s'\mid s,a)}[r + \gamma v_{\pi}(s')]\Big]
> \tag{BE-V1*} \\[6pt]
> v_{\pi}(s)
> & = \mathbb E_{r\sim p(r\mid s)}[r] + \gamma\mathbb E_{s'\sim p(s'\mid s)}[v_{\pi}(s')]
> \tag{BE-V2} \\
> & = \mathbb E_{r,s'\sim p(r,s'\mid s)}[r + \gamma v_{\pi}(s')]
> \tag{BE-V2*} \\
> \end{align}
> $$

TODO: remark on $p(r,s'\mid s)$.

*Proof*: This is direct result of the linearity of expectation.

$$
\begin{align*}
\mathbb E_{x\sim p(x)}[g(x)] + \mathbb E_{y\sim p(y)}[h(y)]
&= \mathbb E_{x,y\sim p(x,y)}[g(x)+h(y)] \\
\mathbb E_{x\sim p(x \mid z)}[g(x)] + \mathbb E_{y\sim p(y \mid z)}[h(y)]
&= \mathbb E_{x,y\sim p(x,y \mid z)}[g(x)+h(y)]
\end{align*}
$$

or equivalently in simplified notation

$$
\begin{align*}
\mathbb E[g(X)] + \mathbb E[h(Y)] &= \mathbb E[g(X)+h(Y)] \\
\mathbb E[g(X) \mid Z=z] + \mathbb E[h(Y) \mid Z=z] &= \mathbb E[g(X)+h(Y) \mid Z=z]
\tag*{$\blacksquare$}
\end{align*}
$$
