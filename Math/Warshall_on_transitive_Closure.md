---
title: "Warshall's Algorithm"
date: "2023"
author: "Ke Zhang"
---

# Warshall's Algorithm on Transitive Closure

## Notations

* $M$: Adjacency matrix of a graph
* $M[i,j]$: the element at $i$-th row and $j$-th col of matrix $M$
* $M[i,:]$: the $i$-th row of matrix $M$
* $M[:,j]$: the $j$-th col of matrix $M$
* $R^*$: the transitive closure of a relation $R$.
* $N^-(v)$: Set of all predecessors of vertex $v$ in a graph.
* $N^+(v)$: Set of all successors of vertex $v$ in a graph.

## Problem Formulation

Given a finite set $V$ and a relation $R$ on $V$. How to find the closure of $R$?

**Recall: The transitive closure $R^*$ is the smallest (w.r.t. $\subseteq$) transitive relation on $V$ such that $R\subseteq R^*$. **

The problem is trivial if $R$ happens to be transitive. However, the solution seems less obvious for a non-transitive $R$.

Note that $R^*$ 

* is unique
* always exists

due to the fact that

> Intersection of two transitive relation is again transitive.

The uniqueness of $R^*$ can be argued that suppose we have two distinct transtive closures $R_1^*$ and $R_2^*$, both of which contains $R$. We can always form the intersection $R^*=R_1^*\cap R_2^*$, which is a smaller transitive relation containing $R$.  

The existence of $R^*$ can be argue as follows. It is trivial that $V^2$ is a transitive relation containing $R$. If there is no other transitive relation containing $R$, then $R^*=V^2$. Otherwise, we form $R^*$ as follows

```pseudocode
Let Closure(R) = V*V;
for all transitive relation T on V:
	Closure(R) = Closure(R) interset T;
end for
```

The algorithm guranntees that we will end up with the closure of $R$. However, the above approach is computationally expensive and hence not suitable for real-world computation. 

## The Algorithm

Given the adjacency matrix $M$ representing the relation $R$ on set $V$. The transitive closure of $R$ can be found via Warshall's algorithm. 

Let $V=\{ v_1, \cdots, v_n\}$. Obviously, $M\in\{0,1\}^{n\times n}$

Let the directed graph $G=(V,\,E)$ represent a relation $R$ on set $V$.

> ```pseudocode
> function warshall(V, E):
>     for all v in V:
>         VI = set of incoming nodes of v
>         VO = set of outgoing nodes of v
>         TR = VI*VO		// Cartesian Product
>         for all e=(vi, vo) in TR:
>             if e not in E:
>             	E = E union {e}
>             end if
>         end for
>     end for  
> ```

Intuition: Consider each node $v\in V$ as the intermidiate node of the path $(v_i, v)-(v, v_o)$. If there is no direct connection from $v_i$ to $v_o$, then we connect them.

To see that the above algorithm indeed produces the transitive closure of $R$, we need to show

1. The resulting relation contains $R$
2. The resulting relation is transitive
3. The resulting relation can not be further reduced

Let $R'$ be the resulting relation. Clearly, $R\subseteq R'$ since we are only adding edges in the algorithm.

We show the transitivity of $R'$ by induction. 

Suppose we visited the vertices in oder $v_1,\cdots, v_n$. Let $V_k$ be the set of visited vertices after $k$ iterations. Note that at each iteration, we add a few edges. Let $E_k$ be the set of all edges after $k$ iterations. Define $G_k=(V, \, E_k)$. 

We claim that the graph $G_k$ restricted on $V_k$ is transitive for all $k=1,\cdots, n$. i.e.
$$
\forall k=1,\cdots, n, \, \forall v\in V_k, v \text{ is a proper intermediate in } G_k
$$
where a vertex $v$ is called a proper interdediate in G  if 
$$
\forall a\in N^-(v), \forall b\in N^+(v): (a,b)\in V
$$


For $k=1,2,3$, our claim is obviouly true. The induction step for $k\ge 4$ is as follows:

Suppose $G_{k-1}$ is transitive, we show that $G_k$ is transitive by contradiction. Suppose $G_k$ fails to be transitive. There are two cases:

* The transitivity breaks at intermidiate vertex $v_k$. i.e.
  $$
  \exists a,b\in V_{k-1} \text{ s.t. } 
  (a,v_k),(v_k, b)\in E_k 
  \: \and \:
  (a,b)\notin E_k
  $$
  Note that  $(a,v_k),(v_k, b)\in E_{k-1}$ in particular since we won't add egdes directly connecting $v_k$.  The pre-existence of $(a,v_k),(v_k, b)\in E_{k-1} $ leads to contradiction since $(a,b)$ is guranteed to be added to $E_k$ at iteration $k$.

* The transitivity breaks at some already visited vertex $w$. i.e.
  $$
  \exists a,w,b\in V_{k-1} \text{ s.t. } 
  (a,w),(w, b)\in E_k 
  \: \and \:
  (a,b)\notin E_k
  $$
  From induction hypothesis, we know that $G_{k-1}$ is transitive. Hence, at least one of $\{(a,w),(w, b)\}$ is added at iteration $k$. 

  * If both $(a,w),(w, b)$ are added at iteration $k$, then $(a,v_k),\, (v_k, ),\, (a,v_k),\, (a,v_k),\,$



For numerical calculation on adjacency matrix, the above algorithm becomes

```pseudocode
function warshall(M):
    for k in 1...n:			
        for i in 1...n:		// set of incoming nodes of v[k]
        	for j in 1...n:	// set of outgoing nodes of v[k]
            	M[i,j] = M[i,j] or (M[i,k] and M[k,j])
            end for
        end for
    end for
  
```



Example:

