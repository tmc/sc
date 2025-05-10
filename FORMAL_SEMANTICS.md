# Formal Semantics of Statecharts

This document provides a formal mathematical definition of the semantics of statecharts as implemented in this library. The formalism follows the approach outlined in Harel's original paper [1] and subsequent formalizations [2,3].

## Basic Definitions

### Statechart Structure

A statechart $SC$ is formally defined as a tuple:

$$SC = (S, \rho, \psi, \delta, \gamma, \lambda, \sigma_0)$$

Where:
- $S$ is a finite set of states
- $\rho \subseteq S \times S$ is the hierarchy relation where $(s, s') \in \rho$ indicates $s'$ is a substate of $s$
- $\psi: S \rightarrow \{BASIC, NORMAL, PARALLEL\}$ is a function that assigns a type to each state
- $\delta \subseteq S \times E \times G \times A \times S$ is the transition relation where $E$ is the set of events, $G$ is the set of guards, and $A$ is the set of actions
- $\gamma: S \rightarrow A^* $ maps states to entry/exit actions
- $\lambda: S \rightarrow \mathbb{P}(S)$ maps OR-states to their default substate
- $\sigma_0 \in \mathbb{P}(S)$ is the initial configuration

### Configuration Semantics

A configuration $\sigma \subseteq S$ is a set of states that satisfies the following properties:

1. **Root inclusion**: The root state $r \in \sigma$
2. **Parent inclusion**: $\forall s \in \sigma, s \neq r \implies \exists s' \in \sigma: (s', s) \in \rho$
3. **OR-state child inclusion**: $\forall s \in \sigma, \psi(s) = NORMAL \implies |\{s' \in \sigma : (s, s') \in \rho\}| = 1$
4. **AND-state children inclusion**: $\forall s \in \sigma, \psi(s) = PARALLEL \implies \{s' : (s, s') \in \rho\} \subseteq \sigma$

### Step Semantics

Given a configuration $\sigma_i$ and an event $e$, the next configuration $\sigma_{i+1}$ is determined by:

1. **Enabled transitions**: A transition $t = (s_{src}, e, g, a, s_{tgt}) \in \delta$ is enabled in $\sigma_i$ if:
   - $s_{src} \in \sigma_i$
   - The guard $g$ evaluates to true

2. **Conflict resolution**: If multiple transitions are enabled, conflict resolution is applied:
   - Priority is given to transitions originating from deeper states in the hierarchy
   - For transitions at the same hierarchy level, source state order is used

3. **Transition execution**:
   - Exit states from $\sigma_i$ that are not in $\sigma_{i+1}$, in reverse hierarchical order
   - Execute transition actions
   - Enter states in $\sigma_{i+1}$ that are not in $\sigma_i$, in hierarchical order
   - Compute default completions for any OR-states without an active substate

## Orthogonal Regions

For a state $s$ with $\psi(s) = PARALLEL$ (also known as ORTHOGONAL), all child states are active simultaneously. The semantics of parallel state execution follows these principles:

1. **Concurrent execution**: Events are processed concurrently in all orthogonal regions
2. **Synchronization**: The step is complete only when all regions have processed the event
3. **Cross-region transitions**: Transitions can cross region boundaries

## Extensions

### History States

For a state $s$ with a history state $h$, the history mechanism is defined as:

$$\lambda_H(s, \sigma) = \begin{cases}
  \{s' \in \sigma : (s, s') \in \rho\} & \text{if} \exists s' \in \sigma : (s, s') \in \rho \\
  \lambda(s) & \text{otherwise}
\end{cases}$$

### Event Processing

The event processing semantics follows a run-to-completion model where:

1. An event is processed completely before another event is considered
2. A step may involve multiple microsteps if internal events are generated
3. The system reaches a stable configuration after processing an event

## References

[1] D. Harel, "Statecharts: A Visual Formalism for Complex Systems," Science of Computer Programming, vol. 8, no. 3, pp. 231-274, 1987.

[2] M. von der Beeck, "A Structured Operational Semantics for UML-Statecharts," Software and Systems Modeling, vol. 1, no. 2, pp. 130-141, 2002.

[3] E. Mikk, Y. Lakhnech, M. Siegel, "Hierarchical Automata as Model for Statecharts," in Advances in Computing Science - ASIAN'97, pp. 181-196, 1997.