# Experiment: Context Evolution (`exp_evolve_context`)

**Goal**: Evolve a minimal "extended state" (context schema) that explains observed state transitions which are non-deterministic in the discrete state space.

## Problem Statement
Given observations $O = \{(s, e) \to s' \}$, if $(s, e)$ maps to multiple different $s'$, we need hidden variables to disambiguate. Find minimal context schema $C$ such that $f(s, e, c) \to s'$ is deterministic.

## Genome Representation
*   **Genome**: `ContextSchema`
*   **Variables**: Dictionary `{name: type}`
    *   Types: `bool`, `int`, `enum(N)`
*   **Initial Values**: `{name: value}`

## Mutation Operators
1.  **Add Variable**: Add a new variable (e.g., `toggle: bool`, `count: int`).
2.  **Remove Variable**: Remove an unused variable.
3.  **Change Type**: Change `int` to `bool` (or vice-versa).
4.  **Rename**: (Semantic only, doesn't affect fitness).

## Fitness Function
$Fitness = w_1 \cdot Explainability + w_2 \cdot Parsimony$

1.  **Explainability**: Can we distinguish the conflicting transitions using this schema?
    *   For each conflict group (same $s, e$, different $s'$), can we assign distinct values of $C$ to each outcome?
2.  **Parsimony**: $1 / (1 + |variables|)$. Fewer variables are better.

## Test Scenarios
1.  **Toggle**: Alternating outcomes A -> B, A -> C, A -> B. Needs 1 bit (bool).
2.  **Counter**: A -> A (9 times) -> B. Needs integer or counter.
3.  **History**: Outcome depends on previous state. Needs `last_state` variable.
4.  **Redundant**: Schema has 5 vars but only 1 needed. Evolution should prune.
