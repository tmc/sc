# Experiment: Action Evolution (`exp_evolve_actions`)

**Goal**: Evolve action sequences (assignments) that correctly transform a `before` context into an `after` context.

## Problem Statement
Given a set of transitions $T = \{(c_{before}, c_{after})_i\}$, find a program $P$ (sequence of actions) such that $\forall i, P(c_{before}) = c_{after}$.

## Genome Representation
We represent an action sequence as a list of abstract syntax tree (AST) nodes for assignments.

*   **Genome**: `List[Assignment]`
*   **Assignment**: `TargetVar = Expression`
*   **Expression**:
    *   `Constant(value)`
    *   `Variable(name)`
    *   `BinaryOp(left, op, right)`

## Mutation Operators
1.  **Modify Constant**: Change a numeric constant (e.g., `1` -> `2`, `1` -> `-1`).
2.  **Modify Variable**: Change a variable reference (e.g., `x` -> `y`).
3.  **Modify Operator**: Change binary operator (e.g., `+` -> `-`, `*` -> `/`).
4.  **Insert Assignment**: Add a new random assignment to the sequence.
5.  **Delete Assignment**: Remove an assignment.
6.  **Swap Assignments**: Reorder statements (important if variables depend on each other).

## Fitness Function
$Fitness = w_1 \cdot Accuracy + w_2 \cdot Parsimony$

1.  **Accuracy**: Proportion of test cases where the transformed context matches the target context exactly.
2.  **Parsimony**: $1 / (1 + length(genome))$. Shorter programs are preferred.

## Test Scenarios
1.  **Increment**: `count = count + 1`
2.  **Reset**: `count = 0`
3.  **Arithmetic**: `y = x * 2 + 1`
4.  **Swapping**: `temp = a; a = b; b = temp` (requires specific order)
5.  **Accumulate**: `total = total + price * quantity`
