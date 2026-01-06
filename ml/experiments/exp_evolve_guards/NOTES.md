# Experiment: Guard Evolution (`exp_evolve_guards`)

**Goal**: Evolve boolean expressions (guards) that act as predicates for transitions, correctly predicting whether a transition fires based on context.

## Problem Statement
Given traces $T = \{(c, \text{fired})_i\}$, find a boolean expression $G$ such that $G(c) == \text{fired}$.

## Genome Representation
Binary Expression Tree (AST):
*   **Nodes**:
    *   `LogicalOp`: AND, OR, NOT
    *   `ComparisonOp`: <, >, <=, >=, ==, !=
    *   `Variable`: context variable
    *   `Constant`: numeric or boolean

## Mutation Operators
1.  **Modify Constant/Variable**: `x > 5` -> `x > 6` or `y > 5`.
2.  **Modify Operator**: `>` -> `>=`.
3.  **Grow**: Replace leaf with subtree `x` -> `x > 5`.
4.  **Shrink**: Replace subtree with leaf `(x > 5 && y < 2)` -> `x > 5`.

## Fitness Function
$Fitness = w_1 \cdot Accuracy + w_2 \cdot Parsimony$

1.  **Accuracy**: Precision/Recall on predicting transition firing.
2.  **Parsimony**: $1 / (1 + tree\_size)$.

## Test Scenarios
1.  **Metric Threshold**: `count > 5`
2.  **Boolean Flag**: `is_ready == True`
3.  **Compound**: `count > 5 && is_ready`
4.  **Range**: `x > 10 && x < 20`
