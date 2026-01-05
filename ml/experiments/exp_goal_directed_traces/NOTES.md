# Experiment Notes: [C9F0] Goal Directed Traces

## Overview
**Objective**: Generate traces that optimally reach a specified goal state within a Statechart.
**Date**: 2026-01-05
**Status**: Completed

## Implementation Details
- **Goal Types**:
  - `TargetState`: Reach a specific label.
  - `Sequence`: Reach A then B.
  - `Avoidance`: Reach A avoiding X.
- **Algo**: `PathFinder` using BFS for shortest path discovery.

## Verification Results
Tested on Traffic Light + Emergency SC:
- **Reachability**: 100% (Found paths to 'YELLOW' and 'EMERGENCY').
- **Optimality**: 100% (Breadth-First Search guaranteed shortest path).

## Key Findings
- Goal-directed generation requires lookahead or planning (BFS/A*).
- Simple constrained sampling (random walk) is inefficient for specific goals.
- Future integration should use the `PathFinder` to guide the logits of the LLM (e.g., masking invalid branches).
