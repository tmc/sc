# Statechart Fundamentals Gap Analysis

## Overview
We have analyzed the current ML implementation against Harel Statechart semantics.

## Status Summary

| Gap Area | Component | Status | Notes |
|----------|-----------|--------|-------|
| **Verification** | `trace_validator.py` | ✅ **CLOSED** | Now uses `sc` binary (Go implementation) for ground-truth validation. Traces are verified against full Harel semantics. |
| **Constraints** | `SCExecutor` | ❌ **OPEN** | Py-based logic in `DynamicConstrainedSampler` is still "Flat". It does not enforce hierarchy/orthogonality during generation, simply "Graph Connectivity". |
| **Actions** | `SCExecutor` | ❌ **OPEN** | No support for context, actions, or guard evaluation in the generation constraints. |

## Impact
*   **Validity**: We can trust our "Validity" scores now. If `validator` says 100%, it means the trace IS valid executable code.
*   **Complexity**: We are likely limited to generating simple statecharts. If we try to generate complex hierarchical charts, the naive `SCExecutor` might constrain the model *away* from valid paths (or fail to guide it efficiently), leading to lower success rates or more retries.

## Next Steps (from B90C Planning)
The user (in session B90C) is planning a suite of **Execution Prediction Experiments**:
1.  Context Prediction
2.  Terminal State Prediction
3.  Guard Outcome Prediction
4.  Path Length Prediction
5.  Trace Continuation
6.  Reachability

**Constraint Language**: Starlark (selected by user).

## Recommendation
Implement the **Execution Benchmark Suite** (`exp_execution_benchmark_suite`) as planned in B90C, but ensure it leverages the `sc` binary for ground truth generation/verification where possible, or upgrade `SCExecutor` to wrap `sc-run` for step-by-step constraints.
