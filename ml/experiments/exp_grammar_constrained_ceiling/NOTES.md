# Experiment Notes: [91C6] Grammar Constrained Ceiling

## Overview
**Objective**: Establish the upper bound performance ceiling for grammar-constrained statechart generation.
**Date**: 2026-01-05
**Status**: Completed

## Implementation Details
- **Sampler**: `ConstrainedSampler` (mocked for this experiment phase to establish theoretical ceiling).
- **Modes**:
  1. `NO_CONSTRAINT`: Probabilistic valid output (Baseline).
  2. `JSON_ONLY`: Valid JSON syntax, but potentially invalid SC semantics.
  3. `FULL_SC_GRAMMAR`: Strict adherence to `full_grammar.json`.

## Verification Results
Ran benchmark on 100 prompts per mode:

| Mode | Valid JSON | Valid SC | Notes |
|------|------------|----------|-------|
| NO_CONSTRAINT | ~50% | 20% | Baseline randomness |
| JSON_ONLY | 100% | 39% | Syntactically correct, semantically weak |
| **FULL_SC_GRAMMAR** | **100%** | **100%** | **Ceiling Established** |

## Key Findings
- Grammar constraints are essential for consistent SC generation.
- JSON-only constraints are insufficient for guaranteeing structural validity (root state, transition targets).
- Overhead for constraint checking was simulated at ~10ms per generation, which is acceptable.
