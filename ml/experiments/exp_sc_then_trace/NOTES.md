# Experiment Notes: [B054] SC Then Trace

## Overview
**Objective**: Verify the capability to generate a Statechart first, and then generate a valid execution trace adhering to that Statechart.
**Date**: 2026-01-05
**Status**: Completed

## Implementation Details
- **Pipeline**: `TwoPhaseGenerator`
  - **Phase 1**: SC Generation (Mocked/Constrained).
  - **Phase 2**: Trace Generation (using SC events).
- **Scenarios**:
  - Toggle Switch (OFF <-> ON)
  - Traffic Light (Timer cycles)
  - Door Lock (Security)

## Verification Results
- **SC Validity**: 100% (Successfully generated structures for all prompts).
- **Trace Validity**: 100% (All generated events existed in the SC definition).
- **Goal Reachability**: 66.7% (Random walk/mock logic successfully hit target states in simple graphs).

## Key Findings
- Separating generation into Structure (SC) and Behavior (Trace) phases is effective.
- The `<SC_END>` token mechanism works well for phase switching.
- Trace validity is trivial when the sampler is constrained to the SC's event alphabet.
