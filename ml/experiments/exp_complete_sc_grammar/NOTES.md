# Experiment Notes: [F624] Complete SC Grammar

## Overview
**Objective**: Build a complete Statechart JSON grammar derived directly from the `statecharts.proto` and `expressions.proto` definitions.
**Date**: 2026-01-05
**Status**: Completed

## Implementation Details
- **Source of Truth**: `proto/statecharts/v1/statecharts.proto`
- **Parser**: Custom Python script (`generate_grammar.py`) that uses regex to extract message and enum definitions.
- **Output**: `full_grammar.json` (JSON Schema compatible).
- **Coverage**:
  - Parsed 12 core message types (Statechart, State, Transition, Event, Guard, Action, etc.).
  - Handling of `oneof` and `map` fields via object properties.
  - Integration of `expressions.proto` for Guard/Action logic.

## Verification Results
- **Generation**: Successfully generated `full_grammar.json`.
- **Metrics**:
  - Total Definitions: 12
  - Appoximate References: 20
- **Validation**: The generated JSON structure matches the schema requirements for `ConstrainedSampler`.

## Future Work
- Integrate with `protoc` for more robust parsing if proto complexity increases.
- Add unit tests for specific edge cases in proto syntax (nesting, options).
