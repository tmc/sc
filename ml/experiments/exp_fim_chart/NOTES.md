# Experiment: Structure Infilling (FIM)

## Goal
Demonstrate that the model can generate valid structural components (transitions, states) when given a prefix and a suffix context, effectively "filling in the middle" of a Statechart definition.

## Hypothesis
By providing a JSON prefix (e.g., up to the `transitions` key) and using grammar constraints to ensure the generated tokens form valid `Transition` objects, we can insert semantic logic into a gap.

## Methodology
1.  **Prefix**: `{"root_state": {...}, "transitions": [`
2.  **Target**: Generate N valid transitions (e.g., `{"from": ..., "to": ..., "event": ...}`).
3.  **Suffix**: `]}`
4.  **Verification**: Concatenate Prefix + Generated + Suffix and parse as JSON. Check if transitions are valid SC transitions.

## Results
(To be filled)
