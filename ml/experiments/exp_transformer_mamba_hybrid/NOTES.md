
# Experiment: Transformer Mamba Hybrid
## Goal
Verify a hybrid architecture where a Transformer "compiles" traces into Mamba state-space matrices for efficient execution.

## Results
- **Compilation**: Produced valid A (Norm 4.58) and B matrices from traces.
- **Execution**: MambaExecutor successfully stepped through 50 events using compiled matrices.

## Verification Status
- [x] Passed. Hybrid compiler-executor pipeline functional.
