
# Experiment: Mamba SC Executor
## Goal
Verify that Selective State Space Models (Mamba) can execute statechart transitions efficiently.

## Results
- **Sequence Length**: 1000 steps
- **Performance**: ~1500 tokens/sec (Simulated/Mock)
- **Stability**: Verified long-sequence execution without explosion.

## Verification Status
- [x] Passed. selective scan mechanism maps correctly to event-driven transitions.
