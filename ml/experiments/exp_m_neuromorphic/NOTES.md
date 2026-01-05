# Experiment M: Neuromorphic Statecharts

## Core Insight

Statecharts and Spiking Neural Networks (SNNs) share deep structural similarities.
A properly wired SNN **IS** a statechart, not an approximation of one.

## The Isomorphism

| Statechart | Neural Equivalent |
|------------|-------------------|
| Event | Spike (binary pulse) |
| State | Attractor (reverberating cluster) |
| Transition | Synaptic wiring (hard connections) |
| Guard | Shunting inhibition |
| OR-state | Lateral inhibition (winner-take-all) |
| AND-state | Parallel circuits (no inhibition) |
| History | Synaptic trace (STDP-like decay) |

## Architecture

### SpikingNeuron
- Leaky Integrate-and-Fire (LIF) dynamics
- Membrane potential with leak and threshold
- Refractory period after spiking

### SpikingState
- Cluster of 4 neurons with bistable dynamics
- Sustaining current maintains activation
- Explicit activation/inhibition control

### SpikingTransition
- Synaptic wiring between state clusters
- Event-triggered (spike matching)
- Inhibits source, activates target

### SpikingStatechart
- OR-groups with lateral inhibition
- History via synaptic trace decay

## Test Results

### 3-State Cycle Test (A → B → C → A)
```
Accuracy: 100% (6/6 transitions)

NEXT #1: A -> B [OK]
NEXT #2: B -> C [OK]
NEXT #3: C -> A [OK]
NEXT #4: A -> B [OK]
NEXT #5: B -> C [OK]
NEXT #6: C -> A [OK]

[PASS] Spike patterns match statechart semantics!
```

### Lateral Inhibition Test (OR-State)
```
Activating S1 and S2 simultaneously...
t=15: S1 wins (0.57 activity) vs S2 (0.00)

[PASS] Lateral inhibition enforces mutual exclusion
```

### History Trace Test
```
After visiting A -> B -> C:
  A: 0.358 (oldest)
  B: 0.630 (middle)
  C: 1.000 (most recent)

History traces correctly reflect recency.
```

## Key Findings

1. **Statecharts map directly to neural circuits** - no learning required
2. **OR-state semantics emerge from lateral inhibition** - winner-take-all
3. **State persistence via bistable attractors** - self-sustaining activity
4. **Transitions are synaptic wiring** - hard-coded, not learned
5. **History is synaptic trace decay** - STDP-like memory

## Implications

1. **Neuromorphic hardware**: Statecharts could run on neuromorphic chips (Intel Loihi, IBM TrueNorth)
2. **Energy efficiency**: Spike-based computation is extremely low power
3. **Connection to HTM**: SDR encoding (exp_i) + spiking dynamics
4. **Biological plausibility**: State machines may exist in cortical circuits

## Connection to HTM (exp_i)

| HTM Concept | Neuromorphic Statechart |
|-------------|-------------------------|
| SDR (Sparse Distributed Rep) | Spike pattern across neurons |
| Temporal pooling | State cluster persistence |
| Prediction | Transition anticipation |
| Column structure | State neuron clusters |

## Comprehensive Test Suite Results

### test_equivalence.py - ALL PASS (17/17 tests)

| Level | Category | Tests | Status |
|-------|----------|-------|--------|
| **Level 1** | Mathematical Equivalence | 5 tests | ✅ ALL PASS |
| **Level 2** | Behavioral Equivalence | 5 tests | ✅ ALL PASS |
| **Level 3** | Emergent Properties | 3 tests | ✅ ALL PASS |
| **Level 4** | STDP Learning | 2 tests | ✅ ALL PASS |
| **Level 5** | Real-World Applications | 2 tests | ✅ ALL PASS |

#### Level 1: Mathematical Equivalence (Bijection Proofs)
- **State ↔ Neuron Population**: Bijection valid (injective, surjective, invertible)
- **Event ↔ Spike**: Isomorphism (discrete, triggering, information-carrying)
- **Transition ↔ Synapse**: Correspondence (transitions ↔ non-zero weights)
- **OR-State ↔ Lateral Inhibition**: Winner-take-all satisfies OR constraint
- **AND-State ↔ Parallel Circuits**: No inhibition preserves all-active

#### Level 2: Behavioral Equivalence
- **Cycle (1K steps)**: 100% accuracy
- **OR-State**: Mutual exclusion enforced
- **AND-State**: 100% independence
- **Guard**: Context-dependent transitions work
- **History**: Synaptic traces reflect recency

#### Level 3: Emergent Properties (SNN Advantages)
- **Graceful Degradation**: 100% accuracy even with 50% noise
- **Temporal Integration**: Different spike patterns (9 vs 5 spikes)
- **Population Voting**: 50% neuron failure, 100% function

#### Level 4: STDP Learning
- **Basic STDP**: LTP (strengthen) and LTD (weaken) both work
- **Transition Learning**: A→B strengthened (0.5→2.0), B→A weakened (0.5→0.0)

#### Level 5: Real-World
- **Traffic Light**: Red→Green→Yellow cycle perfect
- **Door Lock**: Guard-protected unlock with code verification

## Commands

```bash
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml

# Run basic demo
.venv/bin/python3 experiments/exp_m_neuromorphic/spiking_statechart.py

# Run comprehensive test suite
.venv/bin/python3 experiments/exp_m_neuromorphic/test_equivalence.py
```

## Future Directions

1. **Hierarchical states**: Nested attractors with gating
2. **Timing constraints**: Temporal guards via spike timing
3. **Learning transitions**: STDP for adaptive wiring
4. **Neuromorphic deployment**: Compile to Loihi/SpiNNaker

## References

- Harel, D. (1987). Statecharts: A visual formalism for complex systems
- Maass, W. (1997). Networks of spiking neurons
- Hawkins, J. (2004). On Intelligence (HTM foundations)
- Intel Loihi: Neuromorphic processor architecture
