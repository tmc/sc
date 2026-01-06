# exp_continuous_control: Research Notes

## Key Insight

**Statecharts separate "what to do" from "how to do it"**:
- Discrete states: High-level behavior modes (APPROACH, GRASP, LIFT)
- Continuous controllers: Low-level execution within each mode
- Guards: Continuous conditions trigger mode switches

This is the essence of hybrid automata applied to robotics.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID STATECHART                        │
├─────────────────────────────────────────────────────────────┤
│  ┌────────┐    ┌──────────┐    ┌───────┐    ┌──────┐       │
│  │  IDLE  │───▶│ APPROACH │───▶│ GRASP │───▶│ LIFT │──▶DONE│
│  └────────┘    └──────────┘    └───────┘    └──────┘       │
│       │              │              │            │          │
│       ▼              ▼              ▼            ▼          │
│  ┌────────┐    ┌──────────┐    ┌───────┐    ┌──────┐       │
│  │  Stay  │    │  Move to │    │ Close │    │ Move │       │
│  │  still │    │  object  │    │gripper│    │  up  │       │
│  └────────┘    └──────────┘    └───────┘    └──────┘       │
│  Continuous     Continuous     Continuous    Continuous     │
│  Controller     Controller     Controller    Controller     │
└─────────────────────────────────────────────────────────────┘
```

## What Worked

1. **State-specific controllers** - Each discrete state has its own continuous controller. Simple, modular, interpretable.

2. **Continuous guards** - Transitions fire when continuous conditions met (distance < threshold, force > threshold). Natural interface between discrete and continuous.

3. **AND-states for limbs** - Multiple limbs run in parallel, each with own state machine. High-level state = conjunction of limb states.

4. **Simple 2D simulation** - No MuJoCo needed. 2D environment sufficient to demonstrate concepts.

## What Didn't Work

1. **Fully learned transitions** - Tried learning when to transition but unstable. Rule-based guards with learned controllers is more robust.

2. **Single monolithic controller** - Tried one big neural net for all states. Lost interpretability, harder to debug.

3. **Complex physics** - Started with 3D, friction, etc. Unnecessary complexity for demonstrating hybrid control concepts.

## State Transition Guards

| Transition | Guard Condition |
|------------|-----------------|
| IDLE → APPROACH | Always (start signal) |
| APPROACH → GRASP | distance(robot, object) < 0.05 |
| GRASP → LIFT | gripper > 0.9 AND force > 0.3 |
| LIFT → DONE | robot.y > 0.4 |
| LIFT → ERROR | force < 0.1 AND gripper > 0.5 (dropped) |

## AND-States (Parallel Composition)

```
ParallelLimbController
├── Limb 0: READY → MOVING → CONTACT → HOLDING
└── Limb 1: READY → MOVING → CONTACT → HOLDING

High-level transition when ALL limbs reach CONTACT.
```

## Connections to Other Experiments

| Experiment | Connection |
|------------|------------|
| `exp_transition_priorities` | When multiple transitions enabled, priority resolves |
| `exp_guard_synthesis` | Guards could be learned, not hand-coded |
| `exp_temporal_guards` | Add time-based conditions (timeout, rate limit) |
| `exp_learnable_policies` | Learn controllers via RL |
| `exp_action_composition` | Continuous actions compose within states |

## Future Directions

1. **Learn transition guards** - Currently hand-coded. Could evolve or learn from demonstrations.

2. **Hierarchical control** - Nest statecharts: high-level (pick, place, stack) contains low-level (approach, grasp, lift).

3. **Safety constraints** - Add invariants that must hold within states (joint limits, collision avoidance).

4. **Skill library** - Reusable state-controller pairs that can be composed.

5. **Sim-to-real transfer** - How well do learned hybrid policies transfer to real robots?

## Code Patterns

```python
# Hybrid step: discrete transition + continuous action
def step(self, world):
    # Check discrete transitions
    for trans in self.transitions:
        if trans.source == self.current_state and trans.guard(world):
            self.current_state = trans.target
            break

    # Get continuous action from current state's controller
    return self.controllers[self.current_state].compute_action(world)
```

## Performance

Simple pick-and-lift task:
- Success rate: 100% (with hand-coded controllers)
- Steps to completion: ~200-300 (at dt=0.02)
- Time in each state: APPROACH ~2s, GRASP ~0.5s, LIFT ~1.5s

## Pure MLX

No numpy, no MuJoCo, no external physics engines.
- MLX arrays for state representation
- MLX neural networks for learned controllers
- Pure Python for simulation dynamics

## Date
2024-01-04
