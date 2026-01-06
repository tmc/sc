# exp_partial_observability - Research Notes

## Core Insight

**Statechart AND-states (parallel regions) naturally represent factored belief states in POMDPs.**

Each orthogonal region tracks beliefs about ONE hidden dimension:
- Region 1: P(opponent_hand | observations)
- Region 2: P(opponent_strategy | observations)
- Region 3: P(deck_state | observations)

The CONFIGURATION of active states across regions IS the joint belief state.

## Architecture

```
Observations (opponent actions)
        ↓
   Belief Tracker
   (Bayesian updates per dimension)
        ↓
   Parallel Belief State
   ├── OpponentHandBelief (OR): weak | medium | strong
   ├── OpponentStrategyBelief (OR): passive | aggressive | bluffer
   └── DeckStateBelief (OR): high_rich | balanced | low_rich
        ↓
   Belief Configuration
   (active states = joint belief)
        ↓
   Belief-Conditioned Policy
   (guards check belief configuration)
        ↓
   Selected Action
```

## Key Connection: Statecharts ↔ POMDPs

| POMDP Concept | Statechart Equivalent |
|---------------|----------------------|
| Belief state b(s) | Configuration across AND-regions |
| Belief update | Observation-triggered transitions |
| Hidden state space | Union of states across regions |
| Observation model | Observation likelihood in guards |
| Policy π(b,a) | Transition guards on belief config |

## What Worked

### 1. Factored Belief via AND-States
- Each dimension independently tracked
- Bayesian updates per dimension
- Joint configuration for policy conditioning
- Much simpler than monolithic belief tracking

### 2. Discrete Belief Levels
- Continuous P → discrete levels (weak/medium/strong)
- Easier to learn guards over discrete states
- Still captures essential belief structure
- Reduces state space explosion

### 3. Observation Models per Dimension
- P(action | hand_strength) learned/specified
- P(action | strategy_type) learned/specified
- Bayesian updates are standard
- Easy to extend with new dimensions

### 4. Evolution of Belief-Conditioned Policy
- Policy is set of belief-guarded transitions
- Evolution discovers optimal guards
- Self-play provides training signal
- Bluffing detection EMERGES

## What Didn't Work

### 1. Full Probability Tracking
- Tracking exact P(h) for all hidden states
- Expensive, unnecessary for decision making
- Discretization to levels works better

### 2. Single-Region Belief
- Putting all beliefs in one OR-state
- Exponential blowup (3 dims × 3 values = 27 states)
- Factored AND-states scale linearly

### 3. History-Based Policy
- Conditioning on raw observation history
- Misses belief structure
- Harder to generalize

## Connections to Other Experiments

### exp_poker_bluff
- Source of poker game mechanics
- We extend: add belief tracking over hidden cards
- They evolve bluff strategies
- We evolve belief-conditioned strategies

### exp_action_side_effects
- Actions affect context → beliefs
- Observations update beliefs
- Both: context/belief-conditioned decisions

### exp_guard_synthesis
- Synthesize guards from examples
- Here: guards condition on belief configuration
- Could synthesize belief guards automatically

### exp_deep_history
- History captures past states
- Beliefs capture uncertainty about current state
- Complement: history for known, beliefs for unknown

### exp_formal_verification
- Verify properties of belief-conditioned policy
- e.g., "never fold when belief says opponent bluffing"
- SMT over belief-guarded transitions

## Key Metrics

| Metric | Description |
|--------|-------------|
| Belief Accuracy | Error in P(hidden) estimation |
| Policy Profit | Expected profit per hand |
| Certainty | 1 - normalized entropy |
| Update Speed | Time for Bayesian update |
| Convergence | Generations to good policy |

## Belief Update Formula

Standard Bayesian:
```
P(hidden | obs) = P(obs | hidden) × P(hidden) / P(obs)
```

Per dimension:
```python
for i, value in enumerate(dimension.possible_values):
    prior = dimension.distribution[i]
    likelihood = observation_model.get_likelihood(obs, value)
    posterior[i] = prior * likelihood

posterior = normalize(posterior)
```

## Guard Expressions

Belief guards are boolean over dimension states:
```
if opponent_hand IN {strong, very_strong}
   AND opponent_strategy = aggressive
   AND certainty >= 0.6
then FOLD
```

These compile to statechart transition guards.

## Future Directions

### 1. Hierarchical Beliefs
- Beliefs about beliefs (meta-uncertainty)
- Recursive AND-states
- Model opponent's model of us

### 2. Continuous Belief SAE
- SAE features represent belief dimensions
- Learn belief structure from data
- No manual dimension design

### 3. Multi-Agent Beliefs
- Each agent has belief state
- Beliefs about others' beliefs
- Game-theoretic equilibria

### 4. Online Belief Learning
- Learn observation models from experience
- Adapt to new opponent types
- Continuous belief model refinement

### 5. Temporal Belief Patterns
- Beliefs change predictably over time
- Model belief dynamics
- Plan over belief trajectories

## Open Questions

1. How many belief dimensions are needed?
2. Optimal discretization granularity?
3. Can we learn dimension structure?
4. How to handle belief about continuous quantities?
5. Efficient belief propagation in deep hierarchies?

## Implementation Notes

### Belief Dimension
```python
@dataclass
class BeliefDimension:
    name: str
    possible_values: List[str]
    distribution: List[float]  # P(value) for each value

    def update_bayesian(self, likelihood: Dict[str, float]):
        for i, value in enumerate(self.possible_values):
            self.distribution[i] *= likelihood.get(value, 0.5)
        self._normalize()
```

### Parallel Belief State
```python
@dataclass
class ParallelBeliefState:
    name: str
    dimensions: List[BeliefDimension]  # AND-state regions

    def get_configuration(self) -> BeliefConfiguration:
        active = {dim.get_active_state() for dim in self.dimensions}
        return BeliefConfiguration(active_states=frozenset(active))
```

### Policy Guard
```python
@dataclass
class BeliefPattern:
    dimension_values: Dict[str, Set[str]]  # Required state per dimension

    def matches(self, belief: ParallelBeliefState) -> bool:
        for dim in belief.dimensions:
            if dim.name in self.dimension_values:
                if dim.get_active_state() not in self.dimension_values[dim.name]:
                    return False
        return True
```

## References

- Kaelbling, L.P. "Planning and Acting in Partially Observable Domains" (1998)
- Harel, D. "Statecharts: A Visual Formalism for Complex Systems" (1987)
- Silver, D. "Monte-Carlo Tree Search for POMDPs" (2010)
- Hausknecht, M. "Deep Recurrent Q-Learning for Partially Observable MDPs" (2015)
