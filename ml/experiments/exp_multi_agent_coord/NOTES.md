# exp_multi_agent_coord: Research Notes

## Key Insight

**Statecharts + shared event bus = emergent communication protocols**

Each agent has its own statechart (discrete behavior modes), but they share:
1. A common event bus (broadcast + targeted signals)
2. An evolved protocol genome (what signals mean, when to send them)
3. Coordination tasks requiring cooperation

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     SHARED EVENT BUS                              │
│   [LEADER_CLAIM] [FOLLOW_ME] [SIGNAL_A] [TARGET_SPOTTED]         │
└────────────┬────────────┬────────────┬────────────┬──────────────┘
             │            │            │            │
     ┌───────▼───┐  ┌─────▼─────┐  ┌───▼─────┐  ┌───▼─────┐
     │  Agent 0  │  │  Agent 1  │  │ Agent 2 │  │ Agent 3 │
     │ LEADER    │  │ FOLLOWER  │  │FOLLOWER │  │ SCOUT   │
     │  ┌─────┐  │  │  ┌─────┐  │  │ ┌─────┐ │  │ ┌─────┐ │
     │  │LEAD │  │  │  │FOLL │  │  │ │FOLL │ │  │ │EXPL │ │
     │  └──┬──┘  │  │  └──┬──┘  │  │ └──┬──┘ │  │ └──┬──┘ │
     │     │     │  │     │     │  │    │    │  │    │    │
     └─────┼─────┘  └─────┼─────┘  └────┼────┘  └────┼────┘
           │              │             │            │
           ▼              ▼             ▼            ▼
       [APPROACH]     [FOLLOW]      [FOLLOW]     [EXPLORE]
```

## What Worked

1. **Shared protocol genome** - All agents in a team use same protocol. Creates consistent conventions without explicit coordination.

2. **Event bus with time window** - Events persist for 1.0 time units. Agents can react asynchronously.

3. **Emergent roles** - LEADER/FOLLOWER/SCOUT emerge from initial conditions + evolved role assignment rules.

4. **Co-evolution** - Agents evolve together. No adversarial dynamics, just cooperation pressure.

5. **Signal meanings** - Evolution assigns meanings to signals (SIGNAL_A = "move to sender"). Creates implicit language.

## What Didn't Work

1. **Independent evolution** - Tried evolving each agent separately. Created incompatible protocols.

2. **Too many signals** - Started with 8 signal types. Evolution couldn't explore effectively. Reduced to 4.

3. **Explicit role assignment** - Hand-coding "agent 0 = leader" removes emergence. Evolution finds better assignments.

4. **Complex timing** - Started with variable cooldowns per signal. Simple fixed cooldown (0.5s) works better.

## Protocol Genome

```python
@dataclass
class ProtocolGenome:
    signal_meanings: Dict[EventType, SignalMeaning]  # What signals mean
    role_conditions: List[RoleCondition]             # How to assign roles
    transition_probs: Dict[Tuple[Mode, Mode], float] # State transitions
    signal_cooldown: float = 0.5                     # Time between signals
    leader_claim_prob: float = 0.1                   # Chance to claim leader
```

## Evolved Conventions (Examples)

| Signal | Evolved Meaning |
|--------|----------------|
| SIGNAL_A | MOVE_TO_SENDER |
| SIGNAL_B | WAIT |
| SIGNAL_C | FOLLOW |
| SIGNAL_D | EXPLORE |

| Role | Evolved Condition |
|------|------------------|
| LEADER | FIRST_TO_SIGNAL (claim leadership first) |
| FOLLOWER | MOST_CENTRAL (closest to group center) |
| SCOUT | RANDOM (assigned probabilistically) |

## Connections to Other Experiments

| Experiment | Connection |
|------------|------------|
| `exp_hide_seek_evolution` | Extends 2-agent to N-agent. Similar turn-taking ideas |
| `exp_transition_priorities` | Conflict when multiple agents want same role |
| `exp_continuous_control` | Agents have continuous actions within discrete modes |
| `exp_temporal_guards` | Signal cooldowns = temporal constraints |
| `exp_action_composition` | Multiple agents' actions compose |

## Baselines Comparison

| Controller | Description |
|------------|-------------|
| Random | Random velocities, no communication |
| Rule-based | Hand-coded leader-follower |
| MAPPO | Multi-agent PPO, no communication |
| QMIX | Value decomposition, no communication |
| Statechart | Our approach with hand-coded protocol |
| Evolved-Statechart | Our approach with evolved protocol |

**Expected ranking**: Evolved-Statechart > Rule-based > Statechart > QMIX > MAPPO > Random

## Communication Efficiency

Less is more. Evolution discovers:
- Send signals only when needed (low `leader_claim_prob`)
- High-priority signals trigger action, low-priority ignored
- Redundant signals penalized (communication cost)

## Future Directions

1. **Attention over events** - Agents attend to relevant signals, ignore noise.

2. **Hierarchical roles** - LEADER contains SUB_LEADERS. Multi-level coordination.

3. **Adversarial teams** - Two teams with different protocols. Protocol arms race.

4. **Transfer learning** - Protocol learned on GATHER task transfers to COVERAGE.

5. **Natural language grounding** - Map evolved signals to English descriptions.

## Code Patterns

```python
# Event bus publishing
event = Event(
    event_type=EventType.LEADER_CLAIM,
    sender_id=agent.agent_id,
    payload={"x": agent.x, "y": agent.y}
)
self.env.event_bus.publish(event)

# Agent reacting to events
for event in events:
    action = self.genome.signal_meanings[event.event_type].action
    if action == "FOLLOW":
        target_x = event.payload.get("x", 0.5)
        target_y = event.payload.get("y", 0.5)
```

## Performance

GATHER task (4 agents):
- Random: ~10% success
- Rule-based: ~85% success
- Evolved-Statechart: ~90% success

COVERAGE task (4 agents):
- Random: ~5% success
- Rule-based: ~40% success (not designed for this)
- Evolved-Statechart: ~70% success

## Pure MLX

No numpy, no external RL libraries.
- MLX arrays for neural network controllers (MAPPO, QMIX)
- MLX nn.Module for network architecture
- Pure Python for simulation, evolution, event bus

## Date
2024-01-04
