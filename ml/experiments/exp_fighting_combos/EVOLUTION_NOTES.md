# Fighting Game Evolution Results

## Concept

Evolve fighting game strategy from self-play. The genome encodes statechart-like decisions:
- When to attack vs defend based on spacing
- Frame advantage exploitation
- Meter management

## Setup

- Population: 30
- Generations: 40
- Elite: 4
- Fitness: Win rate vs random + damage bonus

## Results

### Fitness Progression
```
Gen  0: best=0.820
Gen 10: best=1.047
Gen 20: best=1.093
Gen 30: best=1.105
Gen 40: best=~1.10
```

### Evolved Spacing Strategy

| Spacing | #1 Action | #2 Action | #3 Action |
|---------|-----------|-----------|-----------|
| CLOSE | BLOCK (+1.0) | JAB (+0.95) | HEAVY (+0.32) |
| MID | HEAVY (+1.3) | BACK_DASH (+1.0) | BLOCK (+0.85) |
| FAR | FORWARD_DASH (+1.1) | OVERHEAD (+0.8) | HEAVY (-0.01) |

**Interpretation:**
- At CLOSE range: Play defensive (block/jab) - respect opponent's fast attacks
- At MID range: Use HEAVY's reach advantage, or create space
- At FAR range: Close distance to get into range

### Frame Advantage Exploitation

| Situation | #1 Action | Interpretation |
|-----------|-----------|----------------|
| CAN PUNISH (+6) | HEAVY (+0.57) | Max damage punish |
| DISADVANTAGE (-4) | SWEEP (+0.46) | Desperation knockdown |

### Meter Management

- Use special when meter >= 46 (just under half)
- Save at least 29 meter

### Bonuses Discovered

- Punish bonus: 3.23 (high - punishing is important)
- Pressure bonus: 1.26 (moderate - grab mixups matter)

## Key Discoveries

1. **Spacing-dependent strategy emerged**: Evolution found that different ranges require different approaches

2. **Risk management**: At close range, defense is prioritized over offense

3. **Reach optimization**: At mid range, using HEAVY (longer reach) is optimal

4. **Distance management**: At far range, closing distance is the priority

5. **Punish knowledge**: HEAVY is preferred for punishes (max damage)

## Implications for Statecharts

This demonstrates that fighting game strategy can be modeled as a statechart:

```
States:
  - FAR_RANGE
  - MID_RANGE
  - CLOSE_RANGE

Transitions:
  - FAR -> MID: FORWARD_DASH
  - MID -> FAR: BACK_DASH
  - MID -> CLOSE: FORWARD_DASH
  - CLOSE -> MID: BACK_DASH

Actions per state:
  - FAR: approach
  - MID: attack with reach
  - CLOSE: defend or jab
```

The guards (spacing thresholds) and actions (move selection) were all evolved from gameplay!

## Reproduce

```bash
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml
.venv/bin/python3 experiments/exp_fighting_combos/evolve_fighter.py
```
