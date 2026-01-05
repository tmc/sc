# Evolving Statechart Structure from Gameplay

## Concept

Instead of hand-coding which positions matter (corners, X-squares), **evolve** the statechart signals through self-play. Let evolution discover what humans figured out over decades of Othello play.

## Genome Encoding

The genome encodes the "shape" of the statechart:

```
Genome = {
    position_values: [64 floats]    # Value of each board position
    direction_weights: [8 floats]   # Importance of each flip direction
    phase_thresholds: [2 floats]    # When phases transition
    signal_weights: [4 floats]      # How to combine signals
}
```

This is essentially learning:
- **Guards**: Which positions enable good moves
- **State transitions**: When game phases change
- **Actions**: How to score candidate moves

## Evolution Parameters

- Population: 30 genomes
- Generations: 30
- Elite size: 5 (preserved unchanged)
- Mutation rate: 15%
- Fitness: Win rate vs random (20 games)

## Results

### Fitness Progression
```
Gen  0: best=0.921, avg=0.584
Gen  5: best=0.983, avg=0.768
Gen 10: best=0.987, avg=0.840
Gen 15: best=1.035, avg=0.855
Gen 20: best=1.037, avg=0.856
Gen 30: best=1.040, avg=0.877
```

### Discovered Position Values

```
Position Value Means:
  Corners:   +0.850  ← Discovered as GOOD
  X-squares: -0.544  ← Discovered as BAD
  Edges:     -0.244  ← Neutral
  Center:    +0.386  ← Moderate
```

**Evolution discovered corners are valuable and X-squares are dangerous!**

### Discovered Phase Thresholds

| Transition | Evolved | Hand-coded |
|------------|---------|------------|
| Opening → Mid | 22.7 | 20 |
| Mid → End | 45.5 | 50 |

Remarkably close to human intuition!

### Signal Importance (Evolved)

```
Position    : 27.8%
Flip count  : 16.4%
Direction   :  8.0%
Phase bonus : 47.8%  ← Dominant!
```

Evolution discovered that **phase-dependent strategy is most important** - opening play differs from endgame.

## Validation: Evolved vs Hand-coded

| Method | Win Rate |
|--------|----------|
| Evolved | 84% |
| Hand-coded | 88% |
| Gap | 4% |

Only 4% gap - evolution got 95% of the way to expert knowledge!

## Key Insights

1. **Domain knowledge can be evolved**: Evolution discovered corners > edges > X-squares without being told

2. **Phase thresholds emerge naturally**: The opening/midgame/endgame transitions were discovered to be ~23 and ~46 pieces

3. **Signal importance is learned**: Evolution figured out that phase-aware strategy matters most (47.8%)

4. **Comparable performance**: 84% vs 88% shows evolution approximates hand-coded expertise

## Implications for Statecharts

This demonstrates that **statechart structure can be learned**, not just hard-coded:

1. **Guard functions** (which moves are good) can be evolved
2. **State transitions** (phase changes) can be discovered
3. **Action priorities** (signal weights) can be optimized

This bridges pure learning (attention, transformers) with structured representations (statecharts).

## Future Work

1. **Neuroevolution**: Use neural networks inside genomes for more complex guards
2. **Co-evolution**: Evolve populations that play against each other
3. **Hierarchical evolution**: First evolve structure, then fine-tune with gradient descent
4. **Transfer**: Evolve on simple games, transfer to complex ones

## Reproduce

```bash
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml
.venv/bin/python3 experiments/exp_othello/evolve_statechart.py
```

Results saved to `evolved_genome.json`.
