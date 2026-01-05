# Turbo Evolution Analysis - Live Progress

## Experiment Setup

**Basic Evolution (completed)**:
- Pop: 30, Gens: 30
- Result: 84% win rate vs random
- Gap vs hand-coded: -4%

**Turbo Evolution (running)**:
- Pop: 40, Gens: 60, Elite: 6
- Key change: Strong priors (corners ~25, X-squares ~-12)
- Target: Beat 88% hand-coded baseline

## Live Progress Tracking

| Gen | Best Fitness | Avg Fitness | Time (s) |
|-----|-------------|-------------|----------|
| 0   | 1.040       | -           | -        |
| 5   | 1.046       | 0.802       | 308      |
| 10  | 1.047       | 0.821       | 623      |
| 15  | 1.043       | 0.819       | 966      |
| 19  | 1.044       | 0.822       | 1238     |

**Observations**:
- Fitness plateauing around 1.04-1.05
- Average improving slowly (0.802 → 0.822)
- ~65s per generation

## Basic Evolution Discoveries

From `evolved_genome.json` analysis:

### Position Values (64-cell board)
```
           A      B      C      D      E      F      G      H
      1   +0.57  -0.58  -0.43  -0.42  +0.94  +0.59  -0.53  +0.74
      2   -0.32  -0.70  -0.32  -0.25  +0.80  +1.03  -0.61  -0.24
      3   -0.01  -0.72  +0.31  -0.44  -0.24  +0.47  +0.10  -0.55
      4   +0.47  +0.62  +0.16  +1.07  +0.56  -0.28  -0.90  -0.34
      5   -0.47  -0.40  -0.57  +0.06  -0.14  -0.60  -0.35  +0.64
      6   -0.01  +1.05  +0.46  -0.20  +0.97  +1.13  -1.08  -0.16
      7   -0.81  -0.96  -0.00  +0.20  +0.89  -0.41  +0.09  -1.26
      8   +0.73  -0.59  -1.12  +0.82  -0.04  -0.31  -1.12  +1.36
```

### Pattern Summary
| Position Type | Mean Value | Status |
|--------------|-----------|--------|
| Corners (A1,H1,A8,H8) | +0.850 | ✅ Highest |
| X-squares (B2,G2,B7,G7) | -0.544 | ✅ Negative |
| C-squares | -0.682 | ✅ Penalized |
| Other edges | -0.244 | Neutral |
| Center 4 | +0.386 | Moderate |

### Phase Thresholds
| Phase | Evolved | Hand-coded | Difference |
|-------|---------|------------|------------|
| Opening→Mid | 22.7 | 20 | +2.7 |
| Mid→End | 45.5 | 50 | -4.5 |

### Signal Weights (normalized)
```
Position value:  27.8%
Flip count:      16.4%
Direction bias:   8.0%
Phase bonus:     47.8%  ← Dominant!
```

## Key Insight

Evolution independently discovered that **phase-dependent strategy** is the most important factor (47.8% weight), followed by position values (27.8%). This matches expert Othello knowledge:

1. **Opening**: Mobility matters most (fewer pieces, control center)
2. **Midgame**: Position matters (edges, stability)
3. **Endgame**: Piece count matters (maximize flips)

## Turbo Genome Structure

Unlike basic evolution with 64 position values, turbo uses semantic parameters:

```python
TurboGenome:
  corner_value       ~25.0 (±5)
  x_square_penalty   ~-12.0 (±3)
  c_square_penalty   ~-4.0 (±2)
  edge_value         ~3.0 (±1)
  center_values      [4 floats by distance]
  flip_weight        ~1.5
  mobility_weight    ~0.8
  phase1, phase2     ~20, ~48
```

This focused search space should find better optima faster.

## Next Steps

1. Wait for turbo evolution to complete (Gen 60)
2. Run final comparison (300 games)
3. Compare evolved params vs hand-coded baseline
4. If evolved beats hand-coded, document the winning parameters
