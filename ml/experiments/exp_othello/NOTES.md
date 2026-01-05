# Experiment: Othello - Fused Attention-Statechart

## Why Othello?

Othello is the "Goldilocks" game for testing statechart-attention fusion:

| Property | TicTacToe | Connect 4 | Othello |
|----------|-----------|-----------|---------|
| Board size | 9 | 42 | 64 |
| State space | ~5,000 | ~4.5T | **~10^28** |
| Game length | 5-9 moves | ~21 moves | **~60 moves** |
| Fusion improvement | +5% | +34% | **+20%** |

Othello tests key statechart features:
- **Flip propagation** = parallel region updates
- **Game phases** = hierarchical (Opening/Mid/End)
- **Capture rules** = guards (`has_flip(pos, dir)`)

## Statechart Structure

```
Othello (AND)
├── GamePhase (OR): Opening | Midgame | Endgame
├── Turn (OR): Black | White
├── Board[64] (PARALLEL): Empty | Black | White
└── ValidMoves: computed from flip constraints
```

Key guard: `has_flip(pos, dir)` - must flip at least one piece in at least one direction.

## Explicit Signals

The fused model computes these statechart-derived signals:

1. **valid**: Is move legal? (flips ≥ 1 piece)
2. **flip_count**: How many pieces flipped
3. **is_corner**: Captures a corner (very valuable, stable)
4. **is_edge**: On edge (stable once surrounded)
5. **is_x_square**: Diagonal to corner (dangerous, can give up corner)
6. **is_c_square**: Adjacent to corner (risky)

### Phase-Dependent Weights

Strategy changes by game phase:

| Signal | Opening | Midgame | Endgame |
|--------|---------|---------|---------|
| Corner | 20.0 | 30.0 | 25.0 |
| Edge | 2.0 | 5.0 | 3.0 |
| Flip count | 0.5 | 1.0 | 2.0 |
| Attention | 3.0 | 2.0 | 1.0 |
| X-square penalty | -8.0 | -10.0 | -5.0 |
| C-square penalty | -2.0 | -3.0 | -1.0 |

Opening: Mobility matters, avoid corners (too early)
Midgame: Corners and edges are key for stability
Endgame: Piece count (flips) matter most

## Results

### Untrained Performance

| Model | Win Rate | W/L/D |
|-------|----------|-------|
| Pure Attention | 64% | 32/15/3 |
| **Fused** | **70%** | **35/13/2** |

Even without training, explicit signals help (+6%).

### Training Dynamics

**Fused Model:**
```
Episode  100: 82.0% (41/9/0)
Episode  200: 86.0% (43/7/0)
Episode  300: 70.0% (35/14/1)  # variance
Episode  400: 74.0% (37/12/1)
Episode  500: 86.0% (43/7/0)   # recovered
```

**Pure Attention:**
```
Episode  100: 72.0% (36/12/2)
Episode  200: 76.0% (38/11/1)
Episode  300: 68.0% (34/14/2)
Episode  400: 72.0% (36/13/1)
Episode  500: 80.0% (40/8/2)
```

Fused consistently outperforms during training.

### Final Evaluation (100 games)

| Model | Win Rate | W/L/D | Margin | Params |
|-------|----------|-------|--------|--------|
| Random baseline | ~35% | - | - | - |
| Pure Attention | 65.0% | 65/31/4 | 8.4 | 17,729 |
| **FUSED (ours)** | **85.0%** | **85/12/3** | **20.0** | 17,747 |

### Key Metrics

- **Improvement**: +20% win rate
- **Loss reduction**: 12 vs 31 (2.5x fewer losses)
- **Margin improvement**: 20.0 vs 8.4 (2.4x larger wins)
- **Parameters**: Nearly identical (~17.7k)

## Analysis

### Why Fusion Works for Othello

1. **Corner knowledge is crucial**: Corners are permanently stable. Pure attention must learn this from data; fusion encodes it directly.

2. **X-square avoidance**: Playing diagonal to a corner often gives away the corner. This is a non-obvious pattern that takes many games to learn.

3. **Phase-aware strategy**: Opening (mobility), midgame (stability), endgame (piece count) require different approaches. Fusion explicitly models this.

4. **Flip propagation correctness**: The `has_flip` guard ensures only valid moves are considered. Pure attention wastes capacity on illegal move filtering.

### Scaling Trend

| Game | Complexity | Fusion Improvement |
|------|------------|-------------------|
| TicTacToe | ~5,000 states | +5% |
| Connect 4 | ~4.5T states | +34% |
| Othello | ~10^28 states | +20% |

The improvement is substantial for complex games where memorization is impossible.

## Reproduce

```bash
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml
.venv/bin/python3 experiments/exp_othello/fused_othello.py
```

## Future Work

1. **Self-play improvements**: Longer training, curriculum learning
2. **Monte Carlo Tree Search**: Combine with MCTS for stronger play
3. **Multi-phase training**: Train separate networks per game phase
4. **Transfer learning**: Pre-train on simpler games, fine-tune on Othello
