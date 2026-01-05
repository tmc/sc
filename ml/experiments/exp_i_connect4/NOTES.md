# Experiment I: Connect 4 - Fused Attention-Statechart

## Hypothesis

Combining explicit statechart signals (win/block detection) with implicit attention patterns should outperform pure attention, especially on games too complex to memorize.

## Why Connect 4?

TicTacToe is too simple - a Transformer can memorize optimal play. Connect 4 provides:

| Metric | TicTacToe | Connect 4 |
|--------|-----------|-----------|
| Board size | 9 | 42 (7x6) |
| Winning lines | 8 | 69 |
| State space | ~5,000 | ~4.5 trillion |
| Strategy depth | Shallow | Deep (forks, threats, columns) |

## Architecture

### Fused Model
1. **Explicit signals** (StatechartSignals):
   - `my_wins`: Columns that win immediately
   - `opp_wins`: Columns that block opponent win
   - `my_threats`: Count of 3-in-a-row threats after move

2. **Implicit patterns** (BoardAttention):
   - Cell embeddings + row/column position encoding
   - Turn-conditioned self-attention
   - Pool over rows to get per-column scores

3. **Fusion** (hierarchical):
   - If can win → take winning move
   - If must block → block opponent
   - Else → attention + threats + column bias

### Pure Attention Model
Same BoardAttention, but no explicit signals. Must learn everything from data.

## Results

### Untrained Performance

| Model | Win Rate | W/L/D |
|-------|----------|-------|
| Pure Attention | 65% | 65/35/0 |
| **Fused** | **96%** | **96/4/0** |

Even without training, explicit signals provide +31% advantage!

### After Training (1000 episodes REINFORCE)

| Model | Win Rate | W/L/D | Params |
|-------|----------|-------|--------|
| Random baseline | ~35% | - | - |
| Pure Attention | 66% | 132/68/0 | 10,801 |
| **Fused** | **100%** | **200/0/0** | 10,808 |

### Key Metrics

- **Improvement**: +34% win rate
- **Loss reduction**: 0 vs 68 losses
- **Zero losses**: Fused model never loses to random

## Training Dynamics

### Fused Model
```
Episode  250: 98.0% (98/2/0)
Episode  500: 99.0% (99/1/0)
Episode  750: 97.0% (97/3/0)
Episode 1000: 100.0% (100/0/0)
```

### Pure Attention Model
```
Episode  250: 72.0% (72/28/0)
Episode  500: 79.0% (79/21/0)
Episode  750: 75.0% (75/25/0)
Episode 1000: 73.0% (73/27/0)
```

Pure attention plateaus around 75% despite training.

## Analysis

### Why Fusion Works

1. **Inductive bias**: Explicit signals encode game rules that would take many samples to learn
2. **Guaranteed correctness**: Win/block logic is perfect by construction
3. **Learning efficiency**: Attention only needs to learn strategy, not rules
4. **Hierarchical priority**: Win > Block > Strategy prevents catastrophic mistakes

### Why Pure Attention Struggles

1. Must learn win conditions from scratch (69 patterns)
2. No hard constraint to take winning moves
3. Easily distracted by learned patterns that aren't optimal
4. High variance in policy gradient training

### Comparison with TicTacToe

| Game | Fused Improvement | Best Pure Attention |
|------|-------------------|---------------------|
| TicTacToe | +5% | 98% (Transformer) |
| Connect 4 | **+34%** | 66% (BoardAttention) |

The fusion advantage grows with game complexity!

## Implications for Statecharts

This validates the core hypothesis:

> **Statecharts provide structured inductive bias that improves learning efficiency.**

For games/tasks where:
- Rules are known and can be encoded
- State space is too large to memorize
- Correctness on critical paths matters

...fusing explicit statechart logic with learned patterns beats pure learning.

## Reproduce

```bash
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml
.venv/bin/python3 experiments/exp_i_connect4/fused_connect4.py
```

## Future Work

1. **Larger games**: Chess, Go (even more complex)
2. **Partial observability**: Where statechart state tracking matters
3. **Multi-agent**: Where coordination requires explicit protocols
4. **Real-time**: Where safety constraints must be guaranteed
