# Experiment H: TicTacToe Baseline Comparison

## Objective
Compare MLP baseline against statechart-based approach for TicTacToe move prediction.

## Models

### MLP Baseline
- **Architecture**: Cell embeddings (3 -> 16) + Turn embedding (2 -> 16) -> MLP (160 -> 64 -> 32 -> 9)
- **Parameters**: 12,761
- **Input**: board (9 cells, 0/1/2) + turn (0/1)
- **Output**: logits for 9 cells

### Transformer Baseline
- **Architecture**: Token Embed + Positional Encoding + Self-Attention + FFN + Output Projection
- **Parameters**: 3,657
- **Input**: board (9 cells, 0/1/2)
- **Output**: softmax over 9 cells

### DifferentiableTicTacToe (Statechart)
- **Architecture**: BoardEncoder + 37 transitions + guards
- **Parameters**: ~3,000-5,000 (varies with embed_dim)
- **Input**: game_state (8 states) + board
- **Output**: transition enablement (37 transitions)

## Metrics

### 1. Legal Move Accuracy
Can the model predict only legal moves (empty cells)?

| Model | Accuracy |
|-------|----------|
| MLP Baseline | 1.0000 |
| Transformer Baseline | 0.7817 |
| Statechart | 1.0000 (by construction) |

**Note**: MLP achieves 100%. Transformer achieves 78%. Statechart has it encoded in guard logic.

### 2. Win Rate vs Random Opponent

| Model | Win Rate | Wins | Losses | Draws |
|-------|----------|------|--------|-------|
| Random Baseline | ~0.35 | - | - | - |
| MLP (100 samples) | 0.56 | - | - | - |
| MLP (1000 samples) | 0.68 | - | - | - |
| MLP (10000 samples) | 0.51 | - | - | - |
| MLP (Full training) | 0.675 | 135 | 51 | 14 |
| **Transformer** | **0.980** | **98** | **0** | **2** |

**Observation**: Transformer dramatically outperforms MLP on win rate (98% vs 67.5%) despite lower legal accuracy and fewer parameters.

### 3. Sample Efficiency

| Samples | Legal Acc | Win Rate | Time (s) |
|---------|-----------|----------|----------|
| 100 | 1.0000 | 0.56 | 12.65 |
| 1000 | 1.0000 | 0.68 | 9.78 |
| 10000 | 1.0000 | 0.51 | 7.28 |

**Note**: Legal accuracy converges quickly. Win rate variance is high.

### 4. Training Time

| Configuration | Time |
|--------------|------|
| 500 steps, batch=32 | 7-13s |
| 2000 steps, batch=64 | 42.65s |

## Reproduce

```bash
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml
.venv/bin/python3 experiments/exp_h_tictactoe/baseline_mlp.py
```

## Analysis

### What MLP Does Well
- Learns legal moves perfectly (100% accuracy)
- Beats random opponent consistently (~67% win rate)
- Fast training convergence

### What MLP Lacks
- No explicit state representation
- No notion of game phases (opening, midgame, endgame)
- Win rate plateaus without strategic knowledge

### Statechart Advantages (Theoretical)
- Explicit state machine structure
- Guards encode game rules directly
- States represent meaningful game phases
- More interpretable

### Statechart Disadvantages
- More complex to train
- Requires manual state definition
- Higher engineering overhead

## Key Findings

1. **Legal move learning is trivial**: Both approaches achieve 100% on legal moves.

2. **Win rate is the harder metric**: 67% against random is good but not great. Optimal play against random should approach 95%+.

3. **MLP is a strong baseline**: Simple, fast, effective for basic policy learning.

4. **Statechart value proposition**: Not in raw accuracy, but in:
   - Interpretability (what state is the game in?)
   - Composability (add new rules as transitions)
   - Verification (guards ensure correctness)

## Next Steps

1. **Train statechart model end-to-end** for fair comparison
2. **Add strategic objectives** (win rate optimization, not just legal moves)
3. **Test on harder games** where state structure matters more
4. **Measure interpretability** - can we extract game phases from learned states?

## Baseline Comparison Summary

| Metric | MLP | LSTM | Transformer | Statechart |
|--------|-----|------|-------------|------------|
| Parameters | 12,761 | 43,269 | 3,657 | 48,853 |
| Legal Move Accuracy | 100% | 100% | 78.2% | 100% (by design) |
| Win Rate vs Random | 67.5% | 66.6% | **98.0%** | TBD |
| Training Time | 42.65s | 24.6s | 58.83s | TBD |
| Training Samples | 9,000 | 34,361 | 9,000 | N/A |

### LSTM Results (Updated)
- **Parameters**: 43,269 (matched to ~48k statechart target)
- **Architecture**: BoardEmbed(28) -> LSTM(hidden=56) -> Linear -> Softmax
- **Legal Move Accuracy**: 100%
- **Win Rate**: 66.6% (333 wins, 102 losses, 65 draws / 500 games)
- **Epochs to 50% WR**: 0 (starts at 57%)
- **Training Time**: 24.6s (20 epochs)

**Analysis**: LSTM performs comparably to MLP but with more parameters. Sequential architecture provides no advantage for board state reasoning since game state is fully observable at each timestep.

## Key Findings

1. **Transformer wins more games**: 98% vs 67.5% despite fewer params and lower legal accuracy
2. **Legal accuracy ≠ strategic play**: MLP learns all moves are legal but doesn't play well
3. **Attention helps strategy**: Transformer's self-attention may capture board relationships

## Conclusion

**Transformer baseline sets a high bar:**
- **Win rate**: 98% (near-optimal against random)
- **Parameters**: 3,657 (4x fewer than MLP)
- **Zero losses** against random opponent

Statechart approach must demonstrate:
- Comparable or better win rate, OR
- Better interpretability (can we explain decisions?)
- Better composability (can we add rules declaratively?)

(The value of statecharts may not be raw performance but structured representation.)

## Reproduce

```bash
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml

# MLP baseline
.venv/bin/python3 experiments/exp_h_tictactoe/baseline_mlp.py

# LSTM baseline
.venv/bin/python3 experiments/exp_h_tictactoe/baseline_lstm.py

# Transformer baseline
.venv/bin/python3 experiments/exp_h_tictactoe/baseline_transformer.py

# Statechart validation
.venv/bin/python3 experiments/exp_h_tictactoe/differentiable_tictactoe.py
```
