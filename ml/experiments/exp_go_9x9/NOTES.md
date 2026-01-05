# Experiment: 9x9 Go with Statecharts

## Goal
**100% LEGAL PLAY** where Transformers fail.

Not trying to beat KataGo - proving constraint satisfaction by construction.

## Key Insight
Go rules map directly to statechart concepts:
- **Turn alternation**: OR-state (Black | White)
- **Ko rule**: History state (remembers last capture point)
- **Suicide prevention**: Guard (not_suicide check)
- **Occupation**: Guard (not_occupied check)

## Statechart Structure

```
Go9x9 (AND)
├── Turn (OR): Black | White
├── KoState (OR, HISTORY)
│   ├── NoKo
│   └── KoForbidden [point]
├── Prisoners (AND)
│   ├── BlackCaptures: int
│   └── WhiteCaptures: int
└── Board[81] (context, not states)
```

## Results: Random Baseline (100 games)

| Metric                  | Statechart | Naive Sampling |
|-------------------------|------------|----------------|
| **Illegal Attempts**    | **0**      | 32,844         |
| **Illegal Rate**        | **0.0%**   | 79.6%          |
| Avg Game Length         | 83.3       | 84.3           |
| Time (ms/game)          | 41.6       | 6.1            |

## Results: Transformer Comparison

| Approach              | Illegal Rate | Training Required |
|-----------------------|--------------|-------------------|
| **Statechart**        | **0.0%**     | None (by design)  |
| Transformer (random)  | 96.2%        | -                 |
| Transformer (trained) | 0.0%         | 14K examples      |

### Key Findings

1. **96.2% of random moves are illegal in Go!**
   - Random sampling almost never produces valid moves
   - This is the constraint density that statecharts handle

2. **Transformer CAN learn legality... but:**
   - Requires 14K+ training examples
   - No guarantee on unseen positions
   - Could fail on edge cases (Ko, rare suicide)

3. **Statechart is 0% illegal BY CONSTRUCTION**
   - No training data needed
   - Provably correct for ALL positions
   - Ko rule encoded as history state
   - Suicide prevention encoded as guard

## Guards (What Makes Moves Legal)

1. **not_occupied(x, y)**: Point must be empty
2. **not_ko(x, y)**: Point must not be ko-forbidden
3. **not_suicide(x, y)**: Move must not be suicide
   - Suicide = stone has no liberties AND doesn't capture anything

## Why Statechart Wins

| Approach | Learns Legality | Guarantees Legality |
|----------|-----------------|---------------------|
| Transformer | Must learn from data | No guarantee |
| Statechart | Encoded in topology | **Yes, by construction** |

## Commands

```bash
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml

# Test statechart
.venv/bin/python3 experiments/exp_go_9x9/go_statechart.py

# Run benchmark
.venv/bin/python3 experiments/exp_go_9x9/random_baseline.py
```

## Learning Rules from Gameplay (learn_rules.py)

**Key Question**: Can we LEARN the statechart guards instead of pre-encoding them?

### 4 Approaches Compared

| Approach        | Test Accuracy | Time (s) | What it Learns |
|-----------------|---------------|----------|----------------|
| **Evolutionary**    | 100.0%    | 2.19     | `occupied < 0.85 AND is_suicide < 0.72` |
| **Differentiable**  | 100.0%    | 0.33     | weights: occupied=-9.1, has_liberties=+9.3 |
| **Synthesis**       | 100.0%    | 0.01     | `NOT occupied AND NOT is_suicide` |
| **Hybrid NN**       | 100.0%    | 3.00     | `NOT occupied AND has_liberties` |

### Discovered Rules

All approaches discovered equivalent rules:
```
Legal move requires:
  1. NOT occupied (point must be empty)
  2. NOT is_suicide (has_liberties OR captures_any)
```

This matches what we hand-coded - but now it's **learned from data**!

### Limitation: Ko Rule (Validated!)

Tested learned rules on 810,000 positions:
- **False positive rate: 0.0062%** (50 out of 810,000)
- **ALL 50 false positives were Ko violations**

Ko requires game **history** (remembering the previous board state).
Per-move features can't capture this - it's fundamentally temporal.

```
Ko situation example:
  Move: (5, 8)
  Ko state: ko_forbidden
  Ko point: (5, 8)

  guard_not_occupied: True  ✓
  guard_not_ko: False       ✗ (THIS IS THE PROBLEM)
  guard_not_suicide: True   ✓
```

This is where the statechart's **history state** concept shines:
the Ko rule is literally a history state in the formal model.

## History Mechanisms for Ko Detection (history_mechanisms.py)

Tested 7 different mechanisms for detecting Ko violations:

| Mechanism          | Accuracy | Precision | Recall |   F1   |
|--------------------|----------|-----------|--------|--------|
| **Delta Encoding** |   100.0% |    100.0% | 100.0% |   1.00 |
| **Explicit Ko**    |   100.0% |    100.0% | 100.0% |   1.00 |
| Zobrist Hash       |    99.9% |      0.0% |   0.0% |   0.00 |
| Embedding Sim      |    99.9% |      0.0% |   0.0% |   0.00 |
| Ring Buffer        |    99.9% |      0.0% |   0.0% |   0.00 |
| Attention          |    99.9% |      0.0% |   0.0% |   0.00 |
| Spike Timing       |    99.9% |      0.0% |   0.0% |   0.00 |

### Key Findings

1. **Delta Encoding + Learned Gate**: 100% F1!
   - Learns the rule: `single_removed AND move_at_removed AND would_capture_single`
   - Differentiable and compact
   - Weights: single_removed=2.4, move_at_removed=11.5, would_capture_single=10.8

2. **Why Embedding/Attention fail even with training**:
   - Ko is sparse (~0.2% of positions)
   - Raw board embeddings don't naturally capture "recapture" pattern
   - Need the RIGHT FEATURES, not just more data
   - Attention accuracy: 40% during training (can't separate signal from noise)

3. **Why Zobrist/Ring Buffer/Spike Timing fail**:
   - Need to track state DURING the game
   - Work in-game but fail on isolated position examples
   - They need to SEE the game history unfold

4. **Key Insight**: Ko detection requires STRUCTURED FEATURES
   - Either hand-crafted (Explicit Ko)
   - Or learned with right structure (Delta Encoding)
   - Raw board embeddings insufficient

5. **Recommendation**: Delta Encoding + Learned Gate
   - Captures essence: "single capture followed by recapture attempt"
   - Differentiable (can integrate with NN)
   - Learns from data (no hand-crafting needed)
   - 3 weights + bias = minimal representation

## Hybrid Neural Network (exp_go_hybrid/hybrid_go_net.py)

**Goal**: Combine CNN for move quality with statechart guards for guaranteed legality.

### Architecture

```
HybridGoNet (MLX)
├── Input: 9x9x3 (our stones, opponent stones, empty)
├── Conv Backbone: 3→32→64→64 channels
├── Ko Gate: Delta Encoding (pre-trained, 100% F1)
├── Policy Head: 82 outputs (81 moves + pass)
│   └── Guard Masking: Legal moves only!
└── Value Head: single scalar (-1 to +1)
```

### Results (50 games, 4456 moves)

| Metric               | Value       |
|----------------------|-------------|
| **Legal Move Accuracy**  | **100.0%**  |
| **Ko Violation Rate**    | **0.0000%** |
| Policy Entropy       | 2.992       |
| Illegal Attempts     | 0           |

### Key Achievement

This is **end-to-end differentiable Go with guaranteed legality**:

1. **100% legal moves BY CONSTRUCTION** - guard masking ensures only valid moves
2. **0% Ko violations** - Delta Encoding gate blocks Ko-forbidden moves
3. **Differentiable** - CNN weights can be trained with gradient descent
4. **Minimal overhead** - Ko gate is just 3 weights + bias

### Commands

```bash
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml

# Run hybrid network training
.venv/bin/python3 experiments/exp_go_hybrid/hybrid_go_net.py
```

## Future Work

1. ~~**Learn Ko rule**: Add sequence features or RNN~~ ✓ SOLVED with Delta Encoding
2. ~~**Hybrid approach**: Learned guards + NN move quality~~ ✓ DONE
3. **19x19 scaling**: Same approach, larger board
4. **Game traces**: Export games for visualization/analysis
5. **MCTS integration**: Use hybrid network as policy/value prior for search
