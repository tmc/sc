# exp_history_transfer: Research Notes

## Core Hypothesis

**History type selection patterns transfer across domains.**

When a composite state needs DEEP vs SHALLOW vs NONE history, the decision depends on structural features that are domain-agnostic:
- How deeply nested is the state?
- How many children does it have?
- Is there reentrant behavior?

These patterns should transfer from games to UI (and vice versa).

## Approach

### 1. Domain-Agnostic Features

Extract 11 features from each composite state:

| Feature | Description | Transfer Relevance |
|---------|-------------|-------------------|
| depth | Hierarchy depth | Higher depth → more likely DEEP |
| n_children | Direct child count | Many children → SHALLOW |
| n_descendants | Total nested states | Many descendants → DEEP |
| max_descendant_depth | Deepest nesting | >1 level → DEEP |
| has_parallel_children | Any AND children | Parallel regions need history |
| has_or_children | Any OR children | OR needs history for mode |
| n_outgoing_transitions | Exit transition count | High → complex, needs history |
| n_incoming_transitions | Entry transition count | High → reentry, needs history |
| is_reentrant | Has self-transitions | Reentrant → needs history |
| parent_depth | Parent's depth | Context for relative depth |
| sibling_count | Number of siblings | Affects navigation patterns |

### 2. Training Pipeline

```
Game Domain                    UI Domain
┌─────────────────┐           ┌─────────────────┐
│ GamePause       │           │ Wizard          │
│ RPGMenu         │  ──────►  │ SettingsPanel   │
│ FightGame       │  Transfer │ ModalNav        │
└────────┬────────┘           │ EmailClient     │
         │                    └────────┬────────┘
         ▼                             │
    HistoryPredictor          Zero-shot │
    (trained on games)        evaluation▼
```

### 3. Metrics

- **Source accuracy**: Training accuracy on game domain
- **Transfer accuracy**: Zero-shot on UI domain (no fine-tuning)
- **Scratch accuracy**: Train from scratch on UI domain
- **Improvement**: (transfer - random) / random

## Domain Examples

### Game Domain

| Statechart | States | Composites | Key History Pattern |
|------------|--------|------------|---------------------|
| GamePause | 7 | 3 | NONE (checkpoint) |
| RPGMenu | 12 | 5 | DEEP (nested navigation) |
| FightGame | 11 | 4 | SHALLOW (parallel modes) |

### UI Domain

| Statechart | States | Composites | Key History Pattern |
|------------|--------|------------|---------------------|
| Wizard | 5 | 1 | NONE (restart) |
| SettingsPanel | 8 | 3 | SHALLOW (tabs) |
| ModalNav | 12 | 5 | DEEP (nested views) |
| EmailClient | 11 | 4 | MIXED (shallow folder, deep compose) |

## Expected Results

Based on exp_deep_history findings:

1. **max_descendant_depth** is the strongest predictor
   - >1 level of nesting → DEEP history
   - Exactly 1 level → SHALLOW history
   - No children → NONE

2. **Pattern similarity across domains**
   - Both games and UI have modal popups (interrupt → return)
   - Both have nested navigation
   - Both have wizard/checkpoint patterns

3. **Expected transfer accuracy**: 60-80%
   - Higher than random (33%)
   - Lower than scratch (80-90%) due to domain-specific nuances

## Connections to Other Experiments

| Experiment | Connection |
|------------|------------|
| `exp_deep_history` | Source of HistoryType and scenarios |
| `exp_transfer_learning` | Topology transfer concepts |
| `exp_inter_level_transitions` | LCA affects history scope |
| `exp_priority_attention` | Could combine with history prediction |

## Implementation Details

### Neural Predictor

Simple MLP with:
- Input: 11 features
- Hidden: 32 units, ReLU
- Output: 3 classes (NONE, SHALLOW, DEEP)
- Loss: Cross-entropy
- Optimizer: SGD (learning_rate=0.05)

### Training

- 100 epochs
- Full batch (small datasets)
- No validation split (evaluating transfer, not overfitting)

## Future Directions

1. **Bidirectional transfer**: UI → games
2. **Fine-tuning analysis**: How many UI examples needed to match scratch?
3. **Feature importance**: Which features matter most for transfer?
4. **Larger domains**: Web apps, mobile apps, embedded systems
5. **Combine with topology transfer**: Transfer both structure AND history strategy

## Files

```
exp_history_transfer/
├── __init__.py           # Package exports
├── history_transfer.py   # Core implementation
└── NOTES.md              # This file
```

## Status

- [x] Domain-agnostic feature extraction
- [x] Game domain examples (3 statecharts)
- [x] UI domain examples (4 statecharts)
- [x] HistoryPredictor neural network
- [x] Training and evaluation pipeline
- [x] Transfer experiment runner
- [x] Pattern analysis
- [ ] Run full experiment and validate hypothesis
- [ ] Fine-tuning analysis

## Date

2026-01-04
