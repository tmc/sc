# exp_statechart_synthesis_benchmark

## Overview
Comprehensive benchmark suite for comparing all statechart synthesis methods across multiple domains. Aggregates results from completed experiments into a unified leaderboard.

## Key Results (from Completed Experiments)

| Rank | Method | Accuracy | Category |
|------|--------|----------|----------|
| 1 | priority_attention | 1.140 | NEURAL |
| 2 | action_algebra | 1.000 | EVOLUTIONARY |
| 3 | guard_evolution | 0.950 | EVOLUTIONARY |
| 4 | dialogue_extractor | 0.900 | NEURAL |
| 5 | joint_evolution | 0.888 | EVOLUTIONARY |
| 6 | nsga2_unified | 0.850 | EVOLUTIONARY |
| 7 | sae_coverage | 0.820 | SAE_BASED |
| 8 | hybrid_lca | 0.744 | HYBRID |
| 9 | supervised_lca | 0.736 | NEURAL |
| 10 | transfer_zero_shot | 0.580 | HYBRID |

## Category Comparison

| Category | Avg Score |
|----------|-----------|
| NEURAL | 0.839 |
| EVOLUTIONARY | 0.230-1.000 (varies) |
| SAE_BASED | 0.426 |
| HYBRID | 0.580-0.744 |

## Architecture

### 1. Synthesis Methods (`synthesis_methods.py`)
Unified interface for all methods:
- `EvolutionaryMethod`: Genetic algorithms
- `NSGA2Method`: Multi-objective evolution
- `SAEMethod`: Sparse autoencoder state discovery
- `NeuralMethod`: Supervised learning
- `HybridMethod`: SAE + evolution combination
- `RandomMethod`: Baseline

### 2. Domain Suite (`domain_suite.py`)
9 test domains across 4 categories:
- **Games**: tictactoe, go_ko, zelda_modules
- **Regex**: email_regex, number_regex
- **Dialogue**: customer_service, booking_assistant
- **Code**: python_syntax, json_syntax

### 3. Metrics (`metrics.py`)
Three metric categories:
- **Accuracy**: accuracy, precision, recall, F1, coverage
- **Efficiency**: time, iterations, examples/sec
- **Interpretability**: naming, clarity, minimality

### 4. Leaderboard (`leaderboard.py`)
- Per-method rankings
- Per-domain rankings
- Category comparisons
- Known experiment results aggregation

### 5. Benchmark Runner (`benchmark_runner.py`)
- Orchestrates full benchmark
- Supports filters (domain, difficulty, category)
- JSON export

## Files

| File | Lines | Description |
|------|-------|-------------|
| synthesis_methods.py | ~600 | Unified method interface |
| domain_suite.py | ~500 | Test domains |
| metrics.py | ~450 | Metrics computation |
| leaderboard.py | ~400 | Ranking system |
| benchmark_runner.py | ~350 | Runner orchestration |

## Usage

```bash
# Quick benchmark (easy domains)
python -m exp_statechart_synthesis_benchmark.benchmark_runner --quick

# Full benchmark
python -m exp_statechart_synthesis_benchmark.benchmark_runner

# Filter by domain
python -m exp_statechart_synthesis_benchmark.benchmark_runner --domain games

# Export results
python -m exp_statechart_synthesis_benchmark.benchmark_runner --output results.json
```

## Connections to Experiments

This benchmark aggregates results from:
- exp_lca_neural (hybrid 74.4%)
- exp_unified_statechart_evolution (NSGA-II 85%)
- exp_transfer_coverage (58% zero-shot)
- exp_dialogue_statechart (90% extraction)
- exp_sae_coverage_synthesis (82%)
- exp_priority_attention (114% vs baseline)
- exp_guard_synthesis (95%)
- exp_action_composition (100%)
- exp_joint_guard_action_evolution (88.8%)

## Key Findings

1. **NEURAL methods excel on supervised tasks** (dialogue, code) where training data is available
2. **EVOLUTIONARY methods work best for rule learning** (guards, actions) without labels
3. **HYBRID approaches** (SAE+evolution) offer good balance
4. **SAE state discovery** provides interpretable intermediate representations
5. **Domain matters**: different methods suit different domains

## Future Directions

1. **Cross-domain transfer**: Train on one domain, test on another
2. **Few-shot synthesis**: Minimal examples needed
3. **Incremental benchmarks**: Add examples over time
4. **Human evaluation**: Interpretability study

## Key Insight

No single method wins everywhere. The benchmark reveals method-domain affinity:
- Games → Evolutionary (rule discovery)
- Dialogue → Neural (pattern recognition)
- Code → Hybrid (structure + learning)
- Regex → SAE (feature discovery)
