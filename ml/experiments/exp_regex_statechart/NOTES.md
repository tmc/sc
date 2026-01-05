# exp_regex_statechart: Research Notes

## Key Insight

**Regex = DFA = Statechart** - This mathematical equivalence means we can frame regex learning as statechart evolution. But the deeper insight is:

> The structure of a statechart can be LEARNED, not specified.

This is the "differentiable statecharts thesis" - given only input/output examples, ML can discover states, transitions, guards, and actions.

## What Worked

### 1. Basic Topology Evolution
Simple patterns like `a+`, `ab+`, `(ab)+` are learnable with pure topology evolution:
- Population 30-50, generations 40-75
- F1 > 0.95 achievable for simple patterns
- Character guards evolve effectively via mutation

### 2. Prefix Equivalence for State Discovery
The Myhill-Nerode theorem provides the theoretical foundation:
- Two prefixes are equivalent iff they accept the same suffixes
- States = equivalence classes of prefixes
- This gives us a principled way to discover state structure

### 3. Semantic Component Abstraction
Separating concerns into evolvable components proved valuable:
- `GuardExpr` hierarchy (char, counter, flag, composite)
- `ActionExpr` hierarchy (increment, reset, set flag)
- `StateVar` for extended state

### 4. Guard Synthesis via Program Synthesis
Treating guard learning as program synthesis works:
- Generate candidate expressions
- Evaluate against positive/negative contexts
- Evolve toward high F1

## What Didn't Work (Yet)

### 1. Counting Patterns Remain Hard
Patterns like `a{2,4}` (exactly 2-4 a's) are challenging:
- Requires discovering the NEED for a counter
- Then learning the correct accept condition
- Current approach: ~0.67 F1 with limited generations

**Root cause**: The search space explodes when you add variables. Need smarter initialization or curriculum learning.

### 2. State Discovery Produces Overly Complex Structures
The prefix clustering approach creates too many states:
- 7+ states for simple patterns
- Many redundant transitions
- Needs aggressive minimization

**Potential fix**: Add state merging based on behavioral equivalence testing.

### 3. Action Discovery is Reactive, Not Predictive
Current approach detects patterns AFTER seeing examples:
- Sees length constraints → adds counter
- Sees repetition → adds tracking

**Better approach**: Proactively hypothesize variables, let evolution prune.

## Metrics Summary

| Pattern Type | Best F1 | Generations | Notes |
|--------------|---------|-------------|-------|
| Simple (a+) | 1.0 | 30 | Works reliably |
| Concatenation (ab+) | 0.97 | 50 | Usually works |
| Alternation (a\|b) | 0.85 | 75 | Needs more exploration |
| Counting (a{2,4}) | 0.67 | 75 | Needs counter discovery |
| Repetition ((ab)+) | 0.90 | 50 | Topology sufficient |

## Future Directions

### 1. Curriculum Learning for Complexity
Start with simple patterns, gradually increase:
```
Level 1: a*, b*, ab
Level 2: a+, (ab)+
Level 3: a{n}, alternations
Level 4: counting, backreferences
```
Transfer learned structure between levels.

### 2. Neural Guide for Evolution
Use a neural network to:
- Predict useful mutations
- Score candidate structures
- Guide search toward promising regions

Connection: `exp_learnable_policies` uses neural networks for policy learning.

### 3. SAE-Based State Discovery
Replace prefix clustering with Sparse Autoencoder:
- Encode execution traces
- Discover latent states as SAE features
- States emerge from representation learning

Connection: `exp_sae_statechart`, `exp_k_sae_interpretability`

### 4. Differentiable Statechart Execution
Make the entire statechart differentiable:
- Soft state membership (attention over states)
- Soft guard evaluation (sigmoid over conditions)
- Gradient-based learning of structure

This would enable end-to-end training from examples.

### 5. Active Learning for Examples
Instead of fixed example sets:
- Generate adversarial examples
- Query oracle for ambiguous cases
- Minimize examples needed for convergence

### 6. Multi-Objective Optimization
Current fitness: F1 - size_penalty

Better objectives:
- Behavioral correctness (F1)
- Structural parsimony (state count)
- Semantic interpretability (guard complexity)
- Generalization (holdout performance)

Use NSGA-II for Pareto-optimal solutions.

## Connections to Other Experiments

### exp_topology_evolution
- **Borrowed**: Evolution framework, fitness components, NSGA-II selection
- **Extended**: Added semantic components (guards, actions, variables)
- **Insight**: Topology alone insufficient for counting patterns

### exp_guard_synthesis
- **Borrowed**: Guard expression AST, program synthesis approach
- **Extended**: Integrated into statechart evolution
- **Insight**: Guards are learnable via evolution

### exp_grammar_induction
- **Related**: Both learn formal language structure
- **Difference**: Grammar = production rules, Statechart = state machine
- **Synergy**: Could use induced grammar to bootstrap statechart

### exp_transfer_learning
- **Opportunity**: Pre-train on simple patterns, transfer to complex
- **Hypothesis**: Learned state structure transfers across regex families

### exp_rule_discovery
- **Related**: Both discover boolean conditions
- **Difference**: Rules are flat, guards are per-transition
- **Synergy**: Rule discovery could inform guard synthesis

### exp_sae_statechart
- **Opportunity**: Use SAE features as discovered states
- **Hypothesis**: Interpretable SAE features = meaningful states

### exp_code_completion
- **Opportunity**: Regex synthesis as code completion
- **Application**: "Complete this regex given examples"

## Theoretical Connections

### Myhill-Nerode Theorem
States = equivalence classes under suffix distinguishability.
Our prefix clustering approximates this.

### Angluin's L* Algorithm
Active learning of DFAs with membership + equivalence queries.
Could integrate oracle queries into evolution.

### Minimum Description Length
Optimal statechart minimizes: description_length(SC) + description_length(data|SC)
This justifies our size penalty in fitness.

### Kolmogorov Complexity
The simplest statechart that matches behavior is the "true" pattern.
Evolution with parsimony pressure approximates this.

## Open Questions

1. **How many examples are sufficient?**
   - Empirically: ~50-100 for simple patterns
   - Theory: Related to VC dimension of statechart class

2. **Can we learn unbounded counting?**
   - Current: Fixed counter bounds
   - Challenge: Generalize beyond training lengths

3. **What's the right abstraction for "state"?**
   - Current: Discrete states with extended variables
   - Alternative: Continuous latent states (RNN-like)

4. **How to handle ambiguous examples?**
   - Current: Assume examples are consistent
   - Reality: Noisy or contradictory labels

5. **Can learned statecharts explain themselves?**
   - Goal: Human-readable description of learned pattern
   - Challenge: Semantic naming of discovered states

## Code Quality Notes

- All modules have `test_*` functions
- Type hints throughout
- Dataclasses for clean data structures
- Factory methods for common patterns
- Separation of concerns (discovery, evolution, synthesis)

## Performance Observations

- Evaluation is the bottleneck (~70% of time)
- Caching helps for repeated evaluations
- Parallel evaluation possible but not implemented
- String matching is fast; guard evaluation adds overhead

## Reproducibility

- Random seeds not currently fixed
- Results vary across runs
- Need to add seed parameter for reproducibility
- Should log evolution trajectories for analysis
