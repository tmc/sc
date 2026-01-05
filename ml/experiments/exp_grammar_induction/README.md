# exp_grammar_induction: Learn Go Syntax from Observation

## Goal

Reconstruct Go's formal syntax specification as a statechart by observing
valid Go code only - no access to the grammar definition.

## Hypothesis

Programming language syntax IS a statechart. We can learn it:
- **States** = Parser contexts (PACKAGE, IMPORT, FUNC_DECL, BLOCK, EXPR, etc.)
- **Transitions** = Token sequences that change context
- **Guards** = Conditions determining valid transitions
- **Hierarchy** = Nested structure (package > func > block > stmt > expr)

## Why Go?

1. Clean, regular syntax with official specification
2. Large corpus available (Go stdlib, GitHub)
3. Ground truth comparison via `go/parser` and `go/ast`
4. Minimal syntax ambiguity compared to C++/Rust

## Approach

### Phase 1: Token Sequence Learning
```
Input:  Stream of Go tokens (from go/scanner)
Output: Next-token probability distribution
Model:  Transformer or RNN tracking syntax state
```

### Phase 2: SAE State Discovery
```
Input:  Hidden states from Phase 1 model
Output: Discrete syntax states (via TopK SAE)
Goal:   Features = monosemantic syntax contexts
```

### Phase 3: Guard Synthesis
```
Input:  (state_before, token, state_after) triples
Output: Guard expressions for each transition
Method: Evolutionary synthesis (exp_guard_synthesis)
```

### Phase 4: Hierarchy Induction
```
Input:  Flat state machine from Phase 3
Output: Hierarchical statechart with AND/OR states
Method: Topology evolution (exp_topology_evolution)
```

## Evaluation

### Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| State Coverage | % of Go spec production rules captured | 90%+ |
| Transition Accuracy | Agreement with go/parser decisions | 95%+ |
| Rejection Rate | Correctly reject invalid Go code | 99%+ |
| False Positive | Accept invalid code as valid | <1% |

### Ground Truth Comparison

```go
// For each learned transition:
// 1. Generate test case exercising that transition
// 2. Parse with go/parser
// 3. Compare accept/reject decision

learned_state_machine.accepts("package main\nfunc main() {}")  // Should match go/parser
learned_state_machine.rejects("package\nfunc")  // Should match go/parser
```

### Spec Alignment

Map learned states to Go spec sections:
- SourceFile → learned state S_root
- PackageClause → learned state S_package
- FunctionDecl → learned state S_func
- Block → learned state S_block
- etc.

## Files to Implement

1. `token_collector.py` - Collect tokenized Go code samples
2. `sequence_model.py` - Next-token prediction model (MLX)
3. `sae_syntax.py` - SAE for syntax state discovery
4. `grammar_inducer.py` - Main grammar induction pipeline
5. `guard_learner.py` - Transition guard synthesis
6. `hierarchy_learner.py` - Discover nested structure
7. `go_parser_oracle.py` - Ground truth from go/parser
8. `spec_alignment.py` - Map states to Go spec
9. `benchmark.py` - Evaluation against ground truth

## Dataset

### Sources
- Go standard library (~500K lines)
- Popular Go repos (kubernetes, docker, etc.)
- Go playground submissions
- Synthetic variations (valid transformations)

### Negative Examples
- Intentionally broken Go code
- Other languages (Python, Rust) for rejection
- Partially valid code with subtle errors

## Integration with Existing Experiments

| Experiment | Contribution |
|------------|--------------|
| exp_sae_statechart | SAE architecture for state discovery |
| exp_guard_synthesis | Guard expression evolution |
| exp_topology_evolution | Hierarchy induction |
| exp_code_completion | Token-level prediction patterns |
| exp_transfer_learning | Transfer syntax structure across languages |

## Expected Results

1. **Discovered States**: ~50-100 syntax contexts matching Go spec
2. **Learned Guards**: Token predicates matching grammar rules
3. **Hierarchy**: AND-states for parallel structures (type params + body)
4. **Accuracy**: 95%+ agreement with go/parser on accept/reject

## Stretch Goals

1. **Cross-Language Transfer**: Learn Python syntax using Go structure
2. **Dialect Learning**: Discover Go 1.18+ generics from examples
3. **Error Recovery**: Learn robust parsing with error states
4. **Incremental Parsing**: Learn to update parse state efficiently

## Research Questions

1. How much data is needed to induce accurate grammar?
2. Can SAE features directly map to grammar non-terminals?
3. Does learned grammar generalize to unseen constructs?
4. How does sample efficiency compare to grammar specification?
