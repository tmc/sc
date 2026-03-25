# Topology Learning from Behavior

## Abstract

We evaluate neural inference of statechart topology from execution traces. Using step-by-step enumeration, models achieve 82% overall accuracy—with perfect state extraction (100%) and strong transition inference (78%).

---

## 1. Task Definition

Given execution traces $T = \{\tau_1, \ldots, \tau_n\}$, infer topology $(S, \delta)$:

$$f_{\text{topo}}: \mathcal{P}(\tau) \rightarrow (S, \delta)$$

where $S$ is the state set and $\delta$ is the transition relation.

---

## 2. Method: Scratchpad Enumeration

Step-by-step extraction from traces:

```
Traces: [A,B,C,A], [A,C,B,A], [A,B,B,C]

Step 1: Unique states = {A, B, C}
Step 2: Transitions = {A→B, B→C, C→A, A→C, C→B, B→B}
Step 3: Patterns: B→B is self-loop, A has multiple outgoing
Step 4: Initial: A (all traces start here)
```

---

## 3. Results (9D1B42F3)

### Table 1: Overall Accuracy

| Metric | Accuracy |
|--------|----------|
| States | **100%** |
| Transitions | 78% |
| **Overall** | **82%** |

### Table 2: By Pattern Type

| Pattern | Accuracy | Notes |
|---------|----------|-------|
| Self-loop | **90%** | Best - explicit B→B detection |
| Branch | **87%** | Multiple outgoing handled well |
| Cycle | 80% | Return transitions found |
| Linear | 70% | Simple chains |

---

## 4. Analysis

### 4.1 Perfect State Extraction

Models excel at set enumeration:
$$S = \bigcup_{\tau \in T} \{s : s \in \tau\}$$

This is a straightforward aggregation task.

### 4.2 Transition Inference Gap

78% transitions vs 100% states suggests:
- Some transitions not observed in traces
- Edge cases in transition direction

### 4.3 Pattern Difficulty

Counter-intuitively, complex patterns (self-loop, branch) outperform simple (linear):
- Self-loops have explicit repeated states
- Branches show clear divergence
- Linear chains may lack distinguishing features

---

## 5. Conclusion

- **States:** Perfect extraction (100%)
- **Transitions:** Strong inference (78%)
- **Overall:** Exceeds 50% target at 82%
- **Key insight:** Scratchpad enumeration enables topology learning from behavior

---

## 6. State Type Prediction (exp_state_type_prediction, DDB52A21)

### Task
Classify states as BASIC, OR, or PARALLEL from context.

### Results

| Type | Accuracy |
|------|----------|
| BASIC | 86% |
| OR | 86% |
| **PARALLEL** | **100%** |
| **Overall** | **85%** |

### Key Finding

**PARALLEL detection is perfect (100%).**

Models easily identify concurrent/independent regions. The AND-semantics keywords ("simultaneously", "both", "independent") are strong signals.

BASIC and OR are slightly harder (86%) due to ambiguity—a state with children could be OR (exclusive) or the children could be independent (suggesting PARALLEL parent).

---

## 7. exp_topology_completion (12FF641D)

### Task
Complete partial SC topologies by identifying and fixing structural issues.

### Results

| Method | Accuracy |
|--------|----------|
| Algorithm | 75% |
| **LLM** | **88%** |

### By Issue Type

| Issue | Algorithm | LLM |
|-------|-----------|-----|
| Missing transition | 0% | **100%** |
| No terminal | 100% | 100% |
| No initial | 100% | 100% |
| Orphan states | 100% | 100% |
| Dead ends | 100% | 100% |
| **Complex (multi-issue)** | **0%** | **0%** |

### Key Findings

1. **LLM beats algorithm on missing transitions** - Algorithm cannot infer which transition to add; LLM reasons about connectivity.

2. **Both methods handle single issues perfectly** - Terminal, initial, orphan, dead-end fixes are straightforward.

3. **Complex cases remain hard** - When multiple issues combine, both methods fail. This suggests a need for iterative repair or decomposition.

---

## 8. Summary: Topology Learning

| Task | Target | Result | Method |
|------|--------|--------|--------|
| Topology Inference | 50% | **82%** | Scratchpad |
| State Type Prediction | 80% | **85%** | Templates |
| Topology Completion | 70% | **88%** | LLM reasoning |

**Overall finding**: Topology learning is tractable with proper prompting techniques. Only complex multi-issue cases and deep hierarchy semantics remain challenging.

---

## 9. SC Discovery from Behavior

### 9.1 Trace Pattern Completion (exp_trace_pattern_completion, 12FF641D)

**Task**: Learn pattern from example traces, complete partial traces.

| Method | Pattern Detection | 1-step Complete | 3-step Complete |
|--------|-------------------|-----------------|-----------------|
| Algorithm | 50% | **100%** | **100%** |
| LLM | - | 10% | **0%** |

### Key Finding: Algorithm Dominates

**Transition map lookup is trivial algorithmically, impossible for LLMs.**

Once transitions are extracted from examples, completion is a simple table lookup:
```
Observed: A→B, B→C, C→A
Query: [A,B,C,A,B,?]
Lookup: After B comes C
Answer: C
```

LLMs fail at 0-10% because they cannot reliably execute this lookup, even when the pattern is explicitly shown. This identifies **execution/lookup** as a fundamental LLM limitation distinct from **understanding/induction**.

### Implication

For trace completion tasks, use **hybrid approach**:
1. LLM induces transition rules from examples
2. Algorithm executes the rules for prediction

Pure LLM prediction is not viable for this task.

### 9.2 Few-Shot SC Discovery (exp_few_shot_sc_discovery, DDB52A21)

**Task**: Discover SC from minimal I/O examples, predict new outputs.

| Metric | Result |
|--------|--------|
| SC Discovery | **100%** |
| Predict (observed) | **100%** |
| Predict (novel) | **78%** |

| Examples Provided | SC Accuracy |
|-------------------|-------------|
| 2 examples | **100%** |
| 3 examples | **100%** |
| 5 examples | **100%** |

### Key Finding: Minimal Examples Suffice

**Just 2 I/O examples enable perfect SC induction.**

The model correctly identifies:
- State set from observed inputs/outputs
- Transitions from I/O pairs
- SC structure (toggle, counter, etc.)

The 78% novel prediction accuracy shows strong but imperfect generalization. Failures occur on:
- Unobserved state combinations
- Ambiguous cases where multiple SCs fit the examples

### Contrast with Trace Completion

| Task | LLM Strength |
|------|--------------|
| SC Discovery (induction) | **100%** ✓ |
| Trace Completion (execution) | 0-10% ✗ |

LLMs excel at **induction** (finding patterns) but fail at **execution** (applying patterns). This confirms the hybrid approach: LLM for discovery, algorithm for prediction.

### 9.3 Extended Pattern Completion (exp_trace_pattern_extended, 12FF641D)

**Task**: Test n-gram context on complex patterns.

| Method | Accuracy |
|--------|----------|
| Unigram (1 previous state) | 74% |
| **2-gram (2 previous states)** | **96%** |

| Pattern Type | Unigram | 2-gram |
|--------------|---------|--------|
| Context-dependent | 0% | **100%** |
| Conditional | 0% | **100%** |
| **Long-range** | **0%** | **0%** |

### Key Finding: Local Context Solves Complex Patterns

**2-gram context provides +22% improvement over unigram.**

- Context-dependent patterns: 0%→100% with 2-gram
- Conditional patterns: 0%→100% with 2-gram
- Ensemble voting: No improvement (not an uncertainty issue)

### Remaining Gap: Long-Range Dependencies

Long-range patterns (where current state depends on states many steps back) remain at 0% regardless of method. This suggests:
- Fixed context window insufficient
- May need recurrent/attention mechanisms
- Or explicit memory/scratchpad for long-range

### 9.4 Hierarchy Scratchpad (exp_hierarchy_scratchpad, 91C67E33)

**Task**: Apply scratchpad technique to hierarchy understanding.

| Method | Overall |
|--------|---------|
| Baseline | 27% |
| **Scratchpad** | **48%** |

| Task | Accuracy |
|------|----------|
| Containment ("Is B inside A?") | **62%** |
| Depth ("How deep is state X?") | **60%** |
| Entry ("What's the config entering X?") | 50% |
| **Cascade ("What exits when X exits?")** | **17%** |

### Key Finding: Scratchpad Partially Helps Hierarchy

Scratchpad improves hierarchy from 27%→48% (+21pp), but **cascade semantics remains hard at 17%**.

Tasks with clear parent-chain traversal improve:
- Containment: trace up parent chain ✓
- Depth: count parent chain length ✓

Tasks requiring simultaneous tracking of multiple descendants fail:
- Cascade: must enumerate ALL descendants ✗

### Hierarchy Gap Analysis

| Technique | Path/Graph | Hierarchy |
|-----------|------------|-----------|
| Scratchpad | **83%** | 48% |
| Cascade specifically | N/A | **17%** |

Hierarchy cascade remains the **hardest unsolved problem**. May require:
- Fine-tuning on cascade examples
- Recursive scratchpad (enumerate children, then their children)
- Tree-specific prompting techniques

### 9.5 Stateful Behavior Induction (exp_stateful_behavior_induction, 9D1B42F3)

**Task**: Induce rules from stateful I/O sequences, then predict.

| Metric | Result |
|--------|--------|
| Rule induction | **88%** |
| Predict (seen patterns) | 25% |
| Predict (novel inputs) | 10% |

| Machine Type | Prediction Accuracy |
|--------------|---------------------|
| Counter | 58% |
| Queue | 50% |
| Stack | 38% |
| Accumulator | 33% |

### The Understanding-Execution Gap

**LLMs understand rules (88%) but can't apply them (25%).**

This confirms findings across all SC discovery experiments:

| Task | Understanding | Execution |
|------|---------------|-----------|
| SC Discovery | **100%** | - |
| Rule Induction | **88%** | 25% |
| Topology Inference | **82%** | - |
| Pattern Recognition | **96%** | - |
| Trace Completion | - | **0-10%** |

---

## 10. Conclusions: LLM Capabilities for Statecharts

### What LLMs Excel At (80-100%)
- **Induction**: Discovering SC from examples
- **Pattern recognition**: Identifying structure in behavior
- **Classification**: State types, topology features
- **Completion**: Fixing partial structures

### What LLMs Struggle With (0-25%)
- **Execution**: Applying discovered rules
- **Lookup**: Table-based state transitions
- **Long-range**: Dependencies beyond local context
- **Cascade**: Hierarchical descendant enumeration

### Recommended Architecture

```
┌─────────────────────────────────────────────┐
│           HYBRID SC SYSTEM                  │
├─────────────────────────────────────────────┤
│  I/O Examples                               │
│       ↓                                     │
│  [LLM: Induce SC]  ← LLM excels (88-100%)  │
│       ↓                                     │
│  Statechart JSON                            │
│       ↓                                     │
│  [Algorithm: Execute] ← Code excels (100%) │
│       ↓                                     │
│  Predictions                                │
└─────────────────────────────────────────────┘
```

**LLMs for thinking. Algorithms for doing.**

---

## 11. Harel Feature Coverage

### 11.1 Temporal Behaviors (exp_temporal_behaviors, 12FF641D)

**Task**: Test time-based SC behaviors (after, timeout, debounce).

| Method | Overall |
|--------|---------|
| Algorithm | **77%** |
| LLM | 18% |

| Pattern | Accuracy |
|---------|----------|
| Debounce | **100%** |
| Reset (timer reset) | **100%** |
| Backoff | 67% |
| Deadline | 67% |
| Simple after() | 50% |

**Finding**: Pattern-based temporal behaviors (debounce, reset) work perfectly. Complex timing (backoff, deadline) partially work. LLMs struggle at 18%.

### 11.2 Action Execution (exp_action_execution_comprehensive, DDB52A21)

**Task**: Test entry/exit/transition actions and context mutations.

| Action Type | Accuracy |
|-------------|----------|
| Entry | 67% |
| Exit | 50% |
| Transition | 50% |
| Order | 50% |
| Context mutation | 75% |

| Prediction Type | Accuracy |
|-----------------|----------|
| State | **88%** |
| Order | 76% |
| Context | 65% |
| **All correct** | **47%** |

**Finding**: Strong state prediction (88%) but compounding errors. When state, order, AND context must all be correct, accuracy drops to 47%. Struggles with:
- Multi-event sequences
- Nested object access
- Complex execution order

### 11.3 Context Guards (exp_context_guards_comprehensive, 91C67E33)

**Task**: Test guard evaluation across complexity levels.

| Guard Type | Accuracy |
|------------|----------|
| Boolean | **100%** |
| Comparison | **100%** |
| Compound (AND/OR) | **100%** |
| Priority resolution | **100%** |
| Chain (action→guard) | 67% |
| **Overall** | **83%** |

**Finding**: Guard evaluation is strong (83%). Single-step guards perfect (100%). Chain reasoning (where action sets variable, next guard checks it) drops to 67%—consistent with execution gap.

---

## Files

- `experiments/exp_topology_inference/`
- `experiments/exp_state_type_prediction/`
- `experiments/exp_topology_completion/`
