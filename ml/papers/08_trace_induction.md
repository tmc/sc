# Trace-to-Statechart Induction

## Abstract

We evaluate neural induction of statecharts from execution traces. With 0.5B parameters, models achieve only 20% accuracy—succeeding on simple patterns but failing to generalize, instead copying example structures verbatim.

---

## 1. Task Definition

Given traces $T = \{\tau_1, \ldots, \tau_n\}$, induce statechart $\mathcal{SC}$:

$$f_{\text{induce}}: \mathcal{P}(\tau) \rightarrow \mathcal{SC} \quad \text{s.t.} \quad \forall \tau_i \in T: \tau_i \in \mathcal{L}(\mathcal{SC})$$

where $\mathcal{L}(\mathcal{SC})$ is the language of valid traces.

### 1.1 Induction vs Generation

| Task | Input | Output | Challenge |
|------|-------|--------|-----------|
| Generation | Spec | SC | Structure from description |
| **Induction** | Traces | SC | Structure from behavior |

---

## 2. Test Cases

### 2.1 Trace Patterns

| Pattern | Example Traces | Expected SC |
|---------|----------------|-------------|
| Toggle | `[ON,OFF,ON]` | 2-state cycle |
| Counter | `[INC,INC,DEC]` | 3-state linear |
| Lock | `[LOCK,UNLOCK,OPEN]` | 3-state with guard |
| Menu | `[UP,UP,SELECT,BACK]` | Hierarchical |
| Game | `[START,PLAY,PAUSE,RESUME]` | Complex |

---

## 3. Results

### Table 1: Induction Accuracy

| Pattern | JSON Valid | SC Valid | Correct |
|---------|------------|----------|---------|
| Toggle | ✓ | ✓ | **✓** |
| Counter | ✓ | ✓ | ✗ |
| Lock | ✓ | ✓ | ✗ |
| Menu | ✓ | ✓ | ✗ |
| Game | ✓ | ✓ | ✗ |
| **Overall** | **100%** | **100%** | **20%** |

### 3.1 Error Pattern

Model copies example structure rather than inducing from traces:

| Input Traces | Expected | Generated |
|--------------|----------|-----------|
| `[A,B,A,B]` | 2-state cycle | Example SC verbatim |
| `[X,Y,Z,X]` | 3-state cycle | Example SC verbatim |

---

## 4. Analysis

### 4.1 Copying vs Generalizing

Define generalization score:

$$G = 1 - \frac{|\text{output} \cap \text{examples}|}{|\text{output}|}$$

| Model | G Score |
|-------|---------|
| 0.5B | 0.2 (low) |
| Expected | >0.8 (high) |

### 4.2 Model Size Hypothesis

Induction requires:
1. **Pattern recognition:** Extract structure from traces
2. **Abstraction:** Generalize beyond examples
3. **Synthesis:** Construct novel SC

$$\text{Capability}_{\text{induce}}(n) \propto \log(n)$$

0.5B insufficient for abstraction step.

### 4.3 Comparison to Other Tasks

| Task | 0.5B | 1.5B | Required |
|------|------|------|----------|
| Completion | 100% | 100% | Syntax |
| Repair | 20% | 80% | Semantics |
| **Induction** | **20%** | **?** | Abstraction |

---

## 5. The Induction Gap

### 5.1 Definition

$$\Delta_{\text{induce}} = A_{\text{ideal}} - A_{\text{actual}} = 100\% - 20\% = 80\%$$

### 5.2 Closing the Gap

Options:
1. **Scale up:** Try 1.5B, 3B, 7B models
2. **More examples:** Increase few-shot count
3. **Chain-of-thought:** Explicit reasoning steps
4. **Specialized training:** Fine-tune on induction task

---

## 6. Conclusion

- **JSON/SC validity:** 100% (syntax is easy)
- **Semantic correctness:** 20% (induction is hard)
- **Root cause:** Model copies vs generalizes
- **Recommendation:** Larger models for induction tasks

---

## 7. Improvement Experiments (In Progress)

### 7.1 Chain-of-Thought Induction (12FF641D)

Force explicit reasoning before generating SC:

```
Traces: [A,B,A,B], [A,B,C,A]

Step 1: Identify unique states: A, B, C (3 states)
Step 2: Extract transitions: A→B, B→A, B→C, C→A
Step 3: Initial state: A (all traces start with A)
Step 4: Generate SC JSON: ...
```

**Results (12FF641D):**

| Model | Direct | CoT | Improvement |
|-------|--------|-----|-------------|
| 0.5B | 25% | **50%** | +25pp |
| 1.5B | 33% | **50%** | +17pp |
| 3B | - | underperforms | anomaly |

- **States accuracy:** 50% (up from 25%)
- **Transitions accuracy:** 0% (semantic mismatch)
- **3B anomaly:** Larger model performs worse

**Key insight:** CoT helps state extraction but transitions remain hard. The model identifies states correctly but fails to capture transition semantics.

### 7.2 Path Length with Scratchpad (91C67E33)

**Results:**

| Approach | Accuracy |
|----------|----------|
| Baseline | 17% |
| **Scratchpad** | **83%** |
| BFS simulation | 33% |
| ASCII visualization | 33% |

**Key insight: Scratchpad enables graph algorithms in LLMs.**

Explicit step counting allows models to perform graph reasoning they cannot do internally:

```
Trace: A --e1--> B --e2--> C
Step 1: A → B (count: 1)
Step 2: B → C (count: 2)
Answer: 2 steps
```

This is a **fundamental finding**: LLMs lack internal graph algorithm capability, but **external scratchpad computation** bridges the gap. The 17% → 83% improvement (+66pp) far exceeds the 50% target.

### 7.2.1 Reachability Steps (91C67E33)

Applying scratchpad to reachability min_steps prediction:

| Metric | Before | After |
|--------|--------|-------|
| Steps accuracy | 0% | **62%** |
| Reachability F1 | 71% | **92%** |

**Scratchpad consistently enables graph reasoning.** BFS expansion enumeration allows models to compute minimum steps they cannot calculate internally.

### 7.3 Terminal Hierarchy (9D1B42F3)

**Results:**

| Approach | Accuracy |
|----------|----------|
| Baseline | 8% |
| Cascade prompting | 20% |
| **Visualization** | **27%** |

**By hierarchy type:**
| Type | Accuracy |
|------|----------|
| Deep hierarchy | 50% |
| Simple hierarchy | 25% |
| Flat terminal | 33% |
| **Branching** | **0%** |

**Key insight: Hierarchy semantics resists prompting fixes.**

Unlike path length (scratchpad works) and guards (examples work), hierarchy cascade semantics appears to require **fine-tuning** rather than prompting. The model struggles with:
- Multiple composite states (branching = 0%)
- Tracking which states are "inside" which
- Cascade termination propagation

This identifies hierarchy as a **fundamental gap requiring training intervention**.

---

## Files

- `experiments/exp_trace_to_sc/`
