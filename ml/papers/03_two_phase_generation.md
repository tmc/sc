# Two-Phase Generation: Structure Then Execution

## Abstract

We introduce two-phase generation where models first generate a statechart structure, then generate execution traces constrained by that structure. Dynamic constraint loading achieves near-zero overhead (~0ms), enabling real-time grammar switching.

---

## 1. Motivation

Single-phase generation conflates structure and behavior:

$$y = f(s; \theta) \quad \text{(structure + trace mixed)}$$

Two-phase separates concerns:

$$\mathcal{SC} = f_{\text{struct}}(s; \theta, \mathcal{G}_{\text{SC}})$$
$$\tau = f_{\text{trace}}(g; \theta, \mathcal{SC})$$

---

## 2. Method

### 2.1 Phase 1: Structure Generation

Generate statechart constrained by SC grammar:

$$P(\mathcal{SC} | s) = \prod_{t=1}^{T} P_{\mathcal{G}_{\text{SC}}}(y_t | y_{<t}, s)$$

### 2.2 Phase 2: Trace Generation

Load generated SC as dynamic constraint. Valid events at configuration $\sigma$:

$$\mathcal{E}_{\text{valid}}(\sigma, \mathcal{SC}) = \{e \in E : \exists t \in \delta, \text{src}(t) \cap \sigma \neq \emptyset \land \text{event}(t) = e\}$$

### 2.3 Dynamic Loading

```
on_token(v):
    if v == "<SC_END>":
        SC ← parse(buffer)
        constraint ← SCExecutor(SC)
        mode ← TRACE
    elif mode == TRACE:
        valid ← constraint.valid_events()
        if v ∉ valid:
            return REJECT
        constraint.step(v)
    return ACCEPT
```

---

## 3. Results

### Table 1: Two-Phase Performance

| Metric | Result |
|--------|--------|
| SC Validity | **100%** |
| Trace Validity | **100%** |
| Goal Reached | 67% |

### Table 2: Dynamic Loading Overhead

| Operation | Time |
|-----------|------|
| SC Parse | 0.02ms |
| Mode Switch | ~0ms |
| Event Check | ~0ms |

---

## 4. Analysis

### 4.1 Perfect Validity

Both phases achieve 100% validity:
- Phase 1: SC grammar constraint
- Phase 2: SC execution constraint

### 4.2 Goal Reachability Gap

67% goal reachability indicates planning difficulty:

$$P(\text{goal} | \text{valid trace}) = 0.67$$

Not a constraint failure—model chooses valid but non-optimal paths.

### 4.3 Zero Overhead

Dynamic loading cost is negligible:

$$\text{overhead} = \frac{t_{\text{constrained}} - t_{\text{unconstrained}}}{t_{\text{unconstrained}}} \approx 0\%$$

---

## 5. Goal-Directed Extension

### 5.1 Explicit Goals

With explicit goal specification:

| Metric | Result |
|--------|--------|
| Reachability | **100%** |
| Optimality | **100%** |
| Invalid Attempts | 0 |

### 5.2 Goal Types Tested

| Goal Type | Example | Success |
|-----------|---------|---------|
| Single state | "reach Yellow" | 100% |
| Predicate | "reach is_final" | 100% |
| Sequence | "reach A then B" | 100% |
| Avoidance | "reach C, avoid X" | 100% |

---

## 6. Conclusion

Two-phase generation enables:
1. **Perfect validity** for both structure and behavior
2. **Dynamic constraints** with zero overhead
3. **Goal-directed planning** when goals are explicit

---

## Files

- `experiments/exp_sc_then_trace/`
- `experiments/exp_dynamic_sampler_loading/`
- `experiments/exp_goal_directed_traces/`
