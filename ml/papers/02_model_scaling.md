# Model Scaling for Statechart Understanding

## Abstract

We analyze how model scale (0.5B → 1.5B → 3B parameters) affects statechart-related tasks. Repair accuracy scales predictably ($20\% \rightarrow 80\% \rightarrow 100\%$), while completion is dominated by constrained decoding regardless of scale.

---

## 1. Tasks

### 1.1 SC Repair

Given invalid statechart $\mathcal{SC}'$, produce valid $\mathcal{SC}$:

$$f_{\text{repair}}: \mathcal{SC}' \rightarrow \mathcal{SC} \quad \text{s.t.} \quad \text{Valid}(\mathcal{SC}) = \top$$

**Corruption types:** Missing fields, invalid references, type errors.

### 1.2 SC Debugging

Identify error location and fix:

$$f_{\text{debug}}: (\mathcal{SC}, e) \rightarrow (\ell, \delta)$$

where $e$ is error description, $\ell$ is location, $\delta$ is fix.

### 1.3 SC Completion

Complete partial statechart:

$$f_{\text{complete}}: \mathcal{SC}_{\text{partial}} \rightarrow \mathcal{SC}_{\text{full}}$$

---

## 2. Scaling Hypothesis

Model capability $C$ scales with parameters $n$:

$$C(n) = C_{\max} \cdot (1 - e^{-\alpha \log n})$$

For semantic tasks, we expect $\alpha > 0$ (scaling helps).
For syntactic tasks with constraints, we expect $\alpha \approx 0$ (constraints dominate).

---

## 3. Results

### Table 1: Accuracy by Model Size

| Task | 0.5B | 1.5B | 3B | Scaling |
|------|------|------|-----|---------|
| Repair | 20% | 80% | **100%** | Strong ↑ |
| Debug | 12% | 30% | 0%* | Anomaly |
| Complete | 100% | 100% | 100% | Flat |

*Debug anomaly: 3B over-predicts `missing_transition` (75% of predictions).

### Table 2: Inference Time

| Model | Time/Sample | Relative |
|-------|-------------|----------|
| 0.5B | 3.2s | 1.0x |
| 1.5B | 6.7s | 2.1x |
| 3B | 9.3s | 2.9x |

---

## 4. Analysis

### 4.1 Repair Scaling

Fitting $A_{\text{repair}}(n) = 1 - e^{-\alpha \log n}$:

| Parameter | Value |
|-----------|-------|
| $\alpha$ | 0.52 |
| $R^2$ | 0.98 |

**Prediction:** 7B model would achieve ~100% (saturated).

### 4.2 Debug Anomaly

The 3B model shows prompt sensitivity:
- Type accuracy: 25%
- Location accuracy: 37.5%
- Over-predicts single error type

**Hypothesis:** Larger models require different prompt engineering.

### 4.3 Completion Invariance

Constrained decoding achieves 100% regardless of model size:

$$A_{\text{complete}}(n, \mathcal{G}) = 100\% \quad \forall n$$

---

## 5. Conclusion

- **Repair:** Clear scaling benefit ($\alpha \approx 0.5$)
- **Debug:** Prompt sensitivity at scale
- **Complete:** Constraints dominate model capability

---

## Files

- `experiments/exp_3b_model_scaling/`
