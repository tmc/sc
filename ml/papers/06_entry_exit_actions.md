# Entry/Exit Action Generation

## Abstract

We study neural generation of state entry and exit actions. Models show asymmetric learning: entry actions achieve 100% accuracy while exit actions achieve only 50%, with models frequently confusing the two.

---

## 1. Action Formalism

### 1.1 Definition

State actions are defined by the mapping:

$$\gamma: S \rightarrow (A^*_{\text{entry}}, A^*_{\text{exit}})$$

**Entry actions** execute when entering state $s$:
$$\text{enter}(s) \Rightarrow \text{exec}(\gamma_{\text{entry}}(s))$$

**Exit actions** execute when leaving state $s$:
$$\text{exit}(s) \Rightarrow \text{exec}(\gamma_{\text{exit}}(s))$$

### 1.2 Execution Order

For transition $s_1 \xrightarrow{e} s_2$:

$$\text{exec}(\gamma_{\text{exit}}(s_1)) \rightarrow \text{exec}(\alpha_t) \rightarrow \text{exec}(\gamma_{\text{entry}}(s_2))$$

where $\alpha_t$ is the transition action.

---

## 2. Test Scenarios

| Scenario | Description |
|----------|-------------|
| Entry only | Single entry action |
| Exit only | Single exit action |
| Both | Entry and exit on same state |
| Transition | Action on transition itself |
| Chained | Multiple actions in sequence |

---

## 3. Results

### Table 1: Action Generation Accuracy

| Scenario | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| Entry only | 8 | 8 | **100%** |
| Exit only | 4 | 8 | 50% |
| Both | 2 | 6 | 33% |
| Transition | 4 | 6 | 67% |
| Chained | 2 | 4 | 50% |

---

## 4. Error Analysis

### 4.1 Entry/Exit Confusion

Common error pattern:

| Request | Expected | Generated |
|---------|----------|-----------|
| Exit action | `on_exit: log("leaving")` | `on_entry: log("leaving")` |

### 4.2 Confusion Matrix

|  | Pred Entry | Pred Exit |
|--|------------|-----------|
| **True Entry** | 8 | 0 |
| **True Exit** | 4 | 4 |

**False positive rate for entry:** $4/8 = 50\%$

### 4.3 Hypothesis

Training data contains more entry than exit examples:

$$\frac{|\text{entry examples}|}{|\text{exit examples}|} >> 1$$

---

## 5. Combined Actions

### 5.1 Both Entry and Exit

When both required, accuracy drops to 33%:

$$P(\text{both correct}) = P(\text{entry}) \cdot P(\text{exit} | \text{entry}) \approx 1.0 \cdot 0.33$$

### 5.2 Pattern

Model tends to generate entry action for both:
```json
{
  "entry_actions": [{"expression": "start()"}],
  "exit_actions": [{"expression": "start()"}]  // Should be stop()
}
```

---

## 6. Action Grammar

```
state_actions ::= entry_actions? exit_actions?
entry_actions ::= '"entry_actions"' ':' action_array
exit_actions  ::= '"exit_actions"' ':' action_array
action_array  ::= '[' (action (',' action)*)? ']'
action        ::= '{' action_fields '}'
action_fields ::= label? expression params?
```

---

## 7. Exit Gap Resolution (DDB52A21)

### 7.1 Improvement Results

| Scenario | Baseline | Improved |
|----------|----------|----------|
| Entry only | 100% | 100% |
| Exit only | 0% | **100%** |
| Both | - | **100%** |

### 7.2 Key Finding

**The exit bias was a prompting issue, not a fundamental limitation.**

With balanced few-shot examples emphasizing exit actions equally:
- Heavy exit examples in prompt
- Explicit "EXIT action runs when LEAVING" markers
- Entry/exit contrast pairs

### 7.3 Updated Conclusion

- **Entry actions:** 100% ✓
- **Exit actions:** 100% ✓ (was 50%)
- **Both together:** 100% ✓
- **Gap status:** CLOSED

---

## Files

- `experiments/exp_entry_exit_actions/`
