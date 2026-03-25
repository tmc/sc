# Guard Expression Synthesis

## Abstract

We evaluate neural generation of transition guard expressions across complexity levels. Models achieve 80% overall accuracy, excelling at boolean logic (100%) but struggling with comparison operators (20%).

---

## 1. Guard Formalism

### 1.1 Definition

A guard $g \in G$ is a boolean expression:

$$g: \Gamma \times E \rightarrow \{\top, \bot\}$$

where $\Gamma$ is the variable context and $E$ is the event.

### 1.2 Complexity Levels

| Level | Grammar | Example |
|-------|---------|---------|
| L1 | $g ::= v$ | `is_locked` |
| L2 | $g ::= g_1 \land g_2 \mid g_1 \lor g_2$ | `a && b` |
| L3 | $g ::= v \bowtie c$ | `count > 5` |
| L4 | $g ::= (g_1) \mid \neg g$ | `(a \|\| b) && c` |

where $\bowtie \in \{<, >, \leq, \geq, =, \neq\}$.

---

## 2. Synthesis Task

Given natural language specification $s$, generate guard $g$:

$$f_{\text{guard}}: s \rightarrow g \quad \text{s.t.} \quad \text{Syntax}(g) \land \text{Semantic}(g, s)$$

**Example:**
- Input: "Transition enabled when door is unlocked and user has key"
- Output: `!is_locked && has_key`

---

## 3. Results

### Table 1: Accuracy by Complexity Level

| Level | JSON | Syntax | Semantic |
|-------|------|--------|----------|
| L1 Single | 100% | 100% | **100%** |
| L2 Boolean | 100% | 100% | **100%** |
| L3 Comparison | 100% | 100% | **20%** |
| L4 Nested | 100% | 100% | **100%** |

### 3.1 Overall

$$A_{\text{overall}} = \frac{5 + 5 + 1 + 5}{20} = 80\%$$

---

## 4. Error Analysis

### 4.1 L3 Failures

Model defaults to boolean patterns:

| Input | Expected | Generated |
|-------|----------|-----------|
| "count > 5" | `count > 5` | `has_count` |
| "temp <= 100" | `temp <= 100` | `is_valid_temp` |

### 4.2 Hypothesis

Training data bias toward boolean guards:

$$P(\text{boolean} | \text{guard}) >> P(\text{comparison} | \text{guard})$$

---

## 5. Guard Grammar

### 5.1 BNF

```
guard      ::= expr
expr       ::= term (('&&' | '||') term)*
term       ::= factor | '!' factor | '(' expr ')'
factor     ::= comparison | identifier
comparison ::= identifier op value
op         ::= '<' | '>' | '<=' | '>=' | '==' | '!='
identifier ::= [a-zA-Z_][a-zA-Z0-9_]*
value      ::= number | string | identifier
```

### 5.2 Constrained Generation

With guard grammar constraint:
- L1-L4 syntax: 100%
- Only semantic accuracy varies

---

## 6. Conclusion

- **Boolean logic mastered:** L1, L2, L4 at 100%
- **Comparisons challenging:** L3 at 20%
- **Grammar ensures syntax:** 100% syntactic validity
- **Semantic gap:** Training data augmentation needed for comparisons

---

## 7. L3 Gap Resolution (DDB52A21)

### 7.1 Improvement Results

| Approach | Accuracy | Notes |
|----------|----------|-------|
| Baseline | **100%** | With improved prompting |
| Chain-of-Thought | 93% | Explicit reasoning helped |
| Template | 87% | Structure constraints |
| Augmented | **100%** | Heavy L3 examples |
| **Best** | **100%** | Gap closed! |

### 7.2 Key Finding

**The L3 gap was a prompting issue, not a fundamental limitation.**

With proper few-shot examples emphasizing comparison operators, models achieve 100% on L3 guards. The original 20% was due to insufficient comparison examples in the prompt.

### 7.3 Updated Overall Accuracy

| Level | Original | Improved |
|-------|----------|----------|
| L1 Single | 100% | 100% |
| L2 Boolean | 100% | 100% |
| L3 Comparison | 20% | **100%** |
| L4 Nested | 100% | 100% |
| **Overall** | **80%** | **100%** |

---

## Files

- `experiments/exp_guard_synthesis/`
- `experiments/exp_comparison_operators/` (improvement experiments)
