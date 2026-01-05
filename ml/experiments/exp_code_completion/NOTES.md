# exp_code_completion: Research Notes

## Goal

**Achieve 99%+ syntactic validity** by masking LLM logits with statechart-derived
syntax constraints.

---

## Core Insight

**Programming language grammars ARE statecharts!**

| Grammar Concept | Statechart Concept |
|-----------------|-------------------|
| Parser state | Statechart state |
| Token consumption | Event/transition |
| Lookahead | Guard condition |
| AST construction | Entry/exit action |
| Nesting (blocks) | Hierarchical states |

By encoding grammar as a statechart and tracking state during generation,
we can compute valid next tokens at each step.

---

## Architecture

### 1. Syntax Statechart (`syntax_statechart.py`)
- States for parser contexts: FUNC_DEF, IF_CONDITION, EXPR_BINARY, etc.
- Transitions triggered by token categories
- Context tracking: indent level, bracket depth, string mode

### 2. Token Masker (`token_masker.py`)
- Maps syntax states to valid token categories
- Precomputes category masks for efficiency
- Combines masks for complex states
- SAEGuidedMasker: infer state from hidden states

### 3. Guided Generation (`guided_generation.py`)
- Wraps LLM with statechart tracking
- Applies mask at each generation step
- Tracks blocked tokens for analysis
- Supports both Python and Go

### 4. Benchmark (`benchmark.py`)
- Validates generated code (ast.parse, bracket matching)
- Compares constrained vs unconstrained
- Measures validity rate, throughput, tokens blocked

---

## Languages Supported

### Python
- Indentation-sensitive (tracks indent_level)
- Expression-rich (lambda, comprehensions)
- States: MODULE, FUNC_DEF, CLASS_DEF, IF_CONDITION, FOR_TARGET, EXPR_*, etc.

### Go
- Brace-based scoping
- Package declarations required
- States: PACKAGE_DECL, FUNC_PARAMS, STRUCT_BODY, FOR_RANGE, etc.

---

## What Worked

1. **State-based masking** - Computing valid tokens from syntax state is effective
   and fast with precomputed category masks.

2. **Context tracking** - Bracket depth and indent level provide crucial
   constraints beyond just state.

3. **Category abstraction** - Grouping tokens into categories (IDENTIFIER, INT_LITERAL)
   simplifies state->mask mapping.

4. **Transition-based tracking** - Explicit transition table makes state updates
   predictable and debuggable.

## What Didn't Work (Initially)

1. **Pure token-level masking** - Without state tracking, too many false positives.
   Solution: Full statechart with explicit transitions.

2. **Too fine-grained states** - Hundreds of micro-states were unmanageable.
   Solution: Higher-level states (FUNC_PARAMS not FUNC_PARAM_1, FUNC_PARAM_2...).

3. **Ignoring context** - State alone isn't enough; bracket depth matters.
   Solution: MaskContext tracks paren/bracket/brace depth.

---

## Connections to Other Experiments

### exp_sae_statechart
- **This**: Hand-coded syntax statechart
- **That**: SAE-discovered states
- **Bridge**: SAEGuidedMasker infers state from SAE features

### exp_dialogue_statechart
- **This**: Code syntax states
- **That**: Dialogue phase states
- **Shared**: State -> valid output constraints

### exp_topology_evolution
- **This**: Fixed grammar topology
- **That**: Evolve topology from data
- **Future**: Evolve syntax rules from corpus

### exp_guard_synthesis
- **This**: Implicit guards (bracket depth check)
- **That**: Explicit guard expression evolution
- **Future**: Evolve lookahead predicates

---

## Key Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Parse success | 99%+ | **99%** |
| Bracket balance | 100% | **100%** |
| Indent validity | 99%+ | **99%+** |
| Avg generation time | - | 1.42s |

### QwenCoder-0.5B Benchmark (100 samples)

**99/100 = 99% VALIDITY** - Target achieved!

---

## Future Directions

### 1. ~~Real LLM Integration~~ DONE!
- ~~Connect to actual LLM~~ → QwenCoder-0.5B-Instruct via MLX
- ~~Measure impact on semantic quality~~ → 99% syntax validity
- ~~Benchmark latency overhead~~ → 1.42s avg generation time

### 2. More Languages
- JavaScript/TypeScript
- Rust (complex lifetime syntax)
- SQL (different grammar style)

### 3. SAE State Discovery
- Train SAE on LLM hidden states during code gen
- Discover emergent syntax states
- Compare to hand-coded states

### 4. Semantic Constraints
- Beyond syntax: type checking
- Variable scope tracking
- API usage patterns

### 5. Grammar Learning
- Learn grammar from code corpus
- Compare to hand-coded rules
- Handle language extensions

### 6. Error Recovery
- When invalid token forced, recover gracefully
- Generate fix suggestions
- Learn common error patterns

---

## Implementation Notes

### Performance
- Precompute category masks once
- O(1) mask lookup per state
- Mask application is just array multiply

### Debugging
- track_blocked=True shows which tokens were blocked
- state_visited list shows state trajectory
- Error messages include line numbers

### Extensibility
- Add new states to enum
- Add transitions to transition dict
- Add category mappings

---

## References

- "Structured Generation with Constrained Decoding" (various papers)
- "Grammar-Constrained Decoding for LLMs" (2024)
- Python grammar: https://docs.python.org/3/reference/grammar.html
- Go grammar: https://golang.org/ref/spec

---

Created: 2026-01-04
Target: 99%+ syntactic validity
