# Qwen Guard Synthesis Experiment

## Overview

This experiment uses Qwen2.5-Coder-0.5B-Instruct to generate boolean
guard expressions from natural language descriptions.

## Targets

- **Syntax validity**: 95%+ of generated expressions parse correctly
- **Semantic correctness**: 80%+ produce correct results

## Pipeline

```
NL Description
     │
     ▼
┌─────────────┐
│  NL Parser  │ ─── Extract intent, variables, operators
└─────────────┘
     │
     ▼
┌─────────────┐
│   Few-shot  │ ─── 8 examples teach guard syntax
│   Prompting │
└─────────────┘
     │
     ▼
┌─────────────┐
│    Qwen     │ ─── Generate candidate expressions
│   Coder     │
└─────────────┘
     │
     ▼
┌─────────────┐
│  Validator  │ ─── AST-based syntax check
└─────────────┘
     │
     ▼
┌─────────────┐
│  Benchmark  │ ─── Semantic evaluation
└─────────────┘
```

## Components

### guard_generator.py

- `QwenGuardGenerator`: Main generator class
- `GeneratorConfig`: Configuration (model, temperature, etc.)
- `FEW_SHOT_EXAMPLES`: 8 examples covering common patterns
- Uses `mlx_lm` for Apple Silicon acceleration
- Falls back to pattern matching if model unavailable

### nl_parser.py

- `NLParser`: Extract intent from natural language
- `IntentType`: COMPARISON, BOOLEAN, COMPOUND, NEGATION, etc.
- Pattern matching for common phrases
- `suggest_guard_structure()`: Hint for generator

### validator.py

- `GuardValidator`: AST-based syntax validation
- Uses Python's `ast.parse()` for robust parsing
- Tracks variables and operators used
- Optional semantic validation with test context

### benchmark.py

- 4 test categories:
  - Simple Comparisons (>, <, ==, etc.)
  - Boolean Expressions (and, or, not)
  - Game Rules (Ko, castling)
  - State Machine transitions
- Measures syntax validity and semantic correctness

## Few-Shot Examples

```python
# Comparison
"counter is greater than 5" → "counter > 5"

# Boolean
"player is alive and has mana" → "is_alive and (mana >= mana_cost)"

# Game rule
"move to same position as last capture" →
  "(move_x == last_capture_x) and (move_y == last_capture_y)"

# Negation
"king has not moved" → "not king_moved"
```

## Model Details

- **Model**: Qwen/Qwen2.5-Coder-0.5B-Instruct
- **Size**: 0.5B parameters (efficient for local use)
- **Framework**: MLX (Apple Silicon optimized)
- **Temperature**: 0.3 (deterministic for guards)

## Usage

```python
from exp_qwen_guard_synthesis import (
    QwenGuardGenerator,
    GuardValidator,
    run_full_benchmark,
)

# Generate guards
generator = QwenGuardGenerator()
guards = generator.generate(
    "counter is greater than 10",
    ["counter", "limit", "step"],
)

# Validate
validator = GuardValidator()
for guard in guards:
    is_valid, error = validator.validate(guard)
    print(f"{guard}: {'OK' if is_valid else error}")

# Benchmark
results = run_full_benchmark()
```

## Connections to Other Experiments

### exp_guard_synthesis
- Base patterns from llm_guard_gen.py
- AST structure from guard_synthesizer.py
- This experiment adds real LLM generation

### exp_code_completion
- Shares mlx_lm loading patterns
- Similar token constraint approach

### exp_temporal_guards
- Temporal guards could extend this with time predicates
- NL parser could be extended for "after 5 seconds"

## Future Directions

1. **Fine-tuning**: Train on guard-specific dataset
2. **Larger models**: Try 1.5B or 3B variants
3. **Chain-of-thought**: Have model explain reasoning
4. **Semantic repair**: Fix semantically incorrect guards
5. **Context-aware**: Include state machine context in prompt

## References

- Qwen2.5-Coder: https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct
- MLX: https://github.com/ml-explore/mlx
- exp_guard_synthesis/llm_guard_gen.py
