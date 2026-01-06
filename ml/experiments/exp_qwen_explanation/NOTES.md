# exp_qwen_explanation - Research Notes

## Core Goal

**Generate natural language explanations of statecharts using Qwen2.5-Coder-0.5B-Instruct.**

Target: 4+/5 clarity rating on generated explanations.

## Approach

### Hybrid Generation

Combine structured analysis with LLM generation:

1. **Structural Analysis** (rule-based):
   - Parse statechart structure
   - Classify states and transitions
   - Identify patterns and roles
   - Extract key elements

2. **LLM Enhancement** (Qwen):
   - Generate fluent natural language
   - Produce contextual descriptions
   - Create coherent narratives

3. **Post-processing**:
   - Validate accuracy
   - Ensure completeness
   - Format for readability

### Why Hybrid?

| Aspect | Pure Rules | Pure LLM | Hybrid |
|--------|-----------|----------|--------|
| Accuracy | High | Variable | High |
| Fluency | Low | High | High |
| Consistency | High | Variable | High |
| Speed | Fast | Slower | Medium |
| Coverage | Complete | May miss | Complete |

## Model Selection

### Qwen2.5-Coder-0.5B-Instruct

Chosen for:
- **Size**: 0.5B parameters fits on Apple Silicon
- **Speed**: Fast inference via mlx_lm
- **Quality**: Good for structured tasks
- **Instruct-tuned**: Follows prompts well

Alternative models:
- Qwen2.5-Coder-1.5B (better quality, slower)
- Qwen2.5-Coder-7B (best quality, GPU preferred)

### Prompt Engineering

Key prompt patterns:

```
You are explaining a statechart to a user.

Statechart: {name}
States: {state_list}
Initial: {initial_states}
Transitions:
{transition_list}

{level_instruction} {style_instruction}

Explain what this statechart does:
```

Level instructions:
- BRIEF: "Write a single sentence summary."
- STANDARD: "Write a clear paragraph explanation."
- DETAILED: "Write a comprehensive explanation covering all aspects."
- TECHNICAL: "Write a technical explanation with formal semantics."

Style instructions:
- CONVERSATIONAL: "Use friendly, accessible language."
- FORMAL: "Use precise, technical language."
- TUTORIAL: "Explain step-by-step as if teaching."

## Clarity Metrics

### Dimensions Evaluated

1. **Understandability** (25%)
   - Sentence length (target: 15-20 words)
   - Jargon avoidance
   - Structural clarity

2. **Completeness** (25%)
   - State coverage (% explained)
   - Transition coverage (% explained)
   - Summary presence

3. **Accuracy** (25%)
   - Correct state names
   - Correct event names
   - Correct relationships

4. **Fluency** (15%)
   - Complete sentences
   - Proper capitalization
   - Natural phrasing

5. **Conciseness** (10%)
   - Words per element (target: 20-40)
   - No excessive repetition

### Scoring Scale

| Score | Meaning |
|-------|---------|
| 5 | Excellent - clear, complete, engaging |
| 4 | Good - minor issues only |
| 3 | Acceptable - some clarity issues |
| 2 | Poor - significant issues |
| 1 | Very poor - fails to explain |

Target: Average 4+ across all dimensions.

## State Classification

States classified by role:

| Role | Characteristics | Description Pattern |
|------|-----------------|---------------------|
| INITIAL | is_initial=true | "Starting point where..." |
| FINAL | is_final=true | "End state representing..." |
| JUNCTION | High connectivity | "Decision point that..." |
| PROCESSING | Active work | "Active state where..." |
| WAITING | Low outgoing | "Waits for..." |
| ERROR | Name contains error/fail | "Error handling state..." |

## Transition Classification

Transitions classified by type:

| Type | Characteristics | Description Pattern |
|------|-----------------|---------------------|
| SIMPLE | A→B with event | "When {event}, transition to..." |
| GUARDED | Has guard | "When {event} and {guard}..." |
| COMPLETION | No event | "Upon completion, move to..." |
| SELF | src=target | "Remain in state, performing..." |
| COMPOUND | Multiple src/tgt | "From any of {srcs}, go to..." |

## Expected Benchmark Results

### By Explanation Level

| Level | Expected Clarity | Words/Element |
|-------|-----------------|---------------|
| BRIEF | 3.5 | 5-10 |
| STANDARD | 4.2 | 20-30 |
| DETAILED | 4.0 | 40-60 |
| TECHNICAL | 3.8 | 50-80 |

### By Statechart Complexity

| Complexity | Expected Clarity |
|------------|-----------------|
| Simple (3 states) | 4.5 |
| Medium (5-10 states) | 4.0 |
| Complex (10+ states) | 3.5 |

## Generation Strategies

### High-Level Summary

1. Identify purpose from name
2. Count states and transitions
3. Detect behavior pattern
4. Generate one-liner

Template:
> "{Name} is a {pattern} state machine with {n} states that {purpose}."

### State Descriptions

1. Classify role
2. Find connections
3. Infer behavior
4. Generate description

Template:
> "The {role} state '{label}' {behavior_description}. It can transition to {targets} on {events}."

### Transition Descriptions

1. Classify type
2. Identify trigger
3. Note guard/action
4. Generate description

Template:
> "When '{event}' occurs{guard_clause}, the system moves from '{src}' to '{tgt}'{action_clause}."

## Future Improvements

### 1. Interactive Explanations

Allow users to:
- Ask follow-up questions
- Request more detail on specific elements
- Get examples of execution traces

### 2. Multi-Format Output

Generate:
- Plain text
- Markdown with formatting
- HTML with diagrams
- Audio narration

### 3. Domain-Specific Templates

Specialized templates for:
- UI/UX statecharts
- Network protocols
- Game AI
- Workflow engines

### 4. Quality Improvement

- Fine-tune on statechart explanations
- Collect human ratings
- Iterative prompt refinement

## Integration with Other Experiments

| Experiment | Integration |
|------------|-------------|
| exp_code_completion | Explain generated statecharts |
| exp_trace_analysis | Explain execution traces |
| exp_sae_coverage | Explain discovered patterns |
| exp_formal_verification | Explain verification results |

## References

- "Natural Language Generation" - Reiter & Dale (2000)
- "Explaining Finite State Machines" - Academic literature
- Qwen technical reports - Alibaba
- mlx_lm documentation - Apple
