# exp_trace_pattern_completion

## Hypothesis

Pattern recognition with CoT-style scratchpad reasoning can achieve
high accuracy on trace completion tasks.

## Approach

1. **Pattern Learning**: Analyze example traces to extract patterns
   - Detect cycle periods
   - Identify alternating sequences
   - Recognize growth patterns
   - Build transition maps

2. **CoT Scratchpad**: Explicit reasoning steps
   ```
   Step 1 - Analyze example traces
   Step 2 - Find repeating pattern
   Step 3 - Verify pattern
   Step 4 - Build SC (transition map)
   Step 5 - Complete partial trace
   ```

3. **Completion**: Use learned pattern to predict next events
   - 1-step prediction (next event)
   - 3-step prediction (next 3 events)

## Pattern Types

| Type | Example | Detection Method |
|------|---------|------------------|
| cycle | [A,B,C,A,B,C] | Check period repetition |
| alternating | [A,B,A,B] | Period-2 cycle |
| growth | [A], [A,B], [A,B,C] | Prefix relationship |
| nested | [START,A,B,END,...] | Composite patterns |

## Test Cases

| Name | Type | Partial | Expected |
|------|------|---------|----------|
| cycle_3 | cycle | [A,B,C,A,B] | [C] |
| cycle_4 | cycle | [W,X,Y] | [Z] |
| traffic_light | cycle | [RED,GREEN] | [YELLOW] |
| toggle | alt | [ON,OFF,ON] | [OFF] |
| ping_pong | alt | [PING,PONG,...] | [PING] |
| growth_linear | growth | [A,B,C,D] | [] (ends) |
| counting | growth | [ONE,TWO] | [THREE] |
| nested_ab | nested | [...,START,A] | [B] |
| door_sequence | cycle | [...,OPEN] | [CLOSING] |
| single_repeat | cycle | [X,X] | [X] |

## Results

**Algorithmic completion achieved 100% accuracy!**

| Method | Pattern Acc | 1-step | 3-step | Avg Time |
|--------|-------------|--------|--------|----------|
| Algorithmic | 50% | **100%** | **100%** | 0.2ms |
| LLM (1.5B) | 50% | 10% | 0% | 8226ms |

### By Pattern Type (Algorithmic)

| Type | 1-step | 3-step |
|------|--------|--------|
| cycle | 100% | 100% |
| alt | 100% | 100% |
| growth | 100% | 100% |
| nested | 100% | 100% |

### By Pattern Type (LLM)

| Type | 1-step | 3-step |
|------|--------|--------|
| cycle | 20% | 0% |
| alt | 0% | 0% |
| growth | 0% | 0% |
| nested | 0% | 0% |

## Key Findings

### What Worked

1. **Transition map is universal** - Works regardless of pattern type detection
2. **Algorithmic completion is perfect** - 100% on all pattern types
3. **Very fast** - 0.2ms average vs 8s for LLM
4. **Cycle detection handles alternating** - Period-2 cycles work correctly
5. **Growth pattern termination** - Correctly returns empty when pattern ends

### What Didn't Work

1. **LLM fails badly** - 10% 1-step, 0% 3-step
   - Often repeats input trace instead of completing
   - Parsing issues with output format
   - Confuses which event comes next
2. **Pattern type detection** - Only 50% (but doesn't matter for completion)
   - "unknown" with transition map still completes correctly

## Extended Benchmark: Differential Learning

Added complex test cases that challenge simple transition maps:

### Challenge Types

| Challenge | Description | Unigram | 2-gram |
|-----------|-------------|---------|--------|
| context | Need 2+ state history | 0% | **100%** |
| conditional | Mode/flag changes behavior | 0% | **100%** |
| counter | Need to count repetitions | 50% | **100%** |
| long_range | Distance-5 dependencies | 0% | 0% |
| hierarchical | Nested patterns | 100% | 100% |
| noisy | Pattern with variations | 100% | 100% |
| stack | Push/pop matching | 100% | 100% |

### Learner Comparison (23 test cases)

| Learner | Accuracy | Notes |
|---------|----------|-------|
| **2-gram** | **95.7%** | Best overall |
| **3-gram** | **95.7%** | Same as 2-gram |
| mode-aware | 82.6% | Helps conditional |
| ensemble | 78.3% | Voting hurts when specialists wrong |
| unigram | 73.9% | Baseline (simple transition map) |
| stack-aware | 73.9% | Same as unigram |
| counter-aware | 73.9% | Same as unigram |

### Key Differential Findings

1. **N-gram context is crucial** - +22% accuracy (74% → 96%)
   - Context-dependent: 0% → 100%
   - Conditional: 0% → 100%
   - Counter: 50% → 100%

2. **Specialized learners help specific cases**:
   - Mode-aware: 100% on conditional (vs 0% unigram)
   - But don't improve overall due to limited applicability

3. **Long-range dependencies unsolved** - 0% for all learners
   - Need attention mechanism or explicit memory
   - Distance-5 echo requires 5-gram or longer

4. **Ensemble doesn't outperform best individual**
   - Voting dilutes correct specialized answers
   - Better to select learner based on pattern detection

### Recommendations

1. **Use 2-gram as default** - Best balance of accuracy and simplicity
2. **Add pattern detection to select learner** - Use mode-aware for conditional
3. **For long-range: need neural approach** - Attention or Transformer

## Files Created

- `pattern_learner.py` - Pattern extraction from traces
- `trace_completer.py` - Trace completion using patterns
- `benchmark.py` - Simple test cases (10)
- `complex_cases.py` - Complex test cases (13)
- `differential_learner.py` - Multiple learning approaches
- `benchmark_extended.py` - Full comparison benchmark
- `__init__.py` - Module exports
- `NOTES.md` - This file

## Report Format

```
[12FF]: TRACE_PATTERN_EXTENDED [2gram:96%, unigram:74%] ensemble_vs_unigram=[+1/-0/=22]
```
