# exp_error_recovery: FSM Error Recovery Testing

## Goal

Test constrained sampler robustness under adversarial conditions:
1. Inject invalid tokens mid-generation
2. Measure FSM recovery to valid state
3. Compare recovery strategies

## Recovery Strategies

| Strategy | Description | Recovery Rate | Valid JSON |
|----------|-------------|---------------|------------|
| **skip** | Ignore invalid token, continue | 80% | 20% |
| backtrack | Restore to last checkpoint | 80% | 0% |
| force_close | Close all open brackets | 60% | 0% |

## Key Findings

1. **Skip strategy wins**: Best combination of recovery rate and final JSON validity
2. **Backtrack preserves state but disrupts flow**: Restoring to checkpoint loses generation momentum
3. **Force_close is too aggressive**: Terminates early, loses content

## Recommended Error Handling Pipeline

```python
def handle_invalid_token(token, fsm, generated):
    # Level 1: Skip (most common)
    if fsm.can_continue_without(token):
        return SKIP

    # Level 2: Backtrack (when skip fails)
    if consecutive_skips > 3:
        return BACKTRACK

    # Level 3: Force close (emergency)
    if fsm.depth > MAX_DEPTH or fsm.elements > MAX_ELEMENTS:
        return FORCE_CLOSE
```

## Test Methodology

1. Generate valid JSON prefix with model
2. Inject invalid tokens at positions [3, 6, 9, 12, 15]
3. Apply recovery strategy
4. Measure:
   - Recovery success (FSM returns to valid state)
   - Tokens wasted (for backtrack)
   - Final JSON validity
   - Output length

## Invalid Token Types Tested

- `}}}` - Too many closing braces
- `]]]` - Too many closing brackets
- `<<<` - Invalid characters
- `}}]` - Mismatched brackets

## Integration with Retok-TopK

The skip strategy integrates naturally with Retok-TopK:
1. Top-K candidates are checked against FSM
2. Invalid tokens are filtered (skipped)
3. Model samples from remaining valid tokens
4. No explicit recovery needed - invalid tokens never reach output

## Files

- `error_recovery.py` - Main test implementation
- `NOTES.md` - This file
