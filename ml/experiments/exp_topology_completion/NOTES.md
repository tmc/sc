# exp_topology_completion

## Hypothesis

Scratchpad enumeration of topology gaps can achieve 70%+ validity
when completing partial statecharts.

## Approach

1. **Gap Analysis**: Identify issues algorithmically
   - Reachability: States unreachable from initial
   - Terminal: Dead ends (no outgoing, not terminal)
   - Initial: Missing start state
   - Orphans: States with no connections

2. **Fix Generation**: Propose corrections
   - Add transitions to connect unreachable
   - Mark dead ends as terminal
   - Set initial state
   - Remove orphan states

3. **Completion Methods**:
   - Algorithmic: Apply fixes directly
   - LLM: Use scratchpad analysis in prompt

## Test Cases

| Name | Issue | Description |
|------|-------|-------------|
| missing_transition | unreachable | State C disconnected from A→B |
| no_terminal | dead_end | Linear A→B→C but no terminal |
| no_initial | no_initial | States exist but no start marked |
| orphan_state | orphan | State D has no connections |
| dead_end | dead_end | State Trap has no exit |
| complex_multiple | complex | Multiple issues combined |
| valid_simple | valid | Already correct toggle |
| cycle_no_terminal | valid | Cycle is valid (no terminal needed) |

## Results

**Both methods exceeded the 70% target!**

| Method | Valid Rate | Avg Time |
|--------|------------|----------|
| Algorithmic | 75% | 0.2ms |
| LLM (1.5B) | 88% | 6619ms |

### By Issue Type

| Issue | Algorithmic | LLM |
|-------|-------------|-----|
| missing_trans | 0% | 100% |
| no_terminal | 100% | 100% |
| no_initial | 100% | 100% |
| orphan | 100% | 100% |
| dead_end | 100% | 100% |
| complex | 0% | 0% |
| valid | 100% | 100% |

## Key Findings

### Scratchpad Analysis Format

```
Step 1 - Reachability check:
  From A: can reach {B}
  Unreachable: {C}

Step 2 - Terminal check:
  B has no outgoing transitions
  No state marked terminal

Step 3 - Completeness check:
  States without incoming (except initial): {C}

Step 4 - Proposed fixes:
  Add transition B→C
  Mark C as terminal
```

### What Worked

1. **Algorithmic gap detection is reliable** - 100% on single-issue cases
2. **LLM handles complex transitions** - Fixed missing_trans that algo couldn't
3. **Simple fixes work well** - mark_terminal, mark_initial, remove_orphan
4. **Cycle detection** - Avoids false positives on "no terminal"
5. **Scratchpad prompting** - Guides LLM to apply correct fixes

### What Didn't Work

1. **Complex multi-issue cases** - Both fail at 0%
   - Multiple interacting gaps confuse fix ordering
   - Need iterative fix-verify loop
2. **Algorithmic missing_trans** - Can't generate meaningful event names
3. **LLM speed** - 6.6s average vs 0.2ms algorithmic

## Files Created

- `topology_completer.py` - Gap analysis and fix generation
- `benchmark.py` - Test cases and metrics
- `__init__.py` - Module exports
- `NOTES.md` - This file

## Report Format

```
[12FF]: TOPOLOGY_COMPLETION valid=X%, by_issue=[missing_trans:A%, no_terminal:B%, orphan:C%, dead_end:D%]
```
