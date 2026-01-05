# exp_entry_exit_actions

## Goal
Generate statecharts with entry and exit actions using constrained decoding.

## Action Types

| Type | Location | Field | Example |
|------|----------|-------|---------|
| Entry | State | on_entry | `"on_entry": ["start_timer()"]` |
| Exit | State | on_exit | `"on_exit": ["cleanup()"]` |
| Both | State | on_entry + on_exit | Both fields present |
| Transition | Transition | action | `"action": "increment()"` |
| Chained | Any | Multiple actions | `["init()", "setup()", "start()"]` |

## JSON Schema Extension

```json
{
  "root_state": {
    "label": "Example",
    "type": 2,
    "children": [
      {
        "label": "Active",
        "type": 1,
        "on_entry": ["start_monitor()"],
        "on_exit": ["stop_monitor()"]
      }
    ]
  },
  "transitions": [
    {
      "from": ["Idle"],
      "to": ["Active"],
      "event": "START",
      "action": "log(starting)"
    }
  ]
}
```

## Action Syntax
Actions follow function call syntax: `function_name(args)`
- Valid: `start_timer()`, `log(message)`, `add(1)`
- Invalid: `start timer`, `log`, `1+2`

## Metrics
- **entry**: % of tests generating valid on_entry actions
- **exit**: % of tests generating valid on_exit actions
- **both**: % of tests generating both entry and exit
- **transition**: % of tests generating transition actions
- **chained**: % of tests generating multiple actions

## Usage
```python
from experiments.exp_entry_exit_actions import run_benchmark, format_report

result = run_benchmark()
print(format_report(result))
```

## Report Format
```
[C9F0]: ENTRY_EXIT_ACTIONS entry=X%, exit=Y%, both=Z%

Results:
- Entry only: X/N correct
- Exit only: X/N correct
- Both: X/N correct
- Transition actions: X/N correct
- Chained actions: X/N correct
```
