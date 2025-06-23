# Statecharts Python SDK

A Python implementation of Harel statecharts providing a clean, Pythonic API for creating and executing hierarchical and orthogonal state machines.

## Overview

This Python SDK provides:

- **Protocol Buffer bindings** for statechart definitions
- **High-level Pythonic API** for building statecharts
- **Factory functions** for common patterns
- **Integration helpers** for web frameworks (Flask, FastAPI)
- **Validation utilities** for statechart correctness
- **Comprehensive examples** and documentation

## Installation

```bash
pip install statecharts
```

For development with optional dependencies:

```bash
pip install statecharts[dev,web]
```

## Quick Start

### Creating a Simple Statechart

```python
from statecharts import basic_state, normal_state, transition, event, statechart

# Create states
off = basic_state("Off", is_initial=True)
on = basic_state("On")
root = normal_state("PowerSwitch", [off, on])

# Create transitions
turn_on = transition("TurnOn", from_states="Off", to_states="On", event="POWER")
turn_off = transition("TurnOff", from_states="On", to_states="Off", event="POWER")

# Create events
power_event = event("POWER")

# Create the statechart
chart = statechart(root, transitions=[turn_on, turn_off], events=[power_event])

print(f"Created statechart with {len(chart.get_all_states())} states")
```

### Using the Builder Pattern

```python
from statecharts import StatechartBuilder, StateType

chart = (StatechartBuilder()
    .root_state("AlarmSystem")
    .add_state("Off", parent="AlarmSystem", is_initial=True)
    .add_state("On", parent="AlarmSystem", state_type=StateType.NORMAL)
    .add_state("Disarmed", parent="On", is_initial=True)
    .add_state("Armed", parent="On")
    .add_transition("PowerOn", "Off", "On", "POWER")
    .add_transition("PowerOff", "On", "Off", "POWER")
    .add_transition("Arm", "Disarmed", "Armed", "ARM")
    .add_transition("Disarm", "Armed", "Disarmed", "DISARM")
    .add_event("POWER")
    .add_event("ARM")
    .add_event("DISARM")
    .build())
```

### Validation

```python
from statecharts import validate_statechart

result = validate_statechart(chart)

if result.is_valid:
    print("✓ Statechart is valid!")
else:
    print("✗ Validation errors:")
    for error in result.get_errors():
        print(f"  - {error}")
```

## Key Concepts

### State Types

- **Basic State**: Has no sub-states (leaf node)
- **Normal State**: Has sub-states with XOR semantics (exactly one child active)
- **Parallel State**: Has sub-states with AND semantics (all children active)

```python
from statecharts import basic_state, normal_state, parallel_state

# Basic state (no children)
idle = basic_state("Idle")

# Normal state (XOR - one child active)
power = normal_state("Power", [
    basic_state("Off", is_initial=True),
    basic_state("On")
])

# Parallel state (AND - all children active)
device = parallel_state("Device", [
    normal_state("Display", [basic_state("Off", is_initial=True), basic_state("On")]),
    normal_state("Audio", [basic_state("Muted", is_initial=True), basic_state("Unmuted")])
])
```

### Transitions with Guards and Actions

```python
from statecharts import transition

# Transition with guard condition
guarded_transition = transition(
    "ConditionalMove",
    from_states="StateA",
    to_states="StateB", 
    event="MOVE",
    guard="battery_level > 10",  # Condition that must be true
    actions=["log_transition", "update_display"]  # Actions to execute
)
```

### Machine Instances

```python
from statecharts import machine

# Create a machine instance
my_machine = machine("device-001", chart, context={"user": "alice", "session": "123"})

print(f"Machine state: {my_machine.state.name}")
print(f"Active states: {my_machine.get_active_states()}")
print(f"Context: {my_machine.context}")
```

## Examples

### Simple Toggle

```python
from statecharts import simple_toggle_statechart

# Create a simple on/off toggle
toggle = simple_toggle_statechart("On", "Off", "TOGGLE")

# Find states
on_state = toggle.find_state("On")
off_state = toggle.find_state("Off")

# Check which events can trigger from a state
events_from_off = toggle.get_events_for_state("Off")
print(f"From Off state, events: {events_from_off}")  # ['TOGGLE']
```

### Hierarchical Statechart

```python
from statecharts import hierarchical_statechart_example

# Get a pre-built hierarchical example
example = hierarchical_statechart_example()

# Explore the hierarchy
for state in example.get_all_states():
    level = len([s for s in example.get_all_states() if state.label in s.label])
    print(f"{'  ' * level}{state.label} ({state.type.name})")
```

### Orthogonal (Parallel) Statechart

```python
# Create smartphone with concurrent subsystems
display = normal_state("Display", [
    basic_state("Off", is_initial=True),
    basic_state("On")
])

audio = normal_state("Audio", [
    basic_state("Muted", is_initial=True), 
    basic_state("Unmuted")
])

network = normal_state("Network", [
    basic_state("Disconnected", is_initial=True),
    basic_state("Connected")
])

# All subsystems run in parallel
smartphone = parallel_state("Smartphone", [display, audio, network])

chart = statechart(smartphone, transitions=[
    transition("DisplayToggle", ["Off", "On"], ["On", "Off"], "DISPLAY_BUTTON"),
    transition("AudioToggle", ["Muted", "Unmuted"], ["Unmuted", "Muted"], "MUTE_BUTTON"),
    transition("Connect", "Disconnected", "Connected", "NETWORK_AVAILABLE"),
    transition("Disconnect", "Connected", "Disconnected", "NETWORK_LOST"),
])
```

## Web Framework Integration

### Flask Integration

```python
from flask import Flask
from statecharts.integrations.flask_integration import FlaskStatechartBlueprint

app = Flask(__name__)
app.secret_key = "your-secret-key"

# Create and register statechart blueprint
statechart_bp = FlaskStatechartBlueprint()
statechart_bp.register_statechart("workflow", your_statechart)
app.register_blueprint(statechart_bp.blueprint)

# Use in routes
@app.route("/start-workflow", methods=["POST"])
def start_workflow():
    # Creates a machine instance in the session
    # POST /statecharts/machines {"statechart_id": "workflow"}
    pass

@app.route("/send-event/<machine_id>", methods=["POST"]) 
def send_event(machine_id):
    # Send event to machine
    # POST /statecharts/machines/{machine_id}/events {"event": "START"}
    pass
```

### FastAPI Integration

```python
from fastapi import FastAPI
from statecharts.integrations.fastapi_integration import create_statechart_router

app = FastAPI()

# Add statechart router
statechart_router = create_statechart_router()
app.include_router(statechart_router)

# Register statecharts at startup
@app.on_event("startup")
async def register_statecharts():
    service = get_statechart_service()
    service.registry.register_statechart("workflow", your_statechart)

# WebSocket support for real-time updates
@app.websocket("/ws/{machine_id}")
async def websocket_endpoint(websocket: WebSocket, machine_id: str):
    # Real-time machine state updates
    pass
```

## API Reference

### Core Classes

- **`State`**: Represents a state in the statechart
- **`Transition`**: Represents a transition between states  
- **`Event`**: Represents an event that triggers transitions
- **`Statechart`**: Complete statechart definition
- **`Machine`**: Runtime instance of a statechart

### Factory Functions

- **`basic_state(label, is_initial=False, is_final=False)`**: Create a basic state
- **`normal_state(label, children, is_initial=False, is_final=False)`**: Create a normal state
- **`parallel_state(label, children, is_initial=False, is_final=False)`**: Create a parallel state
- **`transition(label, from_states, to_states, event, guard=None, actions=None)`**: Create a transition
- **`event(label)`**: Create an event
- **`statechart(root_state, transitions=None, events=None)`**: Create a statechart
- **`machine(id, statechart, context=None)`**: Create a machine instance

### Validation

- **`validate_statechart(statechart)`**: Validate a statechart definition
- **`LocalValidator`**: Client-side validation
- **`ValidationResult`**: Validation result with errors/warnings

### Builder Pattern

- **`StatechartBuilder`**: Fluent interface for building complex statecharts

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
pytest

# Run tests with coverage
pytest --cov=statecharts

# Run specific test file
pytest tests/test_core.py
```

### Code Formatting

```bash
# Format code
black statecharts tests

# Check formatting
black --check statecharts tests

# Type checking
mypy statecharts

# Linting
ruff statecharts tests
```

### Running Examples

```bash
# Simple statechart example
python -m statecharts.examples.simple_statechart

# Hierarchical statechart example  
python -m statecharts.examples.hierarchical_statechart

# Orthogonal statechart example
python -m statecharts.examples.orthogonal_statechart
```

## Architecture

The Python SDK is built in layers:

1. **Protocol Buffer Layer** (`statecharts.generated`): Auto-generated types
2. **Core API Layer** (`statecharts.core`): Pythonic wrappers around protobuf types
3. **Factory Layer** (`statecharts.factory`): Convenient factory functions and builders
4. **Integration Layer** (`statecharts.integrations`): Framework-specific utilities
5. **Application Layer**: Your statechart definitions and business logic

## Compatibility

- **Python**: 3.8+
- **Dependencies**: `grpcio`, `protobuf`
- **Optional**: `flask`, `fastapi`, `django` (for framework integrations)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Related Projects

- [Statecharts Go Implementation](../../) - Core Go implementation
- [Statecharts Rust SDK](../rust/) - Rust SDK
- [XState](https://xstate.js.org/) - JavaScript statechart library

## Support

- [GitHub Issues](https://github.com/tmc/sc/issues) - Bug reports and feature requests
- [Documentation](https://github.com/tmc/sc/tree/main/docs) - Detailed documentation
- [Examples](./statecharts/examples/) - More example code