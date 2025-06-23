"""
Statecharts Python SDK

A Python implementation of Harel statecharts providing a clean, Pythonic API
for creating and executing hierarchical and orthogonal state machines.

This package provides:
- Protocol Buffer bindings for statechart definitions
- High-level Pythonic API for building statecharts
- Factory functions for common patterns
- Integration helpers for web frameworks
- Validation utilities
"""

from typing import TYPE_CHECKING

# Version information
__version__ = "0.1.0"
__author__ = "Travis Cline <travis.cline@gmail.com>"
__license__ = "MIT"

# Import core types and functions for convenience
from .core import (
    Statechart,
    State,
    Transition,
    Event,
    Guard,
    Action,
    StateRef,
    Configuration,
    Machine,
    Step,
    StateType,
    MachineState,
)

from .factory import (
    basic_state,
    normal_state,
    parallel_state,
    orthogonal_state,  # alias for parallel_state
    transition,
    event,
    guard,
    action,
    statechart,
    machine,
    configuration,
    StatechartBuilder,
    simple_toggle_statechart,
    hierarchical_statechart_example,
)

from .validation import (
    validate_statechart,
    LocalValidator,
    ValidationResult,
    Violation,
    Severity,
    RuleId,
)

# Only import grpc-related modules if they're available
if TYPE_CHECKING:
    from .service import StatechartService
    from .validation import SemanticValidator

__all__ = [
    # Core types
    "Statechart",
    "State", 
    "Transition",
    "Event",
    "Guard",
    "Action",
    "StateRef",
    "Configuration",
    "Machine",
    "Step",
    "StateType",
    "MachineState",
    # Factory functions
    "basic_state",
    "normal_state", 
    "parallel_state",
    "orthogonal_state",
    "transition",
    "event",
    "guard",
    "action",
    "statechart",
    "machine",
    "configuration",
    "StatechartBuilder",
    "simple_toggle_statechart",
    "hierarchical_statechart_example",
    # Validation
    "validate_statechart",
    "LocalValidator",
    "ValidationResult",
    "Violation",
    "Severity",
    "RuleId",
    # Version info
    "__version__",
    "__author__",
    "__license__",
]