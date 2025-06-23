"""
Generated Protocol Buffer bindings for statecharts.

This module contains the auto-generated Python classes from the Protocol Buffer
definitions. These provide the low-level data structures used by the higher-level
Pythonic API.
"""

# Re-export key types for convenience
from .statecharts.v1 import (
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
    # Service types
    StatechartRegistry,
    CreateMachineRequest,
    CreateMachineResponse,
    StepRequest,
    StepResponse,
    # gRPC services
    StatechartServiceServicer,
    StatechartServiceStub,
    add_StatechartServiceServicer_to_server,
)

from .validation.v1 import (
    ValidateChartRequest,
    ValidateChartResponse,
    ValidateTraceRequest,
    ValidateTraceResponse,
    Violation,
    Severity,
    RuleId,
    # gRPC services
    SemanticValidatorServicer,
    SemanticValidatorStub,
    add_SemanticValidatorServicer_to_server,
)

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
    # Service types
    "StatechartRegistry",
    "CreateMachineRequest",
    "CreateMachineResponse", 
    "StepRequest",
    "StepResponse",
    # gRPC services
    "StatechartServiceServicer",
    "StatechartServiceStub",
    "add_StatechartServiceServicer_to_server",
    # Validation types
    "ValidateChartRequest",
    "ValidateChartResponse",
    "ValidateTraceRequest", 
    "ValidateTraceResponse",
    "Violation",
    "Severity",
    "RuleId",
    # Validation gRPC services
    "SemanticValidatorServicer",
    "SemanticValidatorStub",
    "add_SemanticValidatorServicer_to_server",
]