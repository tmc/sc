"""
Validation utilities for statecharts.

This module provides validation functionality for statechart definitions
and execution traces, both locally and via gRPC services.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
import grpc

from .core import Statechart, Machine, StateType
from .generated.validation.v1 import (
    SemanticValidatorStub,
    ValidateChartRequest,
    ValidateChartResponse,
    ValidateTraceRequest,
    ValidateTraceResponse,
    Violation as PbViolation,
    Severity as PbSeverity,
    RuleId as PbRuleId,
)


class Severity(Enum):
    """Severity levels for validation violations."""
    UNSPECIFIED = PbSeverity.SEVERITY_UNSPECIFIED
    INFO = PbSeverity.INFO
    WARNING = PbSeverity.WARNING
    ERROR = PbSeverity.ERROR


class RuleId(Enum):
    """Validation rule identifiers."""
    UNSPECIFIED = PbRuleId.RULE_UNSPECIFIED
    UNIQUE_STATE_LABELS = PbRuleId.UNIQUE_STATE_LABELS
    SINGLE_DEFAULT_CHILD = PbRuleId.SINGLE_DEFAULT_CHILD
    BASIC_HAS_NO_CHILDREN = PbRuleId.BASIC_HAS_NO_CHILDREN
    COMPOUND_HAS_CHILDREN = PbRuleId.COMPOUND_HAS_CHILDREN
    DETERMINISTIC_TRANSITION_SELECTION = PbRuleId.DETERMINISTIC_TRANSITION_SELECTION
    NO_EVENT_BROADCAST_CYCLES = PbRuleId.NO_EVENT_BROADCAST_CYCLES


class Violation:
    """Represents a validation rule violation."""
    
    def __init__(
        self,
        rule: RuleId,
        severity: Severity,
        message: str,
        xpath: Optional[List[str]] = None
    ):
        self.rule = rule
        self.severity = severity
        self.message = message
        self.xpath = xpath or []
    
    @classmethod
    def from_pb(cls, pb_violation: PbViolation) -> 'Violation':
        """Create a Violation from a protobuf Violation."""
        return cls(
            rule=RuleId(pb_violation.rule),
            severity=Severity(pb_violation.severity),
            message=pb_violation.message,
            xpath=list(pb_violation.xpath)
        )
    
    def to_pb(self) -> PbViolation:
        """Convert to protobuf Violation."""
        return PbViolation(
            rule=self.rule.value,
            severity=self.severity.value,
            message=self.message,
            xpath=self.xpath
        )
    
    def __str__(self) -> str:
        """String representation of the violation."""
        location = f" at {'/'.join(self.xpath)}" if self.xpath else ""
        return f"[{self.severity.name}] {self.rule.name}: {self.message}{location}"
    
    def __repr__(self) -> str:
        return f"Violation(rule={self.rule.name}, severity={self.severity.name}, message='{self.message}')"


class ValidationResult:
    """Result of a validation operation."""
    
    def __init__(self, violations: List[Violation]):
        self.violations = violations
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return not any(v.severity == Severity.ERROR for v in self.violations)
    
    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return any(v.severity == Severity.WARNING for v in self.violations)
    
    @property
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return any(v.severity == Severity.ERROR for v in self.violations)
    
    def get_errors(self) -> List[Violation]:
        """Get all error violations."""
        return [v for v in self.violations if v.severity == Severity.ERROR]
    
    def get_warnings(self) -> List[Violation]:
        """Get all warning violations."""
        return [v for v in self.violations if v.severity == Severity.WARNING]
    
    def get_info(self) -> List[Violation]:
        """Get all info violations."""
        return [v for v in self.violations if v.severity == Severity.INFO]
    
    def __str__(self) -> str:
        """String representation of the validation result."""
        if self.is_valid:
            if self.has_warnings:
                return f"Valid with {len(self.get_warnings())} warning(s)"
            else:
                return "Valid"
        else:
            errors = len(self.get_errors())
            warnings = len(self.get_warnings())
            return f"Invalid: {errors} error(s), {warnings} warning(s)"
    
    def __bool__(self) -> bool:
        """Boolean representation (True if valid)."""
        return self.is_valid


class SemanticValidator:
    """
    Client for statechart validation services.
    
    This class provides both local and remote validation capabilities
    for statechart definitions and execution traces.
    """
    
    def __init__(self, channel: Optional[grpc.Channel] = None):
        """
        Initialize the validator.
        
        Args:
            channel: Optional gRPC channel for remote validation
        """
        self.stub = SemanticValidatorStub(channel) if channel else None
    
    def validate_chart(
        self,
        statechart: Statechart,
        ignore_rules: Optional[List[RuleId]] = None
    ) -> ValidationResult:
        """
        Validate a statechart definition.
        
        Args:
            statechart: The statechart to validate
            ignore_rules: Optional list of rules to ignore
            
        Returns:
            ValidationResult with any violations found
            
        Raises:
            RuntimeError: If no gRPC channel is configured
            grpc.RpcError: If the gRPC call fails
        """
        if not self.stub:
            raise RuntimeError("No gRPC channel configured for remote validation")
        
        ignore_rule_ids = [rule.value for rule in (ignore_rules or [])]
        
        request = ValidateChartRequest(
            chart=statechart.to_pb(),
            ignore_rules=ignore_rule_ids
        )
        
        response: ValidateChartResponse = self.stub.ValidateChart(request)
        violations = [Violation.from_pb(v) for v in response.violations]
        
        return ValidationResult(violations)
    
    def validate_trace(
        self,
        statechart: Statechart,
        trace: List[Machine],
        ignore_rules: Optional[List[RuleId]] = None
    ) -> ValidationResult:
        """
        Validate a statechart execution trace.
        
        Args:
            statechart: The statechart definition
            trace: List of machine states representing the execution trace
            ignore_rules: Optional list of rules to ignore
            
        Returns:
            ValidationResult with any violations found
            
        Raises:
            RuntimeError: If no gRPC channel is configured
            grpc.RpcError: If the gRPC call fails
        """
        if not self.stub:
            raise RuntimeError("No gRPC channel configured for remote validation")
        
        ignore_rule_ids = [rule.value for rule in (ignore_rules or [])]
        pb_trace = [machine.to_pb() for machine in trace]
        
        request = ValidateTraceRequest(
            chart=statechart.to_pb(),
            trace=pb_trace,
            ignore_rules=ignore_rule_ids
        )
        
        response: ValidateTraceResponse = self.stub.ValidateTrace(request)
        violations = [Violation.from_pb(v) for v in response.violations]
        
        return ValidationResult(violations)


class LocalValidator:
    """
    Local (client-side) validation for statecharts.
    
    This class provides basic validation rules that can be executed
    without requiring a remote service.
    """
    
    def validate_chart(self, statechart: Statechart) -> ValidationResult:
        """
        Perform local validation of a statechart.
        
        Args:
            statechart: The statechart to validate
            
        Returns:
            ValidationResult with any violations found
        """
        violations = []
        
        # Check for unique state labels
        violations.extend(self._check_unique_state_labels(statechart))
        
        # Check basic state constraints
        violations.extend(self._check_basic_state_constraints(statechart))
        
        # Check compound state constraints
        violations.extend(self._check_compound_state_constraints(statechart))
        
        # Check initial state constraints
        violations.extend(self._check_initial_state_constraints(statechart))
        
        return ValidationResult(violations)
    
    def _check_unique_state_labels(self, statechart: Statechart) -> List[Violation]:
        """Check that all state labels are unique."""
        violations = []
        seen_labels = set()
        
        for state in statechart.get_all_states():
            if state.label in seen_labels:
                violations.append(Violation(
                    rule=RuleId.UNIQUE_STATE_LABELS,
                    severity=Severity.ERROR,
                    message=f"Duplicate state label: '{state.label}'"
                ))
            seen_labels.add(state.label)
        
        return violations
    
    def _check_basic_state_constraints(self, statechart: Statechart) -> List[Violation]:
        """Check that basic states have no children."""
        violations = []
        
        for state in statechart.get_all_states():
            if state.type == StateType.BASIC and state.children:
                violations.append(Violation(
                    rule=RuleId.BASIC_HAS_NO_CHILDREN,
                    severity=Severity.ERROR,
                    message=f"Basic state '{state.label}' cannot have children"
                ))
        
        return violations
    
    def _check_compound_state_constraints(self, statechart: Statechart) -> List[Violation]:
        """Check that compound states have children."""
        violations = []
        
        for state in statechart.get_all_states():
            if state.type in (StateType.NORMAL, StateType.PARALLEL) and not state.children:
                violations.append(Violation(
                    rule=RuleId.COMPOUND_HAS_CHILDREN,
                    severity=Severity.ERROR,
                    message=f"Compound state '{state.label}' must have children"
                ))
        
        return violations
    
    def _check_initial_state_constraints(self, statechart: Statechart) -> List[Violation]:
        """Check that XOR composite states have exactly one initial child."""
        violations = []
        
        for state in statechart.get_all_states():
            if state.type == StateType.NORMAL and state.children:
                initial_children = [child for child in state.children if child.is_initial]
                
                if len(initial_children) == 0:
                    violations.append(Violation(
                        rule=RuleId.SINGLE_DEFAULT_CHILD,
                        severity=Severity.ERROR,
                        message=f"XOR composite state '{state.label}' must have exactly one initial child"
                    ))
                elif len(initial_children) > 1:
                    violations.append(Violation(
                        rule=RuleId.SINGLE_DEFAULT_CHILD,
                        severity=Severity.ERROR,
                        message=f"XOR composite state '{state.label}' has multiple initial children"
                    ))
        
        return violations


# Convenience functions

def validate_statechart(statechart: Statechart, local_only: bool = True) -> ValidationResult:
    """
    Validate a statechart using local rules.
    
    Args:
        statechart: The statechart to validate
        local_only: If True, use only local validation (default)
        
    Returns:
        ValidationResult with any violations found
        
    Example:
        >>> from statecharts import basic_state, statechart
        >>> chart = statechart(basic_state("Root"))
        >>> result = validate_statechart(chart)
        >>> if not result:
        ...     print(f"Validation failed: {result}")
    """
    validator = LocalValidator()
    return validator.validate_chart(statechart)


def create_validator(server_address: str = "localhost:50051") -> SemanticValidator:
    """
    Create a remote semantic validator client.
    
    Args:
        server_address: Address of the validation service
        
    Returns:
        A SemanticValidator instance
        
    Example:
        >>> validator = create_validator()
        >>> result = validator.validate_chart(my_statechart)
    """
    channel = grpc.insecure_channel(server_address)
    return SemanticValidator(channel)