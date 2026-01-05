"""
Unified SC Validation Module

Provides ground-truth validation using the real 'sc' binary with proper Harel semantics.
All experiments should use this module for consistency.

Usage:
    from validation import SCValidator, validate_trace, validate_sc, step_trace

    # Validate a trace against an SC
    is_valid = validate_trace(sc_json, ["EVENT1", "EVENT2"])

    # Validate SC structure
    is_valid, issues = validate_sc(sc_json)

    # Step through trace and get final configuration
    config = step_trace(sc_json, ["EVENT1", "EVENT2"])
"""

from .sc_validator import (
    SCValidator,
    ValidationResult,
    StepResult,
    validate_trace,
    validate_sc,
    step_trace,
    check_reachability,
)

__all__ = [
    'SCValidator',
    'ValidationResult',
    'StepResult',
    'validate_trace',
    'validate_sc',
    'step_trace',
    'check_reachability',
]
