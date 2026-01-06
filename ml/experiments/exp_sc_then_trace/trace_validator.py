"""
Trace Validator for exp_sc_then_trace.

Now uses the unified validation module for proper Harel semantics.
"""

from typing import Dict, List

# Import from unified validation module
import sys
from pathlib import Path

# Add ml/ to path for validation import
ml_dir = Path(__file__).parent.parent.parent
if str(ml_dir) not in sys.path:
    sys.path.insert(0, str(ml_dir))

from validation import (
    SCValidator,
    validate_trace as unified_validate_trace,
    check_reachability,
    step_trace,
    SCValidatorError,
)


class TraceValidator:
    """
    Validates traces against statechart definitions.

    Uses unified validation module with proper Harel semantics.
    """

    _validator = None

    @classmethod
    def _get_validator(cls) -> SCValidator:
        """Get or create validator instance."""
        if cls._validator is None:
            try:
                cls._validator = SCValidator()
            except SCValidatorError as e:
                raise RuntimeError(
                    f"Cannot initialize validator: {e}\n"
                    "Trace validation requires the 'sc' binary for proper Harel semantics."
                )
        return cls._validator

    @staticmethod
    def validate_trace(sc_dict: Dict, trace: List[str]) -> bool:
        """
        Checks if trace is valid using real 'sc' binary.

        This validates that:
        - All events can fire in sequence
        - Hierarchy is respected (default substates entered)
        - Parallel regions maintained
        """
        return unified_validate_trace(sc_dict, trace)

    @staticmethod
    def reached_goal(sc_dict: Dict, trace: List[str], goal: str) -> bool:
        """
        Checks if trace reaches goal state using 'sc' binary.

        Args:
            sc_dict: Statechart definition
            trace: List of event names
            goal: Target state label to check for

        Returns:
            True if goal state is in final configuration
        """
        return check_reachability(sc_dict, trace, goal)

    @staticmethod
    def get_final_config(sc_dict: Dict, trace: List[str]) -> List[str]:
        """
        Execute trace and return final configuration.

        Returns:
            List of active state labels after executing trace
        """
        result = step_trace(sc_dict, trace)
        if result.valid:
            return result.final_config
        return []


# For backwards compatibility
def validate_trace(sc_dict: Dict, trace: List[str]) -> bool:
    """Backwards compatible function."""
    return TraceValidator.validate_trace(sc_dict, trace)


def reached_goal(sc_dict: Dict, trace: List[str], goal: str) -> bool:
    """Backwards compatible function."""
    return TraceValidator.reached_goal(sc_dict, trace, goal)
