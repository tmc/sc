"""
SC Validator - Ground-truth validation using 'sc' binary.

Uses the Go 'sc' binary for proper Harel statechart semantics including:
- Hierarchy (composite states, default substates)
- Orthogonality (parallel regions, AND-states)
- History states (shallow and deep)
- Guards and actions

IMPORTANT: This module refuses to fall back to weak validation.
If the 'sc' binary is not available, validation fails explicitly.
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class ValidationResult:
    """Result of SC structure validation."""
    valid: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class StepResult:
    """Result of stepping through a trace."""
    valid: bool
    final_config: List[str] = field(default_factory=list)
    events_processed: int = 0
    error: str = ""


class SCValidatorError(Exception):
    """Raised when validation cannot be performed."""
    pass


class SCValidator:
    """
    Unified SC validator using the 'sc' binary.

    This provides ground-truth validation with proper Harel semantics.
    """

    # Possible locations for the sc binary
    SC_BINARY_PATHS = [
        # Relative to ml/ directory
        ".venv/bin/sc",
        "../sc",
        # Absolute paths
        "/Volumes/tmc/go/src/github.com/tmc/sc/sc",
        "/Volumes/tmc/go/src/github.com/tmc/sc/ml/.venv/bin/sc",
    ]

    def __init__(self, sc_binary: Optional[str] = None):
        """
        Initialize validator.

        Args:
            sc_binary: Path to sc binary. If None, searches standard locations.

        Raises:
            SCValidatorError: If sc binary cannot be found.
        """
        self.sc_binary = sc_binary or self._find_sc_binary()
        if not self.sc_binary:
            raise SCValidatorError(
                "sc binary not found. Cannot perform Harel-compliant validation. "
                "Build with: cd /Volumes/tmc/go/src/github.com/tmc/sc && go build -o sc ./cmd/sc"
            )

    def _find_sc_binary(self) -> Optional[str]:
        """Find the sc binary in standard locations."""
        # Get base directory (ml/)
        base_dir = Path(__file__).parent.parent

        for rel_path in self.SC_BINARY_PATHS:
            if rel_path.startswith("/"):
                # Absolute path
                path = Path(rel_path)
            else:
                # Relative to ml/
                path = base_dir / rel_path

            if path.exists() and os.access(path, os.X_OK):
                return str(path.resolve())

        # Try PATH
        try:
            result = subprocess.run(
                ["which", "sc"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass

        return None

    def validate_structure(self, sc_json: Dict) -> ValidationResult:
        """
        Validate SC structure using 'sc validate'.

        Checks:
        - Well-formedness (root state, children, transitions)
        - State references (from/to states exist)
        - Initial state requirements
        - Type correctness
        """
        issues = []
        warnings = []

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as tmp:
            json.dump(sc_json, tmp)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [self.sc_binary, "validate", tmp_path],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                # Parse validation errors
                error_output = result.stderr or result.stdout
                if error_output:
                    issues.extend(error_output.strip().split('\n'))
                else:
                    issues.append("Validation failed with no error message")
                return ValidationResult(valid=False, issues=issues)

            # Check for warnings in stdout
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if 'warning' in line.lower():
                        warnings.append(line)

            return ValidationResult(valid=True, warnings=warnings)

        except subprocess.TimeoutExpired:
            issues.append("Validation timed out")
            return ValidationResult(valid=False, issues=issues)
        except Exception as e:
            issues.append(f"Validation error: {e}")
            return ValidationResult(valid=False, issues=issues)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def validate_trace(self, sc_json: Dict, trace: List[str]) -> bool:
        """
        Validate that a trace is executable on the SC.

        Uses 'sc step' which implements proper Harel semantics:
        - Hierarchy entry/exit
        - Parallel region consistency
        - History restoration

        Args:
            sc_json: Statechart definition
            trace: List of event names

        Returns:
            True if trace is valid (all events can fire), False otherwise
        """
        result = self.step_trace(sc_json, trace)
        return result.valid

    def step_trace(
        self,
        sc_json: Dict,
        trace: List[str],
        return_json: bool = True,
    ) -> StepResult:
        """
        Execute a trace and return the final configuration.

        Args:
            sc_json: Statechart definition
            trace: List of event names
            return_json: If True, parse JSON output for configuration

        Returns:
            StepResult with validity and final configuration
        """
        if not trace:
            # Empty trace - get initial configuration
            return self._get_initial_config(sc_json)

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as tmp:
            json.dump(sc_json, tmp)
            tmp_path = tmp.name

        try:
            cmd = [self.sc_binary, "step"]
            if return_json:
                cmd.append("-json")

            for event in trace:
                cmd.extend(["-e", str(event)])

            cmd.append(tmp_path)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                error = result.stderr or result.stdout or "Unknown error"
                return StepResult(
                    valid=False,
                    events_processed=0,
                    error=error.strip(),
                )

            # Parse output
            final_config = []
            if return_json and result.stdout:
                try:
                    output = json.loads(result.stdout)
                    final_config = output.get("configuration", [])
                    # Also check for "active_states" key
                    if not final_config:
                        final_config = output.get("active_states", [])
                except json.JSONDecodeError:
                    # Non-JSON output - try to parse as state list
                    final_config = result.stdout.strip().split()

            return StepResult(
                valid=True,
                final_config=final_config,
                events_processed=len(trace),
            )

        except subprocess.TimeoutExpired:
            return StepResult(valid=False, error="Step timed out")
        except Exception as e:
            return StepResult(valid=False, error=str(e))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _get_initial_config(self, sc_json: Dict) -> StepResult:
        """Get the initial configuration of an SC."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as tmp:
            json.dump(sc_json, tmp)
            tmp_path = tmp.name

        try:
            # Run sc step with no events to get initial config
            result = subprocess.run(
                [self.sc_binary, "step", "-json", tmp_path],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return StepResult(valid=False, error=result.stderr or "Failed to get initial config")

            final_config = []
            if result.stdout:
                try:
                    output = json.loads(result.stdout)
                    final_config = output.get("configuration", [])
                    if not final_config:
                        final_config = output.get("active_states", [])
                except json.JSONDecodeError:
                    pass

            return StepResult(valid=True, final_config=final_config, events_processed=0)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def check_reachability(
        self,
        sc_json: Dict,
        trace: List[str],
        target_state: str,
    ) -> bool:
        """
        Check if a trace reaches a target state.

        Args:
            sc_json: Statechart definition
            trace: List of event names
            target_state: State label to check for

        Returns:
            True if target_state is in final configuration
        """
        result = self.step_trace(sc_json, trace)
        if not result.valid:
            return False
        return target_state in result.final_config

    def get_enabled_events(self, sc_json: Dict, config: List[str]) -> Set[str]:
        """
        Get events that are enabled in a given configuration.

        Note: This requires stepping to the configuration first,
        then checking which transitions are enabled.
        """
        # For now, extract from SC definition
        # A proper implementation would use sc with --enabled-events flag
        enabled = set()

        transitions = sc_json.get("transitions", [])
        config_set = set(config)

        for t in transitions:
            from_states = set(t.get("from", []))
            if from_states & config_set:
                event = t.get("event", "")
                if event:
                    enabled.add(event)

        return enabled

    def analyze(self, sc_json: Dict) -> Dict:
        """
        Run comprehensive analysis on an SC.

        Uses 'sc analyze' to get:
        - State counts and types
        - Transition analysis
        - Reachability info
        - Potential issues
        """
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as tmp:
            json.dump(sc_json, tmp)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [self.sc_binary, "analyze", "-json", tmp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return {"error": result.stderr or "Analysis failed"}

            if result.stdout:
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    return {"raw_output": result.stdout}

            return {}

        except Exception as e:
            return {"error": str(e)}
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


# Convenience functions

_validator: Optional[SCValidator] = None


def _get_validator() -> SCValidator:
    """Get or create the global validator instance."""
    global _validator
    if _validator is None:
        _validator = SCValidator()
    return _validator


def validate_trace(sc_json: Dict, trace: List[str]) -> bool:
    """
    Validate that a trace is executable on the SC.

    Uses proper Harel semantics via 'sc step'.
    """
    return _get_validator().validate_trace(sc_json, trace)


def validate_sc(sc_json: Dict) -> Tuple[bool, List[str]]:
    """
    Validate SC structure.

    Returns:
        (is_valid, list_of_issues)
    """
    result = _get_validator().validate_structure(sc_json)
    return result.valid, result.issues


def step_trace(sc_json: Dict, trace: List[str]) -> StepResult:
    """
    Execute a trace and return the final configuration.
    """
    return _get_validator().step_trace(sc_json, trace)


def check_reachability(sc_json: Dict, trace: List[str], target: str) -> bool:
    """
    Check if a trace reaches a target state.
    """
    return _get_validator().check_reachability(sc_json, trace, target)


# Test function
def test_validator():
    """Test the validator with sample SCs."""
    print("=" * 60)
    print("SC VALIDATOR TEST")
    print("=" * 60)

    try:
        validator = SCValidator()
        print(f"Using sc binary: {validator.sc_binary}")
    except SCValidatorError as e:
        print(f"ERROR: {e}")
        return

    # Test 1: Simple flat SC
    flat_sc = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Off", "type": 1, "is_initial": True},
                {"label": "On", "type": 1}
            ]
        },
        "transitions": [
            {"from": ["Off"], "to": ["On"], "event": "TOGGLE"},
            {"from": ["On"], "to": ["Off"], "event": "TOGGLE"}
        ]
    }

    print("\n1. Flat SC (Toggle)")
    result = validator.validate_structure(flat_sc)
    print(f"   Structure valid: {result.valid}")

    step_result = validator.step_trace(flat_sc, ["TOGGLE"])
    print(f"   Trace [TOGGLE] valid: {step_result.valid}")
    print(f"   Final config: {step_result.final_config}")

    step_result = validator.step_trace(flat_sc, ["TOGGLE", "TOGGLE"])
    print(f"   Trace [TOGGLE, TOGGLE] valid: {step_result.valid}")
    print(f"   Final config: {step_result.final_config}")

    # Test 2: Hierarchical SC
    hier_sc = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Off", "type": 1, "is_initial": True},
                {
                    "label": "On",
                    "type": 2,  # OR-state with children
                    "children": [
                        {"label": "Low", "type": 1, "is_initial": True},
                        {"label": "High", "type": 1}
                    ]
                }
            ]
        },
        "transitions": [
            {"from": ["Off"], "to": ["On"], "event": "POWER"},
            {"from": ["On"], "to": ["Off"], "event": "POWER"},
            {"from": ["Low"], "to": ["High"], "event": "BOOST"},
            {"from": ["High"], "to": ["Low"], "event": "REDUCE"}
        ]
    }

    print("\n2. Hierarchical SC (Power with Low/High)")
    result = validator.validate_structure(hier_sc)
    print(f"   Structure valid: {result.valid}")

    step_result = validator.step_trace(hier_sc, ["POWER"])
    print(f"   Trace [POWER] valid: {step_result.valid}")
    print(f"   Final config: {step_result.final_config}")
    print(f"   Should include 'On' AND 'Low' (default substate)")

    step_result = validator.step_trace(hier_sc, ["POWER", "BOOST"])
    print(f"   Trace [POWER, BOOST] valid: {step_result.valid}")
    print(f"   Final config: {step_result.final_config}")

    # Test 3: Parallel SC (AND-state)
    parallel_sc = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {
                    "label": "Active",
                    "type": 3,  # AND-state (parallel)
                    "is_initial": True,
                    "children": [
                        {
                            "label": "RegionA",
                            "type": 2,
                            "children": [
                                {"label": "A1", "type": 1, "is_initial": True},
                                {"label": "A2", "type": 1}
                            ]
                        },
                        {
                            "label": "RegionB",
                            "type": 2,
                            "children": [
                                {"label": "B1", "type": 1, "is_initial": True},
                                {"label": "B2", "type": 1}
                            ]
                        }
                    ]
                }
            ]
        },
        "transitions": [
            {"from": ["A1"], "to": ["A2"], "event": "GO_A"},
            {"from": ["B1"], "to": ["B2"], "event": "GO_B"}
        ]
    }

    print("\n3. Parallel SC (AND-state with RegionA, RegionB)")
    result = validator.validate_structure(parallel_sc)
    print(f"   Structure valid: {result.valid}")
    if result.issues:
        print(f"   Issues: {result.issues}")

    step_result = validator.step_trace(parallel_sc, [])
    print(f"   Initial config: {step_result.final_config}")
    print(f"   Should include Active, RegionA, A1, RegionB, B1")

    step_result = validator.step_trace(parallel_sc, ["GO_A"])
    print(f"   After [GO_A]: {step_result.final_config}")
    print(f"   Should include A2 AND B1 (parallel regions)")

    # Test 4: Invalid trace
    print("\n4. Invalid trace test")
    step_result = validator.step_trace(flat_sc, ["INVALID_EVENT"])
    print(f"   Trace [INVALID_EVENT] valid: {step_result.valid}")
    if step_result.error:
        print(f"   Error: {step_result.error}")

    print("\n" + "=" * 60)
    print("Validator tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_validator()
