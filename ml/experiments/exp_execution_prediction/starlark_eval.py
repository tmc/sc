"""
Starlark Expression Evaluator for Statechart Actions and Guards.

Uses starlark-go via subprocess for sandboxed, deterministic evaluation.
Falls back to restricted Python eval if starlark not available.
"""

import json
import subprocess
import tempfile
import os
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EvalResult:
    """Result of expression evaluation."""
    success: bool
    value: Any
    context: Dict[str, Any]
    error: Optional[str] = None


class StarlarkEvaluator:
    """Evaluates Starlark expressions for statechart actions and guards."""

    def __init__(self, use_fallback: bool = True):
        self.use_fallback = use_fallback
        self._check_starlark()

    def _check_starlark(self):
        """Check if starlark-go is available."""
        try:
            result = subprocess.run(
                ["starlark", "--help"],
                capture_output=True,
                timeout=5,
            )
            self.has_starlark = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.has_starlark = False

    def eval_guard(self, expression: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Evaluate a guard expression.

        Returns: (result: bool, error: Optional[str])
        """
        if self.has_starlark:
            return self._eval_starlark_guard(expression, context)
        elif self.use_fallback:
            return self._eval_python_guard(expression, context)
        else:
            return False, "Starlark not available"

    def eval_action(self, expression: str, context: Dict[str, Any]) -> EvalResult:
        """
        Evaluate an action expression that modifies context.

        Returns: EvalResult with updated context.
        """
        if self.has_starlark:
            return self._eval_starlark_action(expression, context)
        elif self.use_fallback:
            return self._eval_python_action(expression, context)
        else:
            return EvalResult(False, None, context, "Starlark not available")

    def _eval_starlark_guard(self, expression: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Evaluate guard using starlark-go."""
        # Build Starlark script
        script = self._build_starlark_script(context, expression, is_guard=True)

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.star', delete=False) as f:
                f.write(script)
                script_path = f.name

            result = subprocess.run(
                ["starlark", script_path],
                capture_output=True,
                text=True,
                timeout=5,
            )

            os.unlink(script_path)

            if result.returncode != 0:
                return False, result.stderr

            # Parse result
            output = result.stdout.strip()
            return output.lower() == "true", None

        except Exception as e:
            return False, str(e)

    def _eval_starlark_action(self, expression: str, context: Dict[str, Any]) -> EvalResult:
        """Evaluate action using starlark-go."""
        script = self._build_starlark_script(context, expression, is_guard=False)

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.star', delete=False) as f:
                f.write(script)
                script_path = f.name

            result = subprocess.run(
                ["starlark", script_path],
                capture_output=True,
                text=True,
                timeout=5,
            )

            os.unlink(script_path)

            if result.returncode != 0:
                return EvalResult(False, None, context, result.stderr)

            # Parse updated context from output
            try:
                new_context = json.loads(result.stdout.strip())
                return EvalResult(True, None, new_context, None)
            except json.JSONDecodeError:
                return EvalResult(False, None, context, "Failed to parse context")

        except Exception as e:
            return EvalResult(False, None, context, str(e))

    def _build_starlark_script(self, context: Dict[str, Any], expression: str, is_guard: bool) -> str:
        """Build a Starlark script for evaluation."""
        # Define context variables
        var_defs = []
        for key, value in context.items():
            if isinstance(value, str):
                var_defs.append(f'{key} = "{value}"')
            elif isinstance(value, bool):
                var_defs.append(f'{key} = {"True" if value else "False"}')
            elif isinstance(value, (list, dict)):
                var_defs.append(f'{key} = {json.dumps(value)}')
            else:
                var_defs.append(f'{key} = {value}')

        if is_guard:
            # For guards, just evaluate and print boolean result
            script = "\n".join(var_defs) + f"\nprint({expression})"
        else:
            # For actions, execute and print updated context
            script = "\n".join(var_defs)
            script += f"\n{expression}"
            # Collect all variables back into a dict
            script += "\n_ctx = {"
            script += ", ".join(f'"{k}": {k}' for k in context.keys())
            script += "}\nprint(_ctx)"

        return script

    def _eval_python_guard(self, expression: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Fallback: evaluate guard using restricted Python eval."""
        try:
            # Convert && and || to Python syntax
            py_expr = expression.replace("&&", " and ").replace("||", " or ")

            # Safe eval with only context as namespace
            result = eval(py_expr, {"__builtins__": {}}, context.copy())
            return bool(result), None
        except Exception as e:
            return False, str(e)

    def _eval_python_action(self, expression: str, context: Dict[str, Any]) -> EvalResult:
        """Fallback: evaluate action using restricted Python exec."""
        try:
            # Make a copy of context to modify
            new_context = context.copy()

            # Handle simple assignment expressions
            # e.g., "count = count + 1" or "score = score * 2"
            if "=" in expression and not any(op in expression for op in ["==", "!=", "<=", ">="]):
                # Simple assignment
                exec(expression, {"__builtins__": {}}, new_context)
            else:
                # Expression evaluation
                exec(expression, {"__builtins__": {}}, new_context)

            return EvalResult(True, None, new_context, None)
        except Exception as e:
            return EvalResult(False, None, context, str(e))


# Module-level convenience functions
_evaluator = None

def _get_evaluator() -> StarlarkEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = StarlarkEvaluator()
    return _evaluator


def eval_guard(expression: str, context: Dict[str, Any]) -> bool:
    """Evaluate a guard expression, return True/False."""
    result, error = _get_evaluator().eval_guard(expression, context)
    return result


def eval_action(expression: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate an action expression, return updated context."""
    result = _get_evaluator().eval_action(expression, context)
    return result.context


if __name__ == "__main__":
    print("Starlark Evaluator Test")
    print("=" * 60)

    evaluator = StarlarkEvaluator()
    print(f"Starlark available: {evaluator.has_starlark}")

    # Test guards
    context = {"count": 5, "limit": 10, "enabled": True}

    guards = [
        "count < limit",
        "count >= 5",
        "enabled",
        "count < limit and enabled",
    ]

    print("\nGuard Tests:")
    for guard in guards:
        result, error = evaluator.eval_guard(guard, context)
        print(f"  {guard} => {result}")

    # Test actions
    print("\nAction Tests:")
    actions = [
        ("count = count + 1", {"count": 0}),
        ("score = count * 10", {"count": 5, "score": 0}),
        ("total = a + b", {"a": 3, "b": 7, "total": 0}),
    ]

    for action, ctx in actions:
        result = evaluator.eval_action(action, ctx)
        print(f"  {action}")
        print(f"    Before: {ctx}")
        print(f"    After:  {result.context}")
