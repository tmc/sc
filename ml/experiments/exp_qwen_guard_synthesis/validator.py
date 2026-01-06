"""
Guard Expression Validator

Validates guard expressions for:
1. Syntax correctness (parseable as boolean expression)
2. Variable references (only known variables used)
3. Operator validity (proper boolean/comparison operators)
4. Semantic checks (optional, using test contexts)

Target: 95%+ syntax valid for generated guards.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
import re
import ast


# =============================================================================
# VALIDATION RESULT
# =============================================================================

@dataclass
class ValidationError:
    """A single validation error."""
    error_type: str  # "syntax", "variable", "operator", "semantic"
    message: str
    position: Optional[int] = None


@dataclass
class ValidationResult:
    """Result of validating a guard expression."""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    parsed_ast: Optional[ast.expr] = None
    variables_used: Set[str] = field(default_factory=set)
    operators_used: Set[str] = field(default_factory=set)

    @property
    def error_message(self) -> Optional[str]:
        if self.errors:
            return self.errors[0].message
        return None

    def summary(self) -> str:
        if self.is_valid:
            return "VALID"
        return f"INVALID: {self.error_message}"


# =============================================================================
# GUARD VALIDATOR
# =============================================================================

class GuardValidator:
    """
    Validate guard expressions.

    Checks syntax, variable references, and optionally semantics.
    """

    # Valid operators in guards
    COMPARISON_OPS = {"==", "!=", "<", ">", "<=", ">="}
    BOOLEAN_OPS = {"and", "or", "not"}
    ARITHMETIC_OPS = {"+", "-", "*", "/", "//", "%"}

    def __init__(
        self,
        known_variables: List[str] = None,
        allow_unknown_variables: bool = True,
        strict_mode: bool = False,
    ):
        self.known_variables = set(known_variables or [])
        self.allow_unknown_variables = allow_unknown_variables
        self.strict_mode = strict_mode

    def validate(self, guard: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a guard expression.

        Args:
            guard: The guard expression string

        Returns:
            (is_valid, error_message) tuple
        """
        result = self.validate_full(guard)
        return (result.is_valid, result.error_message)

    def validate_full(self, guard: str) -> ValidationResult:
        """
        Full validation with detailed results.

        Args:
            guard: The guard expression string

        Returns:
            ValidationResult with full details
        """
        result = ValidationResult(is_valid=True)

        # Empty check
        if not guard or not guard.strip():
            result.is_valid = False
            result.errors.append(ValidationError(
                error_type="syntax",
                message="Empty guard expression",
            ))
            return result

        # Clean the guard
        clean_guard = self._preprocess(guard)

        # Syntax validation
        syntax_valid, syntax_error, parsed = self._validate_syntax(clean_guard)
        if not syntax_valid:
            result.is_valid = False
            result.errors.append(ValidationError(
                error_type="syntax",
                message=syntax_error or "Syntax error",
            ))
            return result

        result.parsed_ast = parsed

        # Extract variables and operators
        result.variables_used = self._extract_variables(parsed)
        result.operators_used = self._extract_operators(parsed)

        # Variable validation
        if not self.allow_unknown_variables and self.known_variables:
            unknown = result.variables_used - self.known_variables
            if unknown:
                result.is_valid = False
                result.errors.append(ValidationError(
                    error_type="variable",
                    message=f"Unknown variables: {unknown}",
                ))

        # Operator validation
        invalid_ops = self._check_operators(parsed)
        if invalid_ops:
            result.warnings.append(f"Unusual operators: {invalid_ops}")
            if self.strict_mode:
                result.is_valid = False
                result.errors.append(ValidationError(
                    error_type="operator",
                    message=f"Invalid operators: {invalid_ops}",
                ))

        # Type check (should result in boolean)
        if self.strict_mode and not self._is_boolean_expr(parsed):
            result.warnings.append("Expression may not be boolean")

        return result

    def _preprocess(self, guard: str) -> str:
        """Preprocess guard expression for parsing."""
        # Remove leading/trailing whitespace
        guard = guard.strip()

        # Remove common artifacts
        guard = guard.strip('"\'')

        # Handle True/False case sensitivity
        guard = re.sub(r'\bTrue\b', 'True', guard, flags=re.IGNORECASE)
        guard = re.sub(r'\bFalse\b', 'False', guard, flags=re.IGNORECASE)

        return guard

    def _validate_syntax(self, guard: str) -> Tuple[bool, Optional[str], Optional[ast.expr]]:
        """
        Validate guard syntax using Python's AST parser.

        Returns (is_valid, error_message, parsed_ast)
        """
        try:
            # Try to parse as expression
            parsed = ast.parse(guard, mode='eval')
            return True, None, parsed.body
        except SyntaxError as e:
            return False, f"Syntax error: {e.msg}", None
        except Exception as e:
            return False, f"Parse error: {str(e)}", None

    def _extract_variables(self, node: ast.expr) -> Set[str]:
        """Extract all variable names from AST."""
        variables = set()

        class NameVisitor(ast.NodeVisitor):
            def visit_Name(self, n):
                # Exclude Python builtins
                if n.id not in ('True', 'False', 'None'):
                    variables.add(n.id)
                self.generic_visit(n)

        NameVisitor().visit(node)
        return variables

    def _extract_operators(self, node: ast.expr) -> Set[str]:
        """Extract all operators from AST."""
        operators = set()

        class OpVisitor(ast.NodeVisitor):
            def visit_Compare(self, n):
                for op in n.ops:
                    operators.add(type(op).__name__)
                self.generic_visit(n)

            def visit_BoolOp(self, n):
                operators.add(type(n.op).__name__)
                self.generic_visit(n)

            def visit_UnaryOp(self, n):
                operators.add(type(n.op).__name__)
                self.generic_visit(n)

            def visit_BinOp(self, n):
                operators.add(type(n.op).__name__)
                self.generic_visit(n)

        OpVisitor().visit(node)
        return operators

    def _check_operators(self, node: ast.expr) -> Set[str]:
        """Check for invalid or unusual operators."""
        invalid = set()

        class OpChecker(ast.NodeVisitor):
            def visit_BinOp(self, n):
                # Binary ops like +, -, etc. are unusual in guards
                # but not necessarily invalid
                op_name = type(n.op).__name__
                if op_name not in ('Add', 'Sub', 'Mult', 'Div', 'Mod'):
                    invalid.add(op_name)
                self.generic_visit(n)

        OpChecker().visit(node)
        return invalid

    def _is_boolean_expr(self, node: ast.expr) -> bool:
        """Check if expression is likely boolean."""
        if isinstance(node, ast.BoolOp):
            return True
        if isinstance(node, ast.Compare):
            return True
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return True
        if isinstance(node, ast.NameConstant) and node.value in (True, False):
            return True
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return True
        if isinstance(node, ast.Name):
            # Could be boolean variable
            return True
        return False

    def validate_with_context(
        self,
        guard: str,
        context: Dict[str, Any],
        expected_result: bool = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate guard and optionally evaluate with context.

        Args:
            guard: The guard expression
            context: Variable values for evaluation
            expected_result: Expected evaluation result (for semantic check)

        Returns:
            (is_valid, error_message) tuple
        """
        # First do syntax validation
        result = self.validate_full(guard)
        if not result.is_valid:
            return False, result.error_message

        # Try to evaluate
        try:
            # Safe evaluation with only the context
            safe_context = {
                '__builtins__': {},
                'True': True,
                'False': False,
                'None': None,
            }
            safe_context.update(context)

            actual = eval(guard, safe_context)

            # Check semantic correctness if expected provided
            if expected_result is not None and actual != expected_result:
                return False, f"Semantic error: expected {expected_result}, got {actual}"

            return True, None

        except Exception as e:
            return False, f"Evaluation error: {str(e)}"


# =============================================================================
# BATCH VALIDATION
# =============================================================================

@dataclass
class BatchValidationResult:
    """Result of batch validation."""
    total: int
    valid: int
    invalid: int
    validity_rate: float
    errors_by_type: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return f"{self.valid}/{self.total} valid ({self.validity_rate:.1%})"


def batch_validate(
    guards: List[str],
    validator: GuardValidator = None,
    verbose: bool = False,
) -> BatchValidationResult:
    """
    Validate multiple guards.

    Args:
        guards: List of guard expressions
        validator: Validator instance (created if not provided)
        verbose: Print details

    Returns:
        BatchValidationResult with statistics
    """
    if validator is None:
        validator = GuardValidator()

    result = BatchValidationResult(
        total=len(guards),
        valid=0,
        invalid=0,
        validity_rate=0.0,
    )

    for guard in guards:
        validation = validator.validate_full(guard)

        if validation.is_valid:
            result.valid += 1
        else:
            result.invalid += 1
            for error in validation.errors:
                result.errors_by_type[error.error_type] = \
                    result.errors_by_type.get(error.error_type, 0) + 1

        if verbose:
            status = "OK" if validation.is_valid else f"FAIL: {validation.error_message}"
            print(f"  {guard[:40]:<40} {status}")

    result.validity_rate = result.valid / max(result.total, 1)

    return result


# =============================================================================
# TESTING
# =============================================================================

def test_validator():
    """Test the guard validator."""
    print("=" * 60)
    print("GUARD VALIDATOR TEST")
    print("=" * 60)

    validator = GuardValidator(
        known_variables=["x", "y", "counter", "is_active", "health"],
        allow_unknown_variables=True,
    )

    # Test cases: (guard, expected_valid)
    tests = [
        # Valid
        ("x > 5", True),
        ("counter == 0", True),
        ("is_active", True),
        ("not is_active", True),
        ("(x > 0) and (y < 10)", True),
        ("x == y", True),
        ("health <= 0 or not is_active", True),
        ("True", True),
        ("False", True),

        # Invalid
        ("", False),
        ("x >", False),
        ("and x", False),
        ("(x > 5", False),  # Unbalanced paren
        ("x >> 5", True),   # Valid Python, unusual for guards
        ("x = 5", False),   # Assignment, not comparison
    ]

    print("\nValidation tests:")
    passed = 0
    for guard, expected in tests:
        is_valid, error = validator.validate(guard)
        status = "PASS" if is_valid == expected else "FAIL"
        if status == "PASS":
            passed += 1

        guard_display = guard if guard else "(empty)"
        print(f"  [{status}] {guard_display:<35} expected={expected}, got={is_valid}")
        if error and not expected:
            print(f"       Error: {error}")

    print(f"\nPassed: {passed}/{len(tests)}")

    # Batch validation test
    print("\n" + "-" * 60)
    print("Batch validation test:")

    sample_guards = [
        "x > 5",
        "counter == 0",
        "is_active and (health > 0)",
        "not done",
        "x >",  # Invalid
        "",     # Invalid
        "(a and b) or c",
    ]

    batch_result = batch_validate(sample_guards, validator, verbose=True)
    print(f"\n{batch_result.summary()}")
    if batch_result.errors_by_type:
        print(f"Errors by type: {batch_result.errors_by_type}")

    print("\n" + "=" * 60)
    print("GUARD VALIDATOR TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_validator()
