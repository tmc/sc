"""
Security Tester - Test security limits for expression evaluation.

Based on proto/statecharts/v1/expressions.proto:
- max_depth: Maximum nesting depth
- max_execution_ms: Execution timeout
- max_memory_bytes: Memory limit
- allowed_functions: Function whitelist
- blocked_functions: Function blocklist

Tests:
1. Depth limits - Reject overly nested expressions
2. Timeout limits - Abort long-running expressions
3. Function restrictions - Block dangerous functions
4. Resource exhaustion - Prevent memory bombs
"""

import time
import signal
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Callable
from enum import Enum, auto
from contextlib import contextmanager

from .expression_parser import (
    ExpressionLanguage,
    ParsedExpression,
    MultiLanguageParser,
)


class SecurityViolationType(Enum):
    """Types of security violations."""
    DEPTH_EXCEEDED = auto()
    TIMEOUT_EXCEEDED = auto()
    MEMORY_EXCEEDED = auto()
    BLOCKED_FUNCTION = auto()
    DISALLOWED_FUNCTION = auto()
    DANGEROUS_PATTERN = auto()
    SYNTAX_ATTACK = auto()


@dataclass
class SecurityConfig:
    """Security configuration for expression evaluation.

    Mirrors proto ExpressionSecurityConfig.
    """
    max_depth: int = 10
    max_execution_ms: int = 100
    max_memory_bytes: int = 10 * 1024 * 1024  # 10MB
    allowed_functions: Set[str] = field(default_factory=lambda: {
        'len', 'str', 'int', 'float', 'bool', 'abs', 'min', 'max',
        'sum', 'all', 'any', 'sorted', 'list', 'dict', 'set',
    })
    blocked_functions: Set[str] = field(default_factory=lambda: {
        'eval', 'exec', 'compile', 'open', 'input', '__import__',
        'getattr', 'setattr', 'delattr', 'globals', 'locals',
        'vars', 'dir', 'type', 'isinstance', 'issubclass',
    })
    allow_starlark: bool = True
    allow_starlark_loops: bool = False

    def to_dict(self) -> Dict:
        return {
            'max_depth': self.max_depth,
            'max_execution_ms': self.max_execution_ms,
            'max_memory_bytes': self.max_memory_bytes,
            'n_allowed_functions': len(self.allowed_functions),
            'n_blocked_functions': len(self.blocked_functions),
            'allow_starlark': self.allow_starlark,
            'allow_starlark_loops': self.allow_starlark_loops,
        }


@dataclass
class SecurityViolation:
    """A security violation detected in an expression."""
    violation_type: SecurityViolationType
    expression: str
    language: ExpressionLanguage
    details: str
    severity: str = "high"  # low, medium, high, critical

    def to_dict(self) -> Dict:
        return {
            'type': self.violation_type.name,
            'expression': self.expression[:100],  # Truncate for safety
            'language': self.language.name,
            'details': self.details,
            'severity': self.severity,
        }


@dataclass
class SecurityTestResult:
    """Result of security testing."""
    expression: str
    language: ExpressionLanguage
    config: SecurityConfig
    passed: bool
    violations: List[SecurityViolation] = field(default_factory=list)
    execution_time_ms: float = 0.0
    depth: int = 0

    def to_dict(self) -> Dict:
        return {
            'expression': self.expression[:100],
            'language': self.language.name,
            'passed': self.passed,
            'n_violations': len(self.violations),
            'violations': [v.to_dict() for v in self.violations],
            'execution_time_ms': self.execution_time_ms,
            'depth': self.depth,
        }


class TimeoutError(Exception):
    """Expression evaluation timed out."""
    pass


@contextmanager
def timeout_context(seconds: float):
    """Context manager for timeout (Unix only)."""
    def handler(signum, frame):
        raise TimeoutError("Execution timed out")

    # Set signal handler
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)

    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


class DepthChecker:
    """Check expression nesting depth."""

    def __init__(self, parser: MultiLanguageParser):
        self.parser = parser

    def check_depth(
        self,
        expr: str,
        lang: ExpressionLanguage,
        max_depth: int,
    ) -> Optional[SecurityViolation]:
        """Check if expression exceeds depth limit."""
        parsed = self.parser.parse(expr, lang)

        if not parsed.is_valid:
            return None  # Parse errors handled elsewhere

        if parsed.depth > max_depth:
            return SecurityViolation(
                violation_type=SecurityViolationType.DEPTH_EXCEEDED,
                expression=expr,
                language=lang,
                details=f"Depth {parsed.depth} exceeds max {max_depth}",
                severity="medium",
            )

        return None


class FunctionChecker:
    """Check function calls against allow/block lists."""

    def __init__(self, parser: MultiLanguageParser):
        self.parser = parser

    def check_functions(
        self,
        expr: str,
        lang: ExpressionLanguage,
        allowed: Set[str],
        blocked: Set[str],
    ) -> List[SecurityViolation]:
        """Check function calls against security policy."""
        parsed = self.parser.parse(expr, lang)
        violations = []

        for func in parsed.function_calls:
            base_func = func.split('.')[0]

            # Check blocklist first
            if base_func in blocked or func in blocked:
                violations.append(SecurityViolation(
                    violation_type=SecurityViolationType.BLOCKED_FUNCTION,
                    expression=expr,
                    language=lang,
                    details=f"Blocked function: {func}",
                    severity="critical",
                ))

            # Check allowlist if non-empty
            elif allowed and base_func not in allowed:
                violations.append(SecurityViolation(
                    violation_type=SecurityViolationType.DISALLOWED_FUNCTION,
                    expression=expr,
                    language=lang,
                    details=f"Function not in allowlist: {func}",
                    severity="high",
                ))

        return violations


class PatternChecker:
    """Check for dangerous expression patterns."""

    DANGEROUS_PATTERNS = [
        # Code injection
        (r'__\w+__', 'Double underscore attribute access'),
        (r'\beval\s*\(', 'Eval function call'),
        (r'\bexec\s*\(', 'Exec function call'),
        (r'\bcompile\s*\(', 'Compile function call'),

        # File access
        (r'\bopen\s*\(', 'File open'),
        (r'\.read\s*\(', 'File read'),
        (r'\.write\s*\(', 'File write'),

        # Network access
        (r'\bsocket\b', 'Socket access'),
        (r'\burllib\b', 'URL library'),
        (r'\brequests\b', 'HTTP requests'),

        # System access
        (r'\bos\s*\.', 'OS module'),
        (r'\bsubprocess\b', 'Subprocess'),
        (r'\bsys\s*\.', 'Sys module'),

        # Resource exhaustion
        (r'\*\s*\*\s*\d{3,}', 'Large exponentiation'),
        (r'\[\s*\d+\s*\]\s*\*\s*\d{6,}', 'Large list multiplication'),
    ]

    def __init__(self):
        import re
        self.patterns = [(re.compile(p), desc) for p, desc in self.DANGEROUS_PATTERNS]

    def check_patterns(
        self,
        expr: str,
        lang: ExpressionLanguage,
    ) -> List[SecurityViolation]:
        """Check for dangerous patterns in expression."""
        violations = []

        for pattern, description in self.patterns:
            if pattern.search(expr):
                violations.append(SecurityViolation(
                    violation_type=SecurityViolationType.DANGEROUS_PATTERN,
                    expression=expr,
                    language=lang,
                    details=description,
                    severity="critical",
                ))

        return violations


class TimeoutChecker:
    """Check execution time limits."""

    def __init__(self, parser: MultiLanguageParser):
        self.parser = parser

    def check_timeout(
        self,
        expr: str,
        lang: ExpressionLanguage,
        context: Dict[str, Any],
        max_ms: int,
    ) -> Tuple[Any, float, Optional[SecurityViolation]]:
        """
        Execute expression with timeout.
        Returns (result, time_ms, violation).
        """
        start = time.perf_counter()
        result = None
        violation = None

        try:
            with timeout_context(max_ms / 1000.0):
                result, error = self.parser.evaluate(expr, lang, context)
                if error:
                    result = None
        except TimeoutError:
            violation = SecurityViolation(
                violation_type=SecurityViolationType.TIMEOUT_EXCEEDED,
                expression=expr,
                language=lang,
                details=f"Execution exceeded {max_ms}ms limit",
                severity="high",
            )
        except Exception as e:
            # Other errors are not timeout violations
            pass

        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms, violation


class ExpressionSecurityTester:
    """Main security testing class."""

    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.parser = MultiLanguageParser()
        self.depth_checker = DepthChecker(self.parser)
        self.function_checker = FunctionChecker(self.parser)
        self.pattern_checker = PatternChecker()
        self.timeout_checker = TimeoutChecker(self.parser)

    def test_expression(
        self,
        expr: str,
        lang: ExpressionLanguage,
        context: Dict[str, Any] = None,
    ) -> SecurityTestResult:
        """Run all security tests on an expression."""
        context = context or {}
        violations = []

        # Get parsed info
        parsed = self.parser.parse(expr, lang)

        # 1. Depth check
        depth_violation = self.depth_checker.check_depth(
            expr, lang, self.config.max_depth
        )
        if depth_violation:
            violations.append(depth_violation)

        # 2. Function check
        func_violations = self.function_checker.check_functions(
            expr, lang,
            self.config.allowed_functions,
            self.config.blocked_functions,
        )
        violations.extend(func_violations)

        # 3. Pattern check
        pattern_violations = self.pattern_checker.check_patterns(expr, lang)
        violations.extend(pattern_violations)

        # 4. Timeout check (only if no blocking violations)
        exec_time = 0.0
        if not any(v.severity == 'critical' for v in violations):
            _, exec_time, timeout_violation = self.timeout_checker.check_timeout(
                expr, lang, context, self.config.max_execution_ms
            )
            if timeout_violation:
                violations.append(timeout_violation)

        return SecurityTestResult(
            expression=expr,
            language=lang,
            config=self.config,
            passed=len(violations) == 0,
            violations=violations,
            execution_time_ms=exec_time,
            depth=parsed.depth,
        )

    def test_batch(
        self,
        expressions: List[Tuple[str, ExpressionLanguage]],
        context: Dict[str, Any] = None,
    ) -> List[SecurityTestResult]:
        """Test multiple expressions."""
        return [self.test_expression(expr, lang, context)
                for expr, lang in expressions]


class SecurityTestSuite:
    """Pre-defined security test cases."""

    # Test expressions designed to probe security limits
    TEST_CASES = [
        # Safe expressions
        ('x > 0', ExpressionLanguage.RAW, True, 'simple_safe'),
        ('x > 0 and y < 10', ExpressionLanguage.RAW, True, 'compound_safe'),
        ('len(items) > 0', ExpressionLanguage.RAW, True, 'safe_function'),

        # Depth attacks
        ('((((((((((x))))))))))', ExpressionLanguage.RAW, False, 'deep_nesting'),
        ('x and (y and (z and (a and (b))))', ExpressionLanguage.RAW, True, 'moderate_nesting'),

        # Blocked functions
        ('eval("1+1")', ExpressionLanguage.RAW, False, 'eval_attack'),
        ('exec("print(1)")', ExpressionLanguage.RAW, False, 'exec_attack'),
        ('__import__("os")', ExpressionLanguage.RAW, False, 'import_attack'),

        # Dangerous patterns
        ('x.__class__.__bases__', ExpressionLanguage.RAW, False, 'dunder_attack'),
        ('open("/etc/passwd")', ExpressionLanguage.RAW, False, 'file_attack'),

        # Timeout attacks
        ('2 ** 10', ExpressionLanguage.RAW, True, 'small_exponent'),
        ('2 ** 1000', ExpressionLanguage.RAW, False, 'large_exponent'),

        # Cross-language attacks
        ('x > 0 && y < 10', ExpressionLanguage.CEL, True, 'cel_safe'),
        ('x > 0 && eval("y")', ExpressionLanguage.CEL, False, 'cel_eval'),
    ]

    def __init__(self, config: SecurityConfig = None):
        self.tester = ExpressionSecurityTester(config)

    def run_all_tests(self, verbose: bool = True) -> Dict[str, bool]:
        """Run all security tests."""
        results = {}

        if verbose:
            print("Security Test Suite")
            print("-" * 60)

        for expr, lang, expected_pass, name in self.TEST_CASES:
            result = self.tester.test_expression(expr, lang)
            test_passed = result.passed == expected_pass
            results[name] = test_passed

            if verbose:
                status = "PASS" if test_passed else "FAIL"
                expected = "safe" if expected_pass else "blocked"
                actual = "passed" if result.passed else "blocked"
                print(f"  [{status}] {name}: expected={expected}, actual={actual}")
                if not test_passed:
                    for v in result.violations:
                        print(f"         Violation: {v.violation_type.name}")

        return results


def demo():
    """Demonstrate security testing."""
    print("=" * 60)
    print("EXPRESSION SECURITY TESTER")
    print("=" * 60)

    # Create tester with config
    config = SecurityConfig(
        max_depth=8,
        max_execution_ms=50,
    )
    tester = ExpressionSecurityTester(config)

    print(f"\nSecurity Config:")
    for k, v in config.to_dict().items():
        print(f"  {k}: {v}")

    # Test various expressions
    test_exprs = [
        ('x > 0 and y < 10', ExpressionLanguage.RAW),
        ('eval("x + 1")', ExpressionLanguage.RAW),
        ('((((((x))))))', ExpressionLanguage.RAW),
        ('x.__class__', ExpressionLanguage.RAW),
        ('len(items) > 0', ExpressionLanguage.RAW),
    ]

    print("\n" + "-" * 60)
    print("Individual Tests")
    print("-" * 60)

    for expr, lang in test_exprs:
        result = tester.test_expression(expr, lang)
        status = "PASS" if result.passed else "FAIL"
        print(f"\n[{status}] {lang.name}: {expr}")
        print(f"  Depth: {result.depth}")
        print(f"  Time: {result.execution_time_ms:.2f}ms")
        for v in result.violations:
            print(f"  Violation: {v.violation_type.name} - {v.details}")

    # Run test suite
    print("\n" + "=" * 60)
    print("SECURITY TEST SUITE")
    print("=" * 60)

    suite = SecurityTestSuite(config)
    suite_results = suite.run_all_tests(verbose=True)

    print("\n" + "-" * 60)
    passed = sum(1 for v in suite_results.values() if v)
    print(f"Results: {passed}/{len(suite_results)} tests passed")

    return tester, suite_results


if __name__ == "__main__":
    demo()
