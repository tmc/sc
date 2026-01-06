"""
exp_sc_error_patterns: Analyze Common Statechart Generation Errors

GOAL: Classify and detect errors in generated/malformed statecharts.

Error Categories:
1. Missing States - referenced but not defined
2. Dangling Transitions - source/target doesn't exist
3. Invalid Guards - syntax errors, undefined variables
4. Hierarchy Violations - invalid nesting, parallel conflicts

Components:
- error_taxonomy.py: Error classification hierarchy
- error_detector.py: Detection engine with validators
- error_patterns.py: Common error patterns and examples
- error_fixer.py: Automatic fix suggestions
- benchmark.py: Detection accuracy evaluation
"""

from .error_taxonomy import (
    ErrorCategory,
    ErrorSeverity,
    StatechartError,
    MissingStateError,
    DanglingTransitionError,
    InvalidGuardError,
    HierarchyViolationError,
)
from .error_detector import (
    ErrorDetector,
    DetectionResult,
    detect_errors,
)
from .error_patterns import (
    ErrorPattern,
    PatternMatcher,
    COMMON_PATTERNS,
)
from .error_fixer import (
    ErrorFixer,
    FixSuggestion,
    suggest_fixes,
)
from .benchmark import (
    ErrorDetectionBenchmark,
    BenchmarkResult,
    run_benchmark,
)

__all__ = [
    # Taxonomy
    'ErrorCategory',
    'ErrorSeverity',
    'StatechartError',
    'MissingStateError',
    'DanglingTransitionError',
    'InvalidGuardError',
    'HierarchyViolationError',
    # Detector
    'ErrorDetector',
    'DetectionResult',
    'detect_errors',
    # Patterns
    'ErrorPattern',
    'PatternMatcher',
    'COMMON_PATTERNS',
    # Fixer
    'ErrorFixer',
    'FixSuggestion',
    'suggest_fixes',
    # Benchmark
    'ErrorDetectionBenchmark',
    'BenchmarkResult',
    'run_benchmark',
]
