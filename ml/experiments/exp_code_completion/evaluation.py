"""
Evaluation Module

Measures syntax error rate for constrained vs unconstrained generation.
Benchmarks on code completion tasks.

Metrics:
- Parse success rate (ast.parse succeeds)
- AST validity (no malformed nodes)
- Bracket matching (all brackets closed)
- Indentation validity
- Keyword usage validity
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
import ast
import re
import time
from collections import defaultdict

try:
    from .syntax_statechart import PythonSyntaxStatechart
    from .constrained_generator import ConstrainedCodeGenerator, GenerationConfig
except ImportError:
    from syntax_statechart import PythonSyntaxStatechart
    from constrained_generator import ConstrainedCodeGenerator, GenerationConfig


@dataclass
class SyntaxMetrics:
    """Syntax validity metrics for a code sample."""
    parse_success: bool = False
    ast_valid: bool = False
    brackets_balanced: bool = False
    indent_valid: bool = False
    keyword_valid: bool = False
    error_message: str = ""
    error_line: int = 0

    @property
    def is_valid(self) -> bool:
        """Overall validity."""
        return self.parse_success and self.brackets_balanced


@dataclass
class EvaluationResult:
    """Results from a single generation evaluation."""
    prompt: str
    generated: str
    metrics: SyntaxMetrics
    generation_time: float
    tokens_generated: int
    tokens_blocked: int


@dataclass
class BenchmarkResults:
    """Aggregate benchmark results."""
    total_samples: int = 0
    parse_success_rate: float = 0.0
    bracket_match_rate: float = 0.0
    indent_valid_rate: float = 0.0
    avg_generation_time: float = 0.0
    avg_tokens_per_sample: float = 0.0
    avg_tokens_blocked: float = 0.0
    samples_with_errors: List[EvaluationResult] = field(default_factory=list)


class SyntaxEvaluator:
    """
    Evaluates Python code syntax validity.

    Multiple validation methods:
    1. AST parsing (most authoritative)
    2. Bracket matching (fast check)
    3. Indentation validation
    4. Keyword context validation
    """

    def __init__(self):
        self.bracket_pairs = {'(': ')', '[': ']', '{': '}'}

    def evaluate(self, code: str) -> SyntaxMetrics:
        """Full syntax evaluation."""
        metrics = SyntaxMetrics()

        # Check brackets first (fast)
        metrics.brackets_balanced = self._check_brackets(code)

        # Check indentation
        metrics.indent_valid = self._check_indentation(code)

        # Try to parse
        metrics.parse_success, error = self._try_parse(code)
        if not metrics.parse_success:
            metrics.error_message = error[0] if error else "Unknown error"
            metrics.error_line = error[1] if error and len(error) > 1 else 0

        # Full AST validity
        if metrics.parse_success:
            metrics.ast_valid = self._check_ast_validity(code)

        # Keyword validity
        metrics.keyword_valid = self._check_keywords(code)

        return metrics

    def _try_parse(self, code: str) -> Tuple[bool, Optional[Tuple[str, int]]]:
        """Try to parse code with ast.parse."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, (str(e.msg), e.lineno or 0)
        except Exception as e:
            return False, (str(e), 0)

    def _check_brackets(self, code: str) -> bool:
        """Check if all brackets are balanced."""
        stack = []
        in_string = False
        string_char = None

        i = 0
        while i < len(code):
            char = code[i]

            # Handle strings
            if char in ('"', "'"):
                if not in_string:
                    # Check for triple quote
                    if code[i:i+3] in ('"""', "'''"):
                        in_string = True
                        string_char = code[i:i+3]
                        i += 3
                        continue
                    in_string = True
                    string_char = char
                elif char == string_char or code[i:i+3] == string_char:
                    in_string = False
                    string_char = None
                    if len(string_char) == 3:
                        i += 3
                        continue

            # Skip if in string
            if in_string:
                i += 1
                continue

            # Handle brackets
            if char in self.bracket_pairs:
                stack.append(char)
            elif char in self.bracket_pairs.values():
                if not stack:
                    return False
                expected = self.bracket_pairs[stack.pop()]
                if char != expected:
                    return False

            i += 1

        return len(stack) == 0

    def _check_indentation(self, code: str) -> bool:
        """Check if indentation is consistent."""
        lines = code.split('\n')
        indent_stack = [0]

        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if not stripped or stripped.startswith('#'):
                continue

            indent = len(line) - len(stripped)

            # Check for valid indent change
            if indent > indent_stack[-1]:
                # Increased indent - should follow a colon
                if i > 0:
                    prev = lines[i-1].rstrip()
                    if prev and not prev.endswith(':'):
                        # Unexpected indent increase
                        pass  # Some flexibility needed
                indent_stack.append(indent)
            elif indent < indent_stack[-1]:
                # Decreased indent - should match a previous level
                while indent_stack and indent_stack[-1] > indent:
                    indent_stack.pop()
                if not indent_stack or indent_stack[-1] != indent:
                    # Doesn't match any previous indent level
                    return False

        return True

    def _check_ast_validity(self, code: str) -> bool:
        """Check if AST has valid structure."""
        try:
            tree = ast.parse(code)
            # Walk tree and check for issues
            for node in ast.walk(tree):
                # Check for empty bodies
                if hasattr(node, 'body') and isinstance(node.body, list):
                    if len(node.body) == 0:
                        # Empty body is invalid for some constructs
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef,
                                             ast.If, ast.For, ast.While)):
                            return False
            return True
        except:
            return False

    def _check_keywords(self, code: str) -> bool:
        """Check if keywords are used in valid contexts."""
        # Simple check: return/yield only in functions
        # break/continue only in loops

        try:
            tree = ast.parse(code)
        except:
            return False

        in_function = False
        in_loop = False

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                in_function = True
            elif isinstance(node, (ast.For, ast.While)):
                in_loop = True
            elif isinstance(node, ast.Return):
                # return outside function is still valid at module level in some contexts
                pass
            elif isinstance(node, (ast.Break, ast.Continue)):
                if not in_loop:
                    return False

        return True


class CodeCompletionBenchmark:
    """
    Benchmark for code completion with syntax constraints.

    Compares constrained vs unconstrained generation.
    """

    def __init__(self):
        self.evaluator = SyntaxEvaluator()
        self.prompts = self._default_prompts()

    def _default_prompts(self) -> List[str]:
        """Default code completion prompts."""
        return [
            # Function definitions
            "def factorial(n):",
            "def fibonacci(n):",
            "def is_prime(n):",
            "def binary_search(arr, target):",
            "def quicksort(arr):",

            # Class definitions
            "class Point:",
            "class LinkedList:",
            "class Stack:",

            # Control flow
            "if x > 0:",
            "for i in range(10):",
            "while True:",
            "try:",

            # Complex patterns
            "def parse_json(text):\n    \"\"\"Parse JSON string.\"\"\"",
            "class HTTPClient:\n    def __init__(self, base_url):",

            # List comprehensions
            "[x for x in range(10) if",
            "{k: v for k, v in",

            # Nested structures
            "def outer():\n    def inner():",
            "class Outer:\n    class Inner:",
        ]

    def run_benchmark(
        self,
        generator: Callable[[str], str],
        prompts: List[str] = None,
        constrained: bool = True,
        num_samples_per_prompt: int = 5,
    ) -> BenchmarkResults:
        """
        Run benchmark on a generator.

        Args:
            generator: Function that takes prompt and returns completion
            prompts: List of prompts (uses defaults if None)
            constrained: Whether generator uses constraints
            num_samples_per_prompt: Number of generations per prompt

        Returns: BenchmarkResults
        """
        prompts = prompts or self.prompts
        results = BenchmarkResults()

        all_results: List[EvaluationResult] = []

        for prompt in prompts:
            for _ in range(num_samples_per_prompt):
                start_time = time.time()

                # Generate
                generated = generator(prompt)

                generation_time = time.time() - start_time

                # Combine prompt and generated for full code
                full_code = prompt + generated

                # Evaluate
                metrics = self.evaluator.evaluate(full_code)

                result = EvaluationResult(
                    prompt=prompt,
                    generated=generated,
                    metrics=metrics,
                    generation_time=generation_time,
                    tokens_generated=len(generated.split()),  # Approximate
                    tokens_blocked=0,  # Would come from constrained generator
                )

                all_results.append(result)

                if not metrics.is_valid:
                    results.samples_with_errors.append(result)

        # Compute aggregate metrics
        results.total_samples = len(all_results)

        if results.total_samples > 0:
            results.parse_success_rate = sum(
                1 for r in all_results if r.metrics.parse_success
            ) / results.total_samples

            results.bracket_match_rate = sum(
                1 for r in all_results if r.metrics.brackets_balanced
            ) / results.total_samples

            results.indent_valid_rate = sum(
                1 for r in all_results if r.metrics.indent_valid
            ) / results.total_samples

            results.avg_generation_time = sum(
                r.generation_time for r in all_results
            ) / results.total_samples

            results.avg_tokens_per_sample = sum(
                r.tokens_generated for r in all_results
            ) / results.total_samples

        return results

    def compare_constrained_unconstrained(
        self,
        constrained_gen: Callable[[str], str],
        unconstrained_gen: Callable[[str], str],
        prompts: List[str] = None,
        num_samples: int = 5,
    ) -> Dict[str, BenchmarkResults]:
        """
        Compare constrained vs unconstrained generation.

        Returns dict with 'constrained' and 'unconstrained' results.
        """
        prompts = prompts or self.prompts

        print("Running constrained benchmark...")
        constrained_results = self.run_benchmark(
            constrained_gen, prompts, constrained=True, num_samples_per_prompt=num_samples
        )

        print("Running unconstrained benchmark...")
        unconstrained_results = self.run_benchmark(
            unconstrained_gen, prompts, constrained=False, num_samples_per_prompt=num_samples
        )

        return {
            'constrained': constrained_results,
            'unconstrained': unconstrained_results,
        }

    def report(self, results: Dict[str, BenchmarkResults]) -> str:
        """Generate comparison report."""
        lines = [
            "=" * 60,
            "CODE COMPLETION BENCHMARK RESULTS",
            "=" * 60,
            "",
        ]

        for name, res in results.items():
            lines.append(f"--- {name.upper()} ---")
            lines.append(f"Total samples: {res.total_samples}")
            lines.append(f"Parse success rate: {res.parse_success_rate:.1%}")
            lines.append(f"Bracket match rate: {res.bracket_match_rate:.1%}")
            lines.append(f"Indent valid rate: {res.indent_valid_rate:.1%}")
            lines.append(f"Avg generation time: {res.avg_generation_time:.3f}s")
            lines.append(f"Samples with errors: {len(res.samples_with_errors)}")
            lines.append("")

        # Comparison
        if 'constrained' in results and 'unconstrained' in results:
            c = results['constrained']
            u = results['unconstrained']

            improvement = c.parse_success_rate - u.parse_success_rate

            lines.append("--- COMPARISON ---")
            lines.append(f"Parse success improvement: {improvement:+.1%}")
            lines.append(f"Error reduction: {1 - len(c.samples_with_errors)/max(1, len(u.samples_with_errors)):.1%}")

        return '\n'.join(lines)


# Simulated generators for testing
def simulated_unconstrained_generator(prompt: str) -> str:
    """Simulate unconstrained generation (may have syntax errors)."""
    import random

    # Randomly introduce errors
    completions = [
        # Valid completions
        "\n    pass\n",
        "\n    return None\n",
        "\n    x = 1\n    return x\n",

        # Invalid completions
        "\n    return\n    x = 1\n",  # Invalid indent
        "\n    if True\n        pass\n",  # Missing colon
        "\n    for i in range(10\n        print(i)\n",  # Missing bracket
        "\n    def inner(\n    pass\n",  # Unclosed paren
    ]

    return random.choice(completions)


def simulated_constrained_generator(prompt: str) -> str:
    """Simulate constrained generation (always valid)."""
    # Always return valid completion
    return "\n    pass\n"


# Demo
if __name__ == "__main__":
    print("=" * 60)
    print("CODE COMPLETION EVALUATION DEMO")
    print("=" * 60)

    # Test evaluator
    evaluator = SyntaxEvaluator()

    test_cases = [
        ("def foo():\n    pass", True),
        ("def foo(\n    pass", False),  # Unclosed paren
        ("if True:\npass", False),  # Bad indent
        ("for i in range(10):\n    print(i)", True),
        ("class Foo:\n    def bar(self):\n        return 1", True),
    ]

    print("\n--- Syntax Evaluator Tests ---")
    for code, expected in test_cases:
        metrics = evaluator.evaluate(code)
        status = "✓" if metrics.parse_success == expected else "✗"
        print(f"{status} Expected valid={expected}, got valid={metrics.parse_success}")
        if not metrics.parse_success:
            print(f"  Error: {metrics.error_message}")

    # Run benchmark
    print("\n--- Running Benchmark ---")
    benchmark = CodeCompletionBenchmark()

    results = benchmark.compare_constrained_unconstrained(
        simulated_constrained_generator,
        simulated_unconstrained_generator,
        prompts=benchmark.prompts[:5],  # Use fewer for demo
        num_samples=3,
    )

    print("\n" + benchmark.report(results))
