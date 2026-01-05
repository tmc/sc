"""
Coverage Data Collector

Collects (program, input, coverage) triples for training coverage predictors.
Supports Python and Starlark programs.

Usage:
    collector = CoverageCollector()
    dataset = collector.collect_from_function(my_func, test_inputs)

    # Or collect from many functions
    dataset = collector.collect_from_module(my_module, input_generator)
"""

import ast
import sys
import trace
import coverage
import tempfile
import textwrap
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Callable, Any, Iterator
from pathlib import Path
import json


@dataclass
class CoverageTriple:
    """A (program, input, coverage) training example."""
    program_id: str              # Hash of the program source
    program_source: str          # Source code
    program_ast: Optional[str]   # JSON-serialized AST
    input_repr: str              # String representation of input
    input_value: Any             # Actual input value (if serializable)
    covered_lines: Set[int]      # Lines covered during execution
    total_lines: int             # Total executable lines
    coverage_ratio: float        # covered_lines / total_lines
    execution_trace: List[int]   # Sequence of lines executed
    branches_taken: Dict[int, bool]  # line -> True/False for branches
    error: Optional[str] = None  # If execution failed

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            'program_id': self.program_id,
            'program_source': self.program_source,
            'input_repr': self.input_repr,
            'covered_lines': list(self.covered_lines),
            'total_lines': self.total_lines,
            'coverage_ratio': self.coverage_ratio,
            'execution_trace': self.execution_trace,
            'branches_taken': self.branches_taken,
            'error': self.error,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'CoverageTriple':
        """Load from dict."""
        return cls(
            program_id=d['program_id'],
            program_source=d['program_source'],
            program_ast=None,
            input_repr=d['input_repr'],
            input_value=None,
            covered_lines=set(d['covered_lines']),
            total_lines=d['total_lines'],
            coverage_ratio=d['coverage_ratio'],
            execution_trace=d['execution_trace'],
            branches_taken=d.get('branches_taken', {}),
            error=d.get('error'),
        )


class ExecutionTracer:
    """Trace execution to collect line-by-line coverage."""

    def __init__(self, target_file: str):
        self.target_file = target_file
        self.trace: List[int] = []
        self.covered_lines: Set[int] = set()

    def trace_calls(self, frame, event: str, arg: Any) -> Optional[Callable]:
        """Trace function for sys.settrace."""
        if event != 'line':
            return self.trace_calls

        # Only track our target file
        filename = frame.f_code.co_filename
        if filename != self.target_file:
            return self.trace_calls

        lineno = frame.f_lineno
        self.trace.append(lineno)
        self.covered_lines.add(lineno)

        return self.trace_calls


class CoverageCollector:
    """
    Collect coverage data for training coverage predictors.

    Workflow:
    1. Parse program to get executable lines
    2. Execute with given input while tracing
    3. Record which lines were covered
    4. Store as training example
    """

    def __init__(self, temp_dir: Optional[Path] = None):
        self.temp_dir = temp_dir or Path(tempfile.mkdtemp())
        self.dataset: List[CoverageTriple] = []

    def get_program_id(self, source: str) -> str:
        """Generate a unique ID for a program."""
        return hashlib.md5(source.encode()).hexdigest()[:12]

    def get_executable_lines(self, source: str) -> Set[int]:
        """Get set of executable line numbers from source."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return set()

        executable = set()

        for node in ast.walk(tree):
            if hasattr(node, 'lineno'):
                # Skip decorators, docstrings, etc.
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    # Function/class def line is executable
                    executable.add(node.lineno)
                elif isinstance(node, ast.Expr):
                    if not isinstance(node.value, ast.Constant):
                        # Skip docstrings
                        executable.add(node.lineno)
                elif isinstance(node, (
                    ast.Assign, ast.AnnAssign, ast.AugAssign,
                    ast.Return, ast.Delete, ast.For, ast.AsyncFor,
                    ast.While, ast.If, ast.With, ast.AsyncWith,
                    ast.Raise, ast.Try, ast.Assert, ast.Import,
                    ast.ImportFrom, ast.Global, ast.Nonlocal,
                    ast.Pass, ast.Break, ast.Continue,
                )):
                    executable.add(node.lineno)

        return executable

    def get_branch_lines(self, source: str) -> Dict[int, str]:
        """Get lines that contain branches (if/elif/while/for)."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {}

        branches = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                branches[node.lineno] = 'if'
            elif isinstance(node, ast.While):
                branches[node.lineno] = 'while'
            elif isinstance(node, ast.For):
                branches[node.lineno] = 'for'
            elif isinstance(node, ast.Try):
                branches[node.lineno] = 'try'

        return branches

    def collect_from_source(
        self,
        source: str,
        func_name: str,
        inputs: List[Any],
    ) -> List[CoverageTriple]:
        """
        Collect coverage data from source code.

        Args:
            source: Python source code containing the function
            func_name: Name of the function to call
            inputs: List of inputs to try

        Returns:
            List of CoverageTriple for each input
        """
        results = []
        program_id = self.get_program_id(source)
        executable_lines = self.get_executable_lines(source)
        branch_lines = self.get_branch_lines(source)

        # Write source to temp file
        temp_file = self.temp_dir / f"{program_id}.py"
        temp_file.write_text(source)

        for inp in inputs:
            triple = self._execute_and_trace(
                source=source,
                temp_file=str(temp_file),
                program_id=program_id,
                func_name=func_name,
                inp=inp,
                executable_lines=executable_lines,
                branch_lines=branch_lines,
            )
            results.append(triple)
            self.dataset.append(triple)

        return results

    def _execute_and_trace(
        self,
        source: str,
        temp_file: str,
        program_id: str,
        func_name: str,
        inp: Any,
        executable_lines: Set[int],
        branch_lines: Dict[int, str],
    ) -> CoverageTriple:
        """Execute function with input and trace coverage."""

        # Create tracer
        tracer = ExecutionTracer(temp_file)

        # Create execution namespace
        namespace = {}

        try:
            # Compile and exec the source
            code = compile(source, temp_file, 'exec')
            exec(code, namespace)

            # Get the function
            func = namespace.get(func_name)
            if func is None:
                return CoverageTriple(
                    program_id=program_id,
                    program_source=source,
                    program_ast=None,
                    input_repr=repr(inp),
                    input_value=inp,
                    covered_lines=set(),
                    total_lines=len(executable_lines),
                    coverage_ratio=0.0,
                    execution_trace=[],
                    branches_taken={},
                    error=f"Function '{func_name}' not found",
                )

            # Set up tracing
            old_trace = sys.gettrace()
            sys.settrace(tracer.trace_calls)

            try:
                # Call the function
                if isinstance(inp, tuple):
                    func(*inp)
                elif isinstance(inp, dict):
                    func(**inp)
                else:
                    func(inp)
            finally:
                sys.settrace(old_trace)

            # Compute coverage
            covered = tracer.covered_lines & executable_lines
            coverage_ratio = len(covered) / len(executable_lines) if executable_lines else 0.0

            # Compute branch outcomes
            branches_taken = {}
            for line in branch_lines:
                # Check if line was executed and which path was taken
                if line in covered:
                    # Check if the next line in trace after this branch
                    # was the body or the else
                    # (simplified - full analysis needs AST)
                    branches_taken[line] = True
                else:
                    branches_taken[line] = False

            return CoverageTriple(
                program_id=program_id,
                program_source=source,
                program_ast=None,
                input_repr=repr(inp),
                input_value=inp,
                covered_lines=covered,
                total_lines=len(executable_lines),
                coverage_ratio=coverage_ratio,
                execution_trace=tracer.trace,
                branches_taken=branches_taken,
                error=None,
            )

        except Exception as e:
            return CoverageTriple(
                program_id=program_id,
                program_source=source,
                program_ast=None,
                input_repr=repr(inp),
                input_value=inp,
                covered_lines=set(),
                total_lines=len(executable_lines),
                coverage_ratio=0.0,
                execution_trace=[],
                branches_taken={},
                error=str(e),
            )

    def collect_from_function(
        self,
        func: Callable,
        inputs: List[Any],
    ) -> List[CoverageTriple]:
        """
        Collect coverage data from a function object.

        Args:
            func: The function to trace
            inputs: List of inputs to try

        Returns:
            List of CoverageTriple for each input
        """
        import inspect

        try:
            source = inspect.getsource(func)
            source = textwrap.dedent(source)
            func_name = func.__name__
        except (OSError, TypeError):
            raise ValueError("Cannot get source for function")

        return self.collect_from_source(source, func_name, inputs)

    def save_dataset(self, path: Path):
        """Save collected dataset to JSON."""
        data = [t.to_dict() for t in self.dataset]
        path.write_text(json.dumps(data, indent=2))

    def load_dataset(self, path: Path):
        """Load dataset from JSON."""
        data = json.loads(path.read_text())
        self.dataset = [CoverageTriple.from_dict(d) for d in data]


class SyntheticDataGenerator:
    """
    Generate synthetic programs with known coverage patterns.

    This is useful for controlled experiments where we know
    exactly what coverage to expect.
    """

    def __init__(self, collector: CoverageCollector):
        self.collector = collector

    def generate_if_chain(self, n_branches: int) -> Tuple[str, Dict[Any, Set[int]]]:
        """
        Generate a function with n sequential if statements.

        Returns:
            (source, expected_coverage) where expected_coverage
            maps input values to expected covered lines.
        """
        lines = [
            "def check_value(x):",
            "    result = 0",
        ]

        expected = {}

        for i in range(n_branches):
            lines.append(f"    if x == {i}:")
            lines.append(f"        result = {i + 1}")

            # For input i, lines 1-2 + the matching if + its body are covered
            base_lines = {1, 2}  # def and result = 0
            for j in range(n_branches):
                base_lines.add(3 + j * 2)  # Each if statement
                if j == i:
                    base_lines.add(4 + j * 2)  # The body
            base_lines.add(3 + n_branches * 2)  # return
            expected[i] = base_lines

        lines.append("    return result")

        # For input not in range, only ifs (not bodies) + return covered
        other_lines = {1, 2}
        for j in range(n_branches):
            other_lines.add(3 + j * 2)
        other_lines.add(3 + n_branches * 2)
        expected['other'] = other_lines

        return '\n'.join(lines), expected

    def generate_nested_conditions(self, depth: int) -> Tuple[str, Dict[Tuple, Set[int]]]:
        """
        Generate nested if statements.

        Returns:
            (source, expected_coverage) where expected_coverage
            maps (bool1, bool2, ...) tuples to expected covered lines.
        """
        lines = ["def nested_check(a, b, c):"]
        lines.append("    result = 'none'")

        indent = "    "
        if depth >= 1:
            lines.append(f"{indent}if a:")
            lines.append(f"{indent}    result = 'a'")
        if depth >= 2:
            lines.append(f"{indent}    if b:")
            lines.append(f"{indent}        result = 'ab'")
        if depth >= 3:
            lines.append(f"{indent}        if c:")
            lines.append(f"{indent}            result = 'abc'")

        lines.append("    return result")

        source = '\n'.join(lines)

        # Expected coverage for different inputs
        expected = {}

        # (False, *, *): only outer if checked
        expected[(False, False, False)] = {1, 2, 3, len(lines)}
        expected[(False, True, False)] = {1, 2, 3, len(lines)}
        expected[(False, False, True)] = {1, 2, 3, len(lines)}

        # (True, False, *): outer if true, inner false
        if depth >= 2:
            expected[(True, False, False)] = {1, 2, 3, 4, 5, len(lines)}
            expected[(True, False, True)] = {1, 2, 3, 4, 5, len(lines)}

        # (True, True, False): two levels deep
        if depth >= 2:
            expected[(True, True, False)] = {1, 2, 3, 4, 5, 6, len(lines)}

        # (True, True, True): all the way down
        if depth >= 3:
            expected[(True, True, True)] = {1, 2, 3, 4, 5, 6, 7, 8, len(lines)}

        return source, expected

    def generate_loop_with_conditions(
        self,
        n_elements: int,
    ) -> Tuple[str, Callable[[List], Set[int]]]:
        """
        Generate a loop with conditional processing.

        Returns:
            (source, coverage_function) where coverage_function
            computes expected coverage from input list.
        """
        source = textwrap.dedent("""
            def process_list(items):
                total = 0
                for item in items:
                    if item > 0:
                        total += item
                    else:
                        total -= item
                return total
        """).strip()

        def expected_coverage(items: List) -> Set[int]:
            # Line 1: def
            # Line 2: total = 0
            # Line 3: for item in items
            # Line 4: if item > 0
            # Line 5: total += item
            # Line 6: else (implicit)
            # Line 7: total -= item
            # Line 8: return total

            lines = {1, 2, 3, 8}  # Always: def, init, for, return

            if items:
                lines.add(4)  # If checked for each item

                if any(x > 0 for x in items):
                    lines.add(5)  # Positive branch
                if any(x <= 0 for x in items):
                    lines.add(7)  # Negative branch

            return lines

        return source, expected_coverage


def demo():
    """Demonstrate coverage collection."""
    print("=" * 60)
    print("COVERAGE COLLECTOR DEMO")
    print("=" * 60)

    collector = CoverageCollector()
    generator = SyntheticDataGenerator(collector)

    # Generate a simple program
    source, expected = generator.generate_if_chain(3)

    print("\nGenerated program:")
    for i, line in enumerate(source.split('\n'), 1):
        print(f"  {i:2d}: {line}")

    print("\nCollecting coverage for inputs 0, 1, 2, 5...")

    results = collector.collect_from_source(source, 'check_value', [0, 1, 2, 5])

    print("\nResults:")
    for r in results:
        print(f"  Input: {r.input_repr}")
        print(f"    Covered lines: {sorted(r.covered_lines)}")
        print(f"    Coverage ratio: {r.coverage_ratio:.1%}")
        print(f"    Trace length: {len(r.execution_trace)}")

    # Generate nested conditions
    print("\n" + "=" * 60)
    print("NESTED CONDITIONS")
    print("=" * 60)

    source2, expected2 = generator.generate_nested_conditions(3)

    print("\nGenerated program:")
    for i, line in enumerate(source2.split('\n'), 1):
        print(f"  {i:2d}: {line}")

    test_inputs = [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    ]

    print("\nCollecting coverage...")
    results2 = collector.collect_from_source(source2, 'nested_check', test_inputs)

    for r in results2:
        print(f"  Input: {r.input_repr}")
        print(f"    Covered: {sorted(r.covered_lines)}")
        print(f"    Ratio: {r.coverage_ratio:.1%}")

    print(f"\nTotal dataset size: {len(collector.dataset)}")

    return collector


if __name__ == "__main__":
    demo()
