"""
Coverage Prediction Dataset

Structured dataset for training and evaluating coverage predictors.
Includes diverse program patterns with varying complexity.

Dataset Structure:
- Programs organized by category (control flow, loops, functions, etc.)
- Multiple inputs per program covering different execution paths
- Ground truth coverage from actual execution
- Metadata for analysis (complexity, branch count, etc.)

Target: 1000+ examples across 100+ unique programs
"""

import ast
import json
import random
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Tuple, Optional, Any, Callable, Iterator
from pathlib import Path
from enum import Enum, auto
import textwrap

from .coverage_collector import CoverageCollector, CoverageTriple


class ProgramCategory(Enum):
    """Categories of programs for balanced dataset."""
    SIMPLE_BRANCH = auto()      # Single if/else
    MULTI_BRANCH = auto()       # Multiple if/elif/else
    NESTED_BRANCH = auto()      # Nested conditions
    SIMPLE_LOOP = auto()        # Single for/while loop
    NESTED_LOOP = auto()        # Nested loops
    LOOP_WITH_BRANCH = auto()   # Loop containing branches
    EARLY_RETURN = auto()       # Functions with early returns
    EXCEPTION_HANDLING = auto() # Try/except blocks
    RECURSION = auto()          # Recursive functions
    LIST_PROCESSING = auto()    # List comprehensions, map, filter
    STRING_PROCESSING = auto()  # String operations
    MATH_OPERATIONS = auto()    # Mathematical computations
    STATE_MACHINE = auto()      # State-based logic
    VALIDATION = auto()         # Input validation patterns


@dataclass
class ProgramTemplate:
    """Template for generating program variants."""
    category: ProgramCategory
    name: str
    source: str
    func_name: str
    input_generator: Callable[[], List[Any]]
    description: str = ""
    min_complexity: int = 1
    max_complexity: int = 5


@dataclass
class DatasetExample:
    """A single training/test example."""
    # Identifiers
    example_id: str
    program_id: str
    category: str

    # Program
    source: str
    func_name: str
    n_lines: int
    n_branches: int
    n_loops: int
    max_nesting: int

    # Input
    input_repr: str
    input_type: str

    # Coverage (ground truth)
    covered_lines: List[int]
    uncovered_lines: List[int]
    coverage_ratio: float
    execution_trace: List[int]
    branches_taken: Dict[str, bool]

    # Metadata
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'DatasetExample':
        d['covered_lines'] = list(d['covered_lines'])
        d['uncovered_lines'] = list(d['uncovered_lines'])
        return cls(**d)


@dataclass
class Dataset:
    """Complete dataset with train/val/test splits."""
    examples: List[DatasetExample]
    metadata: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[DatasetExample]:
        return iter(self.examples)

    def filter_by_category(self, category: str) -> 'Dataset':
        """Get examples from a specific category."""
        filtered = [e for e in self.examples if e.category == category]
        return Dataset(examples=filtered, metadata=self.metadata.copy())

    def split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple['Dataset', 'Dataset', 'Dataset']:
        """Split into train/val/test sets."""
        random.seed(seed)
        examples = self.examples.copy()
        random.shuffle(examples)

        n = len(examples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train = Dataset(examples[:n_train], {'split': 'train'})
        val = Dataset(examples[n_train:n_train + n_val], {'split': 'val'})
        test = Dataset(examples[n_train + n_val:], {'split': 'test'})

        return train, val, test

    def statistics(self) -> Dict:
        """Compute dataset statistics."""
        if not self.examples:
            return {}

        categories = {}
        for e in self.examples:
            categories[e.category] = categories.get(e.category, 0) + 1

        coverage_ratios = [e.coverage_ratio for e in self.examples]
        line_counts = [e.n_lines for e in self.examples]
        branch_counts = [e.n_branches for e in self.examples]

        return {
            'n_examples': len(self.examples),
            'n_programs': len(set(e.program_id for e in self.examples)),
            'categories': categories,
            'coverage': {
                'min': min(coverage_ratios),
                'max': max(coverage_ratios),
                'mean': sum(coverage_ratios) / len(coverage_ratios),
            },
            'lines': {
                'min': min(line_counts),
                'max': max(line_counts),
                'mean': sum(line_counts) / len(line_counts),
            },
            'branches': {
                'min': min(branch_counts),
                'max': max(branch_counts),
                'mean': sum(branch_counts) / len(branch_counts),
            },
        }

    def save(self, path: Path):
        """Save dataset to JSON."""
        data = {
            'examples': [e.to_dict() for e in self.examples],
            'metadata': self.metadata,
            'statistics': self.statistics(),
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> 'Dataset':
        """Load dataset from JSON."""
        data = json.loads(path.read_text())
        examples = [DatasetExample.from_dict(e) for e in data['examples']]
        return cls(examples=examples, metadata=data.get('metadata', {}))


class ProgramAnalyzer:
    """Analyze program structure for metadata."""

    @staticmethod
    def count_branches(source: str) -> int:
        """Count branch points (if/elif/while conditions)."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return 0

        count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                count += 1
            # elif is represented as nested If in else
        return count

    @staticmethod
    def count_loops(source: str) -> int:
        """Count loop constructs."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return 0

        count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                count += 1
        return count

    @staticmethod
    def max_nesting_depth(source: str) -> int:
        """Compute maximum nesting depth."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return 0

        def depth(node: ast.AST, current: int = 0) -> int:
            max_depth = current

            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    max_depth = max(max_depth, depth(child, current + 1))
                else:
                    max_depth = max(max_depth, depth(child, current))

            return max_depth

        return depth(tree)

    @staticmethod
    def count_lines(source: str) -> int:
        """Count non-empty, non-comment lines."""
        lines = source.strip().split('\n')
        count = 0
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                count += 1
        return count


class DatasetGenerator:
    """
    Generate comprehensive coverage prediction dataset.

    Includes diverse program patterns:
    - Control flow variations
    - Different input types
    - Edge cases and boundary conditions
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.collector = CoverageCollector()
        self.analyzer = ProgramAnalyzer()
        self.templates: List[ProgramTemplate] = []
        self._register_templates()

    def _register_templates(self):
        """Register all program templates."""
        self._register_branch_templates()
        self._register_loop_templates()
        self._register_function_templates()
        self._register_processing_templates()
        self._register_validation_templates()
        self._register_complex_templates()

    def _register_branch_templates(self):
        """Register branching program templates."""

        # Simple if/else
        self.templates.append(ProgramTemplate(
            category=ProgramCategory.SIMPLE_BRANCH,
            name="sign_check",
            source=textwrap.dedent('''
                def sign_check(x):
                    if x > 0:
                        return "positive"
                    else:
                        return "non-positive"
            ''').strip(),
            func_name="sign_check",
            input_generator=lambda: [-10, -1, 0, 1, 10, 100, -100],
            description="Check sign of number",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.SIMPLE_BRANCH,
            name="is_even",
            source=textwrap.dedent('''
                def is_even(n):
                    if n % 2 == 0:
                        return True
                    else:
                        return False
            ''').strip(),
            func_name="is_even",
            input_generator=lambda: list(range(-5, 6)),
            description="Check if number is even",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.SIMPLE_BRANCH,
            name="abs_value",
            source=textwrap.dedent('''
                def abs_value(x):
                    if x < 0:
                        return -x
                    else:
                        return x
            ''').strip(),
            func_name="abs_value",
            input_generator=lambda: [-5, -1, 0, 1, 5],
            description="Absolute value",
        ))

        # Multi-branch (if/elif/else)
        self.templates.append(ProgramTemplate(
            category=ProgramCategory.MULTI_BRANCH,
            name="grade_letter",
            source=textwrap.dedent('''
                def grade_letter(score):
                    if score >= 90:
                        return "A"
                    elif score >= 80:
                        return "B"
                    elif score >= 70:
                        return "C"
                    elif score >= 60:
                        return "D"
                    else:
                        return "F"
            ''').strip(),
            func_name="grade_letter",
            input_generator=lambda: [95, 85, 75, 65, 55, 100, 0, 89, 79, 69, 59],
            description="Convert score to letter grade",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.MULTI_BRANCH,
            name="classify_age",
            source=textwrap.dedent('''
                def classify_age(age):
                    if age < 0:
                        return "invalid"
                    elif age < 13:
                        return "child"
                    elif age < 20:
                        return "teenager"
                    elif age < 60:
                        return "adult"
                    else:
                        return "senior"
            ''').strip(),
            func_name="classify_age",
            input_generator=lambda: [-1, 0, 5, 12, 13, 15, 19, 20, 30, 59, 60, 80],
            description="Classify age into categories",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.MULTI_BRANCH,
            name="day_type",
            source=textwrap.dedent('''
                def day_type(day):
                    if day == 0:
                        return "Sunday"
                    elif day == 6:
                        return "Saturday"
                    elif 1 <= day <= 5:
                        return "Weekday"
                    else:
                        return "Invalid"
            ''').strip(),
            func_name="day_type",
            input_generator=lambda: list(range(-1, 8)),
            description="Classify day of week",
        ))

        # Nested branches
        self.templates.append(ProgramTemplate(
            category=ProgramCategory.NESTED_BRANCH,
            name="classify_point",
            source=textwrap.dedent('''
                def classify_point(x, y):
                    if x >= 0:
                        if y >= 0:
                            return "Q1"
                        else:
                            return "Q4"
                    else:
                        if y >= 0:
                            return "Q2"
                        else:
                            return "Q3"
            ''').strip(),
            func_name="classify_point",
            input_generator=lambda: [(1, 1), (1, -1), (-1, 1), (-1, -1), (0, 0), (5, -3)],
            description="Classify point by quadrant",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.NESTED_BRANCH,
            name="triangle_type",
            source=textwrap.dedent('''
                def triangle_type(a, b, c):
                    if a <= 0 or b <= 0 or c <= 0:
                        return "invalid"
                    if a + b <= c or b + c <= a or a + c <= b:
                        return "invalid"
                    if a == b == c:
                        return "equilateral"
                    elif a == b or b == c or a == c:
                        return "isosceles"
                    else:
                        return "scalene"
            ''').strip(),
            func_name="triangle_type",
            input_generator=lambda: [
                (3, 3, 3), (3, 3, 4), (3, 4, 5), (1, 2, 3),
                (0, 1, 1), (-1, 2, 2), (5, 5, 5), (2, 2, 3),
            ],
            description="Classify triangle type",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.NESTED_BRANCH,
            name="bmi_category",
            source=textwrap.dedent('''
                def bmi_category(weight, height):
                    if weight <= 0 or height <= 0:
                        return "invalid"
                    bmi = weight / (height * height)
                    if bmi < 18.5:
                        return "underweight"
                    elif bmi < 25:
                        return "normal"
                    elif bmi < 30:
                        return "overweight"
                    else:
                        return "obese"
            ''').strip(),
            func_name="bmi_category",
            input_generator=lambda: [
                (50, 1.7), (70, 1.75), (90, 1.8), (100, 1.6),
                (0, 1.7), (70, 0), (60, 1.65), (80, 1.7),
            ],
            description="Calculate BMI category",
        ))

    def _register_loop_templates(self):
        """Register loop program templates."""

        # Simple loops
        self.templates.append(ProgramTemplate(
            category=ProgramCategory.SIMPLE_LOOP,
            name="sum_list",
            source=textwrap.dedent('''
                def sum_list(items):
                    total = 0
                    for item in items:
                        total += item
                    return total
            ''').strip(),
            func_name="sum_list",
            input_generator=lambda: [[], [1], [1, 2], [1, 2, 3, 4, 5], [-1, 1]],
            description="Sum elements in list",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.SIMPLE_LOOP,
            name="count_items",
            source=textwrap.dedent('''
                def count_items(items):
                    count = 0
                    for item in items:
                        count += 1
                    return count
            ''').strip(),
            func_name="count_items",
            input_generator=lambda: [[], [1], [1, 2, 3], list(range(10))],
            description="Count items in list",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.SIMPLE_LOOP,
            name="factorial",
            source=textwrap.dedent('''
                def factorial(n):
                    result = 1
                    i = 1
                    while i <= n:
                        result *= i
                        i += 1
                    return result
            ''').strip(),
            func_name="factorial",
            input_generator=lambda: [0, 1, 2, 3, 4, 5, 10],
            description="Calculate factorial",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.SIMPLE_LOOP,
            name="reverse_list",
            source=textwrap.dedent('''
                def reverse_list(items):
                    result = []
                    for item in items:
                        result = [item] + result
                    return result
            ''').strip(),
            func_name="reverse_list",
            input_generator=lambda: [[], [1], [1, 2], [1, 2, 3, 4]],
            description="Reverse a list",
        ))

        # Loop with branch
        self.templates.append(ProgramTemplate(
            category=ProgramCategory.LOOP_WITH_BRANCH,
            name="sum_positive",
            source=textwrap.dedent('''
                def sum_positive(items):
                    total = 0
                    for item in items:
                        if item > 0:
                            total += item
                    return total
            ''').strip(),
            func_name="sum_positive",
            input_generator=lambda: [
                [], [1, 2, 3], [-1, -2, -3], [1, -1, 2, -2],
                [0, 1, 0, 2], [-5, 5, -10, 10],
            ],
            description="Sum only positive numbers",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.LOOP_WITH_BRANCH,
            name="count_even_odd",
            source=textwrap.dedent('''
                def count_even_odd(items):
                    even = 0
                    odd = 0
                    for item in items:
                        if item % 2 == 0:
                            even += 1
                        else:
                            odd += 1
                    return (even, odd)
            ''').strip(),
            func_name="count_even_odd",
            input_generator=lambda: [
                [], [2], [1], [1, 2], [1, 2, 3, 4, 5, 6],
            ],
            description="Count even and odd numbers",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.LOOP_WITH_BRANCH,
            name="filter_range",
            source=textwrap.dedent('''
                def filter_range(items, low, high):
                    result = []
                    for item in items:
                        if item >= low and item <= high:
                            result.append(item)
                    return result
            ''').strip(),
            func_name="filter_range",
            input_generator=lambda: [
                ([], 0, 10), ([1, 2, 3], 0, 10), ([1, 5, 10, 15], 3, 12),
                ([1, 2, 3], 5, 10), ([-5, 0, 5], -2, 2),
            ],
            description="Filter items in range",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.LOOP_WITH_BRANCH,
            name="find_max",
            source=textwrap.dedent('''
                def find_max(items):
                    if not items:
                        return None
                    max_val = items[0]
                    for item in items:
                        if item > max_val:
                            max_val = item
                    return max_val
            ''').strip(),
            func_name="find_max",
            input_generator=lambda: [
                [], [5], [1, 2, 3], [3, 2, 1], [1, 5, 2, 4, 3],
            ],
            description="Find maximum value",
        ))

        # Nested loops
        self.templates.append(ProgramTemplate(
            category=ProgramCategory.NESTED_LOOP,
            name="matrix_sum",
            source=textwrap.dedent('''
                def matrix_sum(matrix):
                    total = 0
                    for row in matrix:
                        for val in row:
                            total += val
                    return total
            ''').strip(),
            func_name="matrix_sum",
            input_generator=lambda: [
                [], [[]], [[1]], [[1, 2], [3, 4]], [[1, 2, 3], [4, 5, 6]],
            ],
            description="Sum all matrix elements",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.NESTED_LOOP,
            name="find_pairs",
            source=textwrap.dedent('''
                def find_pairs(items, target):
                    pairs = []
                    n = len(items)
                    i = 0
                    while i < n:
                        j = i + 1
                        while j < n:
                            if items[i] + items[j] == target:
                                pairs.append((items[i], items[j]))
                            j += 1
                        i += 1
                    return pairs
            ''').strip(),
            func_name="find_pairs",
            input_generator=lambda: [
                ([], 5), ([1, 2, 3], 5), ([1, 2, 3, 4], 5),
                ([2, 2, 2], 4), ([1], 2),
            ],
            description="Find pairs summing to target",
        ))

    def _register_function_templates(self):
        """Register function-related templates."""

        # Early return
        self.templates.append(ProgramTemplate(
            category=ProgramCategory.EARLY_RETURN,
            name="find_first",
            source=textwrap.dedent('''
                def find_first(items, target):
                    for item in items:
                        if item == target:
                            return item
                    return None
            ''').strip(),
            func_name="find_first",
            input_generator=lambda: [
                ([], 5), ([1, 2, 3], 2), ([1, 2, 3], 5),
                ([5, 1, 2], 5), ([1, 2, 5], 5),
            ],
            description="Find first occurrence",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.EARLY_RETURN,
            name="all_positive",
            source=textwrap.dedent('''
                def all_positive(items):
                    for item in items:
                        if item <= 0:
                            return False
                    return True
            ''').strip(),
            func_name="all_positive",
            input_generator=lambda: [
                [], [1], [1, 2, 3], [1, -1, 2], [-1, 2, 3], [0, 1, 2],
            ],
            description="Check if all positive",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.EARLY_RETURN,
            name="any_negative",
            source=textwrap.dedent('''
                def any_negative(items):
                    for item in items:
                        if item < 0:
                            return True
                    return False
            ''').strip(),
            func_name="any_negative",
            input_generator=lambda: [
                [], [1], [1, 2, 3], [1, -1, 2], [-1, 2, 3],
            ],
            description="Check if any negative",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.EARLY_RETURN,
            name="binary_search",
            source=textwrap.dedent('''
                def binary_search(items, target):
                    low = 0
                    high = len(items) - 1
                    while low <= high:
                        mid = (low + high) // 2
                        if items[mid] == target:
                            return mid
                        elif items[mid] < target:
                            low = mid + 1
                        else:
                            high = mid - 1
                    return -1
            ''').strip(),
            func_name="binary_search",
            input_generator=lambda: [
                ([], 5), ([1], 1), ([1], 2), ([1, 2, 3, 4, 5], 3),
                ([1, 2, 3, 4, 5], 1), ([1, 2, 3, 4, 5], 5), ([1, 2, 3, 4, 5], 6),
            ],
            description="Binary search",
        ))

        # Exception handling
        self.templates.append(ProgramTemplate(
            category=ProgramCategory.EXCEPTION_HANDLING,
            name="safe_divide",
            source=textwrap.dedent('''
                def safe_divide(a, b):
                    try:
                        result = a / b
                        return result
                    except ZeroDivisionError:
                        return None
            ''').strip(),
            func_name="safe_divide",
            input_generator=lambda: [(10, 2), (10, 0), (0, 5), (-10, 2)],
            description="Safe division with error handling",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.EXCEPTION_HANDLING,
            name="safe_index",
            source=textwrap.dedent('''
                def safe_index(items, idx):
                    try:
                        return items[idx]
                    except IndexError:
                        return None
                    except TypeError:
                        return None
            ''').strip(),
            func_name="safe_index",
            input_generator=lambda: [
                ([1, 2, 3], 0), ([1, 2, 3], 2), ([1, 2, 3], 5),
                ([1, 2, 3], -1), ([], 0),
            ],
            description="Safe list indexing",
        ))

    def _register_processing_templates(self):
        """Register data processing templates."""

        # List processing
        self.templates.append(ProgramTemplate(
            category=ProgramCategory.LIST_PROCESSING,
            name="double_positive",
            source=textwrap.dedent('''
                def double_positive(items):
                    result = []
                    for item in items:
                        if item > 0:
                            result.append(item * 2)
                    return result
            ''').strip(),
            func_name="double_positive",
            input_generator=lambda: [
                [], [1, 2, 3], [-1, 1, -2, 2], [0, 1, 2],
            ],
            description="Double positive numbers",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.LIST_PROCESSING,
            name="flatten",
            source=textwrap.dedent('''
                def flatten(nested):
                    result = []
                    for item in nested:
                        if isinstance(item, list):
                            for sub in item:
                                result.append(sub)
                        else:
                            result.append(item)
                    return result
            ''').strip(),
            func_name="flatten",
            input_generator=lambda: [
                [], [1, 2, 3], [[1, 2], [3, 4]], [1, [2, 3], 4],
            ],
            description="Flatten nested list (one level)",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.LIST_PROCESSING,
            name="unique",
            source=textwrap.dedent('''
                def unique(items):
                    seen = []
                    result = []
                    for item in items:
                        if item not in seen:
                            seen.append(item)
                            result.append(item)
                    return result
            ''').strip(),
            func_name="unique",
            input_generator=lambda: [
                [], [1], [1, 1, 1], [1, 2, 1, 3, 2], [1, 2, 3, 4, 5],
            ],
            description="Remove duplicates preserving order",
        ))

        # String processing
        self.templates.append(ProgramTemplate(
            category=ProgramCategory.STRING_PROCESSING,
            name="count_vowels",
            source=textwrap.dedent('''
                def count_vowels(s):
                    count = 0
                    for c in s:
                        if c in "aeiouAEIOU":
                            count += 1
                    return count
            ''').strip(),
            func_name="count_vowels",
            input_generator=lambda: ["", "hello", "HELLO", "xyz", "aeiou", "bcdfg"],
            description="Count vowels in string",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.STRING_PROCESSING,
            name="is_palindrome",
            source=textwrap.dedent('''
                def is_palindrome(s):
                    left = 0
                    right = len(s) - 1
                    while left < right:
                        if s[left] != s[right]:
                            return False
                        left += 1
                        right -= 1
                    return True
            ''').strip(),
            func_name="is_palindrome",
            input_generator=lambda: ["", "a", "ab", "aa", "aba", "abba", "abcd"],
            description="Check if string is palindrome",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.STRING_PROCESSING,
            name="reverse_words",
            source=textwrap.dedent('''
                def reverse_words(s):
                    words = s.split()
                    result = []
                    for word in words:
                        result = [word] + result
                    return " ".join(result)
            ''').strip(),
            func_name="reverse_words",
            input_generator=lambda: ["", "hello", "hello world", "a b c d"],
            description="Reverse word order",
        ))

        # Math operations
        self.templates.append(ProgramTemplate(
            category=ProgramCategory.MATH_OPERATIONS,
            name="gcd",
            source=textwrap.dedent('''
                def gcd(a, b):
                    while b != 0:
                        temp = b
                        b = a % b
                        a = temp
                    return a
            ''').strip(),
            func_name="gcd",
            input_generator=lambda: [(12, 8), (17, 5), (100, 25), (7, 7), (1, 100)],
            description="Greatest common divisor",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.MATH_OPERATIONS,
            name="is_prime",
            source=textwrap.dedent('''
                def is_prime(n):
                    if n < 2:
                        return False
                    if n == 2:
                        return True
                    if n % 2 == 0:
                        return False
                    i = 3
                    while i * i <= n:
                        if n % i == 0:
                            return False
                        i += 2
                    return True
            ''').strip(),
            func_name="is_prime",
            input_generator=lambda: [0, 1, 2, 3, 4, 5, 7, 9, 11, 15, 17, 25, 29],
            description="Check if prime",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.MATH_OPERATIONS,
            name="fibonacci",
            source=textwrap.dedent('''
                def fibonacci(n):
                    if n <= 0:
                        return 0
                    if n == 1:
                        return 1
                    a = 0
                    b = 1
                    i = 2
                    while i <= n:
                        temp = a + b
                        a = b
                        b = temp
                        i += 1
                    return b
            ''').strip(),
            func_name="fibonacci",
            input_generator=lambda: [-1, 0, 1, 2, 3, 4, 5, 10],
            description="Fibonacci number",
        ))

    def _register_validation_templates(self):
        """Register input validation templates."""

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.VALIDATION,
            name="validate_email",
            source=textwrap.dedent('''
                def validate_email(email):
                    if not email:
                        return False
                    if "@" not in email:
                        return False
                    parts = email.split("@")
                    if len(parts) != 2:
                        return False
                    if not parts[0] or not parts[1]:
                        return False
                    if "." not in parts[1]:
                        return False
                    return True
            ''').strip(),
            func_name="validate_email",
            input_generator=lambda: [
                "", "test", "test@", "@test", "test@test",
                "test@test.com", "a@b.c", "@@", "test@test@test.com",
            ],
            description="Basic email validation",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.VALIDATION,
            name="validate_password",
            source=textwrap.dedent('''
                def validate_password(password):
                    if len(password) < 8:
                        return "too_short"
                    has_upper = False
                    has_lower = False
                    has_digit = False
                    for c in password:
                        if c.isupper():
                            has_upper = True
                        elif c.islower():
                            has_lower = True
                        elif c.isdigit():
                            has_digit = True
                    if not has_upper:
                        return "needs_upper"
                    if not has_lower:
                        return "needs_lower"
                    if not has_digit:
                        return "needs_digit"
                    return "valid"
            ''').strip(),
            func_name="validate_password",
            input_generator=lambda: [
                "", "short", "alllower", "ALLUPPER", "NoDigits",
                "12345678", "Password1", "password1", "PASSWORD1",
            ],
            description="Password validation",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.VALIDATION,
            name="validate_date",
            source=textwrap.dedent('''
                def validate_date(year, month, day):
                    if year < 1 or year > 9999:
                        return False
                    if month < 1 or month > 12:
                        return False
                    if day < 1:
                        return False
                    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                    if month == 2:
                        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                            if day > 29:
                                return False
                        elif day > 28:
                            return False
                    elif day > days_in_month[month - 1]:
                        return False
                    return True
            ''').strip(),
            func_name="validate_date",
            input_generator=lambda: [
                (2024, 1, 1), (2024, 2, 29), (2023, 2, 29), (2024, 2, 28),
                (2024, 13, 1), (2024, 0, 1), (2024, 1, 32), (2024, 4, 31),
                (0, 1, 1), (2024, 12, 31),
            ],
            description="Date validation with leap year",
        ))

    def _register_complex_templates(self):
        """Register more complex program templates."""

        # State machine
        self.templates.append(ProgramTemplate(
            category=ProgramCategory.STATE_MACHINE,
            name="traffic_light",
            source=textwrap.dedent('''
                def traffic_light(state, command):
                    if state == "red":
                        if command == "next":
                            return "green"
                        else:
                            return "red"
                    elif state == "green":
                        if command == "next":
                            return "yellow"
                        else:
                            return "green"
                    elif state == "yellow":
                        if command == "next":
                            return "red"
                        else:
                            return "yellow"
                    else:
                        return "red"
            ''').strip(),
            func_name="traffic_light",
            input_generator=lambda: [
                ("red", "next"), ("red", "stay"), ("green", "next"),
                ("green", "stay"), ("yellow", "next"), ("yellow", "stay"),
                ("invalid", "next"),
            ],
            description="Traffic light state machine",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.STATE_MACHINE,
            name="counter_machine",
            source=textwrap.dedent('''
                def counter_machine(value, commands):
                    for cmd in commands:
                        if cmd == "inc":
                            value += 1
                        elif cmd == "dec":
                            if value > 0:
                                value -= 1
                        elif cmd == "double":
                            value *= 2
                        elif cmd == "reset":
                            value = 0
                    return value
            ''').strip(),
            func_name="counter_machine",
            input_generator=lambda: [
                (0, []), (0, ["inc"]), (5, ["dec"]), (0, ["dec"]),
                (2, ["double"]), (5, ["reset"]),
                (0, ["inc", "inc", "double"]), (10, ["dec", "dec", "double", "reset"]),
            ],
            description="Counter with commands",
        ))

        # Recursion patterns
        self.templates.append(ProgramTemplate(
            category=ProgramCategory.RECURSION,
            name="sum_digits",
            source=textwrap.dedent('''
                def sum_digits(n):
                    if n < 0:
                        n = -n
                    total = 0
                    while n > 0:
                        total += n % 10
                        n = n // 10
                    return total
            ''').strip(),
            func_name="sum_digits",
            input_generator=lambda: [0, 5, 12, 123, 999, -42, 100],
            description="Sum of digits",
        ))

        self.templates.append(ProgramTemplate(
            category=ProgramCategory.RECURSION,
            name="power",
            source=textwrap.dedent('''
                def power(base, exp):
                    if exp < 0:
                        return None
                    result = 1
                    while exp > 0:
                        if exp % 2 == 1:
                            result *= base
                        base *= base
                        exp = exp // 2
                    return result
            ''').strip(),
            func_name="power",
            input_generator=lambda: [(2, 0), (2, 1), (2, 3), (2, 10), (3, 4), (2, -1), (0, 5)],
            description="Fast exponentiation",
        ))

    def generate_from_template(
        self,
        template: ProgramTemplate,
    ) -> List[DatasetExample]:
        """Generate examples from a single template."""
        examples = []

        # Get inputs
        inputs = template.input_generator()

        # Analyze program
        n_lines = self.analyzer.count_lines(template.source)
        n_branches = self.analyzer.count_branches(template.source)
        n_loops = self.analyzer.count_loops(template.source)
        max_nesting = self.analyzer.max_nesting_depth(template.source)

        program_id = hashlib.md5(template.source.encode()).hexdigest()[:12]

        # Collect coverage for each input
        triples = self.collector.collect_from_source(
            template.source,
            template.func_name,
            inputs,
        )

        for triple in triples:
            example_id = hashlib.md5(
                f"{program_id}:{triple.input_repr}".encode()
            ).hexdigest()[:16]

            # Compute uncovered lines
            all_lines = set(range(1, n_lines + 1))
            uncovered = sorted(all_lines - triple.covered_lines)

            example = DatasetExample(
                example_id=example_id,
                program_id=program_id,
                category=template.category.name,
                source=template.source,
                func_name=template.func_name,
                n_lines=n_lines,
                n_branches=n_branches,
                n_loops=n_loops,
                max_nesting=max_nesting,
                input_repr=triple.input_repr,
                input_type=type(triple.input_value).__name__,
                covered_lines=sorted(triple.covered_lines),
                uncovered_lines=uncovered,
                coverage_ratio=triple.coverage_ratio,
                execution_trace=triple.execution_trace,
                branches_taken={str(k): v for k, v in triple.branches_taken.items()},
                error=triple.error,
            )
            examples.append(example)

        return examples

    def generate_dataset(
        self,
        templates: Optional[List[ProgramTemplate]] = None,
    ) -> Dataset:
        """Generate complete dataset from templates."""
        if templates is None:
            templates = self.templates

        all_examples = []
        for template in templates:
            examples = self.generate_from_template(template)
            all_examples.extend(examples)

        metadata = {
            'n_templates': len(templates),
            'generator_seed': self.seed,
        }

        return Dataset(examples=all_examples, metadata=metadata)


def demo():
    """Demonstrate dataset generation."""
    print("=" * 60)
    print("COVERAGE PREDICTION DATASET")
    print("=" * 60)

    generator = DatasetGenerator(seed=42)

    print(f"\nRegistered {len(generator.templates)} program templates")

    # Count by category
    categories = {}
    for t in generator.templates:
        cat = t.category.name
        categories[cat] = categories.get(cat, 0) + 1

    print("\nTemplates by category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    # Generate dataset
    print("\nGenerating dataset...")
    dataset = generator.generate_dataset()

    stats = dataset.statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total examples: {stats['n_examples']}")
    print(f"  Unique programs: {stats['n_programs']}")
    print(f"  Lines per program: {stats['lines']['min']}-{stats['lines']['max']} (mean: {stats['lines']['mean']:.1f})")
    print(f"  Branches per program: {stats['branches']['min']}-{stats['branches']['max']} (mean: {stats['branches']['mean']:.1f})")
    print(f"  Coverage ratio: {stats['coverage']['min']:.2f}-{stats['coverage']['max']:.2f} (mean: {stats['coverage']['mean']:.2f})")

    print("\nExamples by category:")
    for cat, count in sorted(stats['categories'].items()):
        print(f"  {cat}: {count}")

    # Split
    train, val, test = dataset.split()
    print(f"\nSplits: train={len(train)}, val={len(val)}, test={len(test)}")

    # Show example
    print("\nExample:")
    ex = dataset.examples[0]
    print(f"  Program: {ex.func_name}")
    print(f"  Category: {ex.category}")
    print(f"  Input: {ex.input_repr}")
    print(f"  Covered: {ex.covered_lines}")
    print(f"  Uncovered: {ex.uncovered_lines}")
    print(f"  Coverage: {ex.coverage_ratio:.1%}")

    return dataset


if __name__ == "__main__":
    demo()
