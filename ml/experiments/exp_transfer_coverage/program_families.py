"""
Program Families for Transfer Learning

Organizes programs into families by structural complexity:
- SIMPLE: Single if-else, single loop
- MEDIUM: Nested if-else, loop with condition
- COMPLEX: Nested loops, multiple branches, error handling

Transfer hypothesis: Structure learned on SIMPLE transfers to MEDIUM/COMPLEX.
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
from enum import Enum, auto

from ..exp_coverage_prediction.dataset import Dataset, DatasetExample, DatasetGenerator
from ..exp_coverage_prediction.coverage_collector import CoverageCollector


class Complexity(Enum):
    SIMPLE = auto()
    MEDIUM = auto()
    COMPLEX = auto()


@dataclass
class ProgramFamily:
    name: str
    complexity: Complexity
    templates: List[str]
    description: str


# Simple programs - single control structure
SIMPLE_BRANCH = """
def check(x):
    result = 0
    if x > 0:
        result = 1
    else:
        result = -1
    return result
"""

SIMPLE_LOOP = """
def sum_to(n):
    total = 0
    for i in range(n):
        total += i
    return total
"""

SIMPLE_GUARD = """
def validate(x):
    if x is None:
        return False
    if x < 0:
        return False
    return True
"""

# Medium programs - nested or combined structures
MEDIUM_NESTED_BRANCH = """
def classify(x, y):
    result = "zero"
    if x > 0:
        if y > 0:
            result = "pp"
        else:
            result = "pn"
    else:
        if y > 0:
            result = "np"
        else:
            result = "nn"
    return result
"""

MEDIUM_LOOP_BRANCH = """
def find_positive(items):
    for item in items:
        if item > 0:
            return item
    return None
"""

MEDIUM_ACCUMULATOR = """
def count_positive(items):
    count = 0
    for item in items:
        if item > 0:
            count += 1
    return count
"""

# Complex programs - multiple nested structures
COMPLEX_NESTED_LOOPS = """
def matrix_sum(matrix):
    total = 0
    for row in matrix:
        for val in row:
            if val > 0:
                total += val
    return total
"""

COMPLEX_MULTI_BRANCH = """
def grade(score):
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
"""

COMPLEX_SEARCH = """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""

PROGRAM_FAMILIES = {
    'simple_branch': ProgramFamily(
        name='simple_branch',
        complexity=Complexity.SIMPLE,
        templates=[SIMPLE_BRANCH, SIMPLE_GUARD],
        description='Single if-else structure',
    ),
    'simple_loop': ProgramFamily(
        name='simple_loop',
        complexity=Complexity.SIMPLE,
        templates=[SIMPLE_LOOP],
        description='Single loop structure',
    ),
    'medium_nested': ProgramFamily(
        name='medium_nested',
        complexity=Complexity.MEDIUM,
        templates=[MEDIUM_NESTED_BRANCH],
        description='Nested if-else structure',
    ),
    'medium_loop_branch': ProgramFamily(
        name='medium_loop_branch',
        complexity=Complexity.MEDIUM,
        templates=[MEDIUM_LOOP_BRANCH, MEDIUM_ACCUMULATOR],
        description='Loop with branch inside',
    ),
    'complex_nested_loops': ProgramFamily(
        name='complex_nested_loops',
        complexity=Complexity.COMPLEX,
        templates=[COMPLEX_NESTED_LOOPS],
        description='Nested loops with conditions',
    ),
    'complex_multi_branch': ProgramFamily(
        name='complex_multi_branch',
        complexity=Complexity.COMPLEX,
        templates=[COMPLEX_MULTI_BRANCH, COMPLEX_SEARCH],
        description='Multiple branches or while loops',
    ),
}


def generate_family_inputs(family_name: str) -> List:
    """Generate appropriate inputs for a program family."""
    if family_name == 'simple_branch':
        return [-5, -1, 0, 1, 5, 10, None]
    elif family_name == 'simple_loop':
        return [0, 1, 3, 5, 10]
    elif family_name == 'medium_nested':
        return [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0)]
    elif family_name == 'medium_loop_branch':
        return [[], [1], [-1], [1, 2, 3], [-1, -2, 3], [-1, 2, -3]]
    elif family_name == 'complex_nested_loops':
        return [[[]], [[1]], [[1, 2], [3, 4]], [[-1, 2], [3, -4]]]
    elif family_name == 'complex_multi_branch':
        return [95, 85, 75, 65, 55, 45]
    return [0, 1, -1]


def get_function_name(source: str) -> str:
    """Extract function name from source."""
    for line in source.split('\n'):
        if line.strip().startswith('def '):
            name = line.split('(')[0].replace('def ', '').strip()
            return name
    return 'func'


def generate_family_dataset(family_name: str, n_per_template: int = 20) -> Dataset:
    """Generate dataset for a specific program family."""
    family = PROGRAM_FAMILIES.get(family_name)
    if not family:
        raise ValueError(f"Unknown family: {family_name}")

    examples = []
    collector = CoverageCollector()
    inputs = generate_family_inputs(family_name)
    example_counter = 0

    for template in family.templates:
        func_name = get_function_name(template)
        all_lines = set(range(1, template.count('\n') + 2))

        for inp in inputs[:n_per_template]:
            try:
                triples = collector.collect_from_source(template, func_name, [inp])
                for triple in triples:
                    covered = set(triple.covered_lines)
                    uncovered = all_lines - covered
                    coverage_ratio = len(covered) / len(all_lines) if all_lines else 0

                    examples.append(DatasetExample(
                        example_id=f"{family_name}_{example_counter}",
                        program_id=f"{family_name}_{func_name}",
                        category=family_name,
                        source=template,
                        func_name=func_name,
                        n_lines=triple.total_lines,
                        n_branches=template.count('if '),
                        n_loops=template.count('for ') + template.count('while '),
                        max_nesting=1,
                        input_repr=repr(inp),
                        input_type=type(inp).__name__,
                        covered_lines=list(triple.covered_lines),
                        uncovered_lines=list(uncovered),
                        coverage_ratio=coverage_ratio,
                        execution_trace=triple.execution_trace,
                        branches_taken=triple.branches_taken,
                    ))
                    example_counter += 1
            except Exception:
                pass

    return Dataset(examples=examples)


def generate_complexity_datasets() -> Dict[Complexity, Dataset]:
    """Generate datasets grouped by complexity level."""
    datasets = {c: [] for c in Complexity}

    for family_name, family in PROGRAM_FAMILIES.items():
        family_dataset = generate_family_dataset(family_name, n_per_template=15)
        datasets[family.complexity].extend(family_dataset.examples)

    return {c: Dataset(examples=exs) for c, exs in datasets.items()}


def get_transfer_pairs() -> List[Tuple[str, str]]:
    """Get recommended source->target pairs for transfer experiments."""
    return [
        ('simple_branch', 'medium_nested'),           # Branch -> Nested branch
        ('simple_loop', 'medium_loop_branch'),        # Loop -> Loop with branch
        ('medium_nested', 'complex_multi_branch'),    # Nested -> Multi-branch
        ('medium_loop_branch', 'complex_nested_loops'), # Loop+branch -> Nested loops
        ('simple_branch', 'complex_multi_branch'),    # Simple -> Complex (big jump)
    ]


def demo():
    """Demonstrate program families."""
    print("=" * 60)
    print("PROGRAM FAMILIES FOR TRANSFER")
    print("=" * 60)

    for name, family in PROGRAM_FAMILIES.items():
        dataset = generate_family_dataset(name, n_per_template=10)
        print(f"\n{name} ({family.complexity.name}):")
        print(f"  {family.description}")
        print(f"  Templates: {len(family.templates)}")
        print(f"  Examples: {len(dataset)}")

    print("\n" + "-" * 40)
    print("Transfer pairs (source -> target):")
    for src, tgt in get_transfer_pairs():
        src_c = PROGRAM_FAMILIES[src].complexity.name
        tgt_c = PROGRAM_FAMILIES[tgt].complexity.name
        print(f"  {src} ({src_c}) -> {tgt} ({tgt_c})")


if __name__ == "__main__":
    demo()
