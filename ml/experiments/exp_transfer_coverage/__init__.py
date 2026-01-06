"""
exp_transfer_coverage: Transfer Evolved Coverage Statecharts Across Programs

Tests whether coverage statechart STRUCTURE generalizes across programs.

Hypothesis: If-else/loop patterns learned on simple programs transfer
to complex programs, requiring only guard fine-tuning.

Approach:
1. Train coverage statechart on SIMPLE programs (single if-else, loop)
2. Extract abstract topology (states, transitions, guard types)
3. Adapt to COMPLEX programs (nested loops, multi-branch)
4. Fine-tune guards only - freeze structure
5. Compare: transfer vs training from scratch

Key insight: Control flow PATTERNS are universal even when
specific line numbers and guard values differ.

Usage:
    from experiments.exp_transfer_coverage import (
        TopologyTransferLearner,
        TransferBenchmark,
        generate_family_dataset,
    )

    # Quick transfer test
    source = generate_family_dataset('simple_branch')
    target = generate_family_dataset('medium_nested')

    learner = TopologyTransferLearner(source, target)
    results = learner.full_transfer()

    # Full benchmark
    benchmark = TransferBenchmark()
    benchmark.run_all_pairs()
"""

from .topology_transfer import (
    TopologyTransferLearner,
    AbstractTopology,
    AbstractState,
    AbstractTransition,
    PatternType,
    extract_topology,
    adapt_topology,
)

from .program_families import (
    ProgramFamily,
    Complexity,
    PROGRAM_FAMILIES,
    generate_family_dataset,
    generate_complexity_datasets,
    get_transfer_pairs,
)

from .transfer_benchmark import (
    TransferBenchmark,
    TransferResult,
)

__all__ = [
    # Topology transfer
    'TopologyTransferLearner',
    'AbstractTopology',
    'AbstractState',
    'AbstractTransition',
    'PatternType',
    'extract_topology',
    'adapt_topology',
    # Program families
    'ProgramFamily',
    'Complexity',
    'PROGRAM_FAMILIES',
    'generate_family_dataset',
    'generate_complexity_datasets',
    'get_transfer_pairs',
    # Benchmark
    'TransferBenchmark',
    'TransferResult',
]
