"""
exp_hierarchy_scratchpad: Improve hierarchy understanding via scratchpad.

Problem: Models struggle with parent-child relationships and cascade semantics.
Baseline accuracy: 27%

Approach: Explicit hierarchy enumeration in prompts:
- Parent chain tracing for containment
- Child enumeration for cascade exit
- Initial state traversal for default entry
- Parent counting for depth

Tasks tested:
1. Containment: "Is X inside Y?"
2. Cascade exit: "If X exits, what else exits?"
3. Default entry: "Entering X, what's the configuration?"
4. Depth: "How deep is state X?"

Input: (hierarchy_tree, task_type, query_state, [target_state])
Output: Varies by task (bool, set, int)

Model: mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit
"""

from .hierarchy_scratchpad import (
    HierarchyTask,
    HierarchyState,
    HierarchyCase,
    build_state_index,
    compute_parent_chain,
    compute_all_descendants,
    format_hierarchy_tree,
    format_hierarchy_nested,
    create_prompt,
    create_containment_prompt,
    create_cascade_exit_prompt,
    create_default_entry_prompt,
    create_depth_prompt,
    parse_response,
    parse_containment_response,
    parse_cascade_response,
    parse_entry_response,
    parse_depth_response,
    get_test_cases,
)

from .benchmark import (
    run_benchmark,
    BenchmarkConfig,
    PredictionResult,
)

__all__ = [
    # Data types
    'HierarchyTask',
    'HierarchyState',
    'HierarchyCase',
    # Utilities
    'build_state_index',
    'compute_parent_chain',
    'compute_all_descendants',
    'format_hierarchy_tree',
    'format_hierarchy_nested',
    # Prompts
    'create_prompt',
    'create_containment_prompt',
    'create_cascade_exit_prompt',
    'create_default_entry_prompt',
    'create_depth_prompt',
    # Parsing
    'parse_response',
    'parse_containment_response',
    'parse_cascade_response',
    'parse_entry_response',
    'parse_depth_response',
    # Test cases
    'get_test_cases',
    # Benchmark
    'run_benchmark',
    'BenchmarkConfig',
    'PredictionResult',
]
