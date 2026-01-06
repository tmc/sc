"""
exp_schema_evolution: Learn Safe Statechart Migrations

GOAL: Synthesize migration functions that safely transform
configurations from old_chart to new_chart versions.

Based on proto/statecharts/v1/evolution.proto:
- ChartDiff: Detect schema changes
- MigrationPlan: Define migration rules
- StateMapping: Map old states to new states
- Verification: Ensure safety properties

Key Capabilities:
1. Schema Diffing: Detect added/removed/renamed/moved states
2. Migration Synthesis: Generate mapping rules from examples
3. Safety Verification: Check completeness, determinism, reachability
4. Dry Run: Preview migration results before applying

Migration Types (from StateMappingType):
- IDENTITY: Same label in both versions
- RENAME: Label changed, structure preserved
- TO_PARENT: State removed, map to parent's initial
- TO_SIBLING: State removed, map to sibling
- TO_INITIAL: State removed, map to chart initial
- SPLIT: One state splits into multiple (conditional)
- MERGE: Multiple states merge into one

NO HARDCODING: Migration patterns learned from example pairs.

Reference: proto/statecharts/v1/evolution.proto
"""

# Schema diffing
from .schema_diff import (
    # Types
    ChangeType,
    BreakingChangeType,
    CompatibilityLevel,
    MigrationRequirement,
    # Data classes
    State,
    Transition,
    Event,
    Statechart,
    StateDiff,
    TransitionDiff,
    EventDiff,
    BreakingChange,
    ChartDiff,
    # Differ
    SchemaDiffer,
)

# Migration synthesis
from .migration_synthesizer import (
    # Types
    StateMappingType,
    MigrationStrategy,
    # Data classes
    Configuration,
    StateMapping,
    ContextTransformation,
    MigrationPlan,
    MigrationValidation,
    # Synthesizer
    MigrationSynthesizer,
    # Convenience
    synthesize_migration,
    evolve_migration,
)

# Verification
from .migration_verifier import (
    # Types
    VerificationStatus,
    ViolationType,
    # Data classes
    Violation,
    VerificationResult,
    VerificationRule,
    DryRunPreview,
    DryRunResult,
    # Verifier
    MigrationVerifier,
    # Functions
    dry_run,
)

# Benchmark
from .benchmark import (
    BenchmarkResult,
    run_scenario,
    run_full_benchmark,
    quick_benchmark,
    # Scenario generators
    create_state_removal_scenario,
    create_state_rename_scenario,
    create_hierarchy_change_scenario,
    create_feature_addition_scenario,
    create_combined_changes_scenario,
)

__all__ = [
    # Schema diff types
    'ChangeType',
    'BreakingChangeType',
    'CompatibilityLevel',
    'MigrationRequirement',
    # Schema diff classes
    'State',
    'Transition',
    'Event',
    'Statechart',
    'StateDiff',
    'TransitionDiff',
    'EventDiff',
    'BreakingChange',
    'ChartDiff',
    'SchemaDiffer',
    # Migration types
    'StateMappingType',
    'MigrationStrategy',
    # Migration classes
    'Configuration',
    'StateMapping',
    'ContextTransformation',
    'MigrationPlan',
    'MigrationValidation',
    'MigrationSynthesizer',
    'synthesize_migration',
    'evolve_migration',
    # Verification types
    'VerificationStatus',
    'ViolationType',
    # Verification classes
    'Violation',
    'VerificationResult',
    'VerificationRule',
    'DryRunPreview',
    'DryRunResult',
    'MigrationVerifier',
    'dry_run',
    # Benchmark
    'BenchmarkResult',
    'run_scenario',
    'run_full_benchmark',
    'quick_benchmark',
    'create_state_removal_scenario',
    'create_state_rename_scenario',
    'create_hierarchy_change_scenario',
    'create_feature_addition_scenario',
    'create_combined_changes_scenario',
]
