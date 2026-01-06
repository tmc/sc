"""
exp_parallel_region_coupling: Test AND-State Region Interactions

GOAL: Understand how parallel (AND) regions interact:
- Event broadcasting to all regions
- Asymmetric region sizes (2 vs 10 states)
- Context-based coordination and synchronization

Based on proto/statecharts/v1/statecharts.proto:
  STATE_TYPE_PARALLEL = 3  // AND-decomposition: concurrent substates
  ψ(s) = PARALLEL ⟹ ∀c ∈ children(s): c ∈ σ₀ (all children active)

Key Capabilities:
1. Region Broadcast: Events delivered to all/some regions
2. Asymmetric Regions: Handle size imbalance (2 vs 20 states)
3. Region Sync: Coordinate via shared context
4. Coupling Analysis: Measure interaction patterns

Patterns Tested:
- Independence: Regions operate in isolation
- Coupling: Regions share events and transition together
- Synchronization: Regions coordinate via shared context

NO HARDCODING: Learn coupling patterns from examples.

Reference: proto/statecharts/v1/statecharts.proto
"""

# Region broadcasting
from .region_broadcast import (
    # Types
    BroadcastScope,
    ConflictResolution,
    EventDelivery,
    # Data classes
    Region,
    RegionTransition,
    BroadcastEvent,
    BroadcastResult,
    ParallelState,
    # Engine
    BroadcastEngine,
    # Pattern learning
    BroadcastPattern,
    BroadcastGenome,
    BroadcastPatternLearner,
)

# Asymmetric regions
from .asymmetric_regions import (
    # Types
    RegionSize,
    # Data classes
    AsymmetricRegion,
    AsymmetricConfiguration,
    CouplingMetrics,
    AsymmetricParallelState,
    # Generators
    create_tiny_region,
    create_small_region,
    create_medium_region,
    create_large_region,
    create_huge_region,
    # Analysis
    AsymmetricCouplingAnalyzer,
    # Evolution
    InteractionGenome,
    AsymmetricInteractionEvolver,
)

# Region synchronization
from .region_sync import (
    # Types
    SyncPattern,
    ContextAccess,
    # Data classes
    SharedVariable,
    SyncRegion,
    SyncTransition,
    SharedContext,
    SynchronizedParallelState,
    # Pattern implementations
    create_producer_consumer_system,
    create_barrier_sync_system,
    create_leader_follower_system,
    # Pattern learning
    SyncGenome,
    SyncPatternLearner,
)

# Benchmark
from .coupling_benchmark import (
    CouplingBenchmarkResult,
    run_independence_scenario,
    run_coupling_scenario,
    run_synchronization_scenario,
    run_asymmetric_coupling_scenario,
    run_learned_coupling_scenario,
    run_full_benchmark,
    quick_benchmark,
)

__all__ = [
    # Broadcast types
    'BroadcastScope',
    'ConflictResolution',
    'EventDelivery',
    # Broadcast classes
    'Region',
    'RegionTransition',
    'BroadcastEvent',
    'BroadcastResult',
    'ParallelState',
    'BroadcastEngine',
    'BroadcastPattern',
    'BroadcastGenome',
    'BroadcastPatternLearner',
    # Asymmetric types
    'RegionSize',
    # Asymmetric classes
    'AsymmetricRegion',
    'AsymmetricConfiguration',
    'CouplingMetrics',
    'AsymmetricParallelState',
    # Region generators
    'create_tiny_region',
    'create_small_region',
    'create_medium_region',
    'create_large_region',
    'create_huge_region',
    # Asymmetric analysis
    'AsymmetricCouplingAnalyzer',
    'InteractionGenome',
    'AsymmetricInteractionEvolver',
    # Sync types
    'SyncPattern',
    'ContextAccess',
    # Sync classes
    'SharedVariable',
    'SyncRegion',
    'SyncTransition',
    'SharedContext',
    'SynchronizedParallelState',
    # Sync pattern creators
    'create_producer_consumer_system',
    'create_barrier_sync_system',
    'create_leader_follower_system',
    # Sync learning
    'SyncGenome',
    'SyncPatternLearner',
    # Benchmark
    'CouplingBenchmarkResult',
    'run_independence_scenario',
    'run_coupling_scenario',
    'run_synchronization_scenario',
    'run_asymmetric_coupling_scenario',
    'run_learned_coupling_scenario',
    'run_full_benchmark',
    'quick_benchmark',
]
