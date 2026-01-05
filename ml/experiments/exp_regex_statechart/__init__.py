"""
exp_regex_statechart: Learn Regular Expressions as Statecharts

Goal: Evolve statecharts that match RE2 regular expression behavior.

Why RE2?
- RE2 is guaranteed linear time (no catastrophic backtracking)
- RE2 regex = DFA/NFA = statechart (direct correspondence)
- Well-defined semantics, easy ground truth validation
- Used in production (Google, Rust regex crate)

Approach:
1. Given: Input strings + match/no-match labels
2. Learn: Statechart that accepts matching strings, rejects others
3. Validate: Compare against RE2 engine behavior

Key Insight:
    regex "a(b|c)*d" compiles to:

    States: [START] → [SAW_A] → [IN_BC] → [ACCEPT]
    Transitions:
        START -'a'→ SAW_A
        SAW_A -'b'→ IN_BC
        SAW_A -'c'→ IN_BC
        SAW_A -'d'→ ACCEPT
        IN_BC -'b'→ IN_BC
        IN_BC -'c'→ IN_BC
        IN_BC -'d'→ ACCEPT

Evolution discovers this structure from examples.

Applications:
- Regex synthesis from examples
- Explainable pattern matching
- Regex optimization (minimized DFA)
- Cross-language regex portability
"""

# Core statechart
from .regex_statechart import (
    RegexStatechart,
    StateType,
    Transition,
    CharGuard,
)

# Oracle
from .re2_oracle import RE2Oracle, RegexExamples

# Basic evolution
from .evolver import RegexEvolver, EvolutionConfig, EvolutionStats
from .synthesis import RegexSynthesizer, SynthesisResult, synthesize_regex
from .benchmark import RegexBenchmark, BenchmarkPattern, BenchmarkResult, BENCHMARK_PATTERNS

# Semantic components (extended statecharts)
from .semantic_components import (
    StateVar, VarType, ExtendedState,
    GuardExpr, CharGuardExpr, CounterGuardExpr, FlagGuardExpr, CompositeGuardExpr,
    ActionExpr, NoOpAction, IncrementAction, ResetAction, SetFlagAction,
    CompositeAction, SemanticTransition, ComponentFactory,
)
from .extended_statechart import ExtendedStatechart, AcceptCondition, StateActions

# Semantic evolution (full ML discovery)
from .semantic_evolver import SemanticEvolver, SemanticEvolutionConfig, SemanticEvolutionStats
from .component_synthesis import (
    ComponentSynthesizer, StateDiscoverer, GuardSynthesizer,
    TransitionDiscoverer, ActionDiscoverer, ExecutionTrace, StateCluster,
)
from .advanced_benchmark import (
    AdvancedBenchmark, AdvancedPattern, AdvancedBenchmarkResult, ADVANCED_PATTERNS,
)

__all__ = [
    # Core statechart
    "RegexStatechart",
    "StateType",
    "Transition",
    "CharGuard",
    # Oracle
    "RE2Oracle",
    "RegexExamples",
    # Basic evolution
    "RegexEvolver",
    "EvolutionConfig",
    "EvolutionStats",
    # Basic synthesis
    "RegexSynthesizer",
    "SynthesisResult",
    "synthesize_regex",
    # Basic benchmark
    "RegexBenchmark",
    "BenchmarkPattern",
    "BenchmarkResult",
    "BENCHMARK_PATTERNS",
    # Semantic components
    "StateVar",
    "VarType",
    "ExtendedState",
    "GuardExpr",
    "CharGuardExpr",
    "CounterGuardExpr",
    "FlagGuardExpr",
    "CompositeGuardExpr",
    "ActionExpr",
    "NoOpAction",
    "IncrementAction",
    "ResetAction",
    "SetFlagAction",
    "CompositeAction",
    "SemanticTransition",
    "ComponentFactory",
    # Extended statechart
    "ExtendedStatechart",
    "AcceptCondition",
    "StateActions",
    # Semantic evolution
    "SemanticEvolver",
    "SemanticEvolutionConfig",
    "SemanticEvolutionStats",
    # Component synthesis (full ML discovery)
    "ComponentSynthesizer",
    "StateDiscoverer",
    "GuardSynthesizer",
    "TransitionDiscoverer",
    "ActionDiscoverer",
    "ExecutionTrace",
    "StateCluster",
    # Advanced benchmark
    "AdvancedBenchmark",
    "AdvancedPattern",
    "AdvancedBenchmarkResult",
    "ADVANCED_PATTERNS",
]
