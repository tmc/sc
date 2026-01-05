"""
exp_temporal_guards: Learn Time-Based Guards for Statecharts

GOAL: Evolve temporal predicates for transition guards:
- TIMEOUT(state, duration): Fire after N seconds in state
- SINCE(state) > N: Time since leaving a state
- WITHIN(duration): Still within deadline
- RATE_LIMIT(count, window): Limit events per time window
- COOLDOWN(duration): Minimum time between transitions

Use Cases:
- Session timeouts (idle detection)
- Game timers (move time limits)
- Rate limiting (API throttling)
- Ability cooldowns (game mechanics)
- Debounce patterns (UI stability)
- Business hours (schedule-based access)

NO HARDCODING: All temporal thresholds are discovered through evolution.

Reference: exp_guard_synthesis for base DSL framework
"""

# Temporal DSL
from .temporal_guard import (
    # Base types
    TemporalExpr,
    ExprType,
    Const,
    Var,
    BinOp,
    UnaryOp,
    # Temporal predicates
    After,
    Within,
    Elapsed,
    Since,
    Timeout,
    RateLimit,
    Cooldown,
    Debounce,
    TimeOfDay,
    DayOfWeek,
    BusinessHours,
    # Compound guard
    TemporalGuard,
    # Constants
    DURATION_CANDIDATES,
    RATE_LIMIT_COUNTS,
    RATE_LIMIT_WINDOWS,
    # Helpers
    create_timeout_guard,
    create_rate_limit_guard,
    create_cooldown_guard,
    create_business_hours_guard,
)

# Evolution
from .temporal_evolver import (
    TemporalGenome,
    TemporalEvolver,
    evolve_timeout,
    evolve_rate_limit,
)

# Benchmark
from .temporal_benchmark import (
    BenchmarkResult,
    generate_session_timeout_data,
    generate_game_timer_data,
    generate_rate_limit_data,
    generate_cooldown_data,
    generate_debounce_data,
    generate_business_hours_data,
    generate_state_since_data,
    run_scenario,
    run_full_benchmark,
    quick_benchmark,
)

__all__ = [
    # DSL Types
    'TemporalExpr',
    'ExprType',
    'Const',
    'Var',
    'BinOp',
    'UnaryOp',
    # Temporal Predicates
    'After',
    'Within',
    'Elapsed',
    'Since',
    'Timeout',
    'RateLimit',
    'Cooldown',
    'Debounce',
    'TimeOfDay',
    'DayOfWeek',
    'BusinessHours',
    # Compound
    'TemporalGuard',
    # Constants
    'DURATION_CANDIDATES',
    'RATE_LIMIT_COUNTS',
    'RATE_LIMIT_WINDOWS',
    # Helpers
    'create_timeout_guard',
    'create_rate_limit_guard',
    'create_cooldown_guard',
    'create_business_hours_guard',
    # Evolution
    'TemporalGenome',
    'TemporalEvolver',
    'evolve_timeout',
    'evolve_rate_limit',
    # Benchmark
    'BenchmarkResult',
    'generate_session_timeout_data',
    'generate_game_timer_data',
    'generate_rate_limit_data',
    'generate_cooldown_data',
    'generate_debounce_data',
    'generate_business_hours_data',
    'generate_state_since_data',
    'run_scenario',
    'run_full_benchmark',
    'quick_benchmark',
]
