"""
Temporal Guard DSL

Extends the guard expression language with TIME-BASED predicates:
- TIMEOUT(state, duration): True after being in state for duration
- SINCE(state) > N: Time since leaving a state
- WITHIN(duration): True if still within deadline
- ELAPSED: Current time in state
- RATE_LIMIT(count, window): Rate limiting predicate
- COOLDOWN(duration): Minimum time between transitions

NO HARDCODING: Durations are evolvable parameters.

Reference: semantics/v1/machine.go for state timing
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum, auto
from abc import ABC, abstractmethod


# =============================================================================
# EXPRESSION TYPES
# =============================================================================

class ExprType(Enum):
    """Types of expressions in the temporal DSL."""
    BOOL = auto()
    INT = auto()
    FLOAT = auto()
    DURATION = auto()  # Time duration in seconds


# =============================================================================
# BASE EXPRESSION
# =============================================================================

@dataclass
class TemporalExpr(ABC):
    """Base class for temporal expressions."""
    expr_type: ExprType = field(default=ExprType.BOOL)

    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> Any:
        """Evaluate expression in given context."""
        pass

    @abstractmethod
    def to_string(self) -> str:
        """Convert to human-readable string."""
        pass

    def copy(self) -> 'TemporalExpr':
        """Create a deep copy."""
        import copy
        return copy.deepcopy(self)


# =============================================================================
# PRIMITIVE EXPRESSIONS
# =============================================================================

@dataclass
class Const(TemporalExpr):
    """Constant value."""
    value: Any = None

    def evaluate(self, context: Dict[str, Any]) -> Any:
        return self.value

    def to_string(self) -> str:
        if isinstance(self.value, float):
            return f"{self.value:.1f}"
        return repr(self.value)


@dataclass
class Var(TemporalExpr):
    """Variable reference."""
    name: str = ""

    def evaluate(self, context: Dict[str, Any]) -> Any:
        return context.get(self.name)

    def to_string(self) -> str:
        return self.name


# =============================================================================
# BINARY OPERATIONS
# =============================================================================

@dataclass
class BinOp(TemporalExpr):
    """Binary operation."""
    op: str = "and"  # 'and', 'or', '==', '!=', '<', '>', '<=', '>=', '+', '-'
    left: TemporalExpr = None
    right: TemporalExpr = None

    def evaluate(self, context: Dict[str, Any]) -> Any:
        l = self.left.evaluate(context) if self.left else None
        r = self.right.evaluate(context) if self.right else None

        if l is None or r is None:
            return False

        ops = {
            'and': lambda a, b: bool(a) and bool(b),
            'or': lambda a, b: bool(a) or bool(b),
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            '<': lambda a, b: a < b,
            '>': lambda a, b: a > b,
            '<=': lambda a, b: a <= b,
            '>=': lambda a, b: a >= b,
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
        }
        try:
            return ops[self.op](l, r)
        except (TypeError, KeyError):
            return False

    def to_string(self) -> str:
        left_str = self.left.to_string() if self.left else "?"
        right_str = self.right.to_string() if self.right else "?"
        return f"({left_str} {self.op} {right_str})"


@dataclass
class UnaryOp(TemporalExpr):
    """Unary operation."""
    op: str = "not"  # 'not', '-'
    operand: TemporalExpr = None

    def evaluate(self, context: Dict[str, Any]) -> Any:
        v = self.operand.evaluate(context) if self.operand else None
        if self.op == 'not':
            return not v
        elif self.op == '-':
            return -v if v is not None else None
        return v

    def to_string(self) -> str:
        operand_str = self.operand.to_string() if self.operand else "?"
        return f"({self.op} {operand_str})"


# =============================================================================
# TEMPORAL PREDICATES - Core DSL Extensions
# =============================================================================

@dataclass
class After(TemporalExpr):
    """
    after(duration) - True if time since entering current state >= duration.

    TIMEOUT semantic: Fires when state has been active for too long.
    Duration is EVOLVABLE - not hardcoded.

    Example: after(30.0) - True after 30 seconds in current state
    """
    duration: float = 0.0  # seconds (EVOLVABLE)

    def __post_init__(self):
        self.expr_type = ExprType.BOOL

    def evaluate(self, context: Dict[str, Any]) -> bool:
        current_time = context.get('__timestamp__', time.time())
        entry_time = context.get('__state_entry_time__', current_time)
        elapsed = current_time - entry_time
        return elapsed >= self.duration

    def to_string(self) -> str:
        return f"after({self.duration:.1f}s)"


@dataclass
class Within(TemporalExpr):
    """
    within(duration) - True if still within deadline since entering state.

    DEADLINE semantic: Must act before time runs out.
    Opposite of after().

    Example: within(5.0) - True if less than 5 seconds in state
    """
    duration: float = 0.0  # seconds (EVOLVABLE)

    def __post_init__(self):
        self.expr_type = ExprType.BOOL

    def evaluate(self, context: Dict[str, Any]) -> bool:
        current_time = context.get('__timestamp__', time.time())
        entry_time = context.get('__state_entry_time__', current_time)
        elapsed = current_time - entry_time
        return elapsed < self.duration

    def to_string(self) -> str:
        return f"within({self.duration:.1f}s)"


@dataclass
class Elapsed(TemporalExpr):
    """
    elapsed - Time (seconds) since entering current state.

    Returns float for use in comparisons: elapsed > 30, elapsed < 5

    Example: elapsed > 10.0 - More than 10 seconds in state
    """
    def __post_init__(self):
        self.expr_type = ExprType.FLOAT

    def evaluate(self, context: Dict[str, Any]) -> float:
        current_time = context.get('__timestamp__', time.time())
        entry_time = context.get('__state_entry_time__', current_time)
        return current_time - entry_time

    def to_string(self) -> str:
        return "elapsed"


@dataclass
class Since(TemporalExpr):
    """
    since(state_name) - Time since leaving a specific state.

    Tracks how long ago we exited a particular state.
    Returns infinity if never been in that state.

    Example: since('LOGIN') > 3600 - Over an hour since login
    """
    state_name: str = ""

    def __post_init__(self):
        self.expr_type = ExprType.FLOAT

    def evaluate(self, context: Dict[str, Any]) -> float:
        current_time = context.get('__timestamp__', time.time())
        state_exit_times = context.get('__state_exit_times__', {})
        exit_time = state_exit_times.get(self.state_name, 0)
        if exit_time == 0:
            return float('inf')  # Never been in that state
        return current_time - exit_time

    def to_string(self) -> str:
        return f"since('{self.state_name}')"


@dataclass
class Timeout(TemporalExpr):
    """
    timeout(state_name, duration) - True if been in named state for >= duration.

    More explicit than after() - names the specific state.
    Useful for checking timeout on non-current states in parallel regions.

    Example: timeout('IDLE', 300) - Idle for 5 minutes
    """
    state_name: str = ""
    duration: float = 0.0  # seconds (EVOLVABLE)

    def __post_init__(self):
        self.expr_type = ExprType.BOOL

    def evaluate(self, context: Dict[str, Any]) -> bool:
        current_time = context.get('__timestamp__', time.time())
        state_entry_times = context.get('__state_entry_times__', {})
        active_states = context.get('__active_states__', set())

        # Check if state is currently active
        if self.state_name not in active_states:
            return False

        entry_time = state_entry_times.get(self.state_name, current_time)
        elapsed = current_time - entry_time
        return elapsed >= self.duration

    def to_string(self) -> str:
        return f"timeout('{self.state_name}', {self.duration:.1f}s)"


# =============================================================================
# RATE LIMITING PREDICATES
# =============================================================================

@dataclass
class RateLimit(TemporalExpr):
    """
    rate_limit(count, window) - True if event count within window is under limit.

    Implements rate limiting: block if too many events in time window.
    Both count and window are EVOLVABLE.

    Example: rate_limit(5, 60.0) - Allow if < 5 events in last 60 seconds
    """
    count: int = 1       # Max events allowed (EVOLVABLE)
    window: float = 1.0  # Time window in seconds (EVOLVABLE)

    def __post_init__(self):
        self.expr_type = ExprType.BOOL

    def evaluate(self, context: Dict[str, Any]) -> bool:
        current_time = context.get('__timestamp__', time.time())
        event_times = context.get('__event_times__', [])

        # Count events within window
        cutoff = current_time - self.window
        recent_count = sum(1 for t in event_times if t > cutoff)

        return recent_count < self.count

    def to_string(self) -> str:
        return f"rate_limit({self.count}, {self.window:.1f}s)"


@dataclass
class Cooldown(TemporalExpr):
    """
    cooldown(duration) - True if enough time has passed since last transition.

    Implements minimum time between actions.
    Duration is EVOLVABLE.

    Example: cooldown(2.0) - Allow if 2+ seconds since last transition
    """
    duration: float = 0.0  # seconds (EVOLVABLE)

    def __post_init__(self):
        self.expr_type = ExprType.BOOL

    def evaluate(self, context: Dict[str, Any]) -> bool:
        current_time = context.get('__timestamp__', time.time())
        last_transition = context.get('__last_transition_time__', 0)

        if last_transition == 0:
            return True  # No previous transition

        return (current_time - last_transition) >= self.duration

    def to_string(self) -> str:
        return f"cooldown({self.duration:.1f}s)"


@dataclass
class Debounce(TemporalExpr):
    """
    debounce(duration) - True if state has been stable for duration.

    Prevents rapid state changes. Must be in current state for
    at least duration before allowing transition.

    Example: debounce(0.5) - Stable for 500ms before allowing change
    """
    duration: float = 0.0  # seconds (EVOLVABLE)

    def __post_init__(self):
        self.expr_type = ExprType.BOOL

    def evaluate(self, context: Dict[str, Any]) -> bool:
        current_time = context.get('__timestamp__', time.time())
        entry_time = context.get('__state_entry_time__', current_time)
        elapsed = current_time - entry_time
        return elapsed >= self.duration

    def to_string(self) -> str:
        return f"debounce({self.duration:.1f}s)"


# =============================================================================
# TIME-OF-DAY PREDICATES
# =============================================================================

@dataclass
class TimeOfDay(TemporalExpr):
    """
    time_of_day() - Returns hour of day (0-23).

    For schedule-based transitions: business hours, night mode, etc.

    Example: time_of_day() >= 9 and time_of_day() < 17 - Business hours
    """
    def __post_init__(self):
        self.expr_type = ExprType.INT

    def evaluate(self, context: Dict[str, Any]) -> int:
        import datetime
        timestamp = context.get('__timestamp__', time.time())
        dt = datetime.datetime.fromtimestamp(timestamp)
        return dt.hour

    def to_string(self) -> str:
        return "time_of_day()"


@dataclass
class DayOfWeek(TemporalExpr):
    """
    day_of_week() - Returns day of week (0=Monday, 6=Sunday).

    For weekly schedule patterns.

    Example: day_of_week() < 5 - Weekday
    """
    def __post_init__(self):
        self.expr_type = ExprType.INT

    def evaluate(self, context: Dict[str, Any]) -> int:
        import datetime
        timestamp = context.get('__timestamp__', time.time())
        dt = datetime.datetime.fromtimestamp(timestamp)
        return dt.weekday()

    def to_string(self) -> str:
        return "day_of_week()"


@dataclass
class BusinessHours(TemporalExpr):
    """
    business_hours(start, end) - True if current hour is in range.

    Convenience predicate for schedule checks.
    Both start and end hours are EVOLVABLE.

    Example: business_hours(9, 17) - 9 AM to 5 PM
    """
    start_hour: int = 9   # EVOLVABLE
    end_hour: int = 17    # EVOLVABLE

    def __post_init__(self):
        self.expr_type = ExprType.BOOL

    def evaluate(self, context: Dict[str, Any]) -> bool:
        import datetime
        timestamp = context.get('__timestamp__', time.time())
        dt = datetime.datetime.fromtimestamp(timestamp)
        hour = dt.hour
        return self.start_hour <= hour < self.end_hour

    def to_string(self) -> str:
        return f"business_hours({self.start_hour}, {self.end_hour})"


# =============================================================================
# COMPOUND TEMPORAL GUARDS
# =============================================================================

@dataclass
class TemporalGuard:
    """
    A complete temporal guard combining base and temporal predicates.

    Structure:
    - base_expr: Non-temporal condition (variable checks)
    - temporal_expr: Temporal condition
    - combine_op: How to combine ('and', 'or')
    """
    base_expr: Optional[TemporalExpr] = None
    temporal_expr: Optional[TemporalExpr] = None
    combine_op: str = "and"

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate the complete guard."""
        base_result = True
        if self.base_expr:
            base_result = bool(self.base_expr.evaluate(context))

        temporal_result = True
        if self.temporal_expr:
            temporal_result = bool(self.temporal_expr.evaluate(context))

        if self.combine_op == 'and':
            return base_result and temporal_result
        elif self.combine_op == 'or':
            return base_result or temporal_result
        return temporal_result

    def to_string(self) -> str:
        parts = []
        if self.base_expr:
            parts.append(self.base_expr.to_string())
        if self.temporal_expr:
            parts.append(self.temporal_expr.to_string())

        if len(parts) == 0:
            return "true"
        elif len(parts) == 1:
            return parts[0]
        else:
            return f"({parts[0]} {self.combine_op} {parts[1]})"

    def get_temporal_summary(self) -> str:
        """Extract summary of temporal constraints."""
        if not self.temporal_expr:
            return "no temporal constraints"

        def summarize(expr):
            summaries = []
            if isinstance(expr, After):
                summaries.append(f"timeout:{expr.duration:.1f}s")
            elif isinstance(expr, Within):
                summaries.append(f"deadline:{expr.duration:.1f}s")
            elif isinstance(expr, Since):
                summaries.append(f"since:{expr.state_name}")
            elif isinstance(expr, Timeout):
                summaries.append(f"timeout({expr.state_name}):{expr.duration:.1f}s")
            elif isinstance(expr, RateLimit):
                summaries.append(f"rate:{expr.count}/{expr.window:.0f}s")
            elif isinstance(expr, Cooldown):
                summaries.append(f"cooldown:{expr.duration:.1f}s")
            elif isinstance(expr, Debounce):
                summaries.append(f"debounce:{expr.duration:.1f}s")
            elif isinstance(expr, BusinessHours):
                summaries.append(f"hours:{expr.start_hour}-{expr.end_hour}")
            elif isinstance(expr, BinOp):
                summaries.extend(summarize(expr.left))
                summaries.extend(summarize(expr.right))
            elif isinstance(expr, UnaryOp):
                summaries.extend(summarize(expr.operand))
            return summaries

        items = summarize(self.temporal_expr)
        return ", ".join(items) if items else "complex temporal"


# =============================================================================
# DURATION CONSTANTS (Common values for evolution to discover)
# =============================================================================

DURATION_CANDIDATES = [
    0.1, 0.25, 0.5,           # Sub-second (debounce)
    1.0, 2.0, 3.0, 5.0,       # Short (cooldown)
    10.0, 15.0, 30.0,         # Medium (game timers)
    60.0, 120.0, 300.0,       # Minutes (session)
    600.0, 1800.0, 3600.0,    # Long (idle timeout)
    7200.0, 14400.0, 86400.0  # Hours/day (expiration)
]

RATE_LIMIT_COUNTS = [1, 2, 3, 5, 10, 20, 50, 100, 1000]
RATE_LIMIT_WINDOWS = [1.0, 5.0, 10.0, 60.0, 300.0, 3600.0]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_timeout_guard(duration: float) -> TemporalGuard:
    """Create a simple timeout guard."""
    return TemporalGuard(temporal_expr=After(duration=duration))


def create_rate_limit_guard(count: int, window: float) -> TemporalGuard:
    """Create a rate limiting guard."""
    return TemporalGuard(temporal_expr=RateLimit(count=count, window=window))


def create_cooldown_guard(duration: float) -> TemporalGuard:
    """Create a cooldown guard."""
    return TemporalGuard(temporal_expr=Cooldown(duration=duration))


def create_business_hours_guard(start: int = 9, end: int = 17) -> TemporalGuard:
    """Create a business hours guard."""
    return TemporalGuard(temporal_expr=BusinessHours(start_hour=start, end_hour=end))


# =============================================================================
# TESTING
# =============================================================================

def test_temporal_guards():
    """Test temporal guard expressions."""
    print("=" * 60)
    print("TEMPORAL GUARD DSL TEST")
    print("=" * 60)

    base_time = time.time()

    # Test 1: After
    print("\n1. After (timeout):")
    ctx = {
        '__timestamp__': base_time,
        '__state_entry_time__': base_time - 35,  # 35 seconds ago
    }
    guard = After(duration=30.0)
    result = guard.evaluate(ctx)
    print(f"   {guard.to_string()} = {result} (elapsed=35s)")
    assert result == True

    # Test 2: Within (deadline)
    print("\n2. Within (deadline):")
    ctx = {
        '__timestamp__': base_time,
        '__state_entry_time__': base_time - 3,  # 3 seconds ago
    }
    guard = Within(duration=5.0)
    result = guard.evaluate(ctx)
    print(f"   {guard.to_string()} = {result} (elapsed=3s)")
    assert result == True

    # Test 3: Since
    print("\n3. Since (time since state exit):")
    ctx = {
        '__timestamp__': base_time,
        '__state_exit_times__': {'LOGIN': base_time - 3700},  # ~1 hour ago
    }
    guard = BinOp(
        op='>',
        left=Since(state_name='LOGIN'),
        right=Const(value=3600.0, expr_type=ExprType.FLOAT),
        expr_type=ExprType.BOOL
    )
    result = guard.evaluate(ctx)
    print(f"   {guard.to_string()} = {result}")
    assert result == True

    # Test 4: RateLimit
    print("\n4. RateLimit:")
    ctx = {
        '__timestamp__': base_time,
        '__event_times__': [base_time - 10, base_time - 20, base_time - 30],
    }
    guard = RateLimit(count=5, window=60.0)
    result = guard.evaluate(ctx)
    print(f"   {guard.to_string()} = {result} (3 events in window)")
    assert result == True

    # Test 5: Cooldown
    print("\n5. Cooldown:")
    ctx = {
        '__timestamp__': base_time,
        '__last_transition_time__': base_time - 3,  # 3 seconds ago
    }
    guard = Cooldown(duration=2.0)
    result = guard.evaluate(ctx)
    print(f"   {guard.to_string()} = {result}")
    assert result == True

    # Test 6: Compound guard
    print("\n6. Compound guard (timeout AND variable check):")
    ctx = {
        '__timestamp__': base_time,
        '__state_entry_time__': base_time - 35,
        'user_active': False,
    }
    compound = TemporalGuard(
        base_expr=BinOp(
            op='==',
            left=Var(name='user_active', expr_type=ExprType.BOOL),
            right=Const(value=False, expr_type=ExprType.BOOL),
            expr_type=ExprType.BOOL
        ),
        temporal_expr=After(duration=30.0),
        combine_op='and'
    )
    result = compound.evaluate(ctx)
    print(f"   {compound.to_string()} = {result}")
    print(f"   Summary: {compound.get_temporal_summary()}")
    assert result == True

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_temporal_guards()
