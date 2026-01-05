"""
Temporal Guard Benchmark

Test patterns for temporal guard evolution:
1. Session Timeout: Learn idle timeout from user activity
2. Game Timer: Learn move time limits
3. Rate Limiting: Learn request rate constraints
4. Cooldown Mechanics: Learn ability cooldowns
5. Debounce Patterns: Learn UI debounce timing
6. Business Hours: Learn schedule-based access

Each scenario generates training data and evaluates evolved guards.
NO HARDCODING - all thresholds discovered through evolution.
"""

import time
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

from .temporal_evolver import TemporalEvolver, TemporalGenome


# =============================================================================
# DATA GENERATORS
# =============================================================================

def generate_session_timeout_data(
    timeout_seconds: float = 30.0,
    n_samples: int = 100,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate training data for session timeout detection.

    Pattern: Session should expire after being idle for timeout_seconds.

    Positive: idle for >= timeout_seconds (should timeout)
    Negative: active or idle < timeout_seconds (stay active)
    """
    positive = []
    negative = []
    base_time = time.time()

    for _ in range(n_samples):
        # Positive: session should timeout
        idle_time = random.uniform(timeout_seconds, timeout_seconds * 3)
        positive.append({
            '__timestamp__': base_time,
            '__state_entry_time__': base_time - idle_time,
            'user_active': False,
            'requests_pending': 0,
        })

        # Negative: session should stay active
        if random.random() < 0.5:
            # Recently active
            idle_time = random.uniform(0, timeout_seconds * 0.9)
            negative.append({
                '__timestamp__': base_time,
                '__state_entry_time__': base_time - idle_time,
                'user_active': random.choice([True, False]),
                'requests_pending': random.randint(0, 3),
            })
        else:
            # Active user (even if long idle)
            negative.append({
                '__timestamp__': base_time,
                '__state_entry_time__': base_time - random.uniform(0, 60),
                'user_active': True,
                'requests_pending': random.randint(1, 5),
            })

    return positive, negative


def generate_game_timer_data(
    move_time_limit: float = 30.0,
    n_samples: int = 100,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate training data for game move timeouts.

    Pattern: Player forfeits turn if thinking too long.

    Positive: think time >= limit (forfeit turn)
    Negative: think time < limit (valid move)
    """
    positive = []
    negative = []
    base_time = time.time()

    for _ in range(n_samples):
        # Positive: time exceeded
        think_time = random.uniform(move_time_limit, move_time_limit * 3)
        positive.append({
            '__timestamp__': base_time,
            '__state_entry_time__': base_time - think_time,
            'current_player': random.choice(['white', 'black']),
            'move_count': random.randint(1, 50),
            'game_phase': random.choice(['opening', 'middle', 'endgame']),
        })

        # Negative: moved in time
        think_time = random.uniform(0, move_time_limit * 0.9)
        negative.append({
            '__timestamp__': base_time,
            '__state_entry_time__': base_time - think_time,
            'current_player': random.choice(['white', 'black']),
            'move_count': random.randint(1, 50),
            'game_phase': random.choice(['opening', 'middle', 'endgame']),
        })

    return positive, negative


def generate_rate_limit_data(
    max_requests: int = 5,
    window_seconds: float = 60.0,
    n_samples: int = 100,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate training data for rate limiting.

    Pattern: Block if too many requests in time window.

    Positive: requests >= limit (should block)
    Negative: requests < limit (allow)
    """
    positive = []
    negative = []
    base_time = time.time()

    for _ in range(n_samples):
        # Positive: should be rate limited
        event_count = random.randint(max_requests, max_requests * 2)
        event_times = [
            base_time - random.uniform(0, window_seconds * 0.9)
            for _ in range(event_count)
        ]
        positive.append({
            '__timestamp__': base_time,
            '__event_times__': event_times,
            'request_type': random.choice(['api', 'web', 'rpc']),
        })

        # Negative: under rate limit
        event_count = random.randint(0, max_requests - 1)
        event_times = [
            base_time - random.uniform(0, window_seconds * 1.5)
            for _ in range(event_count)
        ]
        negative.append({
            '__timestamp__': base_time,
            '__event_times__': event_times,
            'request_type': random.choice(['api', 'web', 'rpc']),
        })

    return positive, negative


def generate_cooldown_data(
    cooldown_seconds: float = 5.0,
    n_samples: int = 100,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate training data for ability cooldowns.

    Pattern: Allow action only after cooldown period.

    Positive: cooldown complete (can use ability)
    Negative: still on cooldown (cannot use)
    """
    positive = []
    negative = []
    base_time = time.time()

    for _ in range(n_samples):
        # Positive: cooldown complete
        last_use = base_time - random.uniform(cooldown_seconds, cooldown_seconds * 3)
        positive.append({
            '__timestamp__': base_time,
            '__last_transition_time__': last_use,
            'ability': random.choice(['fireball', 'heal', 'shield']),
            'mana': random.randint(50, 100),
        })

        # Negative: still on cooldown
        last_use = base_time - random.uniform(0, cooldown_seconds * 0.9)
        negative.append({
            '__timestamp__': base_time,
            '__last_transition_time__': last_use,
            'ability': random.choice(['fireball', 'heal', 'shield']),
            'mana': random.randint(0, 100),
        })

    return positive, negative


def generate_debounce_data(
    debounce_seconds: float = 0.5,
    n_samples: int = 100,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate training data for UI debounce.

    Pattern: Ignore rapid changes, wait for stability.

    Positive: stable for debounce period (process)
    Negative: changed too recently (wait)
    """
    positive = []
    negative = []
    base_time = time.time()

    for _ in range(n_samples):
        # Positive: stable long enough
        stable_time = random.uniform(debounce_seconds, debounce_seconds * 5)
        positive.append({
            '__timestamp__': base_time,
            '__state_entry_time__': base_time - stable_time,
            'input_value': random.choice(['search', 'filter', 'query']),
            'key_pressed': False,
        })

        # Negative: too recent
        stable_time = random.uniform(0, debounce_seconds * 0.8)
        negative.append({
            '__timestamp__': base_time,
            '__state_entry_time__': base_time - stable_time,
            'input_value': random.choice(['search', 'filter', 'query']),
            'key_pressed': True,
        })

    return positive, negative


def generate_business_hours_data(
    start_hour: int = 9,
    end_hour: int = 17,
    n_samples: int = 100,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate training data for business hours check.

    Pattern: Allow access only during business hours.

    Positive: within business hours (allow)
    Negative: outside business hours (deny)
    """
    import datetime

    positive = []
    negative = []
    base_date = datetime.datetime.now().replace(hour=12, minute=0, second=0)

    for _ in range(n_samples):
        # Positive: during business hours
        hour = random.randint(start_hour, end_hour - 1)
        minute = random.randint(0, 59)
        dt = base_date.replace(hour=hour, minute=minute)
        positive.append({
            '__timestamp__': dt.timestamp(),
            'user_role': random.choice(['admin', 'user', 'guest']),
        })

        # Negative: outside business hours
        if random.random() < 0.5:
            hour = random.randint(0, start_hour - 1)
        else:
            hour = random.randint(end_hour, 23)
        minute = random.randint(0, 59)
        dt = base_date.replace(hour=hour, minute=minute)
        negative.append({
            '__timestamp__': dt.timestamp(),
            'user_role': random.choice(['admin', 'user', 'guest']),
        })

    return positive, negative


def generate_state_since_data(
    since_threshold: float = 3600.0,  # 1 hour
    state_name: str = 'LOGIN',
    n_samples: int = 100,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate training data for "time since state" checks.

    Pattern: Require re-authentication after too long since login.

    Positive: too long since state (re-auth required)
    Negative: recent enough (no re-auth)
    """
    positive = []
    negative = []
    base_time = time.time()

    for _ in range(n_samples):
        # Positive: too long since state exit
        since_time = random.uniform(since_threshold, since_threshold * 3)
        positive.append({
            '__timestamp__': base_time,
            '__state_exit_times__': {state_name: base_time - since_time},
            'action': random.choice(['view', 'edit', 'delete']),
        })

        # Negative: recent enough
        since_time = random.uniform(0, since_threshold * 0.9)
        negative.append({
            '__timestamp__': base_time,
            '__state_exit_times__': {state_name: base_time - since_time},
            'action': random.choice(['view', 'edit', 'delete']),
        })

    return positive, negative


# =============================================================================
# BENCHMARK SCENARIOS
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a benchmark scenario."""
    scenario_name: str
    target_threshold: float
    learned_guard: str
    learned_summary: str
    f1_score: float
    precision: float
    recall: float
    accuracy: float


def run_scenario(
    name: str,
    target: float,
    positive: List[Dict],
    negative: List[Dict],
    variables: List[str] = None,
    state_names: List[str] = None,
    n_generations: int = 60,
    verbose: bool = True,
) -> BenchmarkResult:
    """Run a single benchmark scenario."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {name}")
        print(f"Target threshold: {target}")
        print(f"{'='*60}")

    evolver = TemporalEvolver(
        variables=variables or [],
        state_names=state_names or [],
    )

    best = evolver.evolve(
        positive,
        negative,
        population_size=40,
        n_generations=n_generations,
        verbose=verbose,
    )

    if verbose:
        print(f"\nLearned: {best.expr.to_string()}")
        print(f"Summary: {best.get_temporal_summary()}")
        print(f"F1={best.f1:.3f}, P={best.precision:.3f}, R={best.recall:.3f}")

    return BenchmarkResult(
        scenario_name=name,
        target_threshold=target,
        learned_guard=best.expr.to_string(),
        learned_summary=best.get_temporal_summary(),
        f1_score=best.f1,
        precision=best.precision,
        recall=best.recall,
        accuracy=best.accuracy,
    )


def run_full_benchmark(verbose: bool = True) -> List[BenchmarkResult]:
    """
    Run all benchmark scenarios.

    Tests temporal guard evolution on diverse patterns.
    """
    results = []

    # 1. Session Timeout (30s)
    pos, neg = generate_session_timeout_data(30.0, 80)
    result = run_scenario(
        "Session Timeout",
        30.0,
        pos, neg,
        variables=['user_active', 'requests_pending'],
        verbose=verbose,
    )
    results.append(result)

    # 2. Game Timer (30s)
    pos, neg = generate_game_timer_data(30.0, 80)
    result = run_scenario(
        "Game Timer",
        30.0,
        pos, neg,
        variables=['current_player', 'game_phase'],
        verbose=verbose,
    )
    results.append(result)

    # 3. Rate Limiting (5 req / 60s)
    pos, neg = generate_rate_limit_data(5, 60.0, 80)
    result = run_scenario(
        "Rate Limiting",
        5.0,  # Target: 5 requests
        pos, neg,
        variables=['request_type'],
        verbose=verbose,
    )
    results.append(result)

    # 4. Ability Cooldown (5s)
    pos, neg = generate_cooldown_data(5.0, 80)
    result = run_scenario(
        "Ability Cooldown",
        5.0,
        pos, neg,
        variables=['ability', 'mana'],
        verbose=verbose,
    )
    results.append(result)

    # 5. UI Debounce (0.5s)
    pos, neg = generate_debounce_data(0.5, 80)
    result = run_scenario(
        "UI Debounce",
        0.5,
        pos, neg,
        variables=['input_value', 'key_pressed'],
        verbose=verbose,
    )
    results.append(result)

    # 6. Business Hours (9-17)
    pos, neg = generate_business_hours_data(9, 17, 80)
    result = run_scenario(
        "Business Hours",
        9.0,  # Target: start hour
        pos, neg,
        variables=['user_role'],
        verbose=verbose,
    )
    results.append(result)

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"{'Scenario':<20} {'Target':>10} {'F1':>8} {'Temporal Constraint':<30}")
        print("-" * 70)
        for r in results:
            print(f"{r.scenario_name:<20} {r.target_threshold:>10.1f} {r.f1_score:>8.3f} {r.learned_summary[:30]:<30}")

        avg_f1 = sum(r.f1_score for r in results) / len(results)
        print("-" * 70)
        print(f"{'Average F1':<20} {'':<10} {avg_f1:>8.3f}")
        print("=" * 70)

        if avg_f1 >= 0.8:
            print("\nKEY INSIGHT: Temporal guards are LEARNABLE from timestamped traces!")
            print("Evolution discovers timeouts, rate limits, cooldowns automatically.")
        else:
            print("\nNote: Some scenarios may need more generations or tuning.")

    return results


# =============================================================================
# QUICK BENCHMARK
# =============================================================================

def quick_benchmark(verbose: bool = True) -> List[BenchmarkResult]:
    """Run a quick benchmark with fewer samples and generations."""
    results = []

    # Just test session timeout and cooldown
    scenarios = [
        ("Session Timeout", generate_session_timeout_data, 30.0, ['user_active']),
        ("Cooldown", generate_cooldown_data, 5.0, ['ability']),
    ]

    for name, generator, target, variables in scenarios:
        pos, neg = generator(target, 40)
        result = run_scenario(
            name, target, pos, neg,
            variables=variables,
            n_generations=30,
            verbose=verbose,
        )
        results.append(result)

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TEMPORAL GUARD BENCHMARK")
    print("=" * 70)
    print("\nRunning full benchmark suite...")
    print("This tests evolution of temporal guards across 6 use cases.\n")

    results = run_full_benchmark(verbose=True)

    # Check if all passed
    all_passed = all(r.f1_score >= 0.7 for r in results)
    if all_passed:
        print("\n[SUCCESS] All scenarios achieved F1 >= 0.7")
    else:
        print("\n[PARTIAL] Some scenarios need improvement")
