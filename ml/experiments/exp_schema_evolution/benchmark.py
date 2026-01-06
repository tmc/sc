"""
Schema Evolution Benchmark

Test scenarios for schema evolution:
1. State Removal: Remove states, migrate configs
2. State Rename: Rename states, preserve behavior
3. Hierarchy Change: Restructure state hierarchy
4. Feature Addition: Add new states/transitions
5. Combined Changes: Multiple changes together

Each scenario:
- Creates old and new chart versions
- Generates example migration pairs
- Tests synthesized migration plan
- Verifies safety properties

NO HARDCODING: Migration patterns learned from examples.
"""

import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

from .schema_diff import (
    Statechart, State, Transition, Event,
    ChartDiff, SchemaDiffer,
)
from .migration_synthesizer import (
    MigrationPlan, StateMapping, StateMappingType,
    Configuration, MigrationSynthesizer,
)
from .migration_verifier import (
    MigrationVerifier, VerificationResult, VerificationStatus,
    dry_run, DryRunResult,
)


# =============================================================================
# BENCHMARK RESULT
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a benchmark scenario."""
    scenario_name: str
    migration_accuracy: float  # Fraction of examples correctly migrated
    verification_status: str
    coverage: float
    is_reversible: bool
    elapsed_time: float
    learned_mappings: List[str]


# =============================================================================
# SCENARIO GENERATORS
# =============================================================================

def create_state_removal_scenario() -> Tuple[Statechart, Statechart, List[Tuple[Configuration, Configuration]]]:
    """
    Scenario: State Removal

    Old: Idle → Active → Done
    New: Idle → Active → Complete (Done removed, Complete added)

    Migration: Done → Complete
    """
    old_chart = Statechart(
        version="1.0.0",
        root_state=State(
            label="__root__",
            state_type="NORMAL",
            children=[
                State(label="Idle", is_initial=True),
                State(label="Active"),
                State(label="Done"),
            ]
        ),
        transitions=[
            Transition(label="start", from_states=["Idle"], to_states=["Active"], event="START"),
            Transition(label="finish", from_states=["Active"], to_states=["Done"], event="FINISH"),
        ],
    )

    new_chart = Statechart(
        version="2.0.0",
        root_state=State(
            label="__root__",
            state_type="NORMAL",
            children=[
                State(label="Idle", is_initial=True),
                State(label="Active"),
                State(label="Complete"),  # Renamed from Done
            ]
        ),
        transitions=[
            Transition(label="start", from_states=["Idle"], to_states=["Active"], event="START"),
            Transition(label="finish", from_states=["Active"], to_states=["Complete"], event="FINISH"),
        ],
    )

    examples = [
        (Configuration(states={"Idle"}), Configuration(states={"Idle"})),
        (Configuration(states={"Active"}), Configuration(states={"Active"})),
        (Configuration(states={"Done"}), Configuration(states={"Complete"})),
    ]

    return old_chart, new_chart, examples


def create_state_rename_scenario() -> Tuple[Statechart, Statechart, List[Tuple[Configuration, Configuration]]]:
    """
    Scenario: State Rename

    Old: Login → Dashboard → Settings
    New: SignIn → Dashboard → Preferences (Login→SignIn, Settings→Preferences)
    """
    old_chart = Statechart(
        version="1.0.0",
        root_state=State(
            label="__root__",
            state_type="NORMAL",
            children=[
                State(label="Login", is_initial=True),
                State(label="Dashboard"),
                State(label="Settings"),
            ]
        ),
    )

    new_chart = Statechart(
        version="2.0.0",
        root_state=State(
            label="__root__",
            state_type="NORMAL",
            children=[
                State(label="SignIn", is_initial=True),  # Renamed
                State(label="Dashboard"),
                State(label="Preferences"),  # Renamed
            ]
        ),
    )

    examples = [
        (Configuration(states={"Login"}), Configuration(states={"SignIn"})),
        (Configuration(states={"Dashboard"}), Configuration(states={"Dashboard"})),
        (Configuration(states={"Settings"}), Configuration(states={"Preferences"})),
    ]

    return old_chart, new_chart, examples


def create_hierarchy_change_scenario() -> Tuple[Statechart, Statechart, List[Tuple[Configuration, Configuration]]]:
    """
    Scenario: Hierarchy Change

    Old: Flat states - A, B, C
    New: Nested - Parent{A, B}, C

    States A, B move under new Parent state.
    """
    old_chart = Statechart(
        version="1.0.0",
        root_state=State(
            label="__root__",
            state_type="NORMAL",
            children=[
                State(label="A", is_initial=True),
                State(label="B"),
                State(label="C"),
            ]
        ),
    )

    new_chart = Statechart(
        version="2.0.0",
        root_state=State(
            label="__root__",
            state_type="NORMAL",
            children=[
                State(label="Parent", is_initial=True, state_type="NORMAL", children=[
                    State(label="A", is_initial=True),
                    State(label="B"),
                ]),
                State(label="C"),
            ]
        ),
    )

    examples = [
        (Configuration(states={"A"}), Configuration(states={"A", "Parent"})),
        (Configuration(states={"B"}), Configuration(states={"B", "Parent"})),
        (Configuration(states={"C"}), Configuration(states={"C"})),
    ]

    return old_chart, new_chart, examples


def create_feature_addition_scenario() -> Tuple[Statechart, Statechart, List[Tuple[Configuration, Configuration]]]:
    """
    Scenario: Feature Addition

    Old: Idle → Active
    New: Idle → Active → Paused → Active (new Paused state)

    Backward compatible - old configs still valid.
    """
    old_chart = Statechart(
        version="1.0.0",
        root_state=State(
            label="__root__",
            state_type="NORMAL",
            children=[
                State(label="Idle", is_initial=True),
                State(label="Active"),
            ]
        ),
    )

    new_chart = Statechart(
        version="2.0.0",
        root_state=State(
            label="__root__",
            state_type="NORMAL",
            children=[
                State(label="Idle", is_initial=True),
                State(label="Active"),
                State(label="Paused"),  # New
            ]
        ),
    )

    examples = [
        (Configuration(states={"Idle"}), Configuration(states={"Idle"})),
        (Configuration(states={"Active"}), Configuration(states={"Active"})),
    ]

    return old_chart, new_chart, examples


def create_combined_changes_scenario() -> Tuple[Statechart, Statechart, List[Tuple[Configuration, Configuration]]]:
    """
    Scenario: Combined Changes

    Old: Start → Processing{Validating, Executing} → End
    New: Init → Running{Checking, Working, Paused} → Done

    - Start renamed to Init
    - Processing renamed to Running
    - Validating renamed to Checking
    - Executing renamed to Working
    - New Paused state added
    - End renamed to Done
    """
    old_chart = Statechart(
        version="1.0.0",
        root_state=State(
            label="__root__",
            state_type="NORMAL",
            children=[
                State(label="Start", is_initial=True),
                State(label="Processing", state_type="NORMAL", children=[
                    State(label="Validating", is_initial=True),
                    State(label="Executing"),
                ]),
                State(label="End"),
            ]
        ),
    )

    new_chart = Statechart(
        version="2.0.0",
        root_state=State(
            label="__root__",
            state_type="NORMAL",
            children=[
                State(label="Init", is_initial=True),
                State(label="Running", state_type="NORMAL", children=[
                    State(label="Checking", is_initial=True),
                    State(label="Working"),
                    State(label="Paused"),
                ]),
                State(label="Done"),
            ]
        ),
    )

    examples = [
        (Configuration(states={"Start"}), Configuration(states={"Init"})),
        (Configuration(states={"Processing", "Validating"}), Configuration(states={"Running", "Checking"})),
        (Configuration(states={"Processing", "Executing"}), Configuration(states={"Running", "Working"})),
        (Configuration(states={"End"}), Configuration(states={"Done"})),
    ]

    return old_chart, new_chart, examples


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_scenario(
    name: str,
    old_chart: Statechart,
    new_chart: Statechart,
    examples: List[Tuple[Configuration, Configuration]],
    verbose: bool = True,
) -> BenchmarkResult:
    """Run a single benchmark scenario."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {name}")
        print(f"{'='*60}")

    start_time = time.time()

    # Synthesize and evolve migration
    synth = MigrationSynthesizer(n_generations=30)
    plan = synth.evolve_migration(old_chart, new_chart, examples, verbose=verbose)

    # Verify
    verifier = MigrationVerifier()
    verification = verifier.verify(plan, old_chart, new_chart)

    # Test accuracy
    correct = 0
    for old_config, expected in examples:
        actual = plan.migrate(old_config)
        if actual.states == expected.states:
            correct += 1

    accuracy = correct / len(examples) if examples else 0.0
    elapsed = time.time() - start_time

    # Collect learned mappings
    learned = []
    for m in plan.state_mappings:
        learned.append(f"{m.mapping_type.name}: {m.from_states} → {m.to_states}")

    result = BenchmarkResult(
        scenario_name=name,
        migration_accuracy=accuracy,
        verification_status=verification.status.name,
        coverage=verification.coverage,
        is_reversible=verification.is_reversible,
        elapsed_time=elapsed,
        learned_mappings=learned,
    )

    if verbose:
        print(f"\nResults:")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  Verification: {verification.status.name}")
        print(f"  Coverage: {verification.coverage:.1%}")
        print(f"  Reversible: {verification.is_reversible}")
        print(f"  Elapsed: {elapsed:.2f}s")
        print(f"  Learned mappings:")
        for m in learned:
            print(f"    - {m}")

    return result


def run_full_benchmark(verbose: bool = True) -> List[BenchmarkResult]:
    """
    Run all benchmark scenarios.

    Tests schema evolution on diverse migration patterns.
    """
    results = []

    scenarios = [
        ("State Removal", create_state_removal_scenario),
        ("State Rename", create_state_rename_scenario),
        ("Hierarchy Change", create_hierarchy_change_scenario),
        ("Feature Addition", create_feature_addition_scenario),
        ("Combined Changes", create_combined_changes_scenario),
    ]

    for name, generator in scenarios:
        old_chart, new_chart, examples = generator()
        result = run_scenario(name, old_chart, new_chart, examples, verbose=verbose)
        results.append(result)

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"{'Scenario':<20} {'Accuracy':>10} {'Verification':>15} {'Coverage':>10}")
        print("-" * 70)

        for r in results:
            print(f"{r.scenario_name:<20} {r.migration_accuracy:>10.1%} {r.verification_status:>15} {r.coverage:>10.1%}")

        avg_accuracy = sum(r.migration_accuracy for r in results) / len(results)
        print("-" * 70)
        print(f"{'Average':<20} {avg_accuracy:>10.1%}")
        print("=" * 70)

        if avg_accuracy >= 0.8:
            print("\nKEY INSIGHT: Schema evolution is LEARNABLE from example pairs!")
            print("Migration patterns can be synthesized automatically.")
        else:
            print("\nNote: Some scenarios may need more examples or tuning.")

    return results


def quick_benchmark(verbose: bool = True) -> List[BenchmarkResult]:
    """Run quick benchmark with just state removal and rename."""
    results = []

    scenarios = [
        ("State Removal", create_state_removal_scenario),
        ("State Rename", create_state_rename_scenario),
    ]

    for name, generator in scenarios:
        old_chart, new_chart, examples = generator()
        result = run_scenario(name, old_chart, new_chart, examples, verbose=verbose)
        results.append(result)

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SCHEMA EVOLUTION BENCHMARK")
    print("=" * 70)
    print("\nRunning full benchmark suite...")
    print("This tests migration synthesis across 5 scenarios.\n")

    results = run_full_benchmark(verbose=True)

    # Check if all passed
    all_passed = all(r.migration_accuracy >= 0.8 for r in results)
    if all_passed:
        print("\n[SUCCESS] All scenarios achieved >= 80% accuracy")
    else:
        print("\n[PARTIAL] Some scenarios need improvement")
