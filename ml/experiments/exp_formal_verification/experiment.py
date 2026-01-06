"""
Experiment Runner: Verified Evolution Experiments

Comprehensive experiments demonstrating:
1. Property checking on hand-crafted statecharts
2. Evolution with verification fitness
3. Comparison: verified vs unverified evolution
4. Scalability analysis
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import json

from .smt_encoder import (
    SMTEncoder, StateVariable, TransitionFormula, GuardFormula, StateType, HAS_Z3,
)
from .property_checker import (
    PropertyChecker, PropertyResult, PropertyStatus,
    IllegalTransitionProperty, DeadlockFreedomProperty,
    DeterminismProperty, ReachabilityProperty, LivenessProperty,
)
from .verified_evolution import (
    VerifiedEvolver, VerificationFitness, VerifiedGenome,
)


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""
    experiment_name: str
    success: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    error: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    # Evolution params
    population_size: int = 30
    n_generations: int = 50
    verification_weight: float = 0.3

    # State space
    state_names: List[str] = field(default_factory=lambda: [
        "idle", "running", "paused", "stopped", "error"
    ])
    context_vars: List[str] = field(default_factory=lambda: [
        "counter", "timer", "retries"
    ])

    # Verification params
    timeout_ms: int = 2000
    reject_invalid: bool = True


def run_property_checking_experiment(config: ExperimentConfig) -> ExperimentResult:
    """
    Experiment 1: Property checking on hand-crafted statecharts.

    Tests property verification on known-good and known-bad statecharts.
    """
    result = ExperimentResult(experiment_name="property_checking")
    t0 = time.time()

    try:
        encoder = SMTEncoder()
        checker = PropertyChecker(encoder)

        # Test 1: Well-formed traffic light (should pass all properties)
        print("\n=== Test 1: Well-formed Traffic Light ===")
        states_good = [
            StateVariable(name="root", state_type=StateType.OR,
                         children=["red", "yellow", "green"], is_initial=True),
            StateVariable(name="red", state_type=StateType.BASIC,
                         parent="root", is_initial=True),
            StateVariable(name="yellow", state_type=StateType.BASIC,
                         parent="root"),
            StateVariable(name="green", state_type=StateType.BASIC,
                         parent="root", is_final=True),
        ]
        transitions_good = [
            TransitionFormula(source="red", target="green", guard=GuardFormula("true")),
            TransitionFormula(source="green", target="yellow", guard=GuardFormula("true")),
            TransitionFormula(source="yellow", target="red", guard=GuardFormula("true")),
        ]

        formula_good = encoder.encode_statechart(states_good, transitions_good)
        results_good = checker.check_all_properties(formula_good, timeout_ms=config.timeout_ms)

        good_passed = sum(1 for r in results_good if r.passed)
        print(f"Well-formed statechart: {good_passed}/{len(results_good)} properties passed")

        # Test 2: Broken statechart (missing transitions -> deadlock)
        print("\n=== Test 2: Broken Statechart (Deadlock) ===")
        states_bad = [
            StateVariable(name="root", state_type=StateType.OR,
                         children=["a", "b", "c"], is_initial=True),
            StateVariable(name="a", state_type=StateType.BASIC,
                         parent="root", is_initial=True),
            StateVariable(name="b", state_type=StateType.BASIC,
                         parent="root"),
            StateVariable(name="c", state_type=StateType.BASIC,
                         parent="root", is_final=True),
        ]
        transitions_bad = [
            TransitionFormula(source="a", target="b", guard=GuardFormula("true")),
            # Missing: b -> c transition (deadlock in b)
        ]

        checker2 = PropertyChecker(encoder)
        formula_bad = encoder.encode_statechart(states_bad, transitions_bad)
        results_bad = checker2.check_all_properties(formula_bad, timeout_ms=config.timeout_ms)

        bad_passed = sum(1 for r in results_bad if r.passed)
        print(f"Broken statechart: {bad_passed}/{len(results_bad)} properties passed")

        # Test 3: Non-deterministic statechart
        print("\n=== Test 3: Non-deterministic Statechart ===")
        states_nondet = [
            StateVariable(name="root", state_type=StateType.OR,
                         children=["s1", "s2", "s3"], is_initial=True),
            StateVariable(name="s1", state_type=StateType.BASIC,
                         parent="root", is_initial=True),
            StateVariable(name="s2", state_type=StateType.BASIC,
                         parent="root"),
            StateVariable(name="s3", state_type=StateType.BASIC,
                         parent="root", is_final=True),
        ]
        transitions_nondet = [
            # Two unguarded transitions from s1 -> nondeterminism
            TransitionFormula(source="s1", target="s2", guard=GuardFormula("true")),
            TransitionFormula(source="s1", target="s3", guard=GuardFormula("true")),
            TransitionFormula(source="s2", target="s3", guard=GuardFormula("true")),
        ]

        checker3 = PropertyChecker(encoder)
        formula_nondet = encoder.encode_statechart(states_nondet, transitions_nondet)
        results_nondet = checker3.check_all_properties(formula_nondet, timeout_ms=config.timeout_ms)

        nondet_passed = sum(1 for r in results_nondet if r.passed)
        print(f"Non-deterministic statechart: {nondet_passed}/{len(results_nondet)} properties passed")

        result.metrics = {
            'well_formed_passed': good_passed,
            'well_formed_total': len(results_good),
            'broken_passed': bad_passed,
            'broken_total': len(results_bad),
            'nondet_passed': nondet_passed,
            'nondet_total': len(results_nondet),
        }
        result.success = True

    except Exception as e:
        result.error = str(e)
        result.success = False

    result.duration = time.time() - t0
    return result


def run_verified_evolution_experiment(config: ExperimentConfig) -> ExperimentResult:
    """
    Experiment 2: Evolution with verification fitness.

    Evolve statecharts where fitness includes verification score.
    """
    result = ExperimentResult(experiment_name="verified_evolution")
    t0 = time.time()

    try:
        evolver = VerifiedEvolver(
            state_names=config.state_names,
            context_vars=config.context_vars,
            population_size=config.population_size,
            verification_weight=config.verification_weight,
            reject_invalid=config.reject_invalid,
        )

        # Accuracy function: prefer more states and transitions
        def accuracy_fn(genome: VerifiedGenome) -> float:
            state_score = len(genome.states) / len(config.state_names)
            trans_score = min(1.0, len(genome.transitions) / (len(genome.states) * 1.5))
            return 0.6 * state_score + 0.4 * trans_score

        print("\n=== Verified Evolution ===")
        best = evolver.evolve(
            n_generations=config.n_generations,
            accuracy_fn=accuracy_fn,
            target_fitness=0.85,
            verbose=True,
        )

        stats = evolver.get_statistics()

        if best:
            print(f"\nBest genome:")
            print(f"  States: {[s.name for s in best.states]}")
            print(f"  Transitions: {len(best.transitions)}")
            print(f"  Fitness: {best.fitness.total_fitness:.3f}")
            print(f"  Valid: {best.fitness.is_valid}")

        result.metrics = {
            'generations': stats['generations'],
            'total_verified': stats['total_verified'],
            'total_rejected': stats['total_rejected'],
            'verification_time': stats['verification_time'],
            'best_fitness': stats['best_fitness'],
            'best_is_valid': stats['best_is_valid'],
            'best_states': [s.name for s in best.states] if best else [],
            'best_transitions': len(best.transitions) if best else 0,
        }
        result.success = best is not None and best.fitness.is_valid

    except Exception as e:
        result.error = str(e)
        result.success = False

    result.duration = time.time() - t0
    return result


def run_comparison_experiment(config: ExperimentConfig) -> ExperimentResult:
    """
    Experiment 3: Compare verified vs unverified evolution.

    Key question: Does verification fitness improve quality?
    """
    result = ExperimentResult(experiment_name="comparison")
    t0 = time.time()

    try:
        # Run 1: Evolution WITHOUT verification (weight=0)
        print("\n=== Evolution WITHOUT Verification ===")
        evolver_no_verify = VerifiedEvolver(
            state_names=config.state_names,
            context_vars=config.context_vars,
            population_size=config.population_size,
            verification_weight=0.0,  # No verification
            reject_invalid=False,
        )

        def accuracy_fn(genome: VerifiedGenome) -> float:
            return len(genome.states) / len(config.state_names)

        best_no_verify = evolver_no_verify.evolve(
            n_generations=config.n_generations,
            accuracy_fn=accuracy_fn,
            target_fitness=0.9,
            verbose=True,
        )

        # Verify the "best" from unverified evolution
        encoder = SMTEncoder()
        checker = PropertyChecker(encoder)

        if best_no_verify:
            formula = encoder.encode_statechart(
                best_no_verify.states,
                best_no_verify.transitions
            )
            results_no_verify = checker.check_all_properties(formula)
            no_verify_passed = sum(1 for r in results_no_verify if r.passed)
        else:
            no_verify_passed = 0

        # Run 2: Evolution WITH verification
        print("\n=== Evolution WITH Verification ===")
        evolver_verify = VerifiedEvolver(
            state_names=config.state_names,
            context_vars=config.context_vars,
            population_size=config.population_size,
            verification_weight=config.verification_weight,
            reject_invalid=config.reject_invalid,
        )

        best_verify = evolver_verify.evolve(
            n_generations=config.n_generations,
            accuracy_fn=accuracy_fn,
            target_fitness=0.9,
            verbose=True,
        )

        if best_verify:
            verify_passed = best_verify.fitness.properties_passed
        else:
            verify_passed = 0

        print("\n=== Comparison Results ===")
        print(f"Without verification: {no_verify_passed} properties passed")
        print(f"With verification: {verify_passed} properties passed")

        result.metrics = {
            'no_verify_properties_passed': no_verify_passed,
            'verify_properties_passed': verify_passed,
            'no_verify_fitness': best_no_verify.fitness.total_fitness if best_no_verify else 0,
            'verify_fitness': best_verify.fitness.total_fitness if best_verify else 0,
            'no_verify_valid': best_no_verify.fitness.is_valid if best_no_verify else False,
            'verify_valid': best_verify.fitness.is_valid if best_verify else False,
        }
        result.success = True

    except Exception as e:
        result.error = str(e)
        result.success = False

    result.duration = time.time() - t0
    return result


def run_scalability_experiment(config: ExperimentConfig) -> ExperimentResult:
    """
    Experiment 4: Scalability analysis.

    How does verification time scale with statechart complexity?
    """
    result = ExperimentResult(experiment_name="scalability")
    t0 = time.time()

    try:
        encoder = SMTEncoder()

        sizes = [3, 5, 7, 10, 15]
        timings = {}

        for n_states in sizes:
            print(f"\n=== Testing {n_states} states ===")

            # Create linear statechart with n_states
            states = []
            for i in range(n_states):
                states.append(StateVariable(
                    name=f"s{i}",
                    state_type=StateType.BASIC,
                    is_initial=(i == 0),
                    is_final=(i == n_states - 1),
                ))

            # Create transitions
            transitions = []
            for i in range(n_states - 1):
                transitions.append(TransitionFormula(
                    source=f"s{i}",
                    target=f"s{i+1}",
                    guard=GuardFormula("true"),
                ))

            # Time encoding + verification
            t1 = time.time()
            formula = encoder.encode_statechart(states, transitions)

            checker = PropertyChecker(encoder)
            results = checker.check_all_properties(formula, timeout_ms=config.timeout_ms)
            elapsed = time.time() - t1

            passed = sum(1 for r in results if r.passed)
            print(f"  Time: {elapsed:.3f}s, Passed: {passed}/{len(results)}")

            timings[n_states] = {
                'encoding_and_verification_time': elapsed,
                'properties_passed': passed,
                'properties_total': len(results),
            }

        result.metrics = {
            'sizes': sizes,
            'timings': timings,
        }
        result.success = True

    except Exception as e:
        result.error = str(e)
        result.success = False

    result.duration = time.time() - t0
    return result


def run_experiment(config: ExperimentConfig = None) -> Dict[str, ExperimentResult]:
    """
    Run all experiments.

    Returns:
        Dictionary mapping experiment names to results
    """
    if config is None:
        config = ExperimentConfig()

    print("=" * 60)
    print("FORMAL VERIFICATION EXPERIMENTS")
    print("=" * 60)

    if not HAS_Z3:
        print("\nWARNING: Z3 not available. Install with: pip install z3-solver")
        print("Running with mock implementation (limited functionality).")

    results = {}

    # Experiment 1: Property Checking
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Property Checking")
    print("=" * 60)
    results['property_checking'] = run_property_checking_experiment(config)

    # Experiment 2: Verified Evolution
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Verified Evolution")
    print("=" * 60)
    results['verified_evolution'] = run_verified_evolution_experiment(config)

    # Experiment 3: Comparison
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Verified vs Unverified Comparison")
    print("=" * 60)
    results['comparison'] = run_comparison_experiment(config)

    # Experiment 4: Scalability
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Scalability Analysis")
    print("=" * 60)
    results['scalability'] = run_scalability_experiment(config)

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    for name, res in results.items():
        status = "SUCCESS" if res.success else "FAILED"
        print(f"\n{name}: {status} ({res.duration:.2f}s)")
        if res.error:
            print(f"  Error: {res.error}")
        else:
            for key, value in res.metrics.items():
                if not isinstance(value, (dict, list)) or len(str(value)) < 50:
                    print(f"  {key}: {value}")

    return results


def demo():
    """Quick demo of formal verification."""
    print("=" * 60)
    print("FORMAL VERIFICATION DEMO")
    print("=" * 60)

    if not HAS_Z3:
        print("\nZ3 not available. Install with: pip install z3-solver")

    # Quick property check
    encoder = SMTEncoder()
    checker = PropertyChecker(encoder)

    states = [
        StateVariable(name="off", state_type=StateType.BASIC, is_initial=True),
        StateVariable(name="on", state_type=StateType.BASIC, is_final=True),
    ]
    transitions = [
        TransitionFormula(source="off", target="on", guard=GuardFormula("true")),
        TransitionFormula(source="on", target="off", guard=GuardFormula("true")),
    ]

    print("\nChecking simple on/off switch...")
    formula = encoder.encode_statechart(states, transitions)
    results = checker.check_all_properties(formula)

    print(f"\nResults: {sum(1 for r in results if r.passed)}/{len(results)} passed")

    return checker


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        run_experiment()
