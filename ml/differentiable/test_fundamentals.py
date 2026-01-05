"""
Fundamental Validation Tests for Differentiable Statecharts

Tests core mathematical properties that must hold for the approach to be valid:
1. Soft configs sum to 1 (softmax constraint)
2. Guards output [0,1] range (sigmoid bounded)
3. Transitions preserve probability mass
4. Gradients flow through all components
5. State updates are smooth (Lipschitz continuous)
"""

import mlx.core as mx
import mlx.nn as nn
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from differentiable.exp_c_transitions import (
    DifferentiableTransitionSelector,
    DifferentiableGuard,
    StatechartMachine,
)


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []

    def record(self, name: str, passed: bool, details: str = ""):
        self.results.append((name, passed, details))
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def summary(self):
        return f"{self.passed}/{self.passed + self.failed} tests passed"


def test_soft_config_sums_to_one(results: TestResults):
    """Test 1: Soft configurations always sum to 1."""
    print("\n" + "=" * 60)
    print("Test 1: Soft configs sum to 1")
    print("=" * 60)

    all_passed = True
    details = []

    # Test with various temperatures
    temperatures = [0.1, 1.0, 10.0]

    for temp in temperatures:
        # Create random logits
        logits = mx.random.normal((4, 5))  # batch=4, states=5

        # Apply softmax with temperature
        config = mx.softmax(logits / temp, axis=-1)

        # Check sum
        sums = mx.sum(config, axis=-1)
        max_deviation = float(mx.max(mx.abs(sums - 1.0)))

        passed = max_deviation < 1e-5
        status = "✓" if passed else "✗"
        print(f"  Temperature {temp}: sum deviation={max_deviation:.2e} {status}")

        if not passed:
            all_passed = False
            details.append(f"temp={temp} deviation={max_deviation}")

    status = "PASS" if all_passed else "FAIL"
    print(f"\n  {status}")
    results.record("soft_config_sums_to_one", all_passed, "; ".join(details))


def test_guards_output_range(results: TestResults):
    """Test 2: Guards output values in [0, 1]."""
    print("\n" + "=" * 60)
    print("Test 2: Guards output [0,1] range")
    print("=" * 60)

    all_passed = True
    details = []

    # Create guard
    guard = DifferentiableGuard(context_dim=16, hidden_dim=32)

    # Test with various input magnitudes
    test_inputs = [
        ("normal", mx.random.normal((10, 16))),
        ("large positive", mx.ones((10, 16)) * 100),
        ("large negative", mx.ones((10, 16)) * -100),
        ("zeros", mx.zeros((10, 16))),
        ("mixed extreme", mx.random.normal((10, 16)) * 50),
    ]

    for name, context in test_inputs:
        output = guard(context)

        min_val = float(mx.min(output))
        max_val = float(mx.max(output))

        in_range = min_val >= 0.0 and max_val <= 1.0
        status = "✓" if in_range else "✗"
        print(f"  {name}: min={min_val:.6f}, max={max_val:.6f} {status}")

        if not in_range:
            all_passed = False
            details.append(f"{name}: [{min_val}, {max_val}]")

    status = "PASS" if all_passed else "FAIL"
    print(f"\n  {status}")
    results.record("guards_output_range", all_passed, "; ".join(details))


def test_transitions_preserve_mass(results: TestResults):
    """Test 3: Transitions preserve probability mass."""
    print("\n" + "=" * 60)
    print("Test 3: Transitions preserve probability mass")
    print("=" * 60)

    all_passed = True
    details = []

    # Create 3-state machine with cycle
    transitions = [(0, 1, 0), (1, 2, 0), (2, 0, 0)]
    machine = StatechartMachine(
        num_states=3,
        transitions=transitions,
        embed_dim=16,
        num_events=1
    )

    # Test with various input configs
    test_configs = [
        ("uniform", mx.array([[0.333, 0.333, 0.334]])),
        ("peaked", mx.array([[0.9, 0.05, 0.05]])),
        ("sparse", mx.array([[1.0, 0.0, 0.0]])),
        ("random", mx.softmax(mx.random.normal((1, 3)), axis=-1)),
    ]

    for name, config in test_configs:
        new_config, enablement, selection = machine.step(config)

        # Check new config sums to 1
        config_sum = float(mx.sum(new_config))
        config_ok = abs(config_sum - 1.0) < 1e-5

        # Check selection weights are non-negative and bounded
        sel_min = float(mx.min(selection))
        sel_max = float(mx.max(selection))
        sel_ok = sel_min >= 0.0 and sel_max <= 1.0

        passed = config_ok and sel_ok
        status = "✓" if passed else "✗"
        print(f"  {name}: config_sum={config_sum:.6f}, selection=[{sel_min:.3f},{sel_max:.3f}] {status}")

        if not passed:
            all_passed = False
            details.append(f"{name}: sum={config_sum}")

    status = "PASS" if all_passed else "FAIL"
    print(f"\n  {status}")
    results.record("transitions_preserve_mass", all_passed, "; ".join(details))


def test_gradient_flow(results: TestResults):
    """Test 4: Gradients flow through all components."""
    print("\n" + "=" * 60)
    print("Test 4: Gradients flow through all components")
    print("=" * 60)

    all_passed = True
    details = []

    # Test 4a: Guard gradients
    guard = DifferentiableGuard(context_dim=16, hidden_dim=32)
    context = mx.random.normal((2, 16))

    def guard_loss(ctx):
        return mx.mean(guard(ctx))

    loss, grad = mx.value_and_grad(guard_loss)(context)
    grad_norm = float(mx.sqrt(mx.sum(grad ** 2)))
    passed = grad_norm > 0
    status = "✓" if passed else "✗"
    print(f"  Guard gradients: norm={grad_norm:.6f} {status}")
    if not passed:
        all_passed = False
        details.append("guard grad=0")

    # Test 4b: Transition selector gradients
    transitions = [(0, 1, 0), (1, 2, 0), (2, 0, 0)]
    selector = DifferentiableTransitionSelector(
        num_states=3, num_transitions=3, embed_dim=16, num_events=1
    )
    selector.configure_transitions([0, 1, 2], [1, 2, 0], [0, 0, 0])

    soft_config = mx.array([[0.5, 0.3, 0.2]])

    def selector_loss(cfg):
        new_cfg, _, _ = selector(cfg)
        return mx.mean(new_cfg[:, 2])  # Maximize state 2

    loss, grad = mx.value_and_grad(selector_loss)(soft_config)
    grad_norm = float(mx.sqrt(mx.sum(grad ** 2)))
    passed = grad_norm > 0
    status = "✓" if passed else "✗"
    print(f"  Transition selector gradients: norm={grad_norm:.6f} {status}")
    if not passed:
        all_passed = False
        details.append("selector grad=0")

    # Test 4c: Full machine gradients
    machine = StatechartMachine(
        num_states=3, transitions=transitions, embed_dim=16, num_events=1
    )

    def machine_loss(cfg):
        new_cfg, _, _ = machine.step(cfg)
        return mx.mean(new_cfg[:, 1])

    loss, grad = mx.value_and_grad(machine_loss)(soft_config)
    grad_norm = float(mx.sqrt(mx.sum(grad ** 2)))
    passed = grad_norm > 0
    status = "✓" if passed else "✗"
    print(f"  Full machine gradients: norm={grad_norm:.6f} {status}")
    if not passed:
        all_passed = False
        details.append("machine grad=0")

    status = "PASS" if all_passed else "FAIL"
    print(f"\n  {status}")
    results.record("gradient_flow", all_passed, "; ".join(details))


def test_smooth_updates(results: TestResults):
    """Test 5: State updates are smooth (Lipschitz continuous)."""
    print("\n" + "=" * 60)
    print("Test 5: State updates are smooth")
    print("=" * 60)

    all_passed = True
    details = []

    # Create machine
    transitions = [(0, 1, 0), (1, 2, 0), (2, 0, 0)]
    machine = StatechartMachine(
        num_states=3, transitions=transitions, embed_dim=16, num_events=1
    )

    # Test Lipschitz continuity: ||f(x+ε) - f(x)|| ≤ L * ||ε||
    base_config = mx.array([[0.5, 0.3, 0.2]])
    base_output, _, _ = machine.step(base_config)

    epsilons = [0.1, 0.01, 0.001]
    lipschitz_constants = []

    for eps in epsilons:
        # Perturb input
        perturbation = mx.random.normal(base_config.shape) * eps
        perturbed_config = mx.softmax(mx.log(base_config + 1e-8) + perturbation, axis=-1)

        perturbed_output, _, _ = machine.step(perturbed_config)

        # Compute norms
        input_diff = float(mx.sqrt(mx.sum((perturbed_config - base_config) ** 2)))
        output_diff = float(mx.sqrt(mx.sum((perturbed_output - base_output) ** 2)))

        if input_diff > 1e-8:
            L = output_diff / input_diff
            lipschitz_constants.append(L)
            status = "✓" if L < 100 else "✗"  # Reasonable bound
            print(f"  ε={eps}: input_diff={input_diff:.6f}, output_diff={output_diff:.6f}, L={L:.2f} {status}")

            if L >= 100:
                all_passed = False
                details.append(f"L={L:.2f} at ε={eps}")

    # Check Lipschitz constants are bounded
    if lipschitz_constants:
        max_L = max(lipschitz_constants)
        bounded = max_L < 100
        if not bounded:
            all_passed = False

    status = "PASS" if all_passed else "FAIL"
    print(f"\n  {status}")
    results.record("smooth_updates", all_passed, "; ".join(details))


def run_all_tests():
    """Run all fundamental validation tests."""
    print("=" * 60)
    print("FUNDAMENTAL VALIDATION TESTS")
    print("Differentiable Statecharts Core Properties")
    print("=" * 60)

    results = TestResults()

    # Run all tests
    test_soft_config_sums_to_one(results)
    test_guards_output_range(results)
    test_transitions_preserve_mass(results)
    test_gradient_flow(results)
    test_smooth_updates(results)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, passed, details in results.results:
        status = "✓ PASS" if passed else "✗ FAIL"
        detail_str = f" ({details})" if details and not passed else ""
        print(f"  {name}: {status}{detail_str}")

    print(f"\n  {results.summary()}")

    # Return exit code
    return 0 if results.failed == 0 else 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    exit(exit_code)
