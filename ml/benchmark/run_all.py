"""
Comprehensive Benchmark for Differentiable Statechart Experiments.

Runs all experiment modules and collects:
- Gradient flow (does backprop work?)
- Basic functionality (does forward pass work?)
- Timing (how fast?)
- Parameter counts

Focus: What WORKS vs what DOESN'T. No hype.
"""

import mlx.core as mx
import mlx.nn as nn
import sys
import os
import time
import traceback
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

# Add paths for imports
_benchmark_dir = os.path.dirname(os.path.abspath(__file__))
_ml_dir = os.path.dirname(_benchmark_dir)
_root_dir = os.path.dirname(_ml_dir)
sys.path.insert(0, _ml_dir)
sys.path.insert(0, _root_dir)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark."""
    name: str
    status: str  # "PASS", "FAIL", "SKIP"
    forward_works: bool
    backward_works: bool
    grad_norm: float
    loss: float
    time_ms: float
    param_count: int
    error: Optional[str] = None
    notes: str = ""


def count_params(model: nn.Module) -> int:
    """Count trainable parameters."""
    total = 0
    for name, param in model.parameters().items() if hasattr(model, 'parameters') else []:
        if isinstance(param, mx.array):
            total += param.size
        elif isinstance(param, dict):
            for p in param.values():
                if isinstance(p, mx.array):
                    total += p.size
    return total


def count_params_recursive(params) -> int:
    """Recursively count parameters in nested dict."""
    total = 0
    if isinstance(params, mx.array):
        return params.size
    elif isinstance(params, dict):
        for v in params.values():
            total += count_params_recursive(v)
    elif isinstance(params, list):
        for v in params:
            total += count_params_recursive(v)
    return total


def test_gradient_flow(model: nn.Module, input_fn: Callable, loss_fn: Callable) -> tuple:
    """Test if gradients flow through the model.

    Returns: (grad_norm, loss_value, time_ms)
    """
    start = time.perf_counter()

    try:
        inputs = input_fn()

        def compute_loss(m):
            out = m(*inputs) if isinstance(inputs, tuple) else m(inputs)
            return loss_fn(out)

        loss, grads = nn.value_and_grad(model, compute_loss)(model)
        mx.eval(loss, grads)

        # Compute gradient norm
        def sum_grad_squares(g):
            if isinstance(g, mx.array):
                return float(mx.sum(g * g))
            elif isinstance(g, dict):
                return sum(sum_grad_squares(v) for v in g.values())
            elif isinstance(g, list):
                return sum(sum_grad_squares(v) for v in g)
            return 0.0

        grad_sq = sum_grad_squares(grads)
        grad_norm = grad_sq ** 0.5

        elapsed = (time.perf_counter() - start) * 1000
        return grad_norm, float(loss), elapsed

    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        raise RuntimeError(f"Gradient test failed: {e}")


# =============================================================================
# Experiment A: Soft Config
# =============================================================================

def benchmark_exp_a():
    """Benchmark exp_a: SoftStateConfiguration, DifferentiableGuard, SoftToggle."""
    from differentiable.exp_a_soft_config import SoftStateConfiguration, DifferentiableGuard, SoftToggle

    results = []

    # Test SoftStateConfiguration (uses get_config, not __call__)
    try:
        config = SoftStateConfiguration(num_states=4, temperature=1.0)

        # Manual gradient test since it's not callable
        start = time.perf_counter()

        def loss_fn(model):
            probs = model.get_config(batch_size=2)
            return -mx.mean(probs * mx.log(probs + 1e-10))  # entropy

        loss, grads = nn.value_and_grad(config, loss_fn)(config)
        mx.eval(loss, grads)

        def sum_grad_sq(g):
            if isinstance(g, mx.array):
                return float(mx.sum(g * g))
            elif isinstance(g, dict):
                return sum(sum_grad_sq(v) for v in g.values())
            return 0.0

        grad_norm = sum_grad_sq(grads) ** 0.5
        elapsed = (time.perf_counter() - start) * 1000
        params = count_params_recursive(config.parameters())

        results.append(BenchmarkResult(
            name="exp_a/SoftStateConfiguration",
            status="PASS",
            forward_works=True,
            backward_works=grad_norm > 0,
            grad_norm=grad_norm,
            loss=float(loss),
            time_ms=elapsed,
            param_count=params,
            notes=f"4 states, entropy={-float(loss):.3f}"
        ))
    except Exception as e:
        results.append(BenchmarkResult(
            name="exp_a/SoftStateConfiguration",
            status="FAIL",
            forward_works=False,
            backward_works=False,
            grad_norm=0,
            loss=0,
            time_ms=0,
            param_count=0,
            error=str(e)
        ))

    # Test DifferentiableGuard
    try:
        guard = DifferentiableGuard(context_dim=8, hidden_dim=16)

        def input_fn():
            return (mx.random.normal((2, 8)),)
        def loss_fn(out):
            return mx.mean(out)

        grad_norm, loss, time_ms = test_gradient_flow(guard, input_fn, loss_fn)
        params = count_params_recursive(guard.parameters())

        results.append(BenchmarkResult(
            name="exp_a/DifferentiableGuard",
            status="PASS",
            forward_works=True,
            backward_works=grad_norm > 0,
            grad_norm=grad_norm,
            loss=loss,
            time_ms=time_ms,
            param_count=params,
            notes="8-dim context, 16-dim hidden"
        ))
    except Exception as e:
        results.append(BenchmarkResult(
            name="exp_a/DifferentiableGuard",
            status="FAIL",
            forward_works=False,
            backward_works=False,
            grad_norm=0,
            loss=0,
            time_ms=0,
            param_count=0,
            error=str(e)
        ))

    # Test SoftToggle (uses step, not __call__)
    try:
        toggle = SoftToggle(context_dim=4, temperature=1.0)

        start = time.perf_counter()
        context = mx.random.normal((2, 4))

        def loss_fn(model):
            probs = model.step(context)
            return mx.mean(probs[:, 1])  # Minimize P(Off)

        loss, grads = nn.value_and_grad(toggle, loss_fn)(toggle)
        mx.eval(loss, grads)

        def sum_grad_sq(g):
            if isinstance(g, mx.array):
                return float(mx.sum(g * g))
            elif isinstance(g, dict):
                return sum(sum_grad_sq(v) for v in g.values())
            return 0.0

        grad_norm = sum_grad_sq(grads) ** 0.5
        elapsed = (time.perf_counter() - start) * 1000
        params = count_params_recursive(toggle.parameters())

        results.append(BenchmarkResult(
            name="exp_a/SoftToggle",
            status="PASS",
            forward_works=True,
            backward_works=grad_norm > 0,
            grad_norm=grad_norm,
            loss=float(loss),
            time_ms=elapsed,
            param_count=params,
            notes="2-state toggle with guards"
        ))
    except Exception as e:
        results.append(BenchmarkResult(
            name="exp_a/SoftToggle",
            status="FAIL",
            forward_works=False,
            backward_works=False,
            grad_norm=0,
            loss=0,
            time_ms=0,
            param_count=0,
            error=str(e)
        ))

    return results


# =============================================================================
# Experiment B: Memory
# =============================================================================

def benchmark_exp_b():
    """Benchmark exp_b: Memory mechanisms."""
    from differentiable.exp_b_memory import SoftAttentionMemory, NTMMemory, RecurrentMemory, UnifiedMemory, MemoryType

    results = []
    num_states = 4
    batch_size = 2

    # Test SoftAttentionMemory
    try:
        mem = SoftAttentionMemory(num_states=num_states, d_memory=16, max_history=32)

        def input_fn():
            current = mx.random.normal((batch_size, num_states))
            buffer = mx.random.normal((batch_size, 8, num_states))
            return (current, buffer)
        def loss_fn(out):
            retrieved, attn = out
            return mx.mean(retrieved)

        grad_norm, loss, time_ms = test_gradient_flow(mem, input_fn, loss_fn)
        params = count_params_recursive(mem.parameters())

        results.append(BenchmarkResult(
            name="exp_b/SoftAttentionMemory",
            status="PASS",
            forward_works=True,
            backward_works=grad_norm > 0,
            grad_norm=grad_norm,
            loss=loss,
            time_ms=time_ms,
            param_count=params,
            notes="Attention over history buffer"
        ))
    except Exception as e:
        results.append(BenchmarkResult(
            name="exp_b/SoftAttentionMemory",
            status="FAIL",
            forward_works=False,
            backward_works=False,
            grad_norm=0,
            loss=0,
            time_ms=0,
            param_count=0,
            error=str(e)
        ))

    # Test RecurrentMemory (GRU)
    try:
        mem = RecurrentMemory(num_states=num_states, hidden_size=32)

        def input_fn():
            current = mx.random.normal((batch_size, num_states))
            hidden = mx.zeros((batch_size, 32))
            return (current, hidden)
        def loss_fn(out):
            retrieved, new_hidden = out
            return mx.mean(retrieved)

        grad_norm, loss, time_ms = test_gradient_flow(mem, input_fn, loss_fn)
        params = count_params_recursive(mem.parameters())

        results.append(BenchmarkResult(
            name="exp_b/RecurrentMemory",
            status="PASS",
            forward_works=True,
            backward_works=grad_norm > 0,
            grad_norm=grad_norm,
            loss=loss,
            time_ms=time_ms,
            param_count=params,
            notes="GRU-based memory"
        ))
    except Exception as e:
        results.append(BenchmarkResult(
            name="exp_b/RecurrentMemory",
            status="FAIL",
            forward_works=False,
            backward_works=False,
            grad_norm=0,
            loss=0,
            time_ms=0,
            param_count=0,
            error=str(e)
        ))

    return results


# =============================================================================
# Experiment C: Transitions
# =============================================================================

def benchmark_exp_c():
    """Benchmark exp_c: DifferentiableTransitionSelector."""
    from differentiable.exp_c_transitions import DifferentiableTransitionSelector, StatechartMachine

    results = []

    # Test StatechartMachine (uses step, not __call__)
    try:
        transitions = [(0, 1, 0), (1, 2, 0), (2, 0, 0)]
        machine = StatechartMachine(num_states=3, transitions=transitions, embed_dim=16)

        start = time.perf_counter()
        soft_config = mx.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])

        def loss_fn(model):
            new_config, enablement, selection = model.step(soft_config)
            return -mx.mean(new_config[:, 2])  # Maximize state 2

        loss, grads = nn.value_and_grad(machine, loss_fn)(machine)
        mx.eval(loss, grads)

        def sum_grad_sq(g):
            if isinstance(g, mx.array):
                return float(mx.sum(g * g))
            elif isinstance(g, dict):
                return sum(sum_grad_sq(v) for v in g.values())
            return 0.0

        grad_norm = sum_grad_sq(grads) ** 0.5
        elapsed = (time.perf_counter() - start) * 1000
        params = count_params_recursive(machine.parameters())

        results.append(BenchmarkResult(
            name="exp_c/StatechartMachine",
            status="PASS",
            forward_works=True,
            backward_works=grad_norm > 0,
            grad_norm=grad_norm,
            loss=float(loss),
            time_ms=elapsed,
            param_count=params,
            notes="3 states, 3 transitions"
        ))
    except Exception as e:
        results.append(BenchmarkResult(
            name="exp_c/StatechartMachine",
            status="FAIL",
            forward_works=False,
            backward_works=False,
            grad_norm=0,
            loss=0,
            time_ms=0,
            param_count=0,
            error=str(e)
        ))

    return results


# =============================================================================
# Experiment D: Integrated
# =============================================================================

def benchmark_exp_d():
    """Benchmark exp_d: Integrated statecharts."""
    from differentiable.exp_d_integrated import MemoryAugmentedStatechart, TRMInspiredStatechart
    from differentiable.exp_b_memory import MemoryType

    results = []

    # Test MemoryAugmentedStatechart (uses step, not __call__)
    try:
        chart = MemoryAugmentedStatechart(
            num_states=3,
            context_dim=4,
            memory_type=MemoryType.ATTENTION,
            d_memory=16
        )

        start = time.perf_counter()
        batch_size = 2
        context = mx.random.normal((batch_size, 4))
        mem_state = chart.init_memory_state(batch_size)

        def loss_fn(model):
            new_config, new_mem, info = model.step(context, mem_state)
            return mx.mean(new_config[:, 1])

        loss, grads = nn.value_and_grad(chart, loss_fn)(chart)
        mx.eval(loss, grads)

        def sum_grad_sq(g):
            if isinstance(g, mx.array):
                return float(mx.sum(g * g))
            elif isinstance(g, dict):
                return sum(sum_grad_sq(v) for v in g.values())
            return 0.0

        grad_norm = sum_grad_sq(grads) ** 0.5
        elapsed = (time.perf_counter() - start) * 1000
        params = count_params_recursive(chart.parameters())

        results.append(BenchmarkResult(
            name="exp_d/MemoryAugmentedStatechart",
            status="PASS",
            forward_works=True,
            backward_works=grad_norm > 0,
            grad_norm=grad_norm,
            loss=float(loss),
            time_ms=elapsed,
            param_count=params,
            notes="Config + Memory integration"
        ))
    except Exception as e:
        results.append(BenchmarkResult(
            name="exp_d/MemoryAugmentedStatechart",
            status="FAIL",
            forward_works=False,
            backward_works=False,
            grad_norm=0,
            loss=0,
            time_ms=0,
            param_count=0,
            error=str(e)
        ))

    return results


# =============================================================================
# Experiment I: HTM Memory
# =============================================================================

def benchmark_exp_i():
    """Benchmark exp_i: HTM-inspired memory."""
    results = []

    try:
        from differentiable.exp_i_htm_memory import SDREncoder

        num_states = 8
        batch_size = 2

        # Test SDREncoder (uses num_states, not input_dim)
        encoder = SDREncoder(num_states=num_states, sdr_size=64, sparsity=0.1)

        def input_fn():
            return (mx.random.normal((batch_size, num_states)),)
        def loss_fn(out):
            return mx.mean(out)

        grad_norm, loss, time_ms = test_gradient_flow(encoder, input_fn, loss_fn)
        params = count_params_recursive(encoder.parameters())

        results.append(BenchmarkResult(
            name="exp_i/SDREncoder",
            status="PASS",
            forward_works=True,
            backward_works=grad_norm > 0,
            grad_norm=grad_norm,
            loss=loss,
            time_ms=time_ms,
            param_count=params,
            notes="Sparse distributed representation"
        ))
    except Exception as e:
        results.append(BenchmarkResult(
            name="exp_i/SDREncoder",
            status="FAIL",
            forward_works=False,
            backward_works=False,
            grad_norm=0,
            loss=0,
            time_ms=0,
            param_count=0,
            error=str(e)
        ))

    return results


# =============================================================================
# Experiment J: HOPE/Titans
# =============================================================================

def benchmark_exp_j():
    """Benchmark exp_j: HOPE statechart (Titans-inspired)."""
    results = []

    try:
        from experiments.exp_j_titans_hope.continuum_memory import HOPEStatechart, ContinuumMemorySystem

        hidden_dim = 32
        batch_size = 2
        seq_len = 16

        # Test HOPEStatechart
        model = HOPEStatechart(hidden_dim=hidden_dim, num_levels=3, num_heads=2)

        def input_fn():
            return (mx.random.normal((batch_size, seq_len, hidden_dim)),)
        def loss_fn(out):
            output, level_outputs = out
            return mx.mean(output ** 2)

        grad_norm, loss, time_ms = test_gradient_flow(model, input_fn, loss_fn)
        params = count_params_recursive(model.parameters())

        results.append(BenchmarkResult(
            name="exp_j/HOPEStatechart",
            status="PASS",
            forward_works=True,
            backward_works=grad_norm > 0,
            grad_norm=grad_norm,
            loss=loss,
            time_ms=time_ms,
            param_count=params,
            notes="Multi-frequency CMS + self-ref attention"
        ))
    except Exception as e:
        results.append(BenchmarkResult(
            name="exp_j/HOPEStatechart",
            status="FAIL",
            forward_works=False,
            backward_works=False,
            grad_norm=0,
            loss=0,
            time_ms=0,
            param_count=0,
            error=str(e)
        ))

    return results


# =============================================================================
# Experiment K: SAE Interpretability
# =============================================================================

def benchmark_exp_k():
    """Benchmark exp_k: Sparse Autoencoder for interpretability."""
    results = []

    try:
        from experiments.exp_k_sae_interpretability.sae import SparseAutoEncoder

        input_dim = 32
        latent_dim = 64
        batch_size = 8

        # Note: SparseAutoEncoder (with capital E) uses input_dim, latent_dim
        model = SparseAutoEncoder(input_dim=input_dim, latent_dim=latent_dim, sparsity_weight=0.01)

        def input_fn():
            return (mx.random.normal((batch_size, input_dim)),)
        def loss_fn(out):
            recon, latent, loss_val = out
            return loss_val

        grad_norm, loss, time_ms = test_gradient_flow(model, input_fn, loss_fn)
        params = count_params_recursive(model.parameters())

        results.append(BenchmarkResult(
            name="exp_k/SparseAutoEncoder",
            status="PASS",
            forward_works=True,
            backward_works=grad_norm > 0,
            grad_norm=grad_norm,
            loss=loss,
            time_ms=time_ms,
            param_count=params,
            notes="For state interpretation"
        ))
    except Exception as e:
        results.append(BenchmarkResult(
            name="exp_k/SparseAutoEncoder",
            status="FAIL",
            forward_works=False,
            backward_works=False,
            grad_norm=0,
            loss=0,
            time_ms=0,
            param_count=0,
            error=str(e)
        ))

    return results


# =============================================================================
# Experiment L: Statechart Layers
# =============================================================================

def benchmark_exp_l():
    """Benchmark exp_l: Statechart as neural network layer."""
    results = []

    try:
        from experiments.exp_l_statechart_layers.statechart_layer import StatechartLayer

        input_dim = 32
        num_states = 4
        batch_size = 2

        # Note: StatechartLayer uses input_dim, not hidden_dim
        layer = StatechartLayer(input_dim=input_dim, num_states=num_states)

        def input_fn():
            return (mx.random.normal((batch_size, input_dim)),)
        def loss_fn(out):
            return mx.mean(out ** 2)

        grad_norm, loss, time_ms = test_gradient_flow(layer, input_fn, loss_fn)
        params = count_params_recursive(layer.parameters())

        results.append(BenchmarkResult(
            name="exp_l/StatechartLayer",
            status="PASS",
            forward_works=True,
            backward_works=grad_norm > 0,
            grad_norm=grad_norm,
            loss=loss,
            time_ms=time_ms,
            param_count=params,
            notes="Statechart as NN layer"
        ))
    except Exception as e:
        results.append(BenchmarkResult(
            name="exp_l/StatechartLayer",
            status="FAIL",
            forward_works=False,
            backward_works=False,
            grad_norm=0,
            loss=0,
            time_ms=0,
            param_count=0,
            error=str(e)
        ))

    return results


# =============================================================================
# Experiment M: World Model
# =============================================================================

def benchmark_exp_m():
    """Benchmark exp_m: World model for statechart inference."""
    results = []

    try:
        from experiments.exp_m_world_model.world_model import StateDiscovery

        obs_dim = 16
        embed_dim = 32
        batch_size = 4

        # Note: WorldModel doesn't exist - use StateDiscovery
        model = StateDiscovery(obs_dim=obs_dim, embed_dim=embed_dim, max_states=8)

        def input_fn():
            return (mx.random.normal((batch_size, obs_dim)),)
        def loss_fn(out):
            state_ids, soft_assignments, embeddings = out
            return mx.mean(soft_assignments ** 2)

        # StateDiscovery uses assign_states, not __call__
        start = time.perf_counter()

        def loss_fn_model(m):
            obs = mx.random.normal((batch_size, obs_dim))
            state_ids, soft, emb = m.assign_states(obs)
            return mx.mean(soft ** 2)

        loss, grads = nn.value_and_grad(model, loss_fn_model)(model)
        mx.eval(loss, grads)

        def sum_grad_sq(g):
            if isinstance(g, mx.array):
                return float(mx.sum(g * g))
            elif isinstance(g, dict):
                return sum(sum_grad_sq(v) for v in g.values())
            return 0.0

        grad_norm = sum_grad_sq(grads) ** 0.5
        elapsed = (time.perf_counter() - start) * 1000
        params = count_params_recursive(model.parameters())

        results.append(BenchmarkResult(
            name="exp_m/StateDiscovery",
            status="PASS",
            forward_works=True,
            backward_works=grad_norm > 0,
            grad_norm=grad_norm,
            loss=float(loss),
            time_ms=elapsed,
            param_count=params,
            notes="Discover states from observations"
        ))
    except Exception as e:
        results.append(BenchmarkResult(
            name="exp_m/StateDiscovery",
            status="FAIL",
            forward_works=False,
            backward_works=False,
            grad_norm=0,
            loss=0,
            time_ms=0,
            param_count=0,
            error=str(e)
        ))

    return results


# =============================================================================
# Main
# =============================================================================

def run_all_benchmarks():
    """Run all benchmarks and print results."""
    print("=" * 80)
    print("DIFFERENTIABLE STATECHART BENCHMARK")
    print("=" * 80)
    print()

    all_results = []

    benchmarks = [
        ("Exp A: Soft Config", benchmark_exp_a),
        ("Exp B: Memory", benchmark_exp_b),
        ("Exp C: Transitions", benchmark_exp_c),
        ("Exp D: Integrated", benchmark_exp_d),
        ("Exp I: HTM Memory", benchmark_exp_i),
        ("Exp J: HOPE/Titans", benchmark_exp_j),
        ("Exp K: SAE", benchmark_exp_k),
        ("Exp L: Layers", benchmark_exp_l),
        ("Exp M: World Model", benchmark_exp_m),
    ]

    for name, benchmark_fn in benchmarks:
        print(f"Running {name}...")
        try:
            results = benchmark_fn()
            all_results.extend(results)
            for r in results:
                status = "OK" if r.status == "PASS" else "FAIL"
                print(f"  {r.name}: {status}")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Print summary table
    print()
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print()

    # Header
    print(f"{'Module':<35} | {'Status':<6} | {'Fwd':<4} | {'Bwd':<4} | {'Grad Norm':<10} | {'Params':<8} | {'Time':<8}")
    print("-" * 95)

    # Results
    pass_count = 0
    fail_count = 0

    for r in all_results:
        fwd = "Y" if r.forward_works else "N"
        bwd = "Y" if r.backward_works else "N"
        grad_str = f"{r.grad_norm:.4f}" if r.grad_norm > 0 else "0"
        param_str = f"{r.param_count:,}" if r.param_count > 0 else "-"
        time_str = f"{r.time_ms:.1f}ms" if r.time_ms > 0 else "-"

        print(f"{r.name:<35} | {r.status:<6} | {fwd:<4} | {bwd:<4} | {grad_str:<10} | {param_str:<8} | {time_str:<8}")

        if r.status == "PASS":
            pass_count += 1
        else:
            fail_count += 1

    print("-" * 95)
    print()

    # Summary stats
    total = pass_count + fail_count
    print(f"Total: {total} modules tested")
    print(f"  PASS: {pass_count} ({100*pass_count/total:.0f}%)")
    print(f"  FAIL: {fail_count} ({100*fail_count/total:.0f}%)")

    # What works vs doesn't
    print()
    print("=" * 80)
    print("WHAT WORKS")
    print("=" * 80)
    for r in all_results:
        if r.status == "PASS" and r.backward_works:
            print(f"  [OK] {r.name}: {r.notes}")

    print()
    print("=" * 80)
    print("WHAT DOESN'T WORK (or needs fixes)")
    print("=" * 80)
    for r in all_results:
        if r.status == "FAIL":
            print(f"  [FAIL] {r.name}: {r.error or 'Unknown error'}")
        elif r.status == "PASS" and not r.backward_works:
            print(f"  [NO GRAD] {r.name}: Forward works but no gradients")

    print()
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    mx.random.seed(42)
    run_all_benchmarks()
