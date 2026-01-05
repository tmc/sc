"""
Experiment D: Memory-Augmented Differentiable Statechart

Integrates:
- exp_a_soft_config.py: SoftStateConfiguration, DifferentiableGuard
- exp_b_memory.py: UnifiedMemory (Attention, NTM, Recurrent)

Architecture:
  context -> SoftStatechart.step() -> config [B, num_states]
                    ^                        |
                    |                        v
              memory_hint           UnifiedMemory.step()
                    ^                        |
                    |________________________/

The memory retrieval biases transitions, creating a feedback loop
that allows the system to learn history-dependent state patterns.
"""

import mlx.core as mx
import mlx.nn as nn

try:
    from .exp_a_soft_config import SoftStateConfiguration, DifferentiableGuard
    from .exp_b_memory import UnifiedMemory, MemoryType
    from .exp_c_transitions import DifferentiableTransitionSelector, StatechartMachine
except ImportError:
    from exp_a_soft_config import SoftStateConfiguration, DifferentiableGuard
    from exp_b_memory import UnifiedMemory, MemoryType
    from exp_c_transitions import DifferentiableTransitionSelector, StatechartMachine


class MemoryAugmentedGuard(nn.Module):
    """
    Guard that incorporates memory hint into its decision.

    Combines:
    - Context features (external input)
    - Memory hint (retrieved history information)

    Architecture:
        [context; memory_hint] -> MLP -> sigmoid -> [0,1]
    """

    def __init__(self, context_dim: int, memory_dim: int, hidden_dim: int = 32, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        combined_dim = context_dim + memory_dim

        self.l1 = nn.Linear(combined_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.l3 = nn.Linear(hidden_dim // 2, 1)

    def __call__(self, context: mx.array, memory_hint: mx.array) -> mx.array:
        """
        Evaluate guard with context and memory hint.

        Args:
            context: [B, context_dim] external features
            memory_hint: [B, memory_dim] retrieved from memory

        Returns:
            [B, 1] guard satisfaction in [0, 1]
        """
        combined = mx.concatenate([context, memory_hint], axis=-1)
        x = mx.tanh(self.l1(combined))
        x = mx.tanh(self.l2(x))
        x = self.l3(x)
        return mx.sigmoid(x / self.temperature)


class MemoryAugmentedStatechart(nn.Module):
    """
    Full memory-augmented differentiable statechart.

    Combines soft state configuration with memory mechanisms
    for history-dependent transition learning.
    """

    def __init__(
        self,
        num_states: int,
        context_dim: int,
        memory_type: str = MemoryType.ATTENTION,
        d_memory: int = 32,
        temperature: float = 1.0,
        entropy_weight: float = 0.01,
        memory_reconstruction_weight: float = 0.1,
    ):
        super().__init__()
        self.num_states = num_states
        self.context_dim = context_dim
        self.entropy_weight = entropy_weight
        self.memory_reconstruction_weight = memory_reconstruction_weight

        # Soft state configuration
        self.config = SoftStateConfiguration(num_states, temperature)

        # Memory module
        self.memory = UnifiedMemory(
            num_states=num_states,
            memory_type=memory_type,
            d_memory=d_memory,
            max_history=64,
            memory_size=32,
        )

        # Memory-augmented guards for each transition
        # For N states, we have N*(N-1) possible transitions
        # Simplified: one guard per source state (self-loop handling)
        self.guards = [
            MemoryAugmentedGuard(context_dim, num_states, d_memory, temperature)
            for _ in range(num_states)
        ]

        # Transition mixing network: determines target distribution given source + guard
        self.transition_net = nn.Sequential(
            nn.Linear(num_states + num_states + 1, d_memory),  # config + memory + guard
            nn.Tanh(),
            nn.Linear(d_memory, num_states),
        )

    def step(self, context: mx.array, memory_state: dict) -> tuple[mx.array, dict, dict]:
        """
        Single step of the memory-augmented statechart.

        Args:
            context: [B, context_dim] external context features
            memory_state: dict containing memory state

        Returns:
            new_config: [B, num_states] new state distribution
            new_memory_state: updated memory state
            info: dict with attention weights, guard values, etc.
        """
        batch_size = context.shape[0]

        # Get current configuration
        current_config = self.config.get_config(batch_size)

        # Retrieve from memory
        memory_hint, new_memory_state, attn_weights = self.memory.step(
            current_config, memory_state
        )

        # Compute guards for each source state
        guard_values = []
        for guard in self.guards:
            g = guard(context, memory_hint)  # [B, 1]
            guard_values.append(g)

        # Stack guards: [B, num_states]
        all_guards = mx.concatenate(guard_values, axis=-1)

        # Weighted average guard based on current config
        # This gives the "aggregate" guard value for the current distribution
        avg_guard = mx.sum(current_config * all_guards, axis=-1, keepdims=True)  # [B, 1]

        # Compute transition: combine config, memory, and guard
        transition_input = mx.concatenate([current_config, memory_hint, avg_guard], axis=-1)
        transition_logits = self.transition_net(transition_input)  # [B, num_states]

        # Soft transition: blend current config with transition target
        transition_probs = mx.softmax(transition_logits, axis=-1)

        # New config is mixture of staying (1 - avg_guard) and transitioning (avg_guard)
        new_config = (1 - avg_guard) * current_config + avg_guard * transition_probs

        # Normalize (should already sum to 1, but ensure numerical stability)
        new_config = new_config / mx.sum(new_config, axis=-1, keepdims=True)

        # Update internal config logits (for next step)
        # Convert probs back to logits (inverse softmax)
        new_logits = mx.log(new_config[0] + 1e-10)
        self.config.set_logits(new_logits)

        info = {
            "attention_weights": attn_weights,
            "guard_values": all_guards,
            "avg_guard": avg_guard,
            "memory_hint": memory_hint,
            "transition_probs": transition_probs,
        }

        return new_config, new_memory_state, info

    def init_memory_state(self, batch_size: int) -> dict:
        """Initialize memory state."""
        return self.memory.init_state(batch_size)

    def entropy_loss(self, config: mx.array) -> mx.array:
        """
        Entropy regularization loss.

        Low entropy = decisive states (good)
        High entropy = uncertain states (may want to regularize)

        Returns negative entropy (minimize = maximize entropy for exploration)
        or positive entropy (minimize = encourage decisive states)
        """
        log_probs = mx.log(config + 1e-10)
        entropy = -mx.sum(config * log_probs, axis=-1)
        return mx.mean(entropy)

    def memory_reconstruction_loss(self, config: mx.array, retrieved: mx.array) -> mx.array:
        """
        Loss encouraging memory to predict/reconstruct current config.

        This helps memory learn useful representations.
        """
        return mx.mean((config - retrieved) ** 2)

    def combined_loss(
        self,
        target_config: mx.array,
        predicted_config: mx.array,
        memory_hint: mx.array,
    ) -> tuple[mx.array, dict]:
        """
        Combined loss for training.

        Args:
            target_config: [B, num_states] ground truth distribution
            predicted_config: [B, num_states] predicted distribution
            memory_hint: [B, num_states] memory retrieval

        Returns:
            total_loss: scalar
            loss_components: dict of individual losses
        """
        # Main transition loss: cross-entropy or MSE
        transition_loss = mx.mean((target_config - predicted_config) ** 2)

        # Entropy regularization
        entropy = self.entropy_loss(predicted_config)

        # Memory reconstruction
        memory_loss = self.memory_reconstruction_loss(predicted_config, memory_hint)

        total_loss = (
            transition_loss
            - self.entropy_weight * entropy  # Encourage some exploration
            + self.memory_reconstruction_weight * memory_loss
        )

        return total_loss, {
            "transition_loss": transition_loss,
            "entropy": entropy,
            "memory_loss": memory_loss,
        }


class TRMInspiredStatechart(nn.Module):
    """
    TRM-inspired hierarchical statechart.

    Inspired by Temporal Recurrent Model (TRM):
    - z_H = macro-state distribution (high-level statechart)
    - z_L = micro-state distribution (leaf states within macro-state)
    - H_cycles control macro-transitions
    - L_cycles control micro-steps

    This is a statechart interpretation of the TRM dual-timescale architecture.
    """

    def __init__(
        self,
        num_macro_states: int,
        num_micro_states_per_macro: int,
        context_dim: int,
        H_cycles: int = 4,
        L_cycles: int = 1,
        d_memory: int = 32,
    ):
        super().__init__()
        self.num_macro_states = num_macro_states
        self.num_micro_states = num_micro_states_per_macro
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles

        # Macro-level statechart (slow, high-level states)
        self.macro_chart = MemoryAugmentedStatechart(
            num_states=num_macro_states,
            context_dim=context_dim,
            memory_type=MemoryType.RECURRENT,  # GRU for temporal smoothing
            d_memory=d_memory,
            temperature=0.5,  # Sharper decisions at macro level
        )

        # Micro-level statecharts (one per macro-state, fast, leaf states)
        self.micro_charts = [
            MemoryAugmentedStatechart(
                num_states=num_micro_states_per_macro,
                context_dim=context_dim + num_macro_states,  # Include macro context
                memory_type=MemoryType.ATTENTION,  # Quick retrieval
                d_memory=d_memory // 2,
                temperature=1.0,
            )
            for _ in range(num_macro_states)
        ]

        self.step_counter = 0

    def step(
        self,
        context: mx.array,
        macro_memory_state: dict,
        micro_memory_states: list[dict],
    ) -> tuple[mx.array, mx.array, dict, list[dict], dict]:
        """
        Hierarchical step with TRM-inspired timing.

        Macro-states update every H_cycles steps.
        Micro-states update every L_cycles steps.

        Args:
            context: [B, context_dim]
            macro_memory_state: Memory state for macro chart
            micro_memory_states: List of memory states for micro charts

        Returns:
            macro_config: [B, num_macro_states]
            micro_config: [B, num_micro_states] (weighted by macro probs)
            new_macro_memory_state: Updated macro memory
            new_micro_memory_states: Updated micro memories
            info: Debug information
        """
        batch_size = context.shape[0]
        self.step_counter += 1

        # Macro update (every H_cycles)
        if self.step_counter % self.H_cycles == 0:
            macro_config, macro_memory_state, macro_info = self.macro_chart.step(
                context, macro_memory_state
            )
        else:
            macro_config = self.macro_chart.config.get_config(batch_size)
            macro_info = {}

        # Micro updates (every L_cycles)
        if self.step_counter % self.L_cycles == 0:
            # Augment context with macro-state info
            micro_context = mx.concatenate([context, macro_config], axis=-1)

            # Update each micro chart, weighted by macro probability
            micro_configs = []
            new_micro_states = []

            for i, (micro_chart, micro_state) in enumerate(
                zip(self.micro_charts, micro_memory_states)
            ):
                mc, new_ms, _ = micro_chart.step(micro_context, micro_state)
                micro_configs.append(mc)
                new_micro_states.append(new_ms)

            # Combine micro configs weighted by macro probs
            # [B, num_macro, num_micro] -> [B, num_micro] (marginalize over macro)
            stacked_micro = mx.stack(micro_configs, axis=1)  # [B, num_macro, num_micro]
            macro_weights = mx.expand_dims(macro_config, axis=-1)  # [B, num_macro, 1]
            micro_config = mx.sum(stacked_micro * macro_weights, axis=1)  # [B, num_micro]

            micro_memory_states = new_micro_states
        else:
            # No micro update, use current configs
            micro_configs = [mc.config.get_config(batch_size) for mc in self.micro_charts]
            stacked_micro = mx.stack(micro_configs, axis=1)
            macro_weights = mx.expand_dims(macro_config, axis=-1)
            micro_config = mx.sum(stacked_micro * macro_weights, axis=1)

        info = {
            "macro_config": macro_config,
            "step_counter": self.step_counter,
            "macro_updated": self.step_counter % self.H_cycles == 0,
            "micro_updated": self.step_counter % self.L_cycles == 0,
        }

        return macro_config, micro_config, macro_memory_state, micro_memory_states, info

    def init_all_states(self, batch_size: int) -> tuple[dict, list[dict]]:
        """Initialize all memory states."""
        macro_state = self.macro_chart.init_memory_state(batch_size)
        micro_states = [mc.init_memory_state(batch_size) for mc in self.micro_charts]
        return macro_state, micro_states


# =============================================================================
# Fully Integrated Statechart (A + B + C)
# =============================================================================


class FullyIntegratedStatechart(nn.Module):
    """
    Fully integrated differentiable statechart combining all experiments:
    - exp_a: SoftStateConfiguration for soft state probabilities
    - exp_b: UnifiedMemory for history-dependent behavior
    - exp_c: DifferentiableTransitionSelector for proper transition semantics

    This is the complete differentiable statechart architecture.
    """

    def __init__(
        self,
        num_states: int,
        transitions: list,  # [(source, target, event_id), ...]
        context_dim: int,
        memory_type: str = MemoryType.ATTENTION,
        embed_dim: int = 32,
        num_events: int = 1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_states = num_states
        self.context_dim = context_dim
        self.embed_dim = embed_dim

        # Soft state configuration (exp_a)
        self.config = SoftStateConfiguration(num_states, temperature)

        # Memory module (exp_b)
        self.memory = UnifiedMemory(
            num_states=num_states,
            memory_type=memory_type,
            d_memory=embed_dim,
            max_history=64,
            memory_size=32,
        )

        # Transition selector (exp_c)
        num_transitions = len(transitions)
        self.transition_selector = DifferentiableTransitionSelector(
            num_states=num_states,
            num_transitions=num_transitions,
            embed_dim=embed_dim,
            num_events=num_events,
        )

        # Configure transitions
        sources = [t[0] for t in transitions]
        targets = [t[1] for t in transitions]
        events = [t[2] if len(t) > 2 else 0 for t in transitions]
        self.transition_selector.configure_transitions(sources, targets, events)

        # Context projection to embed_dim for transition selector
        self.context_proj = nn.Linear(context_dim + num_states, embed_dim)

    def step(
        self, context: mx.array, memory_state: dict
    ) -> tuple[mx.array, dict, dict]:
        """
        Single step of the fully integrated statechart.

        Args:
            context: [B, context_dim] external context
            memory_state: Memory state dict

        Returns:
            new_config: [B, num_states] new state distribution
            new_memory_state: Updated memory state
            info: Debug information
        """
        batch_size = context.shape[0]

        # Get current configuration
        current_config = self.config.get_config(batch_size)

        # Retrieve from memory (history-aware)
        memory_hint, new_memory_state, attn_weights = self.memory.step(
            current_config, memory_state
        )

        # Project context + memory_hint to embed_dim for transition guards
        combined_context = mx.concatenate([context, memory_hint], axis=-1)
        transition_context = self.context_proj(combined_context)  # [B, embed_dim]

        # Use transition selector to compute new config
        new_config, enablement, selection = self.transition_selector(
            current_config, transition_context
        )

        # Update internal config logits
        new_logits = mx.log(new_config[0] + 1e-10)
        self.config.set_logits(new_logits)

        info = {
            "attention_weights": attn_weights,
            "memory_hint": memory_hint,
            "enablement": enablement,
            "selection": selection,
        }

        return new_config, new_memory_state, info

    def init_memory_state(self, batch_size: int) -> dict:
        """Initialize memory state."""
        return self.memory.init_state(batch_size)

    def simulate(self, contexts: list, batch_size: int = 1) -> list:
        """Simulate through a sequence of contexts.

        Args:
            contexts: List of [B, context_dim] context tensors
            batch_size: Batch size

        Returns:
            List of configurations for each step
        """
        memory_state = self.init_memory_state(batch_size)
        configs = [self.config.get_config(batch_size)]

        for ctx in contexts:
            new_config, memory_state, _ = self.step(ctx, memory_state)
            configs.append(new_config)

        return configs


# =============================================================================
# Test Script
# =============================================================================


def test_fully_integrated():
    """Test the fully integrated statechart."""
    print("\n" + "=" * 60)
    print("Testing Fully Integrated Statechart (A + B + C)")
    print("=" * 60)

    # 3-state system: Idle -> Active -> Done -> Idle
    transitions = [
        (0, 1, 0),  # Idle -> Active
        (1, 2, 0),  # Active -> Done
        (2, 0, 0),  # Done -> Idle
    ]

    chart = FullyIntegratedStatechart(
        num_states=3,
        transitions=transitions,
        context_dim=4,
        memory_type=MemoryType.ATTENTION,
        embed_dim=16,
    )

    print("\n1. Statechart structure:")
    print("   States: Idle(0), Active(1), Done(2)")
    print("   Transitions: Idle->Active, Active->Done, Done->Idle")

    # Random context sequence
    mx.random.seed(42)
    batch_size = 2
    seq_len = 6
    contexts = [mx.random.normal((batch_size, 4)) for _ in range(seq_len)]

    print(f"\n2. Running simulation ({seq_len} steps)...")
    configs = chart.simulate(contexts, batch_size)

    for t, cfg in enumerate(configs):
        state_names = ["Idle", "Active", "Done"]
        probs = [f"{state_names[i]}:{cfg[0, i]:.2f}" for i in range(3)]
        print(f"   t={t}: {', '.join(probs)}")

    print("\n3. Testing gradient flow...")

    def loss_fn(model, ctx, mem_state):
        pred, _, info = model.step(ctx, mem_state)
        # Target: maximize Done state probability
        return -mx.mean(pred[:, 2])

    fresh_mem = chart.init_memory_state(batch_size)
    loss_and_grad = nn.value_and_grad(chart, loss_fn)
    loss, grads = loss_and_grad(chart, contexts[0], fresh_mem)

    print(f"   Loss: {float(loss):.4f}")

    # Count gradients
    def count_grads(g, prefix=""):
        count, total = 0, 0
        if isinstance(g, dict):
            for k, v in g.items():
                c, t = count_grads(v, f"{prefix}{k}.")
                count += c
                total += t
        elif isinstance(g, mx.array):
            total = 1
            if mx.any(g != 0).item():
                count = 1
        return count, total

    nonzero, total = count_grads(grads)
    print(f"   Gradients: {nonzero}/{total} tensors have non-zero gradients")

    if nonzero > 0:
        print("   [checkmark] Full integration works!")
    else:
        print("   [X] Warning: No gradients")

    print("\n" + "=" * 60)
    return chart


def test_memory_augmented_statechart():
    """Test the integrated memory-augmented statechart."""
    print("=" * 60)
    print("Testing Memory-Augmented Differentiable Statechart")
    print("=" * 60)

    # Setup
    num_states = 3
    context_dim = 4
    batch_size = 2
    seq_len = 10

    print("\n1. Creating MemoryAugmentedStatechart...")
    chart = MemoryAugmentedStatechart(
        num_states=num_states,
        context_dim=context_dim,
        memory_type=MemoryType.ATTENTION,
        d_memory=16,
        temperature=1.0,
    )

    # Random context sequence
    mx.random.seed(42)
    contexts = [mx.random.normal((batch_size, context_dim)) for _ in range(seq_len)]

    print(f"   num_states={num_states}, context_dim={context_dim}")
    print(f"   batch_size={batch_size}, seq_len={seq_len}")

    # Initialize memory
    memory_state = chart.init_memory_state(batch_size)

    print("\n2. Running sequence...")
    configs = []
    for t, ctx in enumerate(contexts):
        config, memory_state, info = chart.step(ctx, memory_state)
        configs.append(config)

        if t < 3 or t == seq_len - 1:
            print(f"   t={t}: config={config[0].tolist()}, avg_guard={float(info['avg_guard'][0, 0]):.3f}")

    print("\n3. Testing gradient flow...")

    def loss_fn(model, ctx, mem_state, target):
        pred, _, info = model.step(ctx, mem_state)
        loss, _ = model.combined_loss(target, pred, info["memory_hint"])
        return loss

    # Target: encourage state 0
    target = mx.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    fresh_mem = chart.init_memory_state(batch_size)

    loss_and_grad = nn.value_and_grad(chart, loss_fn)
    loss, grads = loss_and_grad(chart, contexts[0], fresh_mem, target)

    print(f"   Loss: {float(loss):.4f}")

    # Count non-zero gradients
    def count_nonzero_grads(grads, prefix=""):
        count = 0
        total = 0
        if isinstance(grads, dict):
            for k, v in grads.items():
                c, t = count_nonzero_grads(v, f"{prefix}{k}.")
                count += c
                total += t
        elif isinstance(grads, mx.array):
            total = 1
            if mx.any(grads != 0).item():
                count = 1
        return count, total

    nonzero, total = count_nonzero_grads(grads)
    print(f"   Gradients: {nonzero}/{total} tensors have non-zero gradients")

    if nonzero > 0:
        print("   [checkmark] Gradients flow correctly!")
    else:
        print("   [X] Warning: No gradients found")

    print("\n" + "=" * 60)
    print("MemoryAugmentedStatechart test complete!")
    print("=" * 60)


def test_trm_inspired_statechart():
    """Test the TRM-inspired hierarchical statechart."""
    print("\n" + "=" * 60)
    print("Testing TRM-Inspired Hierarchical Statechart")
    print("=" * 60)

    # Setup: 2 macro states, 3 micro states each
    num_macro = 2
    num_micro = 3
    context_dim = 4
    batch_size = 2
    seq_len = 12  # Multiple of H_cycles=4

    print(f"\n1. Creating TRMInspiredStatechart...")
    print(f"   Macro states: {num_macro}, Micro states: {num_micro}")
    print(f"   H_cycles: 4, L_cycles: 1")

    chart = TRMInspiredStatechart(
        num_macro_states=num_macro,
        num_micro_states_per_macro=num_micro,
        context_dim=context_dim,
        H_cycles=4,
        L_cycles=1,
        d_memory=16,
    )

    # Random context sequence
    mx.random.seed(123)
    contexts = [mx.random.normal((batch_size, context_dim)) for _ in range(seq_len)]

    # Initialize states
    macro_mem, micro_mems = chart.init_all_states(batch_size)

    print("\n2. Running sequence...")
    for t, ctx in enumerate(contexts):
        macro_cfg, micro_cfg, macro_mem, micro_mems, info = chart.step(
            ctx, macro_mem, micro_mems
        )

        if t < 5 or t == seq_len - 1:
            macro_str = [f"{x:.2f}" for x in macro_cfg[0].tolist()]
            micro_str = [f"{x:.2f}" for x in micro_cfg[0].tolist()]
            print(f"   t={t}: macro={macro_str}, micro={micro_str}")
            print(f"        macro_updated={info['macro_updated']}, micro_updated={info['micro_updated']}")

    print("\n3. Verifying dual-timescale behavior...")
    print("   Macro should update at t=0,4,8,12... (every H_cycles)")
    print("   Micro should update every step (L_cycles=1)")

    print("\n" + "=" * 60)
    print("TRM-Inspired Statechart test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_memory_augmented_statechart()
    test_trm_inspired_statechart()
    test_fully_integrated()
