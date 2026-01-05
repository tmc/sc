"""
Level 4: Context-Dependent Guards

Tests whether model can learn guards that depend on external context.

Setup:
    States: {Locked, Unlocked}
    Events: TRY_OPEN
    Guard: key_present(context) -> bool
    Transitions:
        TRY_OPEN [key_present]: Locked→Unlocked
        TRY_OPEN [!key_present]: Locked→Locked (no change)

Context: 8-dim vector, key presence encoded in first 4 dims
Task: Predict next state given (state, event, context)
Pass Criterion: >95% accuracy on guard evaluation
Baseline: Always-true guard = ~50%
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


STATES = ["Locked", "Unlocked"]
EVENTS = ["TRY_OPEN", "LOCK"]

# Key is present if sum of first 4 context dims > 2.0
def key_present(context: mx.array) -> mx.array:
    """Check if key is present in context.

    Args:
        context: [B, 8] context vectors

    Returns:
        [B] boolean array (as float 0/1)
    """
    key_signal = mx.sum(context[:, :4], axis=-1)
    return (key_signal > 2.0).astype(mx.float32)


def get_next_state(state_idx: int, event_idx: int, context: mx.array) -> mx.array:
    """Ground truth transition with guard.

    Args:
        state_idx: 0=Locked, 1=Unlocked
        event_idx: 0=TRY_OPEN, 1=LOCK
        context: [B, 8] context

    Returns:
        [B] next state indices
    """
    B = context.shape[0]
    state_arr = mx.full((B,), state_idx, dtype=mx.int32)
    event_arr = mx.full((B,), event_idx, dtype=mx.int32)

    # Start with current state
    next_state = state_arr.astype(mx.float32)

    # TRY_OPEN from Locked: check guard
    mask_try_open_locked = (state_arr == 0) & (event_arr == 0)  # Locked + TRY_OPEN
    has_key = key_present(context)
    # If Locked + TRY_OPEN + key_present -> Unlocked (1)
    next_state = mx.where(mask_try_open_locked & (has_key > 0.5), 1.0, next_state)

    # LOCK from Unlocked -> Locked
    mask_lock_unlocked = (state_arr == 1) & (event_arr == 1)  # Unlocked + LOCK
    next_state = mx.where(mask_lock_unlocked, 0.0, next_state)

    return next_state.astype(mx.int32)


def generate_training_data(num_samples: int = 2000):
    """Generate training data with varied contexts.

    Returns:
        states: [N] current state indices
        events: [N] event indices
        contexts: [N, 8] context vectors
        targets: [N] target state indices
    """
    states = []
    events = []
    contexts = []
    targets = []

    for _ in range(num_samples):
        # Random state, event, context
        state_idx = int(mx.random.randint(0, len(STATES), ()))
        event_idx = int(mx.random.randint(0, len(EVENTS), ()))

        # Context with varied key presence
        # Make key present 50% of time
        if mx.random.uniform() > 0.5:
            # Key present: high values in first 4 dims
            context = mx.concatenate([
                mx.random.uniform(low=0.5, high=1.0, shape=(4,)),
                mx.random.uniform(low=0.0, high=0.5, shape=(4,)),
            ])
        else:
            # Key absent: low values in first 4 dims
            context = mx.concatenate([
                mx.random.uniform(low=0.0, high=0.4, shape=(4,)),
                mx.random.uniform(low=0.0, high=1.0, shape=(4,)),
            ])

        # Get target
        context_batch = context[None, :]  # [1, 8]
        target = get_next_state(state_idx, event_idx, context_batch)

        states.append(state_idx)
        events.append(event_idx)
        contexts.append(context)
        targets.append(int(target[0]))

    return (
        mx.array(states, dtype=mx.int32),
        mx.array(events, dtype=mx.int32),
        mx.stack(contexts),
        mx.array(targets, dtype=mx.int32),
    )


class GuardedTransitionModel(nn.Module):
    """Model that learns context-dependent guards."""

    def __init__(self, num_states: int = 2, num_events: int = 2,
                 context_dim: int = 8, embed_dim: int = 16):
        super().__init__()

        self.state_embed = nn.Embedding(num_states, embed_dim)
        self.event_embed = nn.Embedding(num_events, embed_dim)

        # Guard network: context -> guard activation
        self.guard_net = nn.Sequential(
            nn.Linear(context_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # Transition predictor: (state, event, guard) -> next_state
        self.transition_net = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, 32),
            nn.ReLU(),
            nn.Linear(32, num_states),
        )

    def __call__(self, states: mx.array, events: mx.array,
                 contexts: mx.array) -> tuple:
        """Predict next state.

        Args:
            states: [B] current state indices
            events: [B] event indices
            contexts: [B, context_dim] context vectors

        Returns:
            logits: [B, num_states] next state logits
            guard_act: [B, 1] guard activation (sigmoid)
        """
        state_emb = self.state_embed(states)  # [B, embed_dim]
        event_emb = self.event_embed(events)  # [B, embed_dim]

        # Guard activation
        guard_logit = self.guard_net(contexts)  # [B, 1]
        guard_act = mx.sigmoid(guard_logit)

        # Combine for prediction
        combined = mx.concatenate([state_emb, event_emb, guard_act], axis=-1)
        logits = self.transition_net(combined)

        return logits, guard_act


def compute_accuracy(model, states, events, contexts, targets):
    """Compute prediction accuracy."""
    logits, _ = model(states, events, contexts)
    preds = mx.argmax(logits, axis=-1)
    return float(mx.sum(preds == targets)) / targets.shape[0]


def compute_guard_accuracy(model, states, events, contexts):
    """Compute guard prediction accuracy vs ground truth."""
    _, guard_act = model(states, events, contexts)
    pred_key = (guard_act[:, 0] > 0.5).astype(mx.float32)
    true_key = key_present(contexts)
    return float(mx.sum(pred_key == true_key)) / contexts.shape[0]


def train(num_epochs: int = 200, num_samples: int = 3000, lr: float = 0.01):
    """Train guard model."""
    print("=" * 60)
    print("Level 4: Context-Dependent Guards")
    print("=" * 60)

    print(f"\nStates: {STATES}")
    print(f"Events: {EVENTS}")
    print(f"Guard: key_present if sum(context[:4]) > 2.0")

    # Generate data
    train_states, train_events, train_contexts, train_targets = generate_training_data(num_samples)
    test_states, test_events, test_contexts, test_targets = generate_training_data(500)

    print(f"\nTraining samples: {num_samples}")
    print(f"Test samples: 500")

    # Baseline: always-true guard
    baseline_key = key_present(test_contexts)
    baseline_acc = float(mx.mean(baseline_key))  # ~50% have key
    print(f"Baseline (always-true guard): {baseline_acc:.1%}")

    # Create model
    model = GuardedTransitionModel()
    total_params = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"Model parameters: {total_params}")

    # Optimizer
    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(model):
        logits, guard_act = model(train_states, train_events, train_contexts)
        # Cross-entropy loss
        log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)
        ce_loss = -mx.mean(log_probs[mx.arange(train_targets.shape[0]), train_targets])

        # Guard supervision loss
        true_key = key_present(train_contexts)
        guard_loss = mx.mean((guard_act[:, 0] - true_key) ** 2)

        return ce_loss + 0.5 * guard_loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training
    print("\nTraining...")
    print("-" * 60)

    for epoch in range(num_epochs):
        loss, grads = loss_and_grad(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if epoch % 40 == 0 or epoch == num_epochs - 1:
            state_acc = compute_accuracy(model, test_states, test_events, test_contexts, test_targets)
            guard_acc = compute_guard_accuracy(model, test_states, test_events, test_contexts)
            print(f"Epoch {epoch:3d}: loss={float(loss):.4f}, state_acc={state_acc:.1%}, guard_acc={guard_acc:.1%}")

            if state_acc >= 0.95 and guard_acc >= 0.95:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Final evaluation
    final_state_acc = compute_accuracy(model, test_states, test_events, test_contexts, test_targets)
    final_guard_acc = compute_guard_accuracy(model, test_states, test_events, test_contexts)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"State prediction accuracy: {final_state_acc:.1%}")
    print(f"Guard prediction accuracy: {final_guard_acc:.1%}")

    # Test specific scenarios
    print("\n" + "-" * 60)
    print("Scenario tests:")
    print("-" * 60)

    scenarios = [
        ("Locked", "TRY_OPEN", "key present", mx.array([[0.8, 0.9, 0.7, 0.8, 0.1, 0.1, 0.1, 0.1]]), "Unlocked"),
        ("Locked", "TRY_OPEN", "no key", mx.array([[0.1, 0.2, 0.1, 0.2, 0.5, 0.5, 0.5, 0.5]]), "Locked"),
        ("Unlocked", "LOCK", "any context", mx.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]), "Locked"),
        ("Unlocked", "TRY_OPEN", "any context", mx.array([[0.8, 0.9, 0.7, 0.8, 0.1, 0.1, 0.1, 0.1]]), "Unlocked"),
    ]

    correct = 0
    for state, event, ctx_desc, ctx, expected in scenarios:
        s = mx.array([STATES.index(state)])
        e = mx.array([EVENTS.index(event)])
        logits, guard = model(s, e, ctx)
        pred_idx = int(mx.argmax(logits, axis=-1))
        pred = STATES[pred_idx]
        guard_val = float(guard[0, 0])

        match = pred == expected
        if match:
            correct += 1
        status = "✓" if match else "✗"
        print(f"  {state} + {event} ({ctx_desc}) -> {pred} (expected: {expected}, guard={guard_val:.2f}) {status}")

    scenario_acc = correct / len(scenarios)
    print(f"\nScenario accuracy: {correct}/{len(scenarios)} = {scenario_acc:.1%}")

    # Pass criterion
    passed = final_state_acc >= 0.95 and final_guard_acc >= 0.95
    status = "PASS" if passed else "FAIL"
    print(f"\nPass criterion (>95%): {status}")

    return model, final_state_acc, passed


if __name__ == "__main__":
    mx.random.seed(42)
    Path(__file__).parent.mkdir(parents=True, exist_ok=True)
    model, accuracy, passed = train()
    exit(0 if passed else 1)
