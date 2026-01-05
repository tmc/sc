"""
Level 3: Event-Dependent Transitions

Tests whether model can learn which event triggers which transition.

Setup:
    States: {Idle, Running, Paused}
    Events: START, STOP, PAUSE, RESUME
    Transitions:
      START: Idle→Running
      STOP: Running→Idle, Paused→Idle
      PAUSE: Running→Paused
      RESUME: Paused→Running

Task: Predict next state given (state, event) pair
Pass Criterion: >98% accuracy
Baseline: Ignore event, guess most common = ~40%
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from differentiable.exp_c_transitions import StatechartMachine


# State and event definitions
STATES = ["Idle", "Running", "Paused"]
EVENTS = ["START", "STOP", "PAUSE", "RESUME"]

# Transition table: (source, event) -> target
# None means no transition (stay in same state)
TRANSITION_TABLE = {
    ("Idle", "START"): "Running",
    ("Idle", "STOP"): "Idle",  # No-op
    ("Idle", "PAUSE"): "Idle",  # No-op
    ("Idle", "RESUME"): "Idle",  # No-op
    ("Running", "START"): "Running",  # No-op
    ("Running", "STOP"): "Idle",
    ("Running", "PAUSE"): "Paused",
    ("Running", "RESUME"): "Running",  # No-op
    ("Paused", "START"): "Paused",  # No-op
    ("Paused", "STOP"): "Idle",
    ("Paused", "PAUSE"): "Paused",  # No-op
    ("Paused", "RESUME"): "Running",
}


def state_to_idx(state: str) -> int:
    return STATES.index(state)


def idx_to_state(idx: int) -> str:
    return STATES[idx]


def event_to_idx(event: str) -> int:
    return EVENTS.index(event)


def get_next_state(state: str, event: str) -> str:
    """Ground truth transition function."""
    return TRANSITION_TABLE.get((state, event), state)


def generate_training_data(num_samples: int = 1000):
    """Generate (state, event) -> next_state training data.

    Returns:
        states: [N] state indices
        events: [N] event indices
        targets: [N] target state indices
    """
    states = []
    events = []
    targets = []

    for _ in range(num_samples):
        # Random state and event
        state_idx = int(mx.random.randint(0, len(STATES), ()))
        event_idx = int(mx.random.randint(0, len(EVENTS), ()))

        state = STATES[state_idx]
        event = EVENTS[event_idx]
        next_state = get_next_state(state, event)
        target_idx = state_to_idx(next_state)

        states.append(state_idx)
        events.append(event_idx)
        targets.append(target_idx)

    return (
        mx.array(states, dtype=mx.int32),
        mx.array(events, dtype=mx.int32),
        mx.array(targets, dtype=mx.int32),
    )


class EventTransitionModel(nn.Module):
    """Model that learns event-dependent transitions.

    Architecture:
        - State embedding
        - Event embedding
        - MLP to predict next state distribution
    """

    def __init__(self, num_states: int = 3, num_events: int = 4, embed_dim: int = 16):
        super().__init__()
        self.num_states = num_states
        self.num_events = num_events

        # Embeddings
        self.state_embed = nn.Embedding(num_states, embed_dim)
        self.event_embed = nn.Embedding(num_events, embed_dim)

        # MLP: concat(state_embed, event_embed) -> next_state
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, num_states),
        )

    def __call__(self, states: mx.array, events: mx.array) -> mx.array:
        """Predict next state distribution.

        Args:
            states: [B] current state indices
            events: [B] event indices

        Returns:
            [B, num_states] logits for next state
        """
        state_emb = self.state_embed(states)  # [B, embed_dim]
        event_emb = self.event_embed(events)  # [B, embed_dim]

        combined = mx.concatenate([state_emb, event_emb], axis=-1)  # [B, embed_dim*2]
        logits = self.mlp(combined)  # [B, num_states]

        return logits


def compute_accuracy(model, states, events, targets):
    """Compute prediction accuracy."""
    logits = model(states, events)
    preds = mx.argmax(logits, axis=-1)
    correct = mx.sum(preds == targets)
    return float(correct) / targets.shape[0]


def compute_baseline_accuracy(targets):
    """Baseline: always predict most common state."""
    # Count occurrences
    counts = [0] * len(STATES)
    for t in targets.tolist():
        counts[t] += 1
    most_common = max(range(len(counts)), key=lambda i: counts[i])
    correct = sum(1 for t in targets.tolist() if t == most_common)
    return correct / len(targets.tolist()), STATES[most_common]


def train(num_epochs: int = 100, num_samples: int = 2000, lr: float = 0.01):
    """Train event transition model."""
    print("=" * 60)
    print("Level 3: Event-Dependent Transitions")
    print("=" * 60)

    # Generate data
    print("\nGenerating training data...")
    train_states, train_events, train_targets = generate_training_data(num_samples)
    test_states, test_events, test_targets = generate_training_data(500)

    print(f"Training samples: {num_samples}")
    print(f"Test samples: 500")

    # Compute baseline
    baseline_acc, baseline_state = compute_baseline_accuracy(test_targets)
    print(f"\nBaseline (always predict '{baseline_state}'): {baseline_acc:.1%}")

    # Create model
    model = EventTransitionModel(num_states=len(STATES), num_events=len(EVENTS))

    # Count params
    total_params = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"Model parameters: {total_params}")

    # Optimizer
    optimizer = optim.Adam(learning_rate=lr)

    # Loss function
    def loss_fn(model):
        logits = model(train_states, train_events)
        # Cross-entropy loss
        log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)
        loss = -mx.mean(log_probs[mx.arange(train_targets.shape[0]), train_targets])
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training loop
    print("\nTraining...")
    print("-" * 60)

    for epoch in range(num_epochs):
        loss, grads = loss_and_grad(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if epoch % 20 == 0 or epoch == num_epochs - 1:
            train_acc = compute_accuracy(model, train_states, train_events, train_targets)
            test_acc = compute_accuracy(model, test_states, test_events, test_targets)
            print(f"Epoch {epoch:3d}: loss={float(loss):.4f}, train_acc={train_acc:.1%}, test_acc={test_acc:.1%}")

            if test_acc >= 0.98:
                print(f"\nEarly stopping: reached 98% accuracy at epoch {epoch}")
                break

    # Final evaluation
    final_acc = compute_accuracy(model, test_states, test_events, test_targets)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Baseline accuracy: {baseline_acc:.1%}")
    print(f"Model accuracy: {final_acc:.1%}")
    print(f"Improvement: {final_acc/baseline_acc:.1f}x baseline")

    # Pass criterion
    passed = final_acc >= 0.98
    status = "PASS" if passed else "FAIL"
    print(f"\nPass criterion (>98%): {status}")

    # Per-transition accuracy breakdown
    print("\n" + "-" * 60)
    print("Per-transition accuracy:")
    print("-" * 60)

    for state in STATES:
        for event in EVENTS:
            # Create single test
            s = mx.array([state_to_idx(state)])
            e = mx.array([event_to_idx(event)])
            logits = model(s, e)
            pred_idx = int(mx.argmax(logits, axis=-1))
            pred_state = idx_to_state(pred_idx)
            expected = get_next_state(state, event)
            match = "✓" if pred_state == expected else "✗"
            print(f"  ({state}, {event}) -> {pred_state} (expected: {expected}) {match}")

    return model, final_acc, passed


if __name__ == "__main__":
    mx.random.seed(42)
    model, accuracy, passed = train(num_epochs=200, num_samples=2000)
    exit(0 if passed else 1)
