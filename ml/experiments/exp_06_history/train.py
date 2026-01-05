"""
Level 6: Shallow History States

Tests whether model can learn to restore previous substate after returning.

Setup:
    Main (OR, has history H)
    ├── Sub1
    └── Sub2
    Outside

    Events: GO_SUB1, GO_SUB2, LEAVE, RETURN
    Transitions:
        GO_SUB1: Main.* -> Main.Sub1
        GO_SUB2: Main.* -> Main.Sub2
        LEAVE: Main -> Outside (saves current substate)
        RETURN: Outside -> Main.H (restores saved substate)

Task: After LEAVE then RETURN, predict restored substate
Pass Criterion: >90% accuracy on history restoration
Baseline: Always return to default = 50%
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# States
STATES = ["Sub1", "Sub2", "Outside"]
EVENTS = ["GO_SUB1", "GO_SUB2", "LEAVE", "RETURN"]


def is_in_main(state: str) -> bool:
    return state in ["Sub1", "Sub2"]


def generate_sequences(num_sequences: int = 500, max_len: int = 10):
    """Generate sequences with history restoration.

    Each sequence starts in Sub1 or Sub2, does some moves,
    then LEAVE, possibly do nothing, then RETURN.

    Returns:
        List of (sequence_of_events, final_state, history_state)
    """
    sequences = []

    for _ in range(num_sequences):
        # Start in Sub1 or Sub2
        current = "Sub1" if mx.random.uniform() > 0.5 else "Sub2"
        history = current  # Track what was saved

        events = []
        saved_history = None

        # Generate random sequence
        seq_len = int(mx.random.randint(3, max_len, ()))

        for _ in range(seq_len):
            event_idx = int(mx.random.randint(0, len(EVENTS), ()))
            event = EVENTS[event_idx]
            events.append(event)

            if event == "GO_SUB1" and is_in_main(current):
                current = "Sub1"
                history = current
            elif event == "GO_SUB2" and is_in_main(current):
                current = "Sub2"
                history = current
            elif event == "LEAVE" and is_in_main(current):
                saved_history = history
                current = "Outside"
            elif event == "RETURN" and current == "Outside":
                current = saved_history if saved_history else "Sub1"
                history = current

        sequences.append({
            'events': events,
            'final_state': current,
            'history_saved': saved_history,
        })

    return sequences


def generate_training_data(num_samples: int = 2000):
    """Generate training data for history prediction.

    Focus: Given current state + history + event, predict next state.

    Returns:
        states: [N, 3] one-hot current state
        histories: [N, 2] one-hot history (Sub1, Sub2)
        events: [N] event indices
        targets: [N] target state indices
    """
    states = []
    histories = []
    events = []
    targets = []

    for _ in range(num_samples):
        # Random current state
        state_idx = int(mx.random.randint(0, len(STATES), ()))
        current = STATES[state_idx]

        # Random history (only Sub1 or Sub2)
        history_idx = int(mx.random.randint(0, 2, ()))
        history = STATES[history_idx]

        # Random event
        event_idx = int(mx.random.randint(0, len(EVENTS), ()))
        event = EVENTS[event_idx]

        # Compute next state
        if event == "GO_SUB1" and is_in_main(current):
            next_state = "Sub1"
            next_history = "Sub1"
        elif event == "GO_SUB2" and is_in_main(current):
            next_state = "Sub2"
            next_history = "Sub2"
        elif event == "LEAVE" and is_in_main(current):
            next_state = "Outside"
            next_history = current  # Save current as history
        elif event == "RETURN" and current == "Outside":
            next_state = history  # Restore from history!
            next_history = history
        else:
            next_state = current  # No-op
            next_history = history

        # One-hot encodings
        state_one_hot = [1.0 if i == state_idx else 0.0 for i in range(len(STATES))]
        history_one_hot = [1.0 if i == history_idx else 0.0 for i in range(2)]
        target_idx = STATES.index(next_state)

        states.append(state_one_hot)
        histories.append(history_one_hot)
        events.append(event_idx)
        targets.append(target_idx)

    return (
        mx.array(states),
        mx.array(histories),
        mx.array(events, dtype=mx.int32),
        mx.array(targets, dtype=mx.int32),
    )


class HistoryPredictor(nn.Module):
    """Model that uses history state for prediction."""

    def __init__(self, num_states: int = 3, num_events: int = 4,
                 history_dim: int = 2, embed_dim: int = 16):
        super().__init__()

        self.event_embed = nn.Embedding(num_events, embed_dim)

        # State encoder
        self.state_encoder = nn.Linear(num_states, embed_dim)

        # History encoder (crucial for this task!)
        self.history_encoder = nn.Linear(history_dim, embed_dim)

        # Combine state + history + event -> next state
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 3, 32),
            nn.ReLU(),
            nn.Linear(32, num_states),
        )

    def __call__(self, states: mx.array, histories: mx.array,
                 events: mx.array) -> mx.array:
        """Predict next state.

        Args:
            states: [B, num_states] one-hot current state
            histories: [B, history_dim] one-hot history
            events: [B] event indices

        Returns:
            [B, num_states] next state logits
        """
        state_emb = self.state_encoder(states)
        history_emb = self.history_encoder(histories)
        event_emb = self.event_embed(events)

        combined = mx.concatenate([state_emb, history_emb, event_emb], axis=-1)
        return self.predictor(combined)


class NoHistoryPredictor(nn.Module):
    """Baseline model WITHOUT history information."""

    def __init__(self, num_states: int = 3, num_events: int = 4, embed_dim: int = 16):
        super().__init__()

        self.event_embed = nn.Embedding(num_events, embed_dim)
        self.state_encoder = nn.Linear(num_states, embed_dim)

        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, num_states),
        )

    def __call__(self, states: mx.array, histories: mx.array,
                 events: mx.array) -> mx.array:
        """Predict next state (ignoring history)."""
        state_emb = self.state_encoder(states)
        event_emb = self.event_embed(events)

        combined = mx.concatenate([state_emb, event_emb], axis=-1)
        return self.predictor(combined)


def compute_accuracy(model, states, histories, events, targets):
    """Compute prediction accuracy."""
    logits = model(states, histories, events)
    preds = mx.argmax(logits, axis=-1)
    return float(mx.sum(preds == targets)) / targets.shape[0]


def compute_return_accuracy(model, states, histories, events, targets):
    """Compute accuracy specifically for RETURN event (history restoration)."""
    return_mask = events == EVENTS.index("RETURN")
    return_count = int(mx.sum(return_mask))

    if return_count == 0:
        return 0.0

    logits = model(states, histories, events)
    preds = mx.argmax(logits, axis=-1)

    # Only check where event is RETURN
    correct = mx.sum((preds == targets) & return_mask)
    return float(correct) / return_count


def train(num_epochs: int = 300, num_samples: int = 3000, lr: float = 0.01):
    """Train history predictor and compare to no-history baseline."""
    print("=" * 60)
    print("Level 6: Shallow History States")
    print("=" * 60)

    print(f"\nStates: {STATES}")
    print(f"Events: {EVENTS}")
    print(f"History: saves substate on LEAVE, restores on RETURN")

    # Generate data
    train_states, train_hist, train_events, train_targets = generate_training_data(num_samples)
    test_states, test_hist, test_events, test_targets = generate_training_data(500)

    print(f"\nTraining samples: {num_samples}")
    print(f"Test samples: 500")

    # Create models
    model_with_history = HistoryPredictor()
    model_no_history = NoHistoryPredictor()

    params_with = sum(p.size for _, p in nn.utils.tree_flatten(model_with_history.parameters()))
    params_without = sum(p.size for _, p in nn.utils.tree_flatten(model_no_history.parameters()))

    print(f"\nModel with history: {params_with} params")
    print(f"Model without history: {params_without} params")

    # Train both models
    for model, name in [(model_with_history, "WITH history"), (model_no_history, "WITHOUT history")]:
        print(f"\n{'='*60}")
        print(f"Training {name}")
        print(f"{'='*60}")

        optimizer = optim.Adam(learning_rate=lr)

        def loss_fn(model):
            logits = model(train_states, train_hist, train_events)
            log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)
            return -mx.mean(log_probs[mx.arange(train_targets.shape[0]), train_targets])

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        for epoch in range(num_epochs):
            loss, grads = loss_and_grad(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            if epoch % 60 == 0 or epoch == num_epochs - 1:
                overall_acc = compute_accuracy(model, test_states, test_hist, test_events, test_targets)
                return_acc = compute_return_accuracy(model, test_states, test_hist, test_events, test_targets)
                print(f"Epoch {epoch:3d}: loss={float(loss):.4f}, overall={overall_acc:.1%}, RETURN={return_acc:.1%}")

    # Final comparison
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    for model, name in [(model_with_history, "WITH history"), (model_no_history, "WITHOUT history")]:
        overall = compute_accuracy(model, test_states, test_hist, test_events, test_targets)
        return_acc = compute_return_accuracy(model, test_states, test_hist, test_events, test_targets)
        print(f"\n{name}:")
        print(f"  Overall accuracy: {overall:.1%}")
        print(f"  RETURN accuracy: {return_acc:.1%}")

    # Pass criterion
    final_acc = compute_accuracy(model_with_history, test_states, test_hist, test_events, test_targets)
    return_acc = compute_return_accuracy(model_with_history, test_states, test_hist, test_events, test_targets)

    passed = final_acc >= 0.90 and return_acc >= 0.90
    status = "PASS" if passed else "FAIL"

    print(f"\n{'='*60}")
    print(f"Pass criterion (>90% on RETURN): {status}")
    print(f"{'='*60}")

    return model_with_history, final_acc, passed


if __name__ == "__main__":
    mx.random.seed(42)
    Path(__file__).parent.mkdir(parents=True, exist_ok=True)
    model, accuracy, passed = train()
    exit(0 if passed else 1)
