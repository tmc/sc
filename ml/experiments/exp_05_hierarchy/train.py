"""
Level 5: Hierarchy (OR-States) - Fixed Version

Problem with original: Model predicts over BOTH composite (A, B) AND leaf (A1, A2, B1, B2) states.
Fix: Only predict leaf states. Composite state is implicit from the leaf.

Setup:
    Root (OR)
    ├── ModeA (OR)
    │   ├── A1
    │   └── A2
    └── ModeB (OR)
        ├── B1
        └── B2

    Events: NEXT, SWITCH, PREV
    Transitions:
        NEXT: A1→A2, B1→B2 (within mode)
        SWITCH: A→B, B→A (change mode, enter default)
        PREV: A2→A1, B2→B1 (reverse, may not fire)

Task: Predict leaf state given (current_leaf, event)
Pass Criterion: >95% accuracy
Baseline: Ignore event = ~25%
"""

import json
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Leaf states only (not composites)
LEAF_STATES = ["A1", "A2", "B1", "B2"]
EVENTS = ["NEXT", "SWITCH", "PREV"]

# Ground truth transition table: (leaf, event) -> leaf
# Based on hierarchy_trace.json
TRANSITION_TABLE = {
    # NEXT: move to next within mode
    ("A1", "NEXT"): "A2",
    ("A2", "NEXT"): "A2",  # No-op (already at end)
    ("B1", "NEXT"): "B2",
    ("B2", "NEXT"): "B2",  # No-op
    # SWITCH: change mode, enter default
    ("A1", "SWITCH"): "B1",
    ("A2", "SWITCH"): "B1",
    ("B1", "SWITCH"): "A1",
    ("B2", "SWITCH"): "A1",
    # PREV: move to previous (may no-op)
    ("A1", "PREV"): "A1",  # No-op
    ("A2", "PREV"): "A1",
    ("B1", "PREV"): "B1",  # No-op
    ("B2", "PREV"): "B1",
}


def get_next_leaf(leaf: str, event: str) -> str:
    """Ground truth transition function."""
    return TRANSITION_TABLE.get((leaf, event), leaf)


def extract_leaf_state(config_list: list) -> str:
    """Extract leaf state from full configuration path.

    Args:
        config_list: ["__root__", "A", "A1"]
    Returns:
        "A1" (the deepest non-root state)
    """
    for s in reversed(config_list):
        if s in LEAF_STATES:
            return s
    return config_list[-1]  # Fallback


def load_hierarchy_trace(path: str):
    """Load trace and extract leaf-only transitions."""
    with open(path, 'r') as f:
        data = json.load(f)

    steps = []
    for step in data['steps']:
        leaf_before = extract_leaf_state(step['state_before'])
        leaf_after = extract_leaf_state(step['state_after'])
        event = step['event']
        fired = 'fired=true' in step.get('guards_evaluated', [''])[0]

        steps.append({
            'leaf_before': leaf_before,
            'event': event,
            'leaf_after': leaf_after,
            'fired': fired,
        })

    return steps


def generate_training_data(num_samples: int = 1000, use_trace: bool = True):
    """Generate training data.

    Args:
        num_samples: Number of samples to generate
        use_trace: If True, also include trace data

    Returns:
        leaves: [N] current leaf state indices
        events: [N] event indices
        targets: [N] target leaf state indices
    """
    leaves = []
    events = []
    targets = []

    # Generate random (leaf, event) pairs
    for _ in range(num_samples):
        leaf_idx = int(mx.random.randint(0, len(LEAF_STATES), ()))
        event_idx = int(mx.random.randint(0, len(EVENTS), ()))

        leaf = LEAF_STATES[leaf_idx]
        event = EVENTS[event_idx]
        next_leaf = get_next_leaf(leaf, event)
        target_idx = LEAF_STATES.index(next_leaf)

        leaves.append(leaf_idx)
        events.append(event_idx)
        targets.append(target_idx)

    # Optionally add trace data
    if use_trace:
        trace_path = Path(__file__).parent.parent.parent.parent / "testdata" / "traces" / "hierarchy_trace.json"
        if trace_path.exists():
            trace_steps = load_hierarchy_trace(trace_path)
            for step in trace_steps:
                leaf_idx = LEAF_STATES.index(step['leaf_before'])
                event_idx = EVENTS.index(step['event'])
                target_idx = LEAF_STATES.index(step['leaf_after'])

                # Add multiple times for emphasis
                for _ in range(10):
                    leaves.append(leaf_idx)
                    events.append(event_idx)
                    targets.append(target_idx)

    return (
        mx.array(leaves, dtype=mx.int32),
        mx.array(events, dtype=mx.int32),
        mx.array(targets, dtype=mx.int32),
    )


class HierarchyPredictor(nn.Module):
    """Predict next leaf state from (current_leaf, event)."""

    def __init__(self, num_leaves: int = 4, num_events: int = 3, embed_dim: int = 16):
        super().__init__()

        self.leaf_embed = nn.Embedding(num_leaves, embed_dim)
        self.event_embed = nn.Embedding(num_events, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, num_leaves),
        )

    def __call__(self, leaves: mx.array, events: mx.array) -> mx.array:
        """Predict next leaf distribution.

        Args:
            leaves: [B] current leaf indices
            events: [B] event indices

        Returns:
            [B, num_leaves] logits
        """
        leaf_emb = self.leaf_embed(leaves)
        event_emb = self.event_embed(events)
        combined = mx.concatenate([leaf_emb, event_emb], axis=-1)
        return self.mlp(combined)


def compute_accuracy(model, leaves, events, targets):
    """Compute prediction accuracy."""
    logits = model(leaves, events)
    preds = mx.argmax(logits, axis=-1)
    return float(mx.sum(preds == targets)) / targets.shape[0]


def train(num_epochs: int = 200, num_samples: int = 2000, lr: float = 0.01):
    """Train hierarchy predictor."""
    print("=" * 60)
    print("Level 5: Hierarchy (OR-States) - Leaf States Only")
    print("=" * 60)

    print(f"\nLeaf states: {LEAF_STATES}")
    print(f"Events: {EVENTS}")
    print(f"Transitions: {len(TRANSITION_TABLE)}")

    # Generate data
    train_leaves, train_events, train_targets = generate_training_data(num_samples, use_trace=True)
    test_leaves, test_events, test_targets = generate_training_data(500, use_trace=False)

    print(f"\nTraining samples: {train_leaves.shape[0]}")
    print(f"Test samples: {test_leaves.shape[0]}")

    # Baseline: always predict most common
    target_list = train_targets.tolist()
    most_common = max(set(target_list), key=target_list.count)
    baseline_acc = target_list.count(most_common) / len(target_list)
    print(f"Baseline (always '{LEAF_STATES[most_common]}'): {baseline_acc:.1%}")

    # Create model
    model = HierarchyPredictor()
    total_params = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"Model parameters: {total_params}")

    # Optimizer
    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(model):
        logits = model(train_leaves, train_events)
        log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)
        return -mx.mean(log_probs[mx.arange(train_targets.shape[0]), train_targets])

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training
    print("\nTraining...")
    print("-" * 60)

    for epoch in range(num_epochs):
        loss, grads = loss_and_grad(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if epoch % 40 == 0 or epoch == num_epochs - 1:
            train_acc = compute_accuracy(model, train_leaves, train_events, train_targets)
            test_acc = compute_accuracy(model, test_leaves, test_events, test_targets)
            print(f"Epoch {epoch:3d}: loss={float(loss):.4f}, train={train_acc:.1%}, test={test_acc:.1%}")

            if test_acc >= 0.95:
                print(f"\nEarly stopping: reached 95% at epoch {epoch}")
                break

    # Final evaluation
    final_acc = compute_accuracy(model, test_leaves, test_events, test_targets)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Baseline: {baseline_acc:.1%}")
    print(f"Model: {final_acc:.1%}")
    print(f"Improvement: {final_acc/baseline_acc:.1f}x")

    # Per-transition check
    print("\n" + "-" * 60)
    print("Per-transition accuracy:")
    print("-" * 60)

    correct = 0
    total = 0
    for leaf in LEAF_STATES:
        for event in EVENTS:
            leaf_idx = mx.array([LEAF_STATES.index(leaf)])
            event_idx = mx.array([EVENTS.index(event)])

            logits = model(leaf_idx, event_idx)
            pred_idx = int(mx.argmax(logits, axis=-1))
            pred_leaf = LEAF_STATES[pred_idx]
            expected = get_next_leaf(leaf, event)

            match = pred_leaf == expected
            if match:
                correct += 1
            total += 1

            status = "✓" if match else "✗"
            print(f"  ({leaf}, {event:6s}) -> {pred_leaf} (expected: {expected}) {status}")

    trans_acc = correct / total
    print("-" * 60)
    print(f"Transition accuracy: {correct}/{total} = {trans_acc:.1%}")

    # Pass criterion
    passed = final_acc >= 0.95 and trans_acc >= 0.95
    status = "PASS" if passed else "FAIL"
    print(f"\nPass criterion (>95%): {status}")

    return model, final_acc, passed


if __name__ == "__main__":
    mx.random.seed(42)
    Path(__file__).parent.mkdir(parents=True, exist_ok=True)
    model, accuracy, passed = train()
    exit(0 if passed else 1)
