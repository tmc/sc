"""
Experiment 01: Learn 2-State Toggle (Off <-> On)

THE BLOCKING QUESTION: Can we learn ANYTHING?

Task:
- Load toggle trace from Go semantics test output
- Train model to predict next state given (current_state, event)
- Pass criterion: >99% accuracy after 1000 steps
- Baseline: random = 50%

This is the simplest possible statechart learning task.
If this fails, nothing else will work.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import json
import os
import sys

# Add paths for imports
_file_dir = os.path.dirname(os.path.abspath(__file__))
_ml_dir = os.path.dirname(os.path.dirname(_file_dir))
_root_dir = os.path.dirname(_ml_dir)
sys.path.insert(0, _ml_dir)

# State and event mappings
STATES = {"Off": 0, "On": 1}
EVENTS = {"TOGGLE": 0}
NUM_STATES = 2
NUM_EVENTS = 1


class ToggleLearner(nn.Module):
    """
    Learn toggle dynamics: (state, event) -> next_state

    Architecture:
    - Embed current state
    - Embed event
    - Concatenate and predict next state distribution
    """

    def __init__(self, embed_dim: int = 8):
        super().__init__()
        self.state_embed = nn.Embedding(NUM_STATES, embed_dim)
        self.event_embed = nn.Embedding(NUM_EVENTS, embed_dim)
        self.hidden = nn.Linear(embed_dim * 2, embed_dim)
        self.output = nn.Linear(embed_dim, NUM_STATES)

    def __call__(self, state_ids: mx.array, event_ids: mx.array) -> mx.array:
        """
        Predict next state distribution.

        Args:
            state_ids: [B] current state indices
            event_ids: [B] event indices

        Returns:
            [B, NUM_STATES] logits for next state
        """
        s_emb = self.state_embed(state_ids)  # [B, embed_dim]
        e_emb = self.event_embed(event_ids)  # [B, embed_dim]
        combined = mx.concatenate([s_emb, e_emb], axis=-1)  # [B, 2*embed_dim]
        h = mx.tanh(self.hidden(combined))  # [B, embed_dim]
        return self.output(h)  # [B, NUM_STATES]


def load_traces(trace_path: str) -> list:
    """Load toggle traces from JSON file."""
    with open(trace_path, 'r') as f:
        data = json.load(f)

    samples = []
    for step in data["steps"]:
        # Extract state names (skip "__root__")
        state_before = [s for s in step["state_before"] if s != "__root__"][0]
        state_after = [s for s in step["state_after"] if s != "__root__"][0]
        event = step["event"]

        samples.append({
            "state": STATES[state_before],
            "event": EVENTS[event],
            "next_state": STATES[state_after],
        })

    return samples


def generate_synthetic_traces(num_samples: int = 1000) -> list:
    """
    Generate synthetic toggle traces.

    The toggle dynamics are deterministic:
    - Off + TOGGLE -> On
    - On + TOGGLE -> Off
    """
    samples = []
    for _ in range(num_samples):
        state = mx.random.randint(0, NUM_STATES, ()).item()
        event = 0  # Only TOGGLE event

        # Deterministic next state (toggle)
        next_state = 1 - state

        samples.append({
            "state": state,
            "event": event,
            "next_state": next_state,
        })

    return samples


def compute_accuracy(model: ToggleLearner, samples: list) -> float:
    """Compute prediction accuracy."""
    states = mx.array([s["state"] for s in samples])
    events = mx.array([s["event"] for s in samples])
    targets = mx.array([s["next_state"] for s in samples])

    logits = model(states, events)
    preds = mx.argmax(logits, axis=-1)

    correct = mx.sum(preds == targets)
    return float(correct) / len(samples)


def train(
    model: ToggleLearner,
    samples: list,
    num_steps: int = 1000,
    lr: float = 0.01,
    batch_size: int = 32,
    verbose: bool = True,
) -> dict:
    """
    Train the toggle learner.

    Returns:
        dict with training stats
    """
    optimizer = optim.Adam(learning_rate=lr)

    # Convert to arrays
    all_states = mx.array([s["state"] for s in samples])
    all_events = mx.array([s["event"] for s in samples])
    all_targets = mx.array([s["next_state"] for s in samples])

    losses = []
    accuracies = []

    for step in range(num_steps):
        # Sample batch
        indices = mx.random.randint(0, len(samples), (batch_size,))
        batch_states = all_states[indices]
        batch_events = all_events[indices]
        batch_targets = all_targets[indices]

        def loss_fn(m):
            logits = m(batch_states, batch_events)
            log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)
            # Cross-entropy loss (gather log probs at target indices)
            # For each sample i, get log_probs[i, batch_targets[i]]
            batch_indices = mx.arange(batch_size)
            target_log_probs = log_probs[batch_indices, batch_targets]
            loss = -mx.mean(target_log_probs)
            return loss

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        losses.append(float(loss))

        # Evaluate every 100 steps
        if (step + 1) % 100 == 0 or step == 0:
            acc = compute_accuracy(model, samples)
            accuracies.append((step + 1, acc))
            if verbose:
                print(f"Step {step + 1:4d}: loss={loss:.4f}, acc={acc:.4f}")

    final_acc = compute_accuracy(model, samples)

    return {
        "final_accuracy": final_acc,
        "losses": losses,
        "accuracies": accuracies,
        "passed": final_acc > 0.99,
    }


def run_experiment():
    """Run the Level 1 toggle learning experiment."""
    print("=" * 60)
    print("EXPERIMENT 01: Learn 2-State Toggle")
    print("=" * 60)
    print()
    print("Task: Predict next state given (current_state, event)")
    print("States: Off (0), On (1)")
    print("Events: TOGGLE (0)")
    print("Dynamics: Off + TOGGLE -> On, On + TOGGLE -> Off")
    print()
    print("Pass criterion: >99% accuracy after 1000 steps")
    print("Baseline (random): 50%")
    print()

    # Set seed for reproducibility
    mx.random.seed(42)

    # Load real traces
    trace_path = os.path.join(_root_dir, "testdata/traces/toggle_trace.json")
    if os.path.exists(trace_path):
        real_samples = load_traces(trace_path)
        print(f"Loaded {len(real_samples)} samples from Go traces")
    else:
        real_samples = []
        print("No trace file found, using synthetic data only")

    # Generate synthetic traces (more data for training)
    synthetic_samples = generate_synthetic_traces(1000)
    print(f"Generated {len(synthetic_samples)} synthetic samples")

    # Combine samples
    all_samples = real_samples + synthetic_samples
    print(f"Total training samples: {len(all_samples)}")
    print()

    # Create model
    model = ToggleLearner(embed_dim=8)

    # Check initial accuracy (should be ~50% random)
    initial_acc = compute_accuracy(model, all_samples)
    print(f"Initial accuracy (untrained): {initial_acc:.4f}")
    print()

    # Train
    print("Training...")
    print("-" * 60)
    result = train(
        model=model,
        samples=all_samples,
        num_steps=1000,
        lr=0.01,
        batch_size=32,
    )
    print("-" * 60)
    print()

    # Report
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Final accuracy: {result['final_accuracy']:.4f}")
    print(f"Pass criterion: >0.99")
    print(f"Status: {'PASS' if result['passed'] else 'FAIL'}")
    print()

    # Verify learned dynamics
    print("Verifying learned dynamics:")
    test_cases = [
        (0, 0, 1, "Off + TOGGLE -> On"),
        (1, 0, 0, "On + TOGGLE -> Off"),
    ]

    for state, event, expected, desc in test_cases:
        logits = model(mx.array([state]), mx.array([event]))
        probs = mx.softmax(logits, axis=-1)[0]
        pred = int(mx.argmax(logits, axis=-1)[0].item())
        correct = "OK" if pred == expected else "WRONG"
        print(f"  {desc}: pred={pred}, expected={expected} [{correct}]")
        print(f"    P(Off)={float(probs[0]):.4f}, P(On)={float(probs[1]):.4f}")

    print()
    print("=" * 60)

    return result


if __name__ == "__main__":
    result = run_experiment()
    sys.exit(0 if result["passed"] else 1)
