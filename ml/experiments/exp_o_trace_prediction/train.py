"""
Experiment O: Trace Prediction Training

Train DifferentiableTransitionSelector to predict next state from ground truth traces.
Validates that ML can learn actual statechart behavior from Go semantics.
"""

import json
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import sys
from pathlib import Path
from dataclasses import dataclass

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from differentiable.exp_c_transitions import (
    DifferentiableTransitionSelector,
    StatechartMachine,
)


@dataclass
class TraceStep:
    """A single step from a trace."""
    state_before: list
    event: str
    state_after: list
    fired: bool


@dataclass
class Trace:
    """A complete trace with metadata."""
    name: str
    states: list  # All unique states
    events: list  # All unique events
    steps: list   # List of TraceStep


def load_trace(path: str) -> Trace:
    """Load a trace from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)

    # Extract unique states and events
    all_states = set()
    all_events = set()
    steps = []

    for step in data['steps']:
        all_states.update(step['state_before'])
        all_states.update(step['state_after'])
        all_events.add(step['event'])

        # Parse fired status
        fired = True
        if step.get('guards_evaluated'):
            fired_str = step['guards_evaluated'][0]
            fired = 'fired=true' in fired_str

        steps.append(TraceStep(
            state_before=step['state_before'],
            event=step['event'],
            state_after=step['state_after'],
            fired=fired,
        ))

    # Sort for consistent indexing (exclude __root__)
    states = sorted([s for s in all_states if s != '__root__'])
    events = sorted(all_events)

    return Trace(
        name=data['name'],
        states=states,
        events=events,
        steps=steps,
    )


def states_to_config(states: list, state_to_idx: dict, num_states: int) -> mx.array:
    """Convert state labels to soft configuration."""
    config = mx.zeros((1, num_states))
    count = 0
    for s in states:
        if s in state_to_idx:
            idx = state_to_idx[s]
            config = config.at[0, idx].add(1.0)
            count += 1
    # Normalize
    if count > 0:
        config = config / count
    return config


def event_to_idx(event: str, event_to_idx_map: dict) -> int:
    """Convert event label to index."""
    return event_to_idx_map.get(event, 0)


class TracePredictor(nn.Module):
    """Model to predict next state from current state and event."""

    def __init__(self, num_states: int, num_events: int, embed_dim: int = 32):
        super().__init__()
        self.num_states = num_states
        self.num_events = num_events
        self.embed_dim = embed_dim

        # Event embedding
        self.event_embed = nn.Embedding(num_events, embed_dim)

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(num_states, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Combined predictor
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_states),
        )

    def __call__(self, state_config: mx.array, event_idx: mx.array) -> mx.array:
        """Predict next state distribution.

        Args:
            state_config: [B, num_states] current state
            event_idx: [B] event indices

        Returns:
            [B, num_states] predicted next state distribution
        """
        # Encode state
        state_emb = self.state_encoder(state_config)  # [B, embed_dim]

        # Embed event
        event_emb = self.event_embed(event_idx)  # [B, embed_dim]

        # Combine and predict
        combined = mx.concatenate([state_emb, event_emb], axis=-1)
        logits = self.predictor(combined)

        # Softmax for distribution
        return mx.softmax(logits, axis=-1)


def compute_loss(pred: mx.array, target: mx.array) -> mx.array:
    """Cross-entropy loss between predicted and target distributions."""
    # Clip predictions for numerical stability
    pred_clipped = mx.clip(pred, 1e-7, 1.0 - 1e-7)
    # Cross-entropy with label smoothing
    smooth = 0.1
    n_classes = target.shape[-1]
    target_smooth = target * (1 - smooth) + smooth / n_classes
    return -mx.mean(mx.sum(target_smooth * mx.log(pred_clipped), axis=-1))


def compute_accuracy(pred: mx.array, target: mx.array) -> float:
    """Compute prediction accuracy (argmax match)."""
    pred_idx = mx.argmax(pred, axis=-1)
    target_idx = mx.argmax(target, axis=-1)
    return float(mx.mean(pred_idx == target_idx))


def train_on_trace(trace: Trace, num_epochs: int = 100,
                   learning_rate: float = 0.01, embed_dim: int = 32):
    """Train a predictor on a single trace.

    Args:
        trace: The trace to train on
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        embed_dim: Embedding dimension

    Returns:
        Trained model and training history
    """
    print(f"\n{'='*60}")
    print(f"Training on: {trace.name}")
    print(f"{'='*60}")
    print(f"States ({len(trace.states)}): {trace.states}")
    print(f"Events ({len(trace.events)}): {trace.events}")
    print(f"Steps: {len(trace.steps)}")

    # Create mappings
    state_to_idx = {s: i for i, s in enumerate(trace.states)}
    event_to_idx_map = {e: i for i, e in enumerate(trace.events)}

    num_states = len(trace.states)
    num_events = len(trace.events)

    # Create model
    model = TracePredictor(num_states, num_events, embed_dim)
    optimizer = optim.Adam(learning_rate=learning_rate)

    # Prepare training data
    state_configs = []
    event_indices = []
    target_configs = []

    for step in trace.steps:
        state_config = states_to_config(step.state_before, state_to_idx, num_states)
        target_config = states_to_config(step.state_after, state_to_idx, num_states)
        event_idx = event_to_idx_map[step.event]

        state_configs.append(state_config)
        event_indices.append(event_idx)
        target_configs.append(target_config)

    # Stack into batches
    X_states = mx.concatenate(state_configs, axis=0)  # [N, num_states]
    X_events = mx.array(event_indices)  # [N]
    Y_targets = mx.concatenate(target_configs, axis=0)  # [N, num_states]

    # Training loop
    history = {'loss': [], 'accuracy': []}

    def loss_fn(model):
        pred = model(X_states, X_events)
        return compute_loss(pred, Y_targets)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    print(f"\nTraining for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        loss, grads = loss_and_grad(model)
        # Gradient clipping for stability
        grads = mlx.utils.tree_map(lambda g: mx.clip(g, -1.0, 1.0), grads)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        # Compute accuracy (MLX doesn't need no_grad context)
        pred = model(X_states, X_events)
        mx.eval(pred)  # Force evaluation
        acc = compute_accuracy(pred, Y_targets)

        history['loss'].append(float(loss))
        history['accuracy'].append(acc)

        if epoch % (num_epochs // 5) == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch:3d}: loss={float(loss):.4f}, accuracy={acc:.2%}")

    return model, history, (state_to_idx, event_to_idx_map)


def evaluate_predictions(model, trace: Trace, state_to_idx: dict,
                        event_to_idx_map: dict):
    """Evaluate model predictions vs ground truth."""
    print(f"\n{'='*60}")
    print(f"Evaluation: {trace.name}")
    print(f"{'='*60}")

    num_states = len(trace.states)
    idx_to_state = {i: s for s, i in state_to_idx.items()}

    correct = 0
    total = 0

    print("\nStep-by-step predictions:")
    print("-" * 60)

    for i, step in enumerate(trace.steps):
        # Prepare input
        state_config = states_to_config(step.state_before, state_to_idx, num_states)
        event_idx = mx.array([event_to_idx_map[step.event]])

        # Predict
        pred = model(state_config, event_idx)
        mx.eval(pred)

        pred_idx = int(mx.argmax(pred[0]))
        pred_state = idx_to_state.get(pred_idx, "?")

        # Ground truth (leaf state only)
        true_states = [s for s in step.state_after if s in state_to_idx]
        true_state = true_states[-1] if true_states else "?"

        # Check match
        match = pred_state == true_state
        if match:
            correct += 1
        total += 1

        status = "✓" if match else "✗"
        fired_str = "fired" if step.fired else "no-op"
        print(f"  {i+1}. [{step.event:8s}] {step.state_before[-1]:4s} -> "
              f"pred={pred_state:4s}, true={true_state:4s} ({fired_str}) {status}")

    accuracy = correct / total if total > 0 else 0
    print("-" * 60)
    print(f"Accuracy: {correct}/{total} = {accuracy:.1%}")

    return accuracy


def main():
    """Main training and evaluation."""
    print("=" * 60)
    print("EXPERIMENT O: TRACE PREDICTION")
    print("Training ML to predict statechart transitions")
    print("=" * 60)

    # Load traces
    trace_dir = Path(__file__).parent.parent.parent.parent / "testdata" / "traces"

    toggle_trace = load_trace(trace_dir / "toggle_trace.json")
    hierarchy_trace = load_trace(trace_dir / "hierarchy_trace.json")

    results = {}

    # Train on toggle trace
    model_toggle, history_toggle, mappings_toggle = train_on_trace(
        toggle_trace, num_epochs=500, learning_rate=0.01, embed_dim=16
    )
    acc_toggle = evaluate_predictions(
        model_toggle, toggle_trace, mappings_toggle[0], mappings_toggle[1]
    )
    results['toggle'] = {
        'final_loss': history_toggle['loss'][-1],
        'final_accuracy': history_toggle['accuracy'][-1],
        'eval_accuracy': acc_toggle,
    }

    # Train on hierarchy trace
    model_hierarchy, history_hierarchy, mappings_hierarchy = train_on_trace(
        hierarchy_trace, num_epochs=1000, learning_rate=0.005, embed_dim=32
    )
    acc_hierarchy = evaluate_predictions(
        model_hierarchy, hierarchy_trace, mappings_hierarchy[0], mappings_hierarchy[1]
    )
    results['hierarchy'] = {
        'final_loss': history_hierarchy['loss'][-1],
        'final_accuracy': history_hierarchy['accuracy'][-1],
        'eval_accuracy': acc_hierarchy,
    }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Final loss: {res['final_loss']:.4f}")
        print(f"  Training accuracy: {res['final_accuracy']:.1%}")
        print(f"  Evaluation accuracy: {res['eval_accuracy']:.1%}")

    # Overall assessment
    all_perfect = all(r['eval_accuracy'] == 1.0 for r in results.values())
    print("\n" + "=" * 60)
    if all_perfect:
        print("✓ ML SUCCESSFULLY LEARNED GO STATECHART SEMANTICS!")
    else:
        avg_acc = sum(r['eval_accuracy'] for r in results.values()) / len(results)
        print(f"Average accuracy: {avg_acc:.1%}")
        if avg_acc >= 0.8:
            print("✓ ML learned most transitions correctly")
        else:
            print("✗ ML needs more training or model capacity")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = main()
