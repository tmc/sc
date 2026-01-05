"""
Experiment 02: 3-State Cycle Learning

Train a model to predict the 3-state cycle: A → B → C → A
Validates that ML can learn cyclic state transitions.

Pass criterion: >99% accuracy
Baseline: 33% (random guess among 3 states)
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils


class CyclePredictor(nn.Module):
    """Model to predict next state in a cycle."""

    def __init__(self, num_states: int = 3, embed_dim: int = 16):
        super().__init__()
        self.num_states = num_states

        # State encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_states, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        # Predictor
        self.predictor = nn.Linear(embed_dim, num_states)

    def __call__(self, state: mx.array) -> mx.array:
        """Predict next state distribution.

        Args:
            state: [B, num_states] one-hot current state

        Returns:
            [B, num_states] predicted next state distribution
        """
        h = self.encoder(state)
        logits = self.predictor(h)
        return mx.softmax(logits, axis=-1)


def generate_cycle_data(num_samples: int = 100):
    """Generate training data for A→B→C→A cycle.

    Returns:
        X: [N, 3] one-hot current states
        Y: [N, 3] one-hot next states
    """
    # States: A=0, B=1, C=2
    # Transitions: A→B, B→C, C→A

    states = []
    targets = []

    for i in range(num_samples):
        current = i % 3  # Cycle through A, B, C
        next_state = (current + 1) % 3  # Next in cycle

        # One-hot encoding
        state_vec = [0.0, 0.0, 0.0]
        state_vec[current] = 1.0

        target_vec = [0.0, 0.0, 0.0]
        target_vec[next_state] = 1.0

        states.append(state_vec)
        targets.append(target_vec)

    X = mx.array(states)
    Y = mx.array(targets)

    return X, Y


def compute_loss(pred: mx.array, target: mx.array) -> mx.array:
    """Cross-entropy loss with label smoothing."""
    pred_clipped = mx.clip(pred, 1e-7, 1.0 - 1e-7)
    smooth = 0.1
    n_classes = target.shape[-1]
    target_smooth = target * (1 - smooth) + smooth / n_classes
    return -mx.mean(mx.sum(target_smooth * mx.log(pred_clipped), axis=-1))


def compute_accuracy(pred: mx.array, target: mx.array) -> float:
    """Compute prediction accuracy."""
    pred_idx = mx.argmax(pred, axis=-1)
    target_idx = mx.argmax(target, axis=-1)
    return float(mx.mean(pred_idx == target_idx))


def train(num_epochs: int = 500, learning_rate: float = 0.01):
    """Train the cycle predictor.

    Returns:
        Trained model and training history
    """
    print("=" * 60)
    print("EXPERIMENT 02: 3-STATE CYCLE LEARNING")
    print("=" * 60)
    print("\nTask: Learn A → B → C → A cyclic transitions")
    print("Pass criterion: >99% accuracy")
    print("Baseline: 33% (random guess)")
    print()

    # Generate data
    X, Y = generate_cycle_data(num_samples=300)
    print(f"Training samples: {X.shape[0]}")
    print(f"States: A=0, B=1, C=2")
    print(f"Transitions: A→B, B→C, C→A")

    # Create model
    model = CyclePredictor(num_states=3, embed_dim=32)
    optimizer = optim.Adam(learning_rate=learning_rate)

    # Loss function
    def loss_fn(model):
        pred = model(X)
        return compute_loss(pred, Y)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training loop
    history = {'loss': [], 'accuracy': []}

    print(f"\nTraining for {num_epochs} epochs...")
    print("-" * 60)

    for epoch in range(num_epochs):
        loss, grads = loss_and_grad(model)

        # Gradient clipping
        grads = mlx.utils.tree_map(lambda g: mx.clip(g, -1.0, 1.0), grads)

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        # Compute accuracy
        pred = model(X)
        mx.eval(pred)
        acc = compute_accuracy(pred, Y)

        history['loss'].append(float(loss))
        history['accuracy'].append(acc)

        if epoch % 100 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:4d}: loss={float(loss):.4f}, accuracy={acc:.4f}")

        # Early stopping if converged
        if acc >= 0.99:
            print(f"\n✓ Converged at epoch {epoch}!")
            break

    return model, history


def evaluate(model):
    """Evaluate model predictions."""
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    state_names = ['A', 'B', 'C']

    print("\nStep-by-step predictions:")
    print("-" * 40)

    correct = 0
    total = 3

    for current in range(3):
        expected_next = (current + 1) % 3

        # One-hot input
        state_vec = mx.zeros((1, 3))
        state_vec = state_vec.at[0, current].add(1.0)

        # Predict
        pred = model(state_vec)
        mx.eval(pred)
        pred_idx = int(mx.argmax(pred[0]))

        match = pred_idx == expected_next
        if match:
            correct += 1

        status = "✓" if match else "✗"
        print(f"  {state_names[current]} → pred={state_names[pred_idx]}, "
              f"expected={state_names[expected_next]} {status}")

    accuracy = correct / total
    print("-" * 40)
    print(f"Accuracy: {correct}/{total} = {accuracy:.1%}")

    return accuracy


def main():
    """Main training and evaluation."""
    # Train
    model, history = train(num_epochs=1000, learning_rate=0.01)

    # Evaluate
    accuracy = evaluate(model)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Initial accuracy: {history['accuracy'][0]:.4f}")
    print(f"Final accuracy: {history['accuracy'][-1]:.4f}")
    print(f"Final loss: {history['loss'][-1]:.4f}")
    print(f"Epochs trained: {len(history['loss'])}")

    # Pass/fail
    print("\n" + "=" * 60)
    if accuracy >= 0.99:
        print("✓ PASS: Level 2 complete - 3-state cycle learned!")
    else:
        print(f"✗ FAIL: Accuracy {accuracy:.1%} < 99% threshold")
    print("=" * 60)

    return {
        'initial_accuracy': history['accuracy'][0],
        'final_accuracy': history['accuracy'][-1],
        'final_loss': history['loss'][-1],
        'epochs': len(history['loss']),
        'eval_accuracy': accuracy,
        'passed': accuracy >= 0.99,
    }


if __name__ == "__main__":
    results = main()
