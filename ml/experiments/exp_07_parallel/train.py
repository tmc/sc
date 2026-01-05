"""
Level 7: Parallel Regions (AND-States)

Tests whether model can track multiple independent state machines simultaneously.

Setup:
    System (AND)
    ├── PowerState: {Off, On}
    └── NetworkState: {Disconnected, Connected}

    Events: POWER_TOGGLE, NETWORK_TOGGLE
    (Each affects only its region)

Task: Predict both regions correctly
Pass Criterion: >95% accuracy on BOTH regions simultaneously
Baseline: Track only one region = ~75%
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Region 1: Power
POWER_STATES = ["Off", "On"]
# Region 2: Network
NETWORK_STATES = ["Disconnected", "Connected"]

EVENTS = ["POWER_TOGGLE", "NETWORK_TOGGLE"]


def get_next_state(power_idx: int, network_idx: int, event_idx: int) -> tuple:
    """Ground truth transition for parallel regions.

    Args:
        power_idx: 0=Off, 1=On
        network_idx: 0=Disconnected, 1=Connected
        event_idx: 0=POWER_TOGGLE, 1=NETWORK_TOGGLE

    Returns:
        (next_power_idx, next_network_idx)
    """
    next_power = power_idx
    next_network = network_idx

    if event_idx == 0:  # POWER_TOGGLE
        next_power = 1 - power_idx  # Toggle
    elif event_idx == 1:  # NETWORK_TOGGLE
        next_network = 1 - network_idx  # Toggle

    return next_power, next_network


def generate_training_data(num_samples: int = 2000):
    """Generate training data for parallel region prediction.

    Returns:
        power_states: [N] current power state indices
        network_states: [N] current network state indices
        events: [N] event indices
        target_power: [N] target power state indices
        target_network: [N] target network state indices
    """
    power_states = []
    network_states = []
    events = []
    target_power = []
    target_network = []

    for _ in range(num_samples):
        # Random current state
        power_idx = int(mx.random.randint(0, 2, ()))
        network_idx = int(mx.random.randint(0, 2, ()))
        event_idx = int(mx.random.randint(0, 2, ()))

        # Get next states
        next_power, next_network = get_next_state(power_idx, network_idx, event_idx)

        power_states.append(power_idx)
        network_states.append(network_idx)
        events.append(event_idx)
        target_power.append(next_power)
        target_network.append(next_network)

    return (
        mx.array(power_states, dtype=mx.int32),
        mx.array(network_states, dtype=mx.int32),
        mx.array(events, dtype=mx.int32),
        mx.array(target_power, dtype=mx.int32),
        mx.array(target_network, dtype=mx.int32),
    )


class ParallelRegionPredictor(nn.Module):
    """Model that tracks two independent regions."""

    def __init__(self, embed_dim: int = 16):
        super().__init__()

        # Embeddings for each region
        self.power_embed = nn.Embedding(2, embed_dim)
        self.network_embed = nn.Embedding(2, embed_dim)
        self.event_embed = nn.Embedding(2, embed_dim)

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim * 3, 32),
            nn.ReLU(),
        )

        # Separate heads for each region
        self.power_head = nn.Linear(32, 2)
        self.network_head = nn.Linear(32, 2)

    def __call__(self, power: mx.array, network: mx.array,
                 events: mx.array) -> tuple:
        """Predict next state for both regions.

        Args:
            power: [B] current power state indices
            network: [B] current network state indices
            events: [B] event indices

        Returns:
            power_logits: [B, 2] power state logits
            network_logits: [B, 2] network state logits
        """
        power_emb = self.power_embed(power)
        network_emb = self.network_embed(network)
        event_emb = self.event_embed(events)

        combined = mx.concatenate([power_emb, network_emb, event_emb], axis=-1)
        hidden = self.encoder(combined)

        power_logits = self.power_head(hidden)
        network_logits = self.network_head(hidden)

        return power_logits, network_logits


class SingleRegionPredictor(nn.Module):
    """Baseline: only tracks power region (ignores network)."""

    def __init__(self, embed_dim: int = 16):
        super().__init__()

        self.power_embed = nn.Embedding(2, embed_dim)
        self.event_embed = nn.Embedding(2, embed_dim)

        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def __call__(self, power: mx.array, network: mx.array,
                 events: mx.array) -> tuple:
        """Only predict power, guess network."""
        power_emb = self.power_embed(power)
        event_emb = self.event_embed(events)

        combined = mx.concatenate([power_emb, event_emb], axis=-1)
        power_logits = self.predictor(combined)

        # For network, just return zeros (will be random after softmax)
        network_logits = mx.zeros((power.shape[0], 2))

        return power_logits, network_logits


def compute_accuracy(model, power, network, events, target_power, target_network):
    """Compute accuracy for both regions."""
    power_logits, network_logits = model(power, network, events)

    power_preds = mx.argmax(power_logits, axis=-1)
    network_preds = mx.argmax(network_logits, axis=-1)

    power_acc = float(mx.sum(power_preds == target_power)) / target_power.shape[0]
    network_acc = float(mx.sum(network_preds == target_network)) / target_network.shape[0]

    # BOTH correct
    both_correct = (power_preds == target_power) & (network_preds == target_network)
    joint_acc = float(mx.sum(both_correct)) / target_power.shape[0]

    return power_acc, network_acc, joint_acc


def train(num_epochs: int = 200, num_samples: int = 2000, lr: float = 0.01):
    """Train parallel region predictor."""
    print("=" * 60)
    print("Level 7: Parallel Regions (AND-States)")
    print("=" * 60)

    print(f"\nPowerState: {POWER_STATES}")
    print(f"NetworkState: {NETWORK_STATES}")
    print(f"Events: {EVENTS}")

    # Generate data
    train_power, train_network, train_events, train_tp, train_tn = generate_training_data(num_samples)
    test_power, test_network, test_events, test_tp, test_tn = generate_training_data(500)

    print(f"\nTraining samples: {num_samples}")
    print(f"Test samples: 500")

    # Create model
    model = ParallelRegionPredictor()
    params = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"Model parameters: {params}")

    # Baseline: random for one region = 50% * 100% = 75% joint
    print(f"Baseline (single region): ~75% joint accuracy")

    # Optimizer
    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(model):
        power_logits, network_logits = model(train_power, train_network, train_events)

        # Cross-entropy for both
        power_log_probs = mx.log(mx.softmax(power_logits, axis=-1) + 1e-10)
        network_log_probs = mx.log(mx.softmax(network_logits, axis=-1) + 1e-10)

        power_loss = -mx.mean(power_log_probs[mx.arange(train_tp.shape[0]), train_tp])
        network_loss = -mx.mean(network_log_probs[mx.arange(train_tn.shape[0]), train_tn])

        return power_loss + network_loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training
    print("\nTraining...")
    print("-" * 60)

    for epoch in range(num_epochs):
        loss, grads = loss_and_grad(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if epoch % 40 == 0 or epoch == num_epochs - 1:
            power_acc, network_acc, joint_acc = compute_accuracy(
                model, test_power, test_network, test_events, test_tp, test_tn
            )
            print(f"Epoch {epoch:3d}: loss={float(loss):.4f}, power={power_acc:.1%}, network={network_acc:.1%}, joint={joint_acc:.1%}")

            if joint_acc >= 0.95:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Final evaluation
    power_acc, network_acc, joint_acc = compute_accuracy(
        model, test_power, test_network, test_events, test_tp, test_tn
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Power region accuracy: {power_acc:.1%}")
    print(f"Network region accuracy: {network_acc:.1%}")
    print(f"Joint accuracy (BOTH correct): {joint_acc:.1%}")

    # Per-event breakdown
    print("\n" + "-" * 60)
    print("Per-event accuracy:")
    print("-" * 60)

    for event in EVENTS:
        event_idx = EVENTS.index(event)
        mask = test_events == event_idx
        count = int(mx.sum(mask))

        if count > 0:
            power_logits, network_logits = model(test_power, test_network, test_events)
            power_preds = mx.argmax(power_logits, axis=-1)
            network_preds = mx.argmax(network_logits, axis=-1)

            power_correct = mx.sum((power_preds == test_tp) & mask)
            network_correct = mx.sum((network_preds == test_tn) & mask)
            both_correct = mx.sum((power_preds == test_tp) & (network_preds == test_tn) & mask)

            print(f"  {event}: power={float(power_correct)/count:.1%}, network={float(network_correct)/count:.1%}, joint={float(both_correct)/count:.1%}")

    # Pass criterion
    passed = joint_acc >= 0.95
    status = "PASS" if passed else "FAIL"
    print(f"\n{'='*60}")
    print(f"Pass criterion (>95% joint): {status}")
    print(f"{'='*60}")

    return model, joint_acc, passed


if __name__ == "__main__":
    mx.random.seed(42)
    Path(__file__).parent.mkdir(parents=True, exist_ok=True)
    model, accuracy, passed = train()
    exit(0 if passed else 1)
