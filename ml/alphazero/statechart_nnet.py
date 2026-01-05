"""
StatechartAlphaZero Neural Network

Policy-value network that takes soft statechart configurations as input.
Uses MLX for Apple Silicon optimization.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import pickle

try:
    from .statechart_encoder import (
        StatechartEncoder, DeepStatechartEncoder,
        encode_soft_config, encode_soft_config_batch,
        TURN_DIM, KO_DIM, BOARD_DIM
    )
except ImportError:
    from statechart_encoder import (
        StatechartEncoder, DeepStatechartEncoder,
        encode_soft_config, encode_soft_config_batch,
        TURN_DIM, KO_DIM, BOARD_DIM
    )

from experiments.exp_go_9x9.go_statechart import Go9x9Statechart


ACTION_SIZE = 82  # 81 positions + pass


class ResBlock(nn.Module):
    """Residual block for trunk network."""

    def __init__(self, channels: int):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.ln1 = nn.LayerNorm(channels)
        self.fc2 = nn.Linear(channels, channels)
        self.ln2 = nn.LayerNorm(channels)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = nn.relu(self.ln1(self.fc1(x)))
        x = self.ln2(self.fc2(x))
        return nn.relu(x + residual)


class StatechartAlphaZeroNet(nn.Module):
    """
    Full policy-value network for StatechartAlphaZero.

    Architecture:
        StatechartEncoder -> Trunk (ResBlocks) -> Policy/Value heads
    """

    def __init__(self, embed_dim: int = 256, num_res_blocks: int = 4,
                 use_deep_encoder: bool = True, encoder_res_blocks: int = 6):
        super().__init__()

        # Encoder
        if use_deep_encoder:
            self.encoder = DeepStatechartEncoder(embed_dim, encoder_res_blocks)
        else:
            self.encoder = StatechartEncoder(embed_dim)

        # Trunk
        self.trunk = [ResBlock(embed_dim) for _ in range(num_res_blocks)]

        # Policy head
        self.policy_fc1 = nn.Linear(embed_dim, 128)
        self.policy_fc2 = nn.Linear(128, ACTION_SIZE)

        # Value head
        self.value_fc1 = nn.Linear(embed_dim, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def __call__(self, turn: mx.array, ko: mx.array,
                 board: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Forward pass.

        Args:
            turn: [B, 2] soft turn config
            ko: [B, 82] soft ko config
            board: [B, 9, 9, 3] soft board config

        Returns:
            log_pi: [B, 82] log policy (log_softmax)
            v: [B, 1] value in [-1, 1]
        """
        # Encode
        x = self.encoder(turn, ko, board)

        # Trunk
        for block in self.trunk:
            x = block(x)

        # Policy head
        pi = nn.relu(self.policy_fc1(x))
        pi_logits = self.policy_fc2(pi)
        log_pi = pi_logits - mx.logsumexp(pi_logits, axis=-1, keepdims=True)

        # Value head
        v = nn.relu(self.value_fc1(x))
        v = mx.tanh(self.value_fc2(v))

        return log_pi, v

    def predict(self, state: Go9x9Statechart) -> Tuple[np.ndarray, float]:
        """
        Predict policy and value for single state.

        Returns:
            pi: [82] policy probabilities (not log)
            v: scalar value
        """
        turn, ko, board = encode_soft_config(state)

        # Add batch dimension
        turn = turn[None, :]
        ko = ko[None, :]
        board = board[None, :, :, :]

        log_pi, v = self(turn, ko, board)

        # Convert to numpy
        pi = np.array(mx.exp(log_pi[0]))
        v = float(v[0, 0])

        return pi, v


@dataclass
class TrainingExample:
    """Training example for policy-value network."""
    turn_config: np.ndarray     # [2]
    ko_config: np.ndarray       # [82]
    board_config: np.ndarray    # [9, 9, 3]
    policy: np.ndarray          # [82] MCTS visit counts
    value: float                # {-1, 0, 1} game outcome


class NNetWrapper:
    """
    Wrapper class matching alpha-zero-general NNet interface.

    Handles training, prediction, and model save/load.
    """

    def __init__(self, embed_dim: int = 256, num_res_blocks: int = 4,
                 lr: float = 0.001, use_deep_encoder: bool = True):
        self.embed_dim = embed_dim
        self.num_res_blocks = num_res_blocks
        self.lr = lr

        self.nnet = StatechartAlphaZeroNet(
            embed_dim=embed_dim,
            num_res_blocks=num_res_blocks,
            use_deep_encoder=use_deep_encoder
        )
        self.optimizer = optim.Adam(learning_rate=lr)

    def train(self, examples: List[TrainingExample], epochs: int = 10,
              batch_size: int = 64) -> dict:
        """
        Train on examples.

        Returns dict with loss history.
        """
        loss_fn = nn.value_and_grad(self.nnet, self._loss_fn)

        history = {'policy_loss': [], 'value_loss': [], 'total_loss': []}

        for epoch in range(epochs):
            # Shuffle examples
            np.random.shuffle(examples)

            epoch_losses = []
            for i in range(0, len(examples), batch_size):
                batch = examples[i:i + batch_size]
                if len(batch) < batch_size // 2:
                    continue

                # Stack batch
                turns = mx.array(np.stack([ex.turn_config for ex in batch]))
                kos = mx.array(np.stack([ex.ko_config for ex in batch]))
                boards = mx.array(np.stack([ex.board_config for ex in batch]))
                target_pis = mx.array(np.stack([ex.policy for ex in batch]))
                target_vs = mx.array([float(ex.value) for ex in batch])

                # Forward + backward
                (total_loss, (pi_loss, v_loss)), grads = loss_fn(
                    turns, kos, boards, target_pis, target_vs
                )

                # Update
                self.optimizer.update(self.nnet, grads)
                mx.eval(self.nnet.parameters())

                epoch_losses.append({
                    'total': float(total_loss),
                    'policy': float(pi_loss),
                    'value': float(v_loss)
                })

            if epoch_losses:
                avg_total = np.mean([l['total'] for l in epoch_losses])
                avg_pi = np.mean([l['policy'] for l in epoch_losses])
                avg_v = np.mean([l['value'] for l in epoch_losses])

                history['total_loss'].append(avg_total)
                history['policy_loss'].append(avg_pi)
                history['value_loss'].append(avg_v)

                if epoch % 5 == 0:
                    print(f"Epoch {epoch}: loss={avg_total:.4f} "
                          f"(pi={avg_pi:.4f}, v={avg_v:.4f})")

        return history

    def _loss_fn(self, turns: mx.array, kos: mx.array, boards: mx.array,
                 target_pis: mx.array, target_vs: mx.array) -> Tuple[mx.array, Tuple]:
        """Compute combined loss."""
        log_pi, v = self.nnet(turns, kos, boards)

        # Policy loss: cross-entropy
        loss_pi = -mx.mean(mx.sum(target_pis * log_pi, axis=-1))

        # Value loss: MSE
        loss_v = mx.mean((target_vs - v.squeeze()) ** 2)

        total_loss = loss_pi + loss_v
        return total_loss, (loss_pi, loss_v)

    def predict(self, state: Go9x9Statechart) -> Tuple[np.ndarray, float]:
        """Predict policy and value for single state."""
        return self.nnet.predict(state)

    def save_checkpoint(self, folder: str, filename: str):
        """Save model checkpoint."""
        filepath = Path(folder) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save weights
        weights = self.nnet.parameters()
        mx.savez(str(filepath.with_suffix('.npz')),
                 **{k: v for k, v in weights.items()})

        # Save config
        config = {
            'embed_dim': self.embed_dim,
            'num_res_blocks': self.num_res_blocks,
            'lr': self.lr
        }
        with open(filepath.with_suffix('.config'), 'wb') as f:
            pickle.dump(config, f)

    def load_checkpoint(self, folder: str, filename: str):
        """Load model checkpoint."""
        filepath = Path(folder) / filename

        # Load weights
        weights = dict(mx.load(str(filepath.with_suffix('.npz'))))
        self.nnet.load_weights(list(weights.items()))


def test_nnet():
    """Test the neural network."""
    print("Testing StatechartAlphaZeroNet...")

    # Create network
    nnet = StatechartAlphaZeroNet(embed_dim=128, num_res_blocks=2)

    # Create test state
    state = Go9x9Statechart()
    state.play_move(4, 4)

    # Encode
    turn, ko, board = encode_soft_config(state)
    turn = turn[None, :]
    ko = ko[None, :]
    board = board[None, :, :, :]

    # Forward pass
    log_pi, v = nnet(turn, ko, board)
    print(f"Log policy shape: {log_pi.shape}")  # [1, 82]
    print(f"Value shape: {v.shape}")            # [1, 1]
    print(f"Value range: [{float(v.min()):.3f}, {float(v.max()):.3f}]")

    # Test predict
    pi, val = nnet.predict(state)
    print(f"Policy shape: {pi.shape}")          # [82]
    print(f"Policy sum: {pi.sum():.4f}")        # Should be ~1.0
    print(f"Value: {val:.4f}")

    # Test wrapper
    wrapper = NNetWrapper(embed_dim=128, num_res_blocks=2)

    # Create dummy training examples
    examples = []
    for _ in range(20):
        s = Go9x9Statechart()
        t, k, b = encode_soft_config(s)
        examples.append(TrainingExample(
            turn_config=np.array(t),
            ko_config=np.array(k),
            board_config=np.array(b),
            policy=np.ones(82) / 82,
            value=np.random.choice([-1, 0, 1])
        ))

    # Train
    print("\nTraining on dummy data...")
    history = wrapper.train(examples, epochs=5, batch_size=10)
    print(f"Final loss: {history['total_loss'][-1]:.4f}")

    print("\nAll neural network tests passed!")


if __name__ == "__main__":
    test_nnet()
