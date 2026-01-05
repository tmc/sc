"""
Statechart Encoder: Convert statechart state to soft configurations.

The key innovation is replacing flat board encoding with hierarchical
soft state configurations that encode OR-states, AND-states, and history.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Tuple, Optional

from experiments.exp_go_9x9.go_statechart import (
    Go9x9Statechart, TurnState, KoState,
    BOARD_SIZE, TOTAL_POINTS, EMPTY, BLACK, WHITE,
    xy_to_idx, idx_to_xy
)


# Soft configuration dimensions
TURN_DIM = 2          # Black | White
KO_DIM = 82           # NoKo | Ko@(0..80)
BOARD_DIM = 81 * 3    # 81 positions x 3 values (empty, black, white)
TOTAL_SOFT_DIM = TURN_DIM + KO_DIM + BOARD_DIM  # 327


def encode_soft_config(state: Go9x9Statechart) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Encode statechart state into soft configuration tensors.

    Returns:
        turn_config: [2] - P(Black), P(White)
        ko_config: [82] - P(NoKo), P(Ko@0), ..., P(Ko@80)
        board_config: [9, 9, 3] - Per-cell: P(Empty), P(Black), P(White)
    """
    # Turn encoding (one-hot OR-state)
    turn_config = mx.zeros(TURN_DIM)
    if state.turn == TurnState.BLACK:
        turn_config = mx.array([1.0, 0.0])
    else:
        turn_config = mx.array([0.0, 1.0])

    # Ko encoding (one-hot OR-state with history)
    ko_np = np.zeros(KO_DIM, dtype=np.float32)
    if state.ko_state == KoState.NO_KO or state.ko_point is None:
        ko_np[0] = 1.0
    else:
        x, y = state.ko_point
        idx = y * BOARD_SIZE + x + 1  # +1 because idx 0 is NoKo
        ko_np[idx] = 1.0
    ko_config = mx.array(ko_np)

    # Board encoding (per-cell one-hot)
    board_np = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.float32)
    for idx in range(TOTAL_POINTS):
        x, y = idx_to_xy(idx)
        stone = state.board.get(x, y)
        board_np[y, x, stone] = 1.0
    board_config = mx.array(board_np)

    return turn_config, ko_config, board_config


def encode_soft_config_batch(states: list) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Encode batch of statechart states.

    Returns:
        turn_configs: [B, 2]
        ko_configs: [B, 82]
        board_configs: [B, 9, 9, 3]
    """
    turns, kos, boards = [], [], []
    for state in states:
        t, k, b = encode_soft_config(state)
        turns.append(t)
        kos.append(k)
        boards.append(b)

    return mx.stack(turns), mx.stack(kos), mx.stack(boards)


def encode_flat_config(state: Go9x9Statechart) -> mx.array:
    """
    Encode statechart state as flat vector [327].

    Concatenates: turn[2] + ko[82] + board[243]
    """
    turn, ko, board = encode_soft_config(state)
    return mx.concatenate([turn, ko, board.reshape(-1)])


class StatechartEncoder(nn.Module):
    """
    Neural network encoder for statechart soft configurations.

    Converts hierarchical soft configs into a unified embedding.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()

        # Turn encoding (OR-state)
        self.turn_encoder = nn.Sequential(
            nn.Linear(TURN_DIM, 32),
            nn.ReLU()
        )

        # Ko encoding (OR-state with history)
        self.ko_encoder = nn.Sequential(
            nn.Linear(KO_DIM, 64),
            nn.ReLU()
        )

        # Board encoding (convolutional)
        self.board_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.board_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.board_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.board_pool = nn.Sequential(
            nn.ReLU(),
            # Global average pooling handled in forward
        )
        self.board_fc = nn.Linear(64, 128)

        # Combine all encodings
        self.combine = nn.Sequential(
            nn.Linear(32 + 64 + 128, embed_dim),
            nn.ReLU()
        )

    def __call__(self, turn: mx.array, ko: mx.array,
                 board: mx.array) -> mx.array:
        """
        Encode soft configuration into embedding.

        Args:
            turn: [B, 2] soft turn config
            ko: [B, 82] soft ko config
            board: [B, 9, 9, 3] soft board config

        Returns:
            [B, embed_dim] combined embedding
        """
        # Encode turn
        t = self.turn_encoder(turn)  # [B, 32]

        # Encode ko
        k = self.ko_encoder(ko)  # [B, 64]

        # Encode board (MLX uses NHWC format)
        # Input is [B, H, W, C] which is what MLX expects
        b = nn.relu(self.board_conv1(board))  # [B, 9, 9, 64]
        b = nn.relu(self.board_conv2(b))
        b = nn.relu(self.board_conv3(b))
        # Global average pool over spatial dimensions
        b = b.mean(axis=(1, 2))  # [B, 64]
        b = nn.relu(self.board_fc(b))  # [B, 128]

        # Combine
        combined = mx.concatenate([t, k, b], axis=-1)  # [B, 224]
        return self.combine(combined)  # [B, embed_dim]


class ResidualBlock(nn.Module):
    """Residual block for board encoding (NHWC format)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm(channels)  # LayerNorm works with last dim
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm(channels)

    def __call__(self, x: mx.array) -> mx.array:
        # x is [B, H, W, C] in NHWC format
        residual = x
        x = nn.relu(self.ln1(self.conv1(x)))
        x = self.ln2(self.conv2(x))
        return nn.relu(x + residual)


class DeepStatechartEncoder(nn.Module):
    """
    Deeper encoder with residual blocks for board processing.
    """

    def __init__(self, embed_dim: int = 256, num_res_blocks: int = 6):
        super().__init__()

        # Turn encoding
        self.turn_encoder = nn.Sequential(
            nn.Linear(TURN_DIM, 32),
            nn.ReLU()
        )

        # Ko encoding
        self.ko_encoder = nn.Sequential(
            nn.Linear(KO_DIM, 64),
            nn.ReLU()
        )

        # Board encoding with residual blocks (NHWC format)
        self.board_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.board_ln = nn.LayerNorm(64)

        self.res_blocks = [ResidualBlock(64) for _ in range(num_res_blocks)]
        self.board_fc = nn.Linear(64, 128)

        # Combine
        self.combine = nn.Sequential(
            nn.Linear(32 + 64 + 128, embed_dim),
            nn.ReLU()
        )

    def __call__(self, turn: mx.array, ko: mx.array,
                 board: mx.array) -> mx.array:
        """Encode with residual blocks."""
        # Encode turn and ko
        t = self.turn_encoder(turn)
        k = self.ko_encoder(ko)

        # Encode board with residual blocks (NHWC format)
        # board is [B, H, W, C] = [B, 9, 9, 3]
        b = nn.relu(self.board_ln(self.board_conv(board)))  # [B, 9, 9, 64]
        for block in self.res_blocks:
            b = block(b)
        b = b.mean(axis=(1, 2))  # Global pool over H, W -> [B, 64]
        b = nn.relu(self.board_fc(b))

        # Combine
        combined = mx.concatenate([t, k, b], axis=-1)
        return self.combine(combined)


def test_encoder():
    """Test the encoder."""
    print("Testing StatechartEncoder...")

    # Create a test state
    state = Go9x9Statechart()
    state.play_move(4, 4)  # Black plays center

    # Encode
    turn, ko, board = encode_soft_config(state)
    print(f"Turn shape: {turn.shape}")  # [2]
    print(f"Ko shape: {ko.shape}")      # [82]
    print(f"Board shape: {board.shape}")  # [9, 9, 3]

    # Test flat encoding
    flat = encode_flat_config(state)
    print(f"Flat config shape: {flat.shape}")  # [327]

    # Test neural encoder
    encoder = StatechartEncoder(embed_dim=256)

    # Add batch dimension
    turn_batch = turn[None, :]
    ko_batch = ko[None, :]
    board_batch = board[None, :, :, :]

    embedding = encoder(turn_batch, ko_batch, board_batch)
    print(f"Embedding shape: {embedding.shape}")  # [1, 256]

    # Test batch encoding
    states = [Go9x9Statechart() for _ in range(4)]
    turns, kos, boards = encode_soft_config_batch(states)
    print(f"Batch turn shape: {turns.shape}")    # [4, 2]
    print(f"Batch ko shape: {kos.shape}")        # [4, 82]
    print(f"Batch board shape: {boards.shape}")  # [4, 9, 9, 3]

    embeddings = encoder(turns, kos, boards)
    print(f"Batch embeddings shape: {embeddings.shape}")  # [4, 256]

    print("All encoder tests passed!")


if __name__ == "__main__":
    test_encoder()
