"""
StatechartAlphaZero Self-Play Coach

Generates training data through self-play with statechart-guaranteed
legal moves. Implements the AlphaZero training loop.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import List, Tuple, Optional
from collections import deque
import copy
from tqdm import tqdm

from experiments.exp_go_9x9.go_statechart import (
    Go9x9Statechart, TurnState, KoState, BoardState,
    BOARD_SIZE, TOTAL_POINTS, BLACK, WHITE,
    xy_to_idx, idx_to_xy
)

try:
    from .statechart_mcts import StatechartMCTS
    from .statechart_encoder import encode_soft_config
    from .statechart_nnet import TrainingExample, NNetWrapper
except ImportError:
    from statechart_mcts import StatechartMCTS
    from statechart_encoder import encode_soft_config
    from statechart_nnet import TrainingExample, NNetWrapper


ACTION_SIZE = 82
TEMP_THRESHOLD = 30  # Use temp=1 for first N moves, then temp=0


class StatechartCoach:
    """
    Self-play coach for StatechartAlphaZero.

    Manages the training loop:
    1. Self-play to generate training data
    2. Train neural network on collected data
    3. Evaluate new model vs previous best
    4. Replace best model if new one wins
    """

    def __init__(self, nnet: NNetWrapper, mcts_sims: int = 100,
                 cpuct: float = 1.0, temp_threshold: int = TEMP_THRESHOLD,
                 history_size: int = 100000, checkpoint_dir: str = './checkpoints'):
        """
        Args:
            nnet: Neural network wrapper
            mcts_sims: Number of MCTS simulations per move
            cpuct: PUCT exploration constant
            temp_threshold: Move number after which to use temp=0
            history_size: Maximum training examples to keep
            checkpoint_dir: Directory for model checkpoints
        """
        self.nnet = nnet
        self.mcts_sims = mcts_sims
        self.cpuct = cpuct
        self.temp_threshold = temp_threshold
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.train_examples = deque(maxlen=history_size)

        # Metrics
        self.total_games = 0
        self.total_moves = 0
        self.illegal_moves = 0  # Should always be 0!

    def execute_episode(self, verbose: bool = False) -> List[TrainingExample]:
        """
        Execute one self-play episode.

        Returns:
            List of training examples with values filled in
        """
        examples = []
        state = Go9x9Statechart()
        mcts = StatechartMCTS(self.nnet, cpuct=self.cpuct,
                              num_sims=self.mcts_sims)

        step = 0
        while not state.is_game_over():
            step += 1

            # Temperature schedule
            temp = 1.0 if step < self.temp_threshold else 0.0

            # Get MCTS policy
            pi = mcts.get_action_prob(state, temp=temp)

            # Get soft configuration for training
            turn, ko, board = encode_soft_config(state)

            # Store example (value filled later)
            examples.append({
                'turn': np.array(turn),
                'ko': np.array(ko),
                'board': np.array(board),
                'pi': pi,
                'player': 1 if state.turn == TurnState.BLACK else -1
            })

            # Sample action
            action = np.random.choice(ACTION_SIZE, p=pi)

            # Execute action
            if action == 81:
                state.play_pass()
                if verbose:
                    print(f"Move {step}: Pass")
            else:
                x, y = idx_to_xy(action)
                success = state.play_move(x, y)
                if not success:
                    self.illegal_moves += 1
                    raise RuntimeError(f"Illegal move at step {step}! "
                                      f"This should never happen.")
                if verbose:
                    print(f"Move {step}: ({x}, {y})")

            self.total_moves += 1

            # Prevent infinite games (very long games are rare)
            if step > 500:
                print("Warning: Episode exceeded 500 moves, ending")
                break

        # Get game result
        winner = state.winner()
        if verbose:
            b_score, w_score = state.score()
            print(f"Game over! Black: {b_score}, White: {w_score}")
            print(f"Winner: {'Black' if winner == BLACK else 'White' if winner == WHITE else 'Draw'}")

        # Fill in values from game outcome
        result_examples = []
        for ex in examples:
            if winner is None:
                value = 0.0  # Draw
            else:
                winner_player = 1 if winner == BLACK else -1
                value = 1.0 if ex['player'] == winner_player else -1.0

            result_examples.append(TrainingExample(
                turn_config=ex['turn'],
                ko_config=ex['ko'],
                board_config=ex['board'],
                policy=ex['pi'],
                value=value
            ))

        self.total_games += 1
        return result_examples

    def execute_episode_with_symmetries(self, verbose: bool = False
                                         ) -> List[TrainingExample]:
        """
        Execute episode and apply 8-fold symmetries for data augmentation.
        """
        examples = self.execute_episode(verbose=verbose)

        augmented = []
        for ex in examples:
            # Original
            augmented.append(ex)

            # Apply 7 additional symmetries (rotations + flips)
            board = ex.board_config
            pi_board = ex.policy[:81].reshape(BOARD_SIZE, BOARD_SIZE)
            pi_pass = ex.policy[81]

            for rot in range(1, 4):  # 90, 180, 270 degrees
                new_board = np.rot90(board, rot, axes=(0, 1))
                new_pi_board = np.rot90(pi_board, rot)
                new_pi = np.concatenate([new_pi_board.ravel(), [pi_pass]])

                augmented.append(TrainingExample(
                    turn_config=ex.turn_config,
                    ko_config=ex.ko_config,
                    board_config=new_board,
                    policy=new_pi,
                    value=ex.value
                ))

            # Horizontal flip
            flip_board = np.fliplr(board)
            flip_pi_board = np.fliplr(pi_board)
            flip_pi = np.concatenate([flip_pi_board.ravel(), [pi_pass]])

            augmented.append(TrainingExample(
                turn_config=ex.turn_config,
                ko_config=ex.ko_config,
                board_config=flip_board,
                policy=flip_pi,
                value=ex.value
            ))

            # Flip + rotations
            for rot in range(1, 4):
                new_board = np.rot90(flip_board, rot, axes=(0, 1))
                new_pi_board = np.rot90(flip_pi_board, rot)
                new_pi = np.concatenate([new_pi_board.ravel(), [pi_pass]])

                augmented.append(TrainingExample(
                    turn_config=ex.turn_config,
                    ko_config=ex.ko_config,
                    board_config=new_board,
                    policy=new_pi,
                    value=ex.value
                ))

        return augmented

    def learn(self, num_iters: int = 100, num_episodes: int = 100,
              epochs_per_iter: int = 10, batch_size: int = 64,
              use_symmetries: bool = True, verbose: bool = False):
        """
        Main training loop.

        Args:
            num_iters: Number of training iterations
            num_episodes: Self-play games per iteration
            epochs_per_iter: Training epochs per iteration
            batch_size: Training batch size
            use_symmetries: Whether to use 8-fold symmetry augmentation
            verbose: Print detailed progress
        """
        for iteration in range(1, num_iters + 1):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}/{num_iters}")
            print(f"{'='*60}")

            # Self-play
            print(f"\nSelf-play: {num_episodes} episodes...")
            iter_examples = []
            for ep in tqdm(range(num_episodes), desc="Self-play"):
                if use_symmetries:
                    examples = self.execute_episode_with_symmetries(verbose=verbose)
                else:
                    examples = self.execute_episode(verbose=verbose)
                iter_examples.extend(examples)

            print(f"Generated {len(iter_examples)} training examples")

            # Add to history
            self.train_examples.extend(iter_examples)
            print(f"Total training examples: {len(self.train_examples)}")

            # Train
            print(f"\nTraining for {epochs_per_iter} epochs...")
            history = self.nnet.train(
                list(self.train_examples),
                epochs=epochs_per_iter,
                batch_size=batch_size
            )

            # Save checkpoint
            checkpoint_name = f"checkpoint_{iteration:04d}"
            self.nnet.save_checkpoint(str(self.checkpoint_dir), checkpoint_name)
            print(f"Saved checkpoint: {checkpoint_name}")

            # Print metrics
            print(f"\nMetrics:")
            print(f"  Total games: {self.total_games}")
            print(f"  Total moves: {self.total_moves}")
            print(f"  Illegal moves: {self.illegal_moves}")
            print(f"  Illegal rate: {self.illegal_moves / max(1, self.total_moves):.6f}")
            print(f"  Final loss: {history['total_loss'][-1]:.4f}")

    def arena_compare(self, nnet1: NNetWrapper, nnet2: NNetWrapper,
                      num_games: int = 40) -> Tuple[int, int, int]:
        """
        Compare two networks by playing games between them.

        Returns:
            (nnet1_wins, nnet2_wins, draws)
        """
        n1_wins = 0
        n2_wins = 0
        draws = 0

        for game_idx in tqdm(range(num_games), desc="Arena"):
            # Alternate who plays Black
            if game_idx % 2 == 0:
                black_net, white_net = nnet1, nnet2
            else:
                black_net, white_net = nnet2, nnet1

            # Play game
            state = Go9x9Statechart()
            mcts_black = StatechartMCTS(black_net, cpuct=self.cpuct,
                                        num_sims=self.mcts_sims)
            mcts_white = StatechartMCTS(white_net, cpuct=self.cpuct,
                                        num_sims=self.mcts_sims)

            step = 0
            while not state.is_game_over() and step < 300:
                step += 1

                if state.turn == TurnState.BLACK:
                    pi = mcts_black.get_action_prob(state, temp=0)
                else:
                    pi = mcts_white.get_action_prob(state, temp=0)

                action = np.argmax(pi)

                if action == 81:
                    state.play_pass()
                else:
                    x, y = idx_to_xy(action)
                    state.play_move(x, y)

            # Determine winner
            winner = state.winner()
            if winner is None:
                draws += 1
            elif (winner == BLACK and game_idx % 2 == 0) or \
                 (winner == WHITE and game_idx % 2 == 1):
                n1_wins += 1
            else:
                n2_wins += 1

        return n1_wins, n2_wins, draws


def test_coach():
    """Test the coach."""
    print("Testing StatechartCoach...")

    # Create network and coach
    nnet = NNetWrapper(embed_dim=64, num_res_blocks=2)
    coach = StatechartCoach(nnet, mcts_sims=10, temp_threshold=10)

    # Execute one episode
    print("\nExecuting self-play episode...")
    examples = coach.execute_episode(verbose=True)
    print(f"Generated {len(examples)} examples")

    # Check examples
    for i, ex in enumerate(examples[:3]):
        print(f"\nExample {i}:")
        print(f"  Turn config shape: {ex.turn_config.shape}")
        print(f"  Ko config shape: {ex.ko_config.shape}")
        print(f"  Board config shape: {ex.board_config.shape}")
        print(f"  Policy sum: {ex.policy.sum():.4f}")
        print(f"  Value: {ex.value}")

    # Test symmetries
    print("\nTesting symmetry augmentation...")
    aug_examples = coach.execute_episode_with_symmetries()
    print(f"Augmented examples: {len(aug_examples)}")

    # Check metrics
    print(f"\nCoach metrics:")
    print(f"  Total games: {coach.total_games}")
    print(f"  Total moves: {coach.total_moves}")
    print(f"  Illegal moves: {coach.illegal_moves}")

    assert coach.illegal_moves == 0, "Should have zero illegal moves!"

    print("\nAll coach tests passed! 100% legal moves guaranteed.")


if __name__ == "__main__":
    test_coach()
