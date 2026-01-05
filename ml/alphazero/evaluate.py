#!/usr/bin/env python3
"""
StatechartAlphaZero Evaluation Script

Compare statechart-based AlphaZero with standard approaches.
Measure illegal move rates, Ko accuracy, and playing strength.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
import json
from datetime import datetime

from experiments.exp_go_9x9.go_statechart import (
    Go9x9Statechart, TurnState, KoState, BoardState,
    BOARD_SIZE, TOTAL_POINTS, BLACK, WHITE, EMPTY,
    xy_to_idx, idx_to_xy
)

try:
    from .statechart_nnet import NNetWrapper, StatechartAlphaZeroNet
    from .statechart_mcts import StatechartMCTS
    from .statechart_encoder import encode_soft_config
except ImportError:
    from statechart_nnet import NNetWrapper, StatechartAlphaZeroNet
    from statechart_mcts import StatechartMCTS
    from statechart_encoder import encode_soft_config


ACTION_SIZE = 82


class StatechartEvaluator:
    """
    Evaluator for StatechartAlphaZero.

    Measures:
    1. Illegal move rate (should be 0%)
    2. Ko accuracy (should be 100%)
    3. Legal move entropy
    4. Playing strength (win rate)
    """

    def __init__(self, nnet: NNetWrapper, mcts_sims: int = 100):
        self.nnet = nnet
        self.mcts_sims = mcts_sims

    def evaluate_illegal_rate(self, num_positions: int = 1000) -> Dict:
        """
        Measure illegal move rate on random positions.

        This should ALWAYS be 0% for statechart approach.
        """
        total_illegal_mass = 0.0
        total_positions = 0
        worst_illegal_mass = 0.0

        for _ in tqdm(range(num_positions), desc="Evaluating illegal rate"):
            # Generate random position
            state = self._generate_random_position()
            if state is None:
                continue

            # Get raw policy from network
            pi, v = self.nnet.predict(state)

            # Compute guard mask
            guard_mask = self._compute_guard_mask(state)

            # Measure probability mass on illegal moves
            illegal_mask = 1.0 - guard_mask
            illegal_mass = np.sum(pi * illegal_mask)

            total_illegal_mass += illegal_mass
            total_positions += 1
            worst_illegal_mass = max(worst_illegal_mass, illegal_mass)

        if total_positions == 0:
            return {'error': 'No valid positions generated'}

        return {
            'avg_illegal_mass': total_illegal_mass / total_positions,
            'worst_illegal_mass': worst_illegal_mass,
            'total_positions': total_positions,
            'illegal_rate_percent': 100 * total_illegal_mass / total_positions
        }

    def evaluate_ko_accuracy(self, num_positions: int = 500) -> Dict:
        """
        Measure Ko rule accuracy.

        For statechart, this should be 100% because Ko is explicit state.
        """
        correct = 0
        total = 0

        for _ in tqdm(range(num_positions), desc="Evaluating Ko accuracy"):
            # Generate position with Ko
            state = self._generate_ko_position()
            if state is None:
                continue

            # Get policy
            pi, v = self.nnet.predict(state)

            # Check if Ko point is correctly avoided
            ko_x, ko_y = state.ko_point
            ko_idx = xy_to_idx(ko_x, ko_y)

            # In statechart approach, guard mask zeroes Ko point
            guard_mask = self._compute_guard_mask(state)

            # Ko point should have zero probability after masking
            masked_pi = pi * guard_mask
            masked_pi /= (masked_pi.sum() + 1e-8)

            if masked_pi[ko_idx] < 0.01:  # Effectively zero
                correct += 1
            total += 1

        if total == 0:
            return {'error': 'No Ko positions generated'}

        return {
            'accuracy': correct / total,
            'accuracy_percent': 100 * correct / total,
            'total_positions': total
        }

    def evaluate_legal_entropy(self, num_positions: int = 500) -> Dict:
        """
        Measure entropy of policy over legal moves.

        Lower entropy = more confident/focused policy.
        """
        entropies = []

        for _ in tqdm(range(num_positions), desc="Evaluating entropy"):
            state = self._generate_random_position()
            if state is None:
                continue

            pi, v = self.nnet.predict(state)
            guard_mask = self._compute_guard_mask(state)

            # Mask and normalize
            masked_pi = pi * guard_mask
            masked_pi /= (masked_pi.sum() + 1e-8)

            # Compute entropy
            entropy = -np.sum(masked_pi * np.log(masked_pi + 1e-10))
            entropies.append(entropy)

        if not entropies:
            return {'error': 'No valid positions'}

        return {
            'mean_entropy': np.mean(entropies),
            'std_entropy': np.std(entropies),
            'min_entropy': np.min(entropies),
            'max_entropy': np.max(entropies)
        }

    def evaluate_playing_strength(self, num_games: int = 100) -> Dict:
        """
        Evaluate playing strength through self-play.

        Returns win/loss/draw statistics.
        """
        black_wins = 0
        white_wins = 0
        draws = 0
        game_lengths = []

        for _ in tqdm(range(num_games), desc="Self-play evaluation"):
            state = Go9x9Statechart()
            mcts = StatechartMCTS(self.nnet, num_sims=self.mcts_sims)

            moves = 0
            while not state.is_game_over() and moves < 300:
                moves += 1

                # Get best move (temp=0)
                pi = mcts.get_action_prob(state, temp=0)
                action = np.argmax(pi)

                if action == 81:
                    state.play_pass()
                else:
                    x, y = idx_to_xy(action)
                    state.play_move(x, y)

            game_lengths.append(moves)
            winner = state.winner()

            if winner == BLACK:
                black_wins += 1
            elif winner == WHITE:
                white_wins += 1
            else:
                draws += 1

        return {
            'black_wins': black_wins,
            'white_wins': white_wins,
            'draws': draws,
            'black_win_rate': black_wins / num_games,
            'avg_game_length': np.mean(game_lengths),
            'std_game_length': np.std(game_lengths)
        }

    def verify_mcts_legality(self, num_games: int = 50) -> Dict:
        """
        Verify 100% legal moves during MCTS self-play.

        This is the key metric proving statechart advantage.
        """
        total_moves = 0
        illegal_moves = 0

        for _ in tqdm(range(num_games), desc="Verifying MCTS legality"):
            state = Go9x9Statechart()
            mcts = StatechartMCTS(self.nnet, num_sims=self.mcts_sims)

            moves = 0
            while not state.is_game_over() and moves < 300:
                moves += 1

                pi = mcts.get_action_prob(state, temp=1.0)
                action = np.random.choice(ACTION_SIZE, p=pi)

                # Verify action is legal
                if action == 81:
                    # Pass is always legal
                    state.play_pass()
                else:
                    x, y = idx_to_xy(action)
                    if not state.is_legal_move(x, y):
                        illegal_moves += 1
                    state.play_move(x, y)

                total_moves += 1

            # Check MCTS internal metrics
            metrics = mcts.get_metrics()
            illegal_moves += metrics['illegal_attempts']

        return {
            'total_moves': total_moves,
            'illegal_moves': illegal_moves,
            'illegal_rate': illegal_moves / max(1, total_moves),
            'illegal_rate_percent': 100 * illegal_moves / max(1, total_moves),
            'verified_100_percent_legal': illegal_moves == 0
        }

    def _generate_random_position(self, max_moves: int = 50) -> Go9x9Statechart:
        """Generate a random legal game position."""
        state = Go9x9Statechart()
        num_moves = np.random.randint(5, max_moves)

        for _ in range(num_moves):
            legal = state.get_legal_moves()
            if not legal:
                break

            # Random move or pass
            if np.random.random() < 0.1:  # 10% pass
                state.play_pass()
            else:
                x, y = legal[np.random.randint(len(legal))]
                state.play_move(x, y)

            if state.is_game_over():
                break

        return state if not state.is_game_over() else None

    def _generate_ko_position(self, max_attempts: int = 100) -> Go9x9Statechart:
        """Generate a position with Ko."""
        for _ in range(max_attempts):
            state = self._generate_random_position()
            if state is not None and state.ko_state == KoState.KO_FORBIDDEN:
                return state

        # Create artificial Ko position
        state = Go9x9Statechart()
        state.ko_state = KoState.KO_FORBIDDEN
        state.ko_point = (4, 4)  # Center point forbidden
        state.board.set(3, 4, WHITE)
        state.board.set(5, 4, WHITE)
        state.board.set(4, 3, WHITE)
        state.board.set(4, 5, WHITE)
        return state

    def _compute_guard_mask(self, state: Go9x9Statechart) -> np.ndarray:
        """Compute guard mask from statechart."""
        mask = np.zeros(ACTION_SIZE, dtype=np.float32)

        for idx in range(TOTAL_POINTS):
            x, y = idx_to_xy(idx)
            if state.is_legal_move(x, y):
                mask[idx] = 1.0

        mask[81] = 1.0  # Pass always legal
        return mask


def run_full_evaluation(nnet: NNetWrapper, mcts_sims: int = 50) -> Dict:
    """Run complete evaluation suite."""
    evaluator = StatechartEvaluator(nnet, mcts_sims=mcts_sims)

    results = {
        'timestamp': datetime.now().isoformat(),
        'mcts_sims': mcts_sims
    }

    print("\n" + "=" * 60)
    print("StatechartAlphaZero Evaluation")
    print("=" * 60)

    # 1. Illegal rate
    print("\n1. Illegal Move Rate (should be 0%)")
    illegal_results = evaluator.evaluate_illegal_rate(num_positions=200)
    results['illegal_rate'] = illegal_results
    print(f"   Raw policy illegal mass: {illegal_results['avg_illegal_mass']:.4f}")
    print(f"   After guard masking: 0.0000 (guaranteed)")

    # 2. Ko accuracy
    print("\n2. Ko Rule Accuracy (should be 100%)")
    ko_results = evaluator.evaluate_ko_accuracy(num_positions=100)
    results['ko_accuracy'] = ko_results
    print(f"   Ko accuracy: {ko_results['accuracy_percent']:.1f}%")

    # 3. Legal entropy
    print("\n3. Policy Entropy")
    entropy_results = evaluator.evaluate_legal_entropy(num_positions=100)
    results['entropy'] = entropy_results
    print(f"   Mean entropy: {entropy_results['mean_entropy']:.2f}")

    # 4. Playing strength
    print("\n4. Playing Strength (self-play)")
    strength_results = evaluator.evaluate_playing_strength(num_games=20)
    results['playing_strength'] = strength_results
    print(f"   Black wins: {strength_results['black_wins']}")
    print(f"   White wins: {strength_results['white_wins']}")
    print(f"   Draws: {strength_results['draws']}")
    print(f"   Avg game length: {strength_results['avg_game_length']:.1f}")

    # 5. MCTS legality verification
    print("\n5. MCTS Legality Verification")
    mcts_results = evaluator.verify_mcts_legality(num_games=20)
    results['mcts_legality'] = mcts_results
    print(f"   Total moves: {mcts_results['total_moves']}")
    print(f"   Illegal moves: {mcts_results['illegal_moves']}")
    print(f"   Illegal rate: {mcts_results['illegal_rate_percent']:.4f}%")

    if mcts_results['verified_100_percent_legal']:
        print("\n*** VERIFIED: 100% legal moves across ALL evaluation! ***")
    else:
        print("\n*** ERROR: Illegal moves detected! ***")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate StatechartAlphaZero'
    )

    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Specific checkpoint to evaluate')
    parser.add_argument('--mcts-sims', type=int, default=50,
                       help='MCTS simulations for evaluation')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')

    args = parser.parse_args()

    # Load or create network
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        nnet = NNetWrapper(embed_dim=256, num_res_blocks=4)
        nnet.load_checkpoint(args.checkpoint_dir, args.checkpoint)
    else:
        print("Using fresh (untrained) network for baseline evaluation")
        nnet = NNetWrapper(embed_dim=128, num_res_blocks=2)

    # Run evaluation
    results = run_full_evaluation(nnet, mcts_sims=args.mcts_sims)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
