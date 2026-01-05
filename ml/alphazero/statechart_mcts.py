"""
Statechart-Integrated Monte Carlo Tree Search

The key innovation: guards mask the policy BEFORE normalization,
guaranteeing 100% legal moves by construction.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import math
from typing import Dict, Optional, Tuple
import copy

from experiments.exp_go_9x9.go_statechart import (
    Go9x9Statechart, TurnState, KoState,
    BOARD_SIZE, TOTAL_POINTS, xy_to_idx, idx_to_xy
)


# MCTS constants
EPS = 1e-8  # Small value to avoid division by zero
ACTION_SIZE = 82


class StatechartMCTS:
    """
    Monte Carlo Tree Search with statechart-guaranteed legal moves.

    Key differences from standard MCTS:
    1. Guards computed from statechart, not learned
    2. Policy masked before normalization (100% legal)
    3. State transitions via statechart (preserves history)
    """

    def __init__(self, nnet, cpuct: float = 1.0, num_sims: int = 100):
        """
        Args:
            nnet: Neural network with predict(state) -> (pi, v)
            cpuct: Exploration constant for PUCT
            num_sims: Number of MCTS simulations per move
        """
        self.nnet = nnet
        self.cpuct = cpuct
        self.num_sims = num_sims

        # MCTS data structures
        self.Qsa: Dict[Tuple[str, int], float] = {}  # Q values
        self.Nsa: Dict[Tuple[str, int], int] = {}    # Visit counts per (s, a)
        self.Ns: Dict[str, int] = {}                  # Visit counts per s
        self.Ps: Dict[str, np.ndarray] = {}          # Policy priors
        self.Es: Dict[str, float] = {}               # Game ended
        self.Vs: Dict[str, np.ndarray] = {}          # Valid moves (guard mask)

        # State cache for efficient transitions
        self.states: Dict[str, Go9x9Statechart] = {}

        # Metrics tracking
        self.illegal_attempts = 0  # Should always be 0!
        self.total_expansions = 0

    def get_action_prob(self, state: Go9x9Statechart,
                        temp: float = 1.0) -> np.ndarray:
        """
        Get action probabilities from MCTS.

        Args:
            state: Current game state (statechart)
            temp: Temperature for action selection
                  temp=1 -> proportional to visit counts
                  temp=0 -> argmax (greedy)

        Returns:
            probs: [82] action probabilities
        """
        # Run simulations
        for _ in range(self.num_sims):
            self._search(state)

        # Get visit counts
        s = self._state_key(state)
        counts = np.array([
            self.Nsa.get((s, a), 0) for a in range(ACTION_SIZE)
        ])

        if temp == 0:
            # Greedy selection
            best_actions = np.argwhere(counts == counts.max()).flatten()
            probs = np.zeros(ACTION_SIZE)
            probs[np.random.choice(best_actions)] = 1.0
        else:
            # Temperature-scaled
            counts_temp = counts ** (1.0 / temp)
            probs = counts_temp / (counts_temp.sum() + EPS)

        return probs

    def _search(self, state: Go9x9Statechart) -> float:
        """
        Recursive MCTS search.

        Returns:
            v: Value of state from current player's perspective
        """
        s = self._state_key(state)

        # Check if game ended
        if s not in self.Es:
            ended = state.is_game_over()
            if ended:
                winner = state.winner()
                if winner is None:
                    self.Es[s] = 0  # Draw
                else:
                    # +1 if current player won, -1 if lost
                    current = 1 if state.turn == TurnState.BLACK else -1
                    winner_val = 1 if winner == 1 else -1  # BLACK=1
                    self.Es[s] = 1 if current == winner_val else -1
            else:
                self.Es[s] = 0  # Game not ended

        if self.Es[s] != 0 or state.is_game_over():
            return -self.Es[s]

        # Leaf node - expand
        if s not in self.Ps:
            self.total_expansions += 1

            # Get neural network prediction
            pi, v = self.nnet.predict(state)

            # CRITICAL: Compute guard mask from statechart
            guard_mask = self._compute_guard_mask(state)

            # Mask policy (this guarantees 100% legal moves)
            masked_pi = pi * guard_mask

            # Normalize
            pi_sum = masked_pi.sum()
            if pi_sum > EPS:
                masked_pi /= pi_sum
            else:
                # Fallback to uniform over valid moves (should be rare)
                masked_pi = guard_mask / (guard_mask.sum() + EPS)

            # Store
            self.Ps[s] = masked_pi
            self.Vs[s] = guard_mask
            self.Ns[s] = 0
            self.states[s] = copy.deepcopy(state)

            return -v

        # Select action with PUCT
        guard_mask = self.Vs[s]
        best_action = -1
        best_ucb = -float('inf')

        for a in range(ACTION_SIZE):
            if guard_mask[a] == 0:
                continue  # Skip invalid actions

            if (s, a) in self.Qsa:
                # UCB = Q + c * P * sqrt(N_s) / (1 + N_sa)
                q = self.Qsa[(s, a)]
                n_sa = self.Nsa[(s, a)]
            else:
                q = 0
                n_sa = 0

            ucb = q + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS) / (1 + n_sa)

            if ucb > best_ucb:
                best_ucb = ucb
                best_action = a

        a = best_action

        # Verify action is legal (should always pass)
        if guard_mask[a] == 0:
            self.illegal_attempts += 1
            raise RuntimeError(f"MCTS selected illegal action {a}! "
                              f"This should never happen with guard masking.")

        # Get next state
        next_state = copy.deepcopy(state)
        if a == 81:
            next_state.play_pass()
        else:
            x, y = idx_to_xy(a)
            success = next_state.play_move(x, y)
            if not success:
                self.illegal_attempts += 1
                raise RuntimeError(f"Statechart rejected move ({x},{y})! "
                                  f"Guard mask was wrong.")

        # Recurse
        v = self._search(next_state)

        # Backpropagate
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v

    def _compute_guard_mask(self, state: Go9x9Statechart) -> np.ndarray:
        """
        Compute binary mask from statechart guards.

        This is where statechart topology GUARANTEES legal moves.
        """
        mask = np.zeros(ACTION_SIZE, dtype=np.float32)

        for idx in range(TOTAL_POINTS):
            x, y = idx_to_xy(idx)
            # All three guards must pass
            if state.is_legal_move(x, y):
                mask[idx] = 1.0

        # Pass is always legal
        mask[81] = 1.0

        return mask

    def _state_key(self, state: Go9x9Statechart) -> str:
        """Generate unique key for state."""
        parts = [
            'B' if state.turn == TurnState.BLACK else 'W',
            f"ko:{state.ko_point}" if state.ko_state == KoState.KO_FORBIDDEN else "noko",
            f"p:{state.consecutive_passes}",
            ''.join(str(s) for s in state.board.stones)
        ]
        return '|'.join(parts)

    def reset(self):
        """Clear MCTS tree for new game."""
        self.Qsa.clear()
        self.Nsa.clear()
        self.Ns.clear()
        self.Ps.clear()
        self.Es.clear()
        self.Vs.clear()
        self.states.clear()

    def get_metrics(self) -> dict:
        """Get MCTS metrics for evaluation."""
        return {
            'illegal_attempts': self.illegal_attempts,
            'total_expansions': self.total_expansions,
            'illegal_rate': self.illegal_attempts / max(1, self.total_expansions),
            'tree_size': len(self.Ns)
        }


class DummyNet:
    """Dummy network for testing (uniform policy, zero value)."""

    def predict(self, state):
        pi = np.ones(ACTION_SIZE) / ACTION_SIZE
        v = 0.0
        return pi, v


def test_mcts():
    """Test MCTS with guard masking."""
    print("Testing StatechartMCTS...")

    # Create MCTS with dummy network
    nnet = DummyNet()
    mcts = StatechartMCTS(nnet, cpuct=1.0, num_sims=50)

    # Test from initial position
    state = Go9x9Statechart()
    print("\nInitial position:")
    print(f"Legal moves: {len(state.get_legal_moves()) + 1}")  # +1 for pass

    # Get action probs
    probs = mcts.get_action_prob(state, temp=1.0)
    print(f"Action probs shape: {probs.shape}")
    print(f"Sum of probs: {probs.sum():.4f}")
    print(f"Non-zero probs: {(probs > 0).sum()}")

    # Verify no illegal moves have probability
    guard_mask = mcts._compute_guard_mask(state)
    illegal_prob = np.sum(probs * (1 - guard_mask))
    print(f"Probability mass on illegal moves: {illegal_prob:.6f}")
    assert illegal_prob < EPS, "Illegal moves should have zero probability!"

    # Play some moves and test again
    print("\nPlaying some moves...")
    state.play_move(4, 4)  # Black center
    state.play_move(4, 5)  # White adjacent

    probs = mcts.get_action_prob(state, temp=1.0)
    guard_mask = mcts._compute_guard_mask(state)
    illegal_prob = np.sum(probs * (1 - guard_mask))
    print(f"After 2 moves, illegal prob: {illegal_prob:.6f}")
    assert illegal_prob < EPS

    # Test Ko situation
    print("\nTesting Ko situation...")
    ko_state = Go9x9Statechart()
    # Set up a Ko position manually
    # This is a simplified test - real Ko would require a capture
    ko_state.ko_state = KoState.KO_FORBIDDEN
    ko_state.ko_point = (0, 0)
    ko_state.board.set(0, 1, 2)  # White
    ko_state.board.set(1, 0, 2)  # White

    guard_mask = mcts._compute_guard_mask(ko_state)
    print(f"Ko point (0,0) is forbidden: {guard_mask[0] == 0}")
    assert guard_mask[0] == 0, "Ko point should be forbidden!"

    # Get metrics
    metrics = mcts.get_metrics()
    print(f"\nMCTS Metrics:")
    print(f"  Illegal attempts: {metrics['illegal_attempts']}")
    print(f"  Total expansions: {metrics['total_expansions']}")
    print(f"  Illegal rate: {metrics['illegal_rate']:.6f}")
    print(f"  Tree size: {metrics['tree_size']}")

    assert metrics['illegal_attempts'] == 0, "Should have zero illegal attempts!"

    print("\nAll MCTS tests passed! 100% legal moves guaranteed.")


if __name__ == "__main__":
    test_mcts()
