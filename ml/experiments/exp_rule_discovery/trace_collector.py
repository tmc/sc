"""
Trace Collector - Observes games and collects transition data.

Collects (state, action, next_state, outcome, legal) tuples from gameplay.
These traces are the raw material for rule discovery.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from enum import Enum, auto
import random
import json


class Outcome(Enum):
    """Possible outcomes of a transition."""
    VALID = auto()      # Legal move, game continues
    INVALID = auto()    # Illegal move, rejected
    WIN = auto()        # Move caused a win
    LOSS = auto()       # Move caused a loss
    DRAW = auto()       # Move caused a draw


@dataclass
class Transition:
    """A single state transition from gameplay."""
    state: Dict[str, Any]       # Raw game state (board, turn, etc)
    action: Any                  # Action taken
    next_state: Dict[str, Any]  # Resulting state
    outcome: Outcome            # What happened
    features: Dict[str, float] = field(default_factory=dict)  # Computed features

    def to_dict(self) -> Dict[str, Any]:
        return {
            'state': self.state,
            'action': self.action,
            'next_state': self.next_state,
            'outcome': self.outcome.name,
            'features': self.features,
        }


@dataclass
class GameTrace:
    """Complete trace of a single game."""
    transitions: List[Transition] = field(default_factory=list)
    winner: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_transition(self, trans: Transition):
        self.transitions.append(trans)

    def __len__(self) -> int:
        return len(self.transitions)


class TraceCollector:
    """
    Collects game traces by observing gameplay.

    Can collect from:
    - Random play (exploration)
    - Self-play with policy
    - Expert demonstrations
    """

    def __init__(self, game_interface: 'GameInterface'):
        """
        Args:
            game_interface: Object implementing get_state(), get_actions(),
                           apply_action(), is_terminal(), get_winner()
        """
        self.game = game_interface
        self.traces: List[GameTrace] = []
        self.feature_extractors: List[Callable] = []

    def add_feature_extractor(self, extractor: Callable[[Dict], Dict[str, float]]):
        """Add a feature extractor that computes features from raw state."""
        self.feature_extractors.append(extractor)

    def extract_features(self, state: Dict, action: Any = None) -> Dict[str, float]:
        """Extract all features from a state and action."""
        features = {}
        for extractor in self.feature_extractors:
            features.update(extractor(state))

        # Add action-specific features
        if action is not None and 'board' in state:
            board = state['board']
            if isinstance(action, int) and 0 <= action < len(board):
                features['target_is_empty'] = 1.0 if board[action] == 0 else 0.0
                features['target_value'] = float(board[action])
                features['action_pos'] = float(action)
                # Position type for 3x3
                if len(board) == 9:
                    features['is_center'] = 1.0 if action == 4 else 0.0
                    features['is_corner'] = 1.0 if action in [0, 2, 6, 8] else 0.0

        return features

    def collect_random_game(self) -> GameTrace:
        """Collect trace from random play."""
        trace = GameTrace()
        self.game.reset()

        while not self.game.is_terminal():
            state = self.game.get_state()
            actions = self.game.get_actions()

            if not actions:
                break

            # Random action selection
            action = random.choice(actions)

            # Try to apply action
            success = self.game.apply_action(action)
            next_state = self.game.get_state()

            # Determine outcome
            if not success:
                outcome = Outcome.INVALID
            elif self.game.is_terminal():
                winner = self.game.get_winner()
                current_player = state.get('current_player', 0)
                if winner is None:
                    outcome = Outcome.DRAW
                elif winner == current_player:
                    outcome = Outcome.WIN
                else:
                    outcome = Outcome.LOSS
            else:
                outcome = Outcome.VALID

            # Create transition with features (including action-specific)
            trans = Transition(
                state=state,
                action=action,
                next_state=next_state,
                outcome=outcome,
                features=self.extract_features(state, action),
            )
            trace.add_transition(trans)

        trace.winner = self.game.get_winner()
        self.traces.append(trace)
        return trace

    def collect_with_policy(self, policy: Callable, num_games: int = 100) -> List[GameTrace]:
        """Collect traces using a policy function."""
        traces = []
        for _ in range(num_games):
            trace = GameTrace()
            self.game.reset()

            while not self.game.is_terminal():
                state = self.game.get_state()
                actions = self.game.get_actions()

                if not actions:
                    break

                # Policy selects action
                action = policy(state, actions)
                success = self.game.apply_action(action)
                next_state = self.game.get_state()

                # Determine outcome
                if not success:
                    outcome = Outcome.INVALID
                elif self.game.is_terminal():
                    winner = self.game.get_winner()
                    current_player = state.get('current_player', 0)
                    if winner is None:
                        outcome = Outcome.DRAW
                    elif winner == current_player:
                        outcome = Outcome.WIN
                    else:
                        outcome = Outcome.LOSS
                else:
                    outcome = Outcome.VALID

                trans = Transition(
                    state=state,
                    action=action,
                    next_state=next_state,
                    outcome=outcome,
                    features=self.extract_features(state, action),
                )
                trace.add_transition(trans)

            trace.winner = self.game.get_winner()
            traces.append(trace)
            self.traces.append(trace)

        return traces

    def collect_random_games(self, num_games: int = 100) -> List[GameTrace]:
        """Collect multiple random games."""
        return [self.collect_random_game() for _ in range(num_games)]

    def get_all_transitions(self) -> List[Transition]:
        """Get all transitions from all collected traces."""
        return [t for trace in self.traces for t in trace.transitions]

    def get_transitions_by_outcome(self, outcome: Outcome) -> List[Transition]:
        """Filter transitions by outcome."""
        return [t for t in self.get_all_transitions() if t.outcome == outcome]

    def get_invalid_transitions(self) -> List[Transition]:
        """Get all invalid/illegal move attempts."""
        return self.get_transitions_by_outcome(Outcome.INVALID)

    def get_valid_transitions(self) -> List[Transition]:
        """Get all valid moves."""
        return [t for t in self.get_all_transitions()
                if t.outcome != Outcome.INVALID]

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        all_trans = self.get_all_transitions()
        outcome_counts = {}
        for outcome in Outcome:
            outcome_counts[outcome.name] = len(
                [t for t in all_trans if t.outcome == outcome]
            )

        return {
            'num_games': len(self.traces),
            'total_transitions': len(all_trans),
            'outcomes': outcome_counts,
            'avg_game_length': len(all_trans) / max(1, len(self.traces)),
        }


class GameInterface:
    """
    Abstract interface for games.

    Implement this for your specific game.
    """

    def reset(self):
        """Reset game to initial state."""
        raise NotImplementedError

    def get_state(self) -> Dict[str, Any]:
        """Get current game state as dictionary."""
        raise NotImplementedError

    def get_actions(self) -> List[Any]:
        """Get list of possible actions (may include illegal ones)."""
        raise NotImplementedError

    def apply_action(self, action: Any) -> bool:
        """Apply action, return True if legal."""
        raise NotImplementedError

    def is_terminal(self) -> bool:
        """Check if game is over."""
        raise NotImplementedError

    def get_winner(self) -> Optional[int]:
        """Get winner (None for draw or ongoing)."""
        raise NotImplementedError


class TicTacToeGame(GameInterface):
    """Simple TicTacToe implementation for testing."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [0] * 9  # 0=empty, 1=X, 2=O
        self.current_player = 1
        self._winner = None

    def get_state(self) -> Dict[str, Any]:
        return {
            'board': self.board.copy(),
            'current_player': self.current_player,
        }

    def get_actions(self) -> List[int]:
        # Return all positions (including occupied - to test illegality detection)
        return list(range(9))

    def apply_action(self, action: int) -> bool:
        if self.board[action] != 0:
            return False  # Illegal - occupied

        self.board[action] = self.current_player
        self._check_winner()
        self.current_player = 3 - self.current_player  # Switch 1<->2
        return True

    def _check_winner(self):
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Cols
            [0, 4, 8], [2, 4, 6],              # Diags
        ]
        for line in lines:
            if (self.board[line[0]] != 0 and
                    self.board[line[0]] == self.board[line[1]] == self.board[line[2]]):
                self._winner = self.board[line[0]]
                return

    def is_terminal(self) -> bool:
        if self._winner is not None:
            return True
        return all(c != 0 for c in self.board)

    def get_winner(self) -> Optional[int]:
        return self._winner
