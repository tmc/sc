"""
Self-Play Refiner - Validates and refines discovered rules through gameplay.

Uses the discovered statechart to play games, then uses errors to refine rules.
Similar to hide-seek evolution but focused on rule refinement.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import random

try:
    from .trace_collector import TraceCollector, Transition, Outcome, GameInterface
    from .state_discoverer import StateDiscoverer
    from .guard_synthesizer import GuardSynthesizer, SymbolicGuard
    from .rule_statechart import RuleStatechart, DiscoveredRule
except ImportError:
    from trace_collector import TraceCollector, Transition, Outcome, GameInterface
    from state_discoverer import StateDiscoverer
    from guard_synthesizer import GuardSynthesizer, SymbolicGuard
    from rule_statechart import RuleStatechart, DiscoveredRule


@dataclass
class RefinementStats:
    """Statistics from a refinement iteration."""
    iteration: int
    games_played: int
    legality_errors: int  # Rule said legal but was illegal
    illegality_errors: int  # Rule said illegal but was legal
    win_detection_accuracy: float
    state_prediction_accuracy: float


class SelfPlayRefiner:
    """
    Refines discovered rules through self-play.

    Process:
    1. Use current statechart to make predictions
    2. Play games and record discrepancies
    3. Collect new examples where rules were wrong
    4. Re-synthesize guards with augmented data
    5. Repeat until convergence
    """

    def __init__(self, game: GameInterface, statechart: RuleStatechart):
        self.game = game
        self.statechart = statechart
        self.history: List[RefinementStats] = []

        # Error examples for refinement
        self.legality_false_positives: List[Transition] = []  # Said legal, was illegal
        self.legality_false_negatives: List[Transition] = []  # Said illegal, was legal
        self.win_false_positives: List[Transition] = []
        self.win_false_negatives: List[Transition] = []

    def _extract_features(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from game state."""
        features = {}

        if 'board' in state:
            board = state['board']
            features['empty_count'] = sum(1 for c in board if c == 0)
            features['p1_count'] = sum(1 for c in board if c == 1)
            features['p2_count'] = sum(1 for c in board if c == 2)
            features['total_pieces'] = features['p1_count'] + features['p2_count']
            features['fill_ratio'] = features['total_pieces'] / max(1, len(board))

            if len(board) == 9:
                features['center_control'] = 1.0 if board[4] != 0 else 0.0

        if 'current_player' in state:
            features['is_p1_turn'] = 1.0 if state['current_player'] == 1 else 0.0

        return features

    def _add_action_features(self, features: Dict[str, float],
                             action: Any, state: Dict[str, Any]) -> Dict[str, float]:
        """Add action-specific features."""
        result = features.copy()

        if isinstance(action, int) and 'board' in state:
            board = state['board']
            result['action_position'] = float(action)
            result['action_is_empty'] = 1.0 if board[action] == 0 else 0.0

            # Position features for 3x3
            if len(board) == 9:
                result['action_is_center'] = 1.0 if action == 4 else 0.0
                result['action_is_corner'] = 1.0 if action in [0, 2, 6, 8] else 0.0
                result['action_is_edge'] = 1.0 if action in [1, 3, 5, 7] else 0.0

        return result

    def play_game_with_statechart(self) -> Tuple[List[Transition], int, Dict[str, int]]:
        """
        Play a game using the statechart for predictions.

        Returns:
            (transitions, winner, error_counts)
        """
        transitions = []
        errors = {'legality_fp': 0, 'legality_fn': 0, 'win_fp': 0, 'win_fn': 0}

        self.game.reset()

        while not self.game.is_terminal():
            state = self.game.get_state()
            features = self._extract_features(state)
            actions = self.game.get_actions()

            if not actions:
                break

            # Test each action against our rules
            legal_actions = []
            for action in actions:
                action_features = self._add_action_features(features, action, state)
                predicted_legal = self.statechart.is_legal(action_features)

                # Test actual legality
                self.game.reset()
                # Restore state (simplified - works for simple games)
                if 'board' in state:
                    self.game.board = state['board'].copy()
                    self.game.current_player = state['current_player']
                    self.game._winner = None

                actual_legal = self.game.apply_action(action)

                # Reset to continue
                self.game.reset()
                if 'board' in state:
                    self.game.board = state['board'].copy()
                    self.game.current_player = state['current_player']
                    self.game._winner = None

                if actual_legal:
                    legal_actions.append(action)

                # Record errors
                if predicted_legal and not actual_legal:
                    errors['legality_fp'] += 1
                    trans = Transition(
                        state=state, action=action, next_state=state,
                        outcome=Outcome.INVALID, features=action_features
                    )
                    self.legality_false_positives.append(trans)
                elif not predicted_legal and actual_legal:
                    errors['legality_fn'] += 1
                    trans = Transition(
                        state=state, action=action, next_state=state,
                        outcome=Outcome.VALID, features=action_features
                    )
                    self.legality_false_negatives.append(trans)

            if not legal_actions:
                break

            # Choose random legal action and execute
            action = random.choice(legal_actions)
            self.game.apply_action(action)
            next_state = self.game.get_state()

            # Determine outcome
            if self.game.is_terminal():
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
                features=self._add_action_features(features, action, state),
            )
            transitions.append(trans)

            # Check win prediction
            predicted_win = self.statechart.is_winning(features)
            actual_win = outcome == Outcome.WIN

            if predicted_win and not actual_win:
                errors['win_fp'] += 1
                self.win_false_positives.append(trans)
            elif not predicted_win and actual_win:
                errors['win_fn'] += 1
                self.win_false_negatives.append(trans)

        return transitions, self.game.get_winner(), errors

    def run_refinement_iteration(self, num_games: int = 50) -> RefinementStats:
        """
        Run one iteration of self-play refinement.

        Returns statistics about errors found.
        """
        total_errors = defaultdict(int)
        total_transitions = 0

        for _ in range(num_games):
            transitions, winner, errors = self.play_game_with_statechart()
            total_transitions += len(transitions)
            for key, count in errors.items():
                total_errors[key] += count

        # Compute accuracies
        legality_errors = total_errors['legality_fp'] + total_errors['legality_fn']
        win_errors = total_errors['win_fp'] + total_errors['win_fn']

        stats = RefinementStats(
            iteration=len(self.history) + 1,
            games_played=num_games,
            legality_errors=legality_errors,
            illegality_errors=total_errors['legality_fn'],
            win_detection_accuracy=1.0 - (win_errors / max(1, total_transitions)),
            state_prediction_accuracy=0.0,  # TODO: implement
        )
        self.history.append(stats)

        return stats

    def refine_legality_rules(self, synthesizer: GuardSynthesizer) -> List[DiscoveredRule]:
        """
        Refine legality rules using collected error examples.

        Combines original rules with corrections from errors.
        """
        # Get all transitions we've seen
        all_legal = self.legality_false_negatives  # Were actually legal
        all_illegal = self.legality_false_positives  # Were actually illegal

        if not all_illegal:
            # No errors to correct
            return []

        # Re-synthesize with error examples
        guard = synthesizer.synthesize_guard(all_legal, all_illegal)

        rule = DiscoveredRule(
            name="RefinedLegality",
            rule_type="legality",
            guard=guard,
            confidence=guard.precision,
            examples_seen=len(all_legal) + len(all_illegal),
        )

        return [rule]

    def refine_win_rules(self, synthesizer: GuardSynthesizer) -> List[DiscoveredRule]:
        """
        Refine win detection rules using collected error examples.
        """
        if not self.win_false_negatives:
            return []

        # Re-synthesize
        guard = synthesizer.synthesize_guard(
            self.win_false_negatives,  # Actual wins we missed
            self.win_false_positives,  # Non-wins we predicted as wins
        )

        rule = DiscoveredRule(
            name="RefinedWin",
            rule_type="win",
            guard=guard,
            confidence=guard.precision,
            examples_seen=len(self.win_false_negatives) + len(self.win_false_positives),
        )

        return [rule]

    def run_full_refinement(self, iterations: int = 5,
                            games_per_iteration: int = 50,
                            verbose: bool = True) -> RuleStatechart:
        """
        Run full refinement loop.

        Returns refined statechart.
        """
        for i in range(iterations):
            stats = self.run_refinement_iteration(games_per_iteration)

            if verbose:
                print(f"Iteration {stats.iteration}: "
                      f"legality_errors={stats.legality_errors}, "
                      f"win_acc={stats.win_detection_accuracy:.2%}")

            # Check for convergence
            if stats.legality_errors == 0 and stats.win_detection_accuracy > 0.95:
                if verbose:
                    print("Converged!")
                break

            # Refine rules
            synthesizer = GuardSynthesizer()

            new_legality = self.refine_legality_rules(synthesizer)
            for rule in new_legality:
                self.statechart.add_rule(rule)

            new_win = self.refine_win_rules(synthesizer)
            for rule in new_win:
                self.statechart.add_rule(rule)

        return self.statechart

    def summary(self) -> str:
        """Generate summary of refinement process."""
        lines = ["=== Self-Play Refinement Summary ===\n"]

        for stats in self.history:
            lines.append(f"Iteration {stats.iteration}:")
            lines.append(f"  Games: {stats.games_played}")
            lines.append(f"  Legality Errors: {stats.legality_errors}")
            lines.append(f"  Win Detection Accuracy: {stats.win_detection_accuracy:.1%}")
            lines.append("")

        lines.append(f"Error Examples Collected:")
        lines.append(f"  Legality False Positives: {len(self.legality_false_positives)}")
        lines.append(f"  Legality False Negatives: {len(self.legality_false_negatives)}")
        lines.append(f"  Win False Positives: {len(self.win_false_positives)}")
        lines.append(f"  Win False Negatives: {len(self.win_false_negatives)}")

        return '\n'.join(lines)
