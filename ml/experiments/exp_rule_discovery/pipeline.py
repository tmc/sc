"""
Rule Discovery Pipeline - Combines all components into end-to-end system.

Input: Game traces (moves + outcomes)
Output: Extracted statechart with discovered guards

Pipeline stages:
1. Collect traces from gameplay
2. Discover abstract states from observations
3. Synthesize guards for transitions
4. Build interpretable statechart
5. Refine through self-play
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

try:
    from .trace_collector import TraceCollector, Transition, Outcome, GameInterface
    from .state_discoverer import StateDiscoverer, AbstractState
    from .guard_synthesizer import GuardSynthesizer, SymbolicGuard
    from .rule_statechart import RuleStatechart, DiscoveredRule
    from .self_play_refiner import SelfPlayRefiner
except ImportError:
    from trace_collector import TraceCollector, Transition, Outcome, GameInterface
    from state_discoverer import StateDiscoverer, AbstractState
    from guard_synthesizer import GuardSynthesizer, SymbolicGuard
    from rule_statechart import RuleStatechart, DiscoveredRule
    from self_play_refiner import SelfPlayRefiner


@dataclass
class PipelineConfig:
    """Configuration for the rule discovery pipeline."""
    # Trace collection
    num_exploration_games: int = 100
    include_invalid_moves: bool = True

    # State discovery
    num_abstract_states: int = 5

    # Guard synthesis
    min_guard_coverage: float = 0.5
    min_guard_precision: float = 0.8
    max_predicates_per_guard: int = 3

    # Self-play refinement
    refinement_iterations: int = 3
    games_per_refinement: int = 30


class RuleDiscoveryPipeline:
    """
    End-to-end pipeline for discovering game rules as statecharts.

    Key innovation: Rules are DISCOVERED from observation, not programmed.
    Output is fully interpretable and executable.
    """

    def __init__(self, game: GameInterface, config: PipelineConfig = None):
        self.game = game
        self.config = config or PipelineConfig()

        # Components
        self.collector: Optional[TraceCollector] = None
        self.state_discoverer: Optional[StateDiscoverer] = None
        self.guard_synthesizer: Optional[GuardSynthesizer] = None
        self.statechart: Optional[RuleStatechart] = None
        self.refiner: Optional[SelfPlayRefiner] = None

        # Results
        self.stages_completed: List[str] = []
        self.timestamps: Dict[str, str] = {}

    def stage_1_collect_traces(self, verbose: bool = True) -> List[Transition]:
        """
        Stage 1: Collect game traces through exploration.
        """
        if verbose:
            print("=" * 60)
            print("STAGE 1: Collecting Game Traces")
            print("=" * 60)

        self.timestamps['collect_start'] = datetime.now().isoformat()

        self.collector = TraceCollector(self.game)

        # Add feature extractors
        def board_features(state: Dict) -> Dict[str, float]:
            """Extract board-level features from state."""
            features = {}
            if 'board' in state:
                board = state['board']
                features['empty_count'] = sum(1 for c in board if c == 0)
                features['p1_count'] = sum(1 for c in board if c == 1)
                features['p2_count'] = sum(1 for c in board if c == 2)
                features['total_pieces'] = features['p1_count'] + features['p2_count']
                features['fill_ratio'] = features['total_pieces'] / max(1, len(board))

                if len(board) == 9:  # TicTacToe
                    features['center_control'] = 1.0 if board[4] != 0 else 0.0
                    corners = [board[0], board[2], board[6], board[8]]
                    features['corner_count'] = sum(1 for c in corners if c != 0)

            # Note: action-specific features (target_is_empty) are added by
            # TraceCollector.extract_features() when action is passed
            return features

        self.collector.add_feature_extractor(board_features)

        # Collect random games
        traces = self.collector.collect_random_games(self.config.num_exploration_games)

        self.timestamps['collect_end'] = datetime.now().isoformat()
        self.stages_completed.append('collect_traces')

        summary = self.collector.summary()
        if verbose:
            print(f"Collected {summary['num_games']} games")
            print(f"Total transitions: {summary['total_transitions']}")
            print(f"Outcomes: {summary['outcomes']}")
            print(f"Avg game length: {summary['avg_game_length']:.1f}")

        return self.collector.get_all_transitions()

    def stage_2_discover_states(self, transitions: List[Transition],
                                verbose: bool = True) -> List[AbstractState]:
        """
        Stage 2: Discover abstract states from observations.
        """
        if verbose:
            print("\n" + "=" * 60)
            print("STAGE 2: Discovering Abstract States")
            print("=" * 60)

        self.timestamps['discover_start'] = datetime.now().isoformat()

        self.state_discoverer = StateDiscoverer(
            num_states=self.config.num_abstract_states
        )

        states = self.state_discoverer.discover_states(transitions)

        self.timestamps['discover_end'] = datetime.now().isoformat()
        self.stages_completed.append('discover_states')

        if verbose:
            print(f"Discovered {len(states)} abstract states:")
            for state in states:
                print(f"  {state.name}: {len(state.members)} transitions, "
                      f"win_rate={state.win_rate:.1%}")

        return states

    def stage_3_synthesize_guards(self, transitions: List[Transition],
                                  states: List[AbstractState],
                                  verbose: bool = True
                                  ) -> Tuple[List[DiscoveredRule], Dict]:
        """
        Stage 3: Synthesize guards for transitions and legality.
        """
        if verbose:
            print("\n" + "=" * 60)
            print("STAGE 3: Synthesizing Guards")
            print("=" * 60)

        self.timestamps['synthesize_start'] = datetime.now().isoformat()

        self.guard_synthesizer = GuardSynthesizer(
            min_coverage=self.config.min_guard_coverage,
            min_precision=self.config.min_guard_precision,
        )

        rules = []
        guard_stats = {}

        # 1. Legality guard
        legality_guard = self.guard_synthesizer.synthesize_legality_guard(transitions)
        if legality_guard.predicates:
            rule = DiscoveredRule(
                name="Legality",
                rule_type="legality",
                guard=legality_guard,
                confidence=legality_guard.precision,
                examples_seen=len(transitions),
            )
            rules.append(rule)
            guard_stats['legality'] = {
                'guard': str(legality_guard),
                'precision': legality_guard.precision,
                'coverage': legality_guard.coverage,
            }
            if verbose:
                print(f"Legality Guard: {legality_guard}")
                print(f"  Precision: {legality_guard.precision:.1%}")

        # 2. Win condition guard
        win_guard = self.guard_synthesizer.synthesize_winning_guard(transitions)
        if win_guard.predicates:
            rule = DiscoveredRule(
                name="WinCondition",
                rule_type="win",
                guard=win_guard,
                confidence=win_guard.precision,
                examples_seen=sum(1 for t in transitions if t.outcome == Outcome.WIN),
            )
            rules.append(rule)
            guard_stats['win'] = {
                'guard': str(win_guard),
                'precision': win_guard.precision,
            }
            if verbose:
                print(f"Win Guard: {win_guard}")

        # 3. State transition guards
        def classify_state(state_dict: Dict) -> str:
            features = {}
            if 'board' in state_dict:
                board = state_dict['board']
                features['fill_ratio'] = sum(1 for c in board if c != 0) / len(board)
                if len(board) == 9:
                    features['center_control'] = 1.0 if board[4] != 0 else 0.0
            return self.state_discoverer.classify_new_state(features).name if features else "Unknown"

        transition_guards = self.guard_synthesizer.synthesize_transition_guards(
            transitions,
            from_state_classifier=classify_state,
            to_state_classifier=classify_state,
        )

        for (from_s, to_s), guard in transition_guards.items():
            rule = DiscoveredRule(
                name=f"Trans_{from_s}_to_{to_s}",
                rule_type="transition",
                guard=guard,
                from_state=from_s,
                to_state=to_s,
                confidence=guard.precision,
            )
            rules.append(rule)

        guard_stats['transitions'] = len(transition_guards)

        self.timestamps['synthesize_end'] = datetime.now().isoformat()
        self.stages_completed.append('synthesize_guards')

        if verbose:
            print(f"Synthesized {len(transition_guards)} transition guards")

        return rules, guard_stats

    def stage_4_build_statechart(self, states: List[AbstractState],
                                 rules: List[DiscoveredRule],
                                 verbose: bool = True) -> RuleStatechart:
        """
        Stage 4: Combine states and rules into executable statechart.
        """
        if verbose:
            print("\n" + "=" * 60)
            print("STAGE 4: Building Statechart")
            print("=" * 60)

        self.timestamps['build_start'] = datetime.now().isoformat()

        self.statechart = RuleStatechart(name="DiscoveredGame")

        # Add states
        for state in states:
            self.statechart.add_state(state)

        # Add rules
        for rule in rules:
            self.statechart.add_rule(rule)

        self.timestamps['build_end'] = datetime.now().isoformat()
        self.stages_completed.append('build_statechart')

        if verbose:
            print(f"Built statechart with {len(states)} states and {len(rules)} rules")

        return self.statechart

    def stage_5_refine_rules(self, verbose: bool = True) -> RuleStatechart:
        """
        Stage 5: Refine rules through self-play.
        """
        if verbose:
            print("\n" + "=" * 60)
            print("STAGE 5: Refining Through Self-Play")
            print("=" * 60)

        self.timestamps['refine_start'] = datetime.now().isoformat()

        self.refiner = SelfPlayRefiner(self.game, self.statechart)
        refined = self.refiner.run_full_refinement(
            iterations=self.config.refinement_iterations,
            games_per_iteration=self.config.games_per_refinement,
            verbose=verbose,
        )

        self.timestamps['refine_end'] = datetime.now().isoformat()
        self.stages_completed.append('refine_rules')

        return refined

    def run_full_pipeline(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the complete rule discovery pipeline.

        Returns dictionary with results.
        """
        self.timestamps['pipeline_start'] = datetime.now().isoformat()

        if verbose:
            print("\n" + "=" * 60)
            print("RULE DISCOVERY PIPELINE")
            print("=" * 60)
            print(f"Game: {self.game.__class__.__name__}")
            print(f"Config: {self.config}")
            print()

        # Run all stages
        transitions = self.stage_1_collect_traces(verbose)
        states = self.stage_2_discover_states(transitions, verbose)
        rules, guard_stats = self.stage_3_synthesize_guards(transitions, states, verbose)
        statechart = self.stage_4_build_statechart(states, rules, verbose)
        refined = self.stage_5_refine_rules(verbose)

        self.timestamps['pipeline_end'] = datetime.now().isoformat()

        # Build results
        results = {
            'config': {
                'exploration_games': self.config.num_exploration_games,
                'abstract_states': self.config.num_abstract_states,
                'refinement_iterations': self.config.refinement_iterations,
            },
            'stages_completed': self.stages_completed,
            'timestamps': self.timestamps,
            'statistics': {
                'total_transitions': len(transitions),
                'states_discovered': len(states),
                'rules_synthesized': len(rules),
                'guard_stats': guard_stats,
            },
            'statechart': refined.to_dict(),
        }

        if verbose:
            print("\n" + "=" * 60)
            print("PIPELINE COMPLETE")
            print("=" * 60)
            print(f"States: {len(states)}")
            print(f"Rules: {len(rules)}")
            print("\n--- Discovered Statechart ---")
            print(refined.describe())

        return results

    def get_statechart(self) -> RuleStatechart:
        """Get the discovered statechart."""
        return self.statechart

    def export_results(self, filepath: str):
        """Export results to JSON file."""
        results = {
            'statechart': self.statechart.to_dict() if self.statechart else None,
            'timestamps': self.timestamps,
            'stages': self.stages_completed,
        }
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
