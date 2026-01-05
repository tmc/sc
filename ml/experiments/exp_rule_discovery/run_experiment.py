#!/usr/bin/env python3
"""
Rule Discovery Experiment

Demonstrates the full pipeline for discovering game rules as statecharts.

Input: Game traces from gameplay
Output: Interpretable statechart with discovered guards

Key insight: Rules are DISCOVERED from observation, not hand-coded!
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Handle imports
try:
    from .trace_collector import TicTacToeGame
    from .pipeline import RuleDiscoveryPipeline, PipelineConfig
except ImportError:
    from trace_collector import TicTacToeGame
    from pipeline import RuleDiscoveryPipeline, PipelineConfig


def run_tictactoe_discovery(verbose: bool = True) -> dict:
    """
    Run rule discovery on TicTacToe.

    Discovers:
    - Legality rules (can't play on occupied square)
    - Win conditions (three in a row)
    - Game phases (early/mid/late)
    """
    print("=" * 60)
    print("GAME RULE DISCOVERY EXPERIMENT")
    print("=" * 60)
    print("Game: TicTacToe")
    print("Goal: Discover rules from observation alone")
    print()

    # Create game
    game = TicTacToeGame()

    # Configure pipeline
    config = PipelineConfig(
        num_exploration_games=150,
        num_abstract_states=4,  # Early, Mid, Late, Terminal
        min_guard_coverage=0.4,
        min_guard_precision=0.7,
        max_predicates_per_guard=3,
        refinement_iterations=3,
        games_per_refinement=40,
    )

    # Run pipeline
    pipeline = RuleDiscoveryPipeline(game, config)
    results = pipeline.run_full_pipeline(verbose=verbose)

    # Show key discoveries
    print("\n" + "=" * 60)
    print("KEY DISCOVERIES")
    print("=" * 60)

    statechart = pipeline.get_statechart()

    print("\n--- Discovered Legality Rule ---")
    legality_rules = [r for r in statechart.rules if r.rule_type == 'legality']
    for rule in legality_rules:
        print(f"  A move is LEGAL when: {rule.guard}")
        print(f"  Confidence: {rule.confidence:.1%}")

    print("\n--- Discovered Win Condition ---")
    win_rules = [r for r in statechart.rules if r.rule_type == 'win']
    for rule in win_rules:
        print(f"  WIN detected when: {rule.guard}")
        print(f"  Confidence: {rule.confidence:.1%}")

    print("\n--- Discovered Game Phases ---")
    for state in statechart.states:
        print(f"  {state.name}:")
        print(f"    Win probability: {state.win_rate:.1%}")
        print(f"    Terminal: {state.is_terminal}")

    # Export SCXML
    print("\n--- SCXML Export ---")
    scxml = statechart.to_scxml()
    print(scxml[:500] + "..." if len(scxml) > 500 else scxml)

    return results


def run_comparison_experiment(verbose: bool = True) -> dict:
    """
    Compare discovered rules vs ground truth.
    """
    print("\n" + "=" * 60)
    print("VALIDATION: Discovered Rules vs Ground Truth")
    print("=" * 60)

    game = TicTacToeGame()

    # Quick discovery
    config = PipelineConfig(
        num_exploration_games=100,
        num_abstract_states=3,
        refinement_iterations=2,
        games_per_refinement=30,
    )

    pipeline = RuleDiscoveryPipeline(game, config)
    results = pipeline.run_full_pipeline(verbose=False)
    statechart = pipeline.get_statechart()

    # Test legality rule accuracy
    print("\n--- Testing Legality Rule ---")

    correct = 0
    total = 0

    # Test all possible board states
    import itertools
    for board_tuple in itertools.product([0, 1, 2], repeat=9):
        board = list(board_tuple)

        # Skip invalid boards (too many pieces of one type)
        p1 = sum(1 for c in board if c == 1)
        p2 = sum(1 for c in board if c == 2)
        if abs(p1 - p2) > 1:
            continue

        # For each position, test if our rule matches ground truth
        for pos in range(9):
            features = {
                'empty_count': sum(1 for c in board if c == 0),
                'fill_ratio': sum(1 for c in board if c != 0) / 9,
                'target_is_empty': 1.0 if board[pos] == 0 else 0.0,  # Must match discovered feature name
                'target_value': float(board[pos]),
            }

            predicted_legal = statechart.is_legal(features)
            actual_legal = (board[pos] == 0)

            if predicted_legal == actual_legal:
                correct += 1
            total += 1

            if total >= 10000:  # Sample
                break
        if total >= 10000:
            break

    accuracy = correct / total
    print(f"Legality Rule Accuracy: {accuracy:.1%}")
    print(f"  (Tested {total} board/action combinations)")

    # Summary
    print("\n--- Summary ---")
    if accuracy > 0.9:
        print("SUCCESS: Discovered legality rule is highly accurate!")
    elif accuracy > 0.7:
        print("PARTIAL: Discovered rule captures main pattern")
    else:
        print("NEEDS WORK: Rule needs more refinement")

    print("\n*** KEY INSIGHT ***")
    print("The rule was DISCOVERED, not programmed!")
    print("We never told the system that 'empty squares are legal moves'")
    print("It learned this from observing valid/invalid move attempts.")

    return {
        'legality_accuracy': accuracy,
        'total_tested': total,
    }


def save_results(results: dict, filename: str = None):
    """Save results to JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rule_discovery_results_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {filename}")
    return filename


if __name__ == '__main__':
    verbose = '--quiet' not in sys.argv

    # Run main experiment
    results = run_tictactoe_discovery(verbose=verbose)

    # Run validation
    validation = run_comparison_experiment(verbose=verbose)
    results['validation'] = validation

    # Save results
    save_results(results)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
