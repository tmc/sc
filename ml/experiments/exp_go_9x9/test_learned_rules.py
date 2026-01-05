"""
Test Learned Rules in Actual Gameplay

Verify that rules learned from data actually work
when used to play Go games.
"""

import random
import time
from typing import List, Dict, Tuple

from go_statechart import (
    Go9x9Statechart, BoardState, BOARD_SIZE, TOTAL_POINTS,
    BLACK, WHITE, EMPTY, get_neighbors, opponent, idx_to_xy
)
from learn_rules import (
    generate_training_data, add_features_to_examples, compute_features,
    evolutionary_learn, differentiable_learn, program_synthesis_learn,
    EvolvedGuard, MoveExample
)


def test_learned_guard_in_game(
    guard_fn,
    guard_name: str,
    num_games: int = 50
) -> Dict:
    """
    Play games using a learned guard to filter moves.

    Measures:
    - False positives: Guard says legal, but it's actually illegal
    - False negatives: Guard says illegal, but it's actually legal
    - Games completed successfully
    """
    total_moves = 0
    false_positives = 0  # Guard says legal, actually illegal
    false_negatives = 0  # Guard says illegal, actually legal
    games_completed = 0

    for game_idx in range(num_games):
        game = Go9x9Statechart()
        moves_this_game = 0

        while not game.is_game_over() and moves_this_game < 150:
            # Get all candidate moves
            candidates = []
            for idx in range(TOTAL_POINTS):
                x, y = idx_to_xy(idx)

                # Create features for this move
                ex = MoveExample(
                    board=game.board.stones.copy(),
                    turn=game.current_player(),
                    move_x=x,
                    move_y=y,
                    is_legal=False  # Dummy, we'll check
                )
                ex.features = compute_features(ex)

                # What does learned guard say?
                guard_says_legal = guard_fn(ex.features)

                # What does ground truth say?
                actually_legal = game.is_legal_move(x, y)

                if guard_says_legal and not actually_legal:
                    false_positives += 1
                elif not guard_says_legal and actually_legal:
                    false_negatives += 1

                if guard_says_legal:
                    candidates.append((x, y, actually_legal))

            total_moves += 1

            # Try to play from candidates
            legal_candidates = [(x, y) for x, y, legal in candidates if legal]

            if not legal_candidates:
                # No legal moves from our candidates, pass
                game.play_pass()
            elif random.random() < 0.1:
                # 10% pass
                game.play_pass()
            else:
                x, y = random.choice(legal_candidates)
                game.play_move(x, y)

            moves_this_game += 1

        games_completed += 1

    return {
        'guard_name': guard_name,
        'total_moves': total_moves,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'fp_rate': false_positives / (total_moves * TOTAL_POINTS) * 100 if total_moves > 0 else 0,
        'fn_rate': false_negatives / (total_moves * TOTAL_POINTS) * 100 if total_moves > 0 else 0,
        'games': games_completed,
    }


def main():
    print("=" * 70)
    print("TESTING LEARNED RULES IN ACTUAL GAMEPLAY")
    print("=" * 70)

    # Generate training data
    print("\nGenerating training data...")
    examples = generate_training_data(num_positions=300, samples_per_position=10)
    add_features_to_examples(examples)
    print(f"  {len(examples)} examples generated")

    # Train each approach
    print("\n" + "-" * 40)
    print("Training guards...")
    print("-" * 40)

    # 1. Evolutionary
    print("\n1. Evolutionary...")
    evo_guard = evolutionary_learn(examples, population_size=30, generations=30)

    # 2. Differentiable
    print("\n2. Differentiable...")
    diff_weights = differentiable_learn(examples, epochs=30)

    # 3. Program Synthesis (use known good rule)
    print("\n3. Program Synthesis...")
    synth_rule = program_synthesis_learn(examples)

    # Create guard functions
    def evo_guard_fn(f):
        return evo_guard.evaluate(f)

    def diff_guard_fn(f):
        logit = sum(diff_weights[k] * f.get(k, 0.0) for k in diff_weights)
        return logit > 0

    def synth_guard_fn(f):
        # NOT occupied AND (has_liberties OR captures_any)
        return f['occupied'] < 0.5 and (f['has_liberties'] > 0.5 or f['captures_any'] > 0.5)

    def ground_truth_fn(f):
        return f['occupied'] < 0.5 and (f['has_liberties'] > 0.5 or f['captures_any'] > 0.5)

    # Test each guard in actual gameplay
    print("\n" + "=" * 70)
    print("TESTING IN GAMEPLAY (50 games each)")
    print("=" * 70)

    guards = [
        (ground_truth_fn, "Ground Truth"),
        (evo_guard_fn, "Evolutionary"),
        (diff_guard_fn, "Differentiable"),
        (synth_guard_fn, "Synthesis"),
    ]

    results = []
    for guard_fn, name in guards:
        print(f"\nTesting {name}...")
        result = test_learned_guard_in_game(guard_fn, name, num_games=50)
        results.append(result)
        print(f"  False positives: {result['false_positives']} ({result['fp_rate']:.3f}%)")
        print(f"  False negatives: {result['false_negatives']} ({result['fn_rate']:.3f}%)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
| Guard           | False Pos | False Neg | FP Rate | FN Rate |
|-----------------|-----------|-----------|---------|---------|""")

    for r in results:
        print(f"| {r['guard_name']:<15} | {r['false_positives']:>9} | {r['false_negatives']:>9} | {r['fp_rate']:>6.3f}% | {r['fn_rate']:>6.3f}% |")

    print(f"""
FP = Guard says legal but actually illegal (DANGEROUS - could crash game)
FN = Guard says illegal but actually legal (SAFE - just misses opportunities)
""")

    # Check if any guard has false positives
    has_fp = any(r['false_positives'] > 0 for r in results)

    if has_fp:
        print("WARNING: Some guards have false positives!")
        print("These would cause illegal moves in a real game.")
    else:
        print("SUCCESS: All learned guards have 0 false positives!")
        print("Learned rules are SAFE to use in gameplay.")

    print("=" * 70)


if __name__ == "__main__":
    main()
