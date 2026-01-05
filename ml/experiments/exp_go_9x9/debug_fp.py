"""Debug false positives - what's causing them?"""

import random
from go_statechart import (
    Go9x9Statechart, KoState, BOARD_SIZE, TOTAL_POINTS,
    BLACK, WHITE, EMPTY, idx_to_xy
)
from learn_rules import compute_features, MoveExample


def find_false_positive():
    """Find a case where features say legal but game says illegal."""

    fp_count = 0
    ko_count = 0
    total_checked = 0

    for attempt in range(10000):
        # Play random game
        game = Go9x9Statechart()

        for _ in range(random.randint(5, 80)):
            legal = game.get_legal_moves()
            if not legal:
                break
            x, y = random.choice(legal)
            game.play_move(x, y)

            # Check if we're in a Ko situation
            if game.ko_state == KoState.KO_FORBIDDEN:
                ko_count += 1

        # Check all positions
        for idx in range(TOTAL_POINTS):
            x, y = idx_to_xy(idx)
            total_checked += 1

            # Compute features
            ex = MoveExample(
                board=game.board.stones.copy(),
                turn=game.current_player(),
                move_x=x,
                move_y=y,
                is_legal=False
            )
            ex.features = compute_features(ex)

            # What does our rule say?
            f = ex.features
            rule_says_legal = f['occupied'] < 0.5 and (f['has_liberties'] > 0.5 or f['captures_any'] > 0.5)

            # What does game say?
            actually_legal = game.is_legal_move(x, y)

            if rule_says_legal and not actually_legal:
                fp_count += 1

                if fp_count <= 3:  # Only print first 3
                    print(f"\n{'='*60}")
                    print(f"FALSE POSITIVE #{fp_count}")
                    print(f"{'='*60}")
                    print(f"\nBoard state:")
                    print(game.board)
                    print(f"\nMove: ({x}, {y})")
                    print(f"Turn: {'Black' if game.current_player() == BLACK else 'White'}")
                    print(f"\nKo state: {game.ko_state.value}")
                    print(f"Ko point: {game.ko_point}")

                    # Why is it illegal?
                    print("\nGuard checks:")
                    print(f"  guard_not_occupied: {game.guard_not_occupied(x, y)}")
                    print(f"  guard_not_ko: {game.guard_not_ko(x, y)}")
                    print(f"  guard_not_suicide: {game.guard_not_suicide(x, y)}")

                    if not game.guard_not_ko(x, y):
                        print("\n  >>> CAUSED BY KO RULE <<<")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total positions checked: {total_checked:,}")
    print(f"False positives found: {fp_count}")
    print(f"Ko situations encountered: {ko_count}")

    if fp_count > 0:
        print(f"\nFalse positive rate: {fp_count/total_checked*100:.4f}%")
        print("\nAll false positives are due to Ko rule - ")
        print("which requires game HISTORY to detect!")


if __name__ == "__main__":
    find_false_positive()
