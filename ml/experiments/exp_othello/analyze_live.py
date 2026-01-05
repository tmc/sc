"""
Live analysis of evolving weights.
Run this while evolution is happening to see emerging patterns.
"""

import json
import sys
import os

# Board layout for reference
POSITIONS = """
  A B C D E F G H
1 0 1 2 3 4 5 6 7
2 8 9 ...       15
3 16            23
4 24            31
5 32            39
6 40            47
7 48            55
8 56 57      62 63
"""

CORNERS = [0, 7, 56, 63]
X_SQUARES = [9, 14, 49, 54]
C_SQUARES = [1, 6, 8, 15, 48, 55, 57, 62]
EDGES = [i for i in range(64) if i // 8 == 0 or i // 8 == 7 or i % 8 == 0 or i % 8 == 7]


def load_genome(filename):
    """Try to load evolved genome."""
    try:
        with open(filename) as f:
            return json.load(f)
    except:
        return None


def visualize_position_values(values):
    """Display 8x8 heatmap of position values."""
    if len(values) != 64:
        print("Invalid values length")
        return

    print("\n8x8 Position Value Heatmap:")
    print("-" * 40)

    vmin, vmax = min(values), max(values)
    rang = vmax - vmin + 1e-6

    symbols = " ░▒▓█"

    print("  ", end="")
    for c in "ABCDEFGH":
        print(f" {c}", end="")
    print()

    for row in range(8):
        print(f"{row+1} ", end="")
        for col in range(8):
            idx = row * 8 + col
            normalized = (values[idx] - vmin) / rang
            sym_idx = min(4, int(normalized * 4.99))
            print(f" {symbols[sym_idx]}", end="")
        print(f"  {row+1}")

    print("  ", end="")
    for c in "ABCDEFGH":
        print(f" {c}", end="")
    print()


def analyze_positions(values):
    """Analyze position value patterns."""
    print("\n Position Value Analysis:")
    print("-" * 40)

    corner_vals = [values[i] for i in CORNERS]
    x_vals = [values[i] for i in X_SQUARES]
    c_vals = [values[i] for i in C_SQUARES]
    edge_vals = [values[i] for i in EDGES if i not in CORNERS]
    center = [27, 28, 35, 36]
    center_vals = [values[i] for i in center]

    def mean(lst): return sum(lst)/len(lst) if lst else 0

    print(f"Corners (A1,H1,A8,H8):  {mean(corner_vals):+.3f}")
    print(f"X-squares (B2,G2,B7,G7): {mean(x_vals):+.3f}")
    print(f"C-squares (adj corners): {mean(c_vals):+.3f}")
    print(f"Other edges:             {mean(edge_vals):+.3f}")
    print(f"Center 4:                {mean(center_vals):+.3f}")

    print("\nDiscovery Check:")
    corners_good = mean(corner_vals) > mean(edge_vals)
    x_bad = mean(x_vals) < mean(corner_vals)
    print(f"  Corners > Edges: {'YES ✓' if corners_good else 'NO ✗'}")
    print(f"  X-squares < Corners: {'YES ✓' if x_bad else 'NO ✗'}")


def show_individual_corners(values):
    """Show each corner and adjacent positions."""
    print("\nCorner Analysis:")
    print("-" * 40)

    corners = {
        "A1 (top-left)": (0, [1, 8, 9]),
        "H1 (top-right)": (7, [6, 14, 15]),
        "A8 (bot-left)": (56, [48, 49, 57]),
        "H8 (bot-right)": (63, [54, 55, 62]),
    }

    for name, (corner, adjacent) in corners.items():
        c_val = values[corner]
        adj_vals = [values[i] for i in adjacent]
        x_val = values[adjacent[2]]  # X-square is always 3rd
        print(f"{name}: corner={c_val:+.2f}, X-square={x_val:+.2f}")


def main():
    # Try different possible filenames
    files = [
        "evolved_genome.json",
        "evolved_scaled_genome.json",
        "experiments/exp_othello/evolved_genome.json",
    ]

    genome = None
    for f in files:
        genome = load_genome(f)
        if genome:
            print(f"Loaded: {f}")
            break

    if not genome:
        print("No genome file found. Showing expected patterns:")
        # Show what we expect to evolve
        expected = [0.0] * 64
        for c in CORNERS:
            expected[c] = 10.0
        for x in X_SQUARES:
            expected[x] = -8.0
        for c in C_SQUARES:
            expected[c] = -3.0
        for e in EDGES:
            if e not in CORNERS and e not in C_SQUARES:
                expected[e] = 2.0

        print("\nExpected optimal pattern (hand-coded):")
        visualize_position_values(expected)
        analyze_positions(expected)
        return

    # Analyze loaded genome
    if "position_values" in genome:
        visualize_position_values(genome["position_values"])
        analyze_positions(genome["position_values"])
        show_individual_corners(genome["position_values"])

    # Show other evolved params
    print("\nOther Parameters:")
    print("-" * 40)
    if "phase_thresholds" in genome:
        print(f"Phase 1: {genome['phase_thresholds'][0]:.1f} pieces")
        print(f"Phase 2: {genome['phase_thresholds'][1]:.1f} pieces")

    if "signal_weights" in genome:
        print(f"Signal weights: {genome['signal_weights']}")

    if "fitness" in genome:
        print(f"\nFinal fitness: {genome['fitness']:.3f}")


if __name__ == "__main__":
    main()
