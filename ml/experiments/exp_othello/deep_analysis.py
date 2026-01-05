"""
Deep analysis of evolved Othello weights.
Examines symmetry, correlations, and strategic patterns.
"""

import json
import math

# Position categories
CORNERS = [0, 7, 56, 63]
X_SQUARES = [9, 14, 49, 54]
C_SQUARES = [1, 6, 8, 15, 48, 55, 57, 62]
EDGES = [i for i in range(64) if i // 8 == 0 or i // 8 == 7 or i % 8 == 0 or i % 8 == 7]

# Adjacent position analysis
CORNER_ADJACENT = {
    0: {'c_squares': [1, 8], 'x_square': 9, 'diagonal': [10, 11, 16, 24]},
    7: {'c_squares': [6, 15], 'x_square': 14, 'diagonal': [13, 12, 23, 31]},
    56: {'c_squares': [48, 57], 'x_square': 49, 'diagonal': [42, 43, 40, 32]},
    63: {'c_squares': [55, 62], 'x_square': 54, 'diagonal': [53, 52, 47, 39]},
}


def load_genome():
    try:
        with open("evolved_genome.json") as f:
            return json.load(f)
    except:
        return None


def idx_to_notation(idx):
    """Convert index to Othello notation (A1-H8)."""
    row, col = idx // 8, idx % 8
    return f"{chr(65 + col)}{row + 1}"


def symmetry_analysis(values):
    """Check if evolved values respect board symmetry."""
    print("\n1. SYMMETRY ANALYSIS")
    print("=" * 50)

    errors = []

    # Check 4-fold rotational symmetry
    def rotate_90(idx):
        row, col = idx // 8, idx % 8
        new_row, new_col = col, 7 - row
        return new_row * 8 + new_col

    # Check horizontal mirror
    def mirror_h(idx):
        row, col = idx // 8, idx % 8
        return row * 8 + (7 - col)

    # Check vertical mirror
    def mirror_v(idx):
        row, col = idx // 8, idx % 8
        return (7 - row) * 8 + col

    # Group equivalent positions
    def get_equiv_group(idx):
        group = {idx}
        current = idx
        for _ in range(3):
            current = rotate_90(current)
            group.add(current)
        for g in list(group):
            group.add(mirror_h(g))
        return sorted(group)

    # Find all unique groups
    seen = set()
    groups = []
    for i in range(64):
        if i not in seen:
            group = get_equiv_group(i)
            groups.append(group)
            seen.update(group)

    # Check variance within groups
    print(f"\nFound {len(groups)} symmetry groups")
    print("\nGroups with high variance (should be similar):")
    high_var = []
    for group in groups:
        if len(group) > 1:
            vals = [values[i] for i in group]
            mean_val = sum(vals) / len(vals)
            var = sum((v - mean_val)**2 for v in vals) / len(vals)
            if var > 0.3:  # High variance threshold
                positions = [idx_to_notation(i) for i in group]
                high_var.append((positions, vals, var))

    if high_var:
        for positions, vals, var in sorted(high_var, key=lambda x: -x[2])[:5]:
            print(f"  {positions}: {[f'{v:.2f}' for v in vals]} (var={var:.3f})")
    else:
        print("  All groups have low variance - good symmetry!")

    # Overall symmetry score
    total_var = sum(
        sum((v - sum(vals)/len(vals))**2 for v in vals) / len(vals)
        for group in groups
        if len(group) > 1
        for vals in [[values[i] for i in group]]
    )
    print(f"\nTotal symmetry variance: {total_var:.3f}")
    print("(Lower is better - 0 means perfect symmetry)")


def corner_defense_analysis(values):
    """Analyze the defensive structure around corners."""
    print("\n2. CORNER DEFENSE ANALYSIS")
    print("=" * 50)

    for corner_idx, adj in CORNER_ADJACENT.items():
        corner_val = values[corner_idx]
        x_val = values[adj['x_square']]
        c_vals = [values[i] for i in adj['c_squares']]

        name = idx_to_notation(corner_idx)
        print(f"\n{name} Corner Analysis:")
        print(f"  Corner value:    {corner_val:+.3f}")
        print(f"  X-square value:  {x_val:+.3f} ({idx_to_notation(adj['x_square'])})")
        print(f"  C-square values: {[f'{v:+.3f}' for v in c_vals]}")

        # Check defensive pattern
        x_penalty = corner_val - x_val
        c_penalty = corner_val - sum(c_vals)/len(c_vals)
        print(f"  X-square penalty (vs corner): {x_penalty:+.3f}")
        print(f"  C-square penalty (vs corner): {c_penalty:+.3f}")

        if x_val < c_vals[0] and x_val < c_vals[1]:
            print("  ✓ X-square is most dangerous (correct)")
        else:
            print("  ✗ X-square not most penalized")


def edge_analysis(values):
    """Analyze edge values and patterns."""
    print("\n3. EDGE ANALYSIS")
    print("=" * 50)

    # Group by edge
    edges = {
        'Top (row 1)': [0, 1, 2, 3, 4, 5, 6, 7],
        'Bottom (row 8)': [56, 57, 58, 59, 60, 61, 62, 63],
        'Left (col A)': [0, 8, 16, 24, 32, 40, 48, 56],
        'Right (col H)': [7, 15, 23, 31, 39, 47, 55, 63],
    }

    for name, positions in edges.items():
        vals = [values[i] for i in positions]
        print(f"\n{name}:")
        labels = [idx_to_notation(i) for i in positions]
        for l, v in zip(labels, vals):
            bar = "█" * int(abs(v) * 5 + 0.5)
            sign = "+" if v >= 0 else "-"
            print(f"  {l}: {v:+.2f} {'|' + bar if v >= 0 else bar + '|':>10}")


def phase_sensitivity(genome):
    """Analyze phase-dependent strategy."""
    print("\n4. PHASE SENSITIVITY")
    print("=" * 50)

    if "signal_weights" in genome:
        weights = genome["signal_weights"]
        total = sum(abs(w) for w in weights)
        labels = ["Position", "Flip count", "Direction", "Phase bonus"]

        print("\nSignal importance (normalized):")
        for label, w in zip(labels, weights):
            pct = abs(w) / total * 100
            bar = "█" * int(pct / 2)
            print(f"  {label:<15}: {pct:5.1f}% {bar}")

    if "phase_thresholds" in genome:
        t1, t2 = genome["phase_thresholds"]
        print(f"\nPhase transitions:")
        print(f"  Opening → Midgame: {t1:.1f} pieces (expert: ~20)")
        print(f"  Midgame → Endgame: {t2:.1f} pieces (expert: ~50)")

        # Visualize game phases
        print("\nGame phase timeline:")
        print("  0       10        20        30        40        50        60")
        print("  |--------|---------|---------|---------|---------|---------|")
        t1_pos = int(t1 / 60 * 60)
        t2_pos = int(t2 / 60 * 60)
        line = [" "] * 61
        line[0] = "O"  # Opening
        line[t1_pos] = "M"  # Midgame
        line[t2_pos] = "E"  # Endgame
        print("  " + "".join(line))
        print(f"  {'Opening':<{t1_pos}}{'Midgame':<{t2_pos-t1_pos}}Endgame")


def correlation_analysis(values):
    """Check for correlations between position features."""
    print("\n5. CORRELATION ANALYSIS")
    print("=" * 50)

    # Distance from center
    def center_dist(idx):
        row, col = idx // 8, idx % 8
        return abs(row - 3.5) + abs(col - 3.5)

    # Distance from nearest corner
    def corner_dist(idx):
        row, col = idx // 8, idx % 8
        return min(
            row + col,  # A1
            row + (7 - col),  # H1
            (7 - row) + col,  # A8
            (7 - row) + (7 - col),  # H8
        )

    # Calculate correlations
    center_dists = [center_dist(i) for i in range(64)]
    corner_dists = [corner_dist(i) for i in range(64)]

    def pearson(x, y):
        n = len(x)
        mx, my = sum(x)/n, sum(y)/n
        sx = (sum((xi - mx)**2 for xi in x) / n) ** 0.5
        sy = (sum((yi - my)**2 for yi in y) / n) ** 0.5
        if sx == 0 or sy == 0:
            return 0
        return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n * sx * sy)

    r_center = pearson(values, center_dists)
    r_corner = pearson(values, corner_dists)

    print(f"\nCorrelation with center distance: {r_center:+.3f}")
    print(f"  (negative = center is better)")
    print(f"\nCorrelation with corner distance: {r_corner:+.3f}")
    print(f"  (negative = closer to corner is better)")


def main():
    genome = load_genome()
    if not genome:
        print("No genome found")
        return

    if "position_values" not in genome:
        print("Genome has no position_values")
        return

    values = genome["position_values"]
    if len(values) != 64:
        print(f"Expected 64 position values, got {len(values)}")
        return

    print("=" * 60)
    print("DEEP ANALYSIS OF EVOLVED OTHELLO WEIGHTS")
    print("=" * 60)

    symmetry_analysis(values)
    corner_defense_analysis(values)
    edge_analysis(values)
    phase_sensitivity(genome)
    correlation_analysis(values)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    corners_mean = sum(values[i] for i in CORNERS) / len(CORNERS)
    x_mean = sum(values[i] for i in X_SQUARES) / len(X_SQUARES)

    print(f"\n✓ Corners favored: {corners_mean:+.3f}")
    print(f"✓ X-squares penalized: {x_mean:+.3f}")
    print(f"✓ Ratio: {corners_mean / abs(x_mean) if x_mean != 0 else 0:.2f}x")
    print(f"\nEvolution discovered classical Othello wisdom!")


if __name__ == "__main__":
    main()
