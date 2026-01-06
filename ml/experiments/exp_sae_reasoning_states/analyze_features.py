"""
Analyze SAE features to discover interpretable reasoning states.

Key analyses:
1. Feature-to-iteration mapping (which features activate at which H,L cycle)
2. Feature-to-constraint mapping (row vs col vs box focused features)
3. Feature clustering into discrete reasoning states
4. Progression analysis (easy → hard cells)
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from .extract_activations import ActivationDataset, PuzzleActivations
from .train_sae import ReasoningSAE, ReasoningSAEConfig, load_reasoning_sae


@dataclass
class FeatureProfile:
    """Profile of a single SAE feature."""
    feature_id: int
    activation_count: int = 0
    mean_activation: float = 0.0

    # Iteration distribution
    iteration_dist: Dict[Tuple[int, int], int] = field(default_factory=dict)

    # Constraint association (based on guard patterns)
    row_association: float = 0.0
    col_association: float = 0.0
    box_association: float = 0.0

    # Cell difficulty association (based on empty count)
    easy_cell_count: int = 0   # High initial probability cells
    hard_cell_count: int = 0   # Low initial probability cells

    # Co-occurring features
    co_features: Set[int] = field(default_factory=set)

    @property
    def dominant_constraint(self) -> str:
        """Get the dominant constraint type."""
        scores = {
            'row': self.row_association,
            'col': self.col_association,
            'box': self.box_association,
        }
        return max(scores.items(), key=lambda x: x[1])[0]

    @property
    def dominant_iteration(self) -> Optional[Tuple[int, int]]:
        """Get the iteration where this feature activates most."""
        if not self.iteration_dist:
            return None
        return max(self.iteration_dist.items(), key=lambda x: x[1])[0]


@dataclass
class ReasoningState:
    """A discovered reasoning state (cluster of features)."""
    state_id: int
    name: str
    features: Set[int] = field(default_factory=set)

    # Aggregate properties
    avg_h_cycle: float = 0.0
    avg_l_cycle: float = 0.0
    dominant_constraint: str = ""
    difficulty_level: str = ""  # "early", "middle", "late"

    # Interpretation
    description: str = ""


@dataclass
class FeatureAnalysis:
    """Complete analysis of SAE features."""
    config: ReasoningSAEConfig
    features: Dict[int, FeatureProfile] = field(default_factory=dict)
    states: List[ReasoningState] = field(default_factory=list)

    # Statistics
    total_activations: int = 0
    alive_feature_count: int = 0
    dead_feature_count: int = 0


def analyze_reasoning_features(
    sae: ReasoningSAE,
    activation_dataset: ActivationDataset,
    config: ReasoningSAEConfig,
    min_activation_count: int = 100,
) -> FeatureAnalysis:
    """
    Analyze SAE features to discover reasoning patterns.

    Args:
        sae: Trained ReasoningSAE
        activation_dataset: Dataset of activations
        config: SAE configuration
        min_activation_count: Minimum activations for a feature to be "alive"

    Returns:
        FeatureAnalysis with feature profiles and discovered states
    """
    print("=" * 60)
    print("Analyzing Reasoning Features")
    print("=" * 60)

    analysis = FeatureAnalysis(config=config)

    # Initialize feature profiles
    for i in range(config.latent_dim):
        analysis.features[i] = FeatureProfile(
            feature_id=i,
            activation_count=int(sae.feature_counts[i]),
            mean_activation=float(sae.feature_means[i]),
        )

    # Populate iteration distribution from sae statistics
    print("\n1. Analyzing iteration distributions...")
    H_cycles = 3  # Default
    L_cycles = 6  # Default

    for feature_id in range(config.latent_dim):
        for it_idx in range(18):  # H×L = 3×6 = 18
            count = int(sae.feature_iteration_counts[feature_id, it_idx])
            if count > 0:
                h = it_idx // L_cycles
                l = it_idx % L_cycles
                analysis.features[feature_id].iteration_dist[(h, l)] = count

    # Count alive/dead features
    analysis.alive_feature_count = sum(
        1 for f in analysis.features.values()
        if f.activation_count >= min_activation_count
    )
    analysis.dead_feature_count = config.latent_dim - analysis.alive_feature_count
    analysis.total_activations = sum(f.activation_count for f in analysis.features.values())

    print(f"   Alive features: {analysis.alive_feature_count}")
    print(f"   Dead features: {analysis.dead_feature_count}")
    print(f"   Total activations: {analysis.total_activations}")

    # Analyze constraint associations
    print("\n2. Analyzing constraint associations...")
    _analyze_constraint_associations(sae, activation_dataset, analysis, config)

    # Cluster features into states
    print("\n3. Clustering into reasoning states...")
    _cluster_into_states(analysis, min_activation_count)

    # Generate descriptions
    print("\n4. Generating state descriptions...")
    _generate_descriptions(analysis)

    return analysis


def _analyze_constraint_associations(
    sae: ReasoningSAE,
    activation_dataset: ActivationDataset,
    analysis: FeatureAnalysis,
    config: ReasoningSAEConfig,
    max_samples: int = 10000,
):
    """
    Analyze which constraint types each feature is associated with.

    Uses guard values to determine if a feature is more associated
    with row, column, or box constraints.
    """
    if not HAS_MLX:
        return

    # Get activations with guard values
    board_states, contexts, guards = activation_dataset.get_all_activations(
        flatten_to_cells=True
    )

    # Limit samples
    n = min(len(board_states), max_samples)
    indices = np.random.choice(len(board_states), n, replace=False)

    board_states = board_states[indices]
    contexts = contexts[indices]
    guards = guards[indices]

    # Encode to get active features
    acts, feature_indices = sae.encode(
        mx.array(board_states),
        mx.array(contexts),
        mx.array(guards),
    )

    feature_indices_np = np.array(feature_indices.tolist())
    guards_np = guards

    # For each feature, track which guard types were active when it fired
    for sample_idx in range(n):
        active_features = feature_indices_np[sample_idx]
        sample_guards = guards_np[sample_idx]  # [9] guard values

        # Determine which constraint type has lowest guard (most constrained)
        # This is a heuristic - low guard means constraint is tight
        row_guard = np.mean(sample_guards[:3])  # First 3 digits
        col_guard = np.mean(sample_guards[3:6])  # Middle 3 digits
        box_guard = np.mean(sample_guards[6:])  # Last 3 digits

        for fid in active_features:
            if fid < len(analysis.features):
                analysis.features[fid].row_association += (1.0 - row_guard)
                analysis.features[fid].col_association += (1.0 - col_guard)
                analysis.features[fid].box_association += (1.0 - box_guard)

    # Normalize
    for fid, profile in analysis.features.items():
        total = profile.row_association + profile.col_association + profile.box_association
        if total > 0:
            profile.row_association /= total
            profile.col_association /= total
            profile.box_association /= total


def _cluster_into_states(
    analysis: FeatureAnalysis,
    min_activation_count: int,
):
    """
    Cluster features into discrete reasoning states.

    Uses iteration timing and constraint associations for clustering.
    """
    # Get alive features
    alive_features = [
        f for f in analysis.features.values()
        if f.activation_count >= min_activation_count
    ]

    if not alive_features:
        print("   No alive features to cluster")
        return

    # Simple clustering: group by dominant iteration phase
    # Phase 1: H=0 (early), Phase 2: H=1 (middle), Phase 3: H=2 (late)
    phase_clusters = defaultdict(list)

    for feature in alive_features:
        dom_iter = feature.dominant_iteration
        if dom_iter is None:
            continue
        h, l = dom_iter
        phase_clusters[h].append(feature.feature_id)

    # Create states from clusters
    phase_names = {0: "early", 1: "middle", 2: "late"}

    for phase, feature_ids in phase_clusters.items():
        if not feature_ids:
            continue

        state = ReasoningState(
            state_id=phase,
            name=f"reasoning_{phase_names.get(phase, str(phase))}",
            features=set(feature_ids),
            avg_h_cycle=float(phase),
        )

        # Compute aggregate constraint association
        constraint_counts = {"row": 0, "col": 0, "box": 0}
        for fid in feature_ids:
            constraint_counts[analysis.features[fid].dominant_constraint] += 1

        state.dominant_constraint = max(constraint_counts.items(), key=lambda x: x[1])[0]
        state.difficulty_level = phase_names.get(phase, "unknown")

        analysis.states.append(state)

    print(f"   Discovered {len(analysis.states)} reasoning states")


def _generate_descriptions(analysis: FeatureAnalysis):
    """Generate human-readable descriptions for discovered states."""
    for state in analysis.states:
        parts = []

        # Timing
        if state.difficulty_level == "early":
            parts.append("Early refinement phase")
        elif state.difficulty_level == "middle":
            parts.append("Middle refinement phase")
        elif state.difficulty_level == "late":
            parts.append("Final refinement phase")

        # Constraint focus
        if state.dominant_constraint == "row":
            parts.append("focused on row constraints")
        elif state.dominant_constraint == "col":
            parts.append("focused on column constraints")
        elif state.dominant_constraint == "box":
            parts.append("focused on box constraints")

        # Feature count
        parts.append(f"({len(state.features)} features)")

        state.description = " - ".join(parts)


def generate_statechart(analysis: FeatureAnalysis) -> str:
    """
    Generate a Mermaid statechart diagram of discovered reasoning states.

    Returns:
        Mermaid diagram string
    """
    lines = ["stateDiagram-v2"]

    # States
    for state in sorted(analysis.states, key=lambda s: s.state_id):
        lines.append(f"    {state.name}: {state.description}")

    # Transitions (sequential phases)
    sorted_states = sorted(analysis.states, key=lambda s: s.state_id)
    for i in range(len(sorted_states) - 1):
        src = sorted_states[i].name
        dst = sorted_states[i + 1].name
        lines.append(f"    {src} --> {dst}: refine")

    # Initial and final
    if sorted_states:
        lines.append(f"    [*] --> {sorted_states[0].name}")
        lines.append(f"    {sorted_states[-1].name} --> [*]")

    return "\n".join(lines)


def generate_report(analysis: FeatureAnalysis) -> str:
    """Generate analysis report."""
    lines = ["=" * 60]
    lines.append("REASONING STATE ANALYSIS REPORT")
    lines.append("=" * 60)

    lines.append(f"\n## Overview")
    lines.append(f"- Total features: {len(analysis.features)}")
    lines.append(f"- Alive features: {analysis.alive_feature_count}")
    lines.append(f"- Dead features: {analysis.dead_feature_count}")
    lines.append(f"- Total activations: {analysis.total_activations}")

    lines.append(f"\n## Discovered States ({len(analysis.states)})")
    for state in analysis.states:
        lines.append(f"\n### {state.name}")
        lines.append(f"- Features: {len(state.features)}")
        lines.append(f"- Dominant constraint: {state.dominant_constraint}")
        lines.append(f"- Description: {state.description}")
        lines.append(f"- Top features: {list(state.features)[:5]}")

    lines.append(f"\n## Statechart Diagram")
    lines.append("```mermaid")
    lines.append(generate_statechart(analysis))
    lines.append("```")

    # Top features by activation
    lines.append(f"\n## Top 20 Features by Activation")
    sorted_features = sorted(
        analysis.features.values(),
        key=lambda f: -f.activation_count
    )[:20]

    for f in sorted_features:
        dom_iter = f.dominant_iteration or (0, 0)
        lines.append(
            f"- Feature {f.feature_id}: count={f.activation_count}, "
            f"iter=H{dom_iter[0]}L{dom_iter[1]}, "
            f"constraint={f.dominant_constraint}"
        )

    return "\n".join(lines)


def save_analysis(analysis: FeatureAnalysis, save_dir: str):
    """Save analysis results to disk."""
    os.makedirs(save_dir, exist_ok=True)

    # Save report
    report_path = os.path.join(save_dir, "analysis_report.md")
    with open(report_path, 'w') as f:
        f.write(generate_report(analysis))
    print(f"Saved report to {report_path}")

    # Save statechart
    mermaid_path = os.path.join(save_dir, "reasoning_statechart.mermaid")
    with open(mermaid_path, 'w') as f:
        f.write(generate_statechart(analysis))
    print(f"Saved statechart to {mermaid_path}")

    # Save JSON summary
    summary = {
        "alive_features": analysis.alive_feature_count,
        "dead_features": analysis.dead_feature_count,
        "states": [
            {
                "id": s.state_id,
                "name": s.name,
                "features": list(s.features),
                "dominant_constraint": s.dominant_constraint,
                "description": s.description,
            }
            for s in analysis.states
        ],
    }
    summary_path = os.path.join(save_dir, "analysis_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


def test_analysis():
    """Test feature analysis."""
    print("=" * 60)
    print("Testing Feature Analysis")
    print("=" * 60)

    # Create mock analysis
    config = ReasoningSAEConfig(
        board_dim=10,
        context_dim=64,
        guard_dim=9,
        expansion_factor=4,
        k_active=8,
    )

    analysis = FeatureAnalysis(config=config)

    # Add mock features
    for i in range(config.latent_dim):
        profile = FeatureProfile(
            feature_id=i,
            activation_count=100 + i * 10,
            mean_activation=0.5,
        )
        # Assign to iteration phases
        h = i % 3
        l = (i // 3) % 6
        profile.iteration_dist[(h, l)] = 50

        # Assign constraint associations
        if i % 3 == 0:
            profile.row_association = 0.6
            profile.col_association = 0.2
            profile.box_association = 0.2
        elif i % 3 == 1:
            profile.row_association = 0.2
            profile.col_association = 0.6
            profile.box_association = 0.2
        else:
            profile.row_association = 0.2
            profile.col_association = 0.2
            profile.box_association = 0.6

        analysis.features[i] = profile

    analysis.alive_feature_count = config.latent_dim
    analysis.dead_feature_count = 0

    # Cluster
    print("\n1. Clustering features...")
    _cluster_into_states(analysis, min_activation_count=50)

    # Generate descriptions
    print("\n2. Generating descriptions...")
    _generate_descriptions(analysis)

    # Generate report
    print("\n3. Generating report...")
    report = generate_report(analysis)
    print(report[:1000] + "...")

    # Save
    print("\n4. Saving analysis...")
    save_analysis(analysis, "/tmp/test_analysis")

    print("\n" + "=" * 60)
    print("Analysis test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_analysis()
