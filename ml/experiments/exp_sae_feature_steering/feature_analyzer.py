"""
Feature Analyzer for SC-Specific Features.

Analyzes SAE features to identify interpretable SC concepts:
- STATE features: Activate for state definitions
- TRANSITION features: Activate for transitions
- GUARD features: Activate for guard conditions
- EVENT features: Activate for event names
- STRUCTURAL features: Activate for JSON structure

Provides feature clustering and interpretation tools.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum, auto
from collections import defaultdict
import math

from .activation_collector import SemanticContext
from .sae_trainer import TopKSAE, TrainerConfig

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None


class SCFeatureType(Enum):
    """Types of SC-specific features."""
    STATE = auto()          # State definition features
    TRANSITION = auto()     # Transition features
    GUARD = auto()          # Guard condition features
    EVENT = auto()          # Event name features
    STRUCTURAL = auto()     # JSON structure features
    MIXED = auto()          # Multiple contexts
    UNKNOWN = auto()


@dataclass
class SCFeature:
    """An analyzed SC feature."""
    feature_id: int
    feature_type: SCFeatureType = SCFeatureType.UNKNOWN
    activation_count: int = 0
    mean_activation: float = 0.0

    # Context associations
    primary_context: Optional[SemanticContext] = None
    context_distribution: Dict[SemanticContext, float] = field(default_factory=dict)

    # Token associations
    associated_tokens: Set[str] = field(default_factory=set)
    top_tokens: List[Tuple[str, int]] = field(default_factory=list)

    # Co-occurrence
    co_occurring_features: List[Tuple[int, float]] = field(default_factory=list)

    # Interpretability score (0-1)
    interpretability: float = 0.0
    description: str = ""

    def describe(self) -> str:
        """Generate human-readable description."""
        tokens = ", ".join(self.associated_tokens)[:50]
        ctx = self.primary_context.name if self.primary_context else "?"
        return (f"Feature {self.feature_id} [{self.feature_type.name}]: "
                f"ctx={ctx}, tokens=[{tokens}], "
                f"count={self.activation_count}, interp={self.interpretability:.2f}")


@dataclass
class FeatureCluster:
    """Cluster of related features."""
    cluster_id: int
    feature_type: SCFeatureType
    feature_ids: Set[int] = field(default_factory=set)
    centroid: Optional[Any] = None
    description: str = ""

    @property
    def size(self) -> int:
        return len(self.feature_ids)


@dataclass
class AnalysisResult:
    """Result of feature analysis."""
    total_features: int = 0
    active_features: int = 0
    dead_features: int = 0

    features: Dict[int, SCFeature] = field(default_factory=dict)
    clusters: List[FeatureCluster] = field(default_factory=list)

    # Feature type counts
    type_counts: Dict[SCFeatureType, int] = field(default_factory=lambda: defaultdict(int))

    # Top features per type
    top_state_features: List[int] = field(default_factory=list)
    top_transition_features: List[int] = field(default_factory=list)
    top_guard_features: List[int] = field(default_factory=list)
    top_event_features: List[int] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== SC Feature Analysis ===",
            f"Total features: {self.total_features}",
            f"Active features: {self.active_features}",
            f"Dead features: {self.dead_features}",
            f"Clusters: {len(self.clusters)}",
            "",
            "Feature types:"
        ]
        for ft, count in sorted(self.type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {ft.name}: {count}")

        if self.top_state_features:
            lines.append(f"\nTop STATE features: {self.top_state_features[:5]}")
        if self.top_transition_features:
            lines.append(f"Top TRANSITION features: {self.top_transition_features[:5]}")
        if self.top_event_features:
            lines.append(f"Top EVENT features: {self.top_event_features[:5]}")

        return "\n".join(lines)


class FeatureAnalyzer:
    """
    Analyzer for SAE features.

    Discovers interpretable SC-specific features from trained SAE.
    """

    def __init__(self, sae: TopKSAE):
        self.sae = sae
        self.features: Dict[int, SCFeature] = {}
        self.clusters: List[FeatureCluster] = []

        # Initialize features
        for i in range(sae.latent_dim):
            self.features[i] = SCFeature(feature_id=i)

    def analyze(
        self,
        min_activation_count: int = 10,
        cluster_threshold: float = 0.5,
    ) -> AnalysisResult:
        """
        Analyze SAE features.

        Args:
            min_activation_count: Minimum activations to be considered active
            cluster_threshold: Similarity threshold for clustering

        Returns:
            AnalysisResult with feature analysis
        """
        result = AnalysisResult(total_features=self.sae.latent_dim)

        # Get feature counts
        if HAS_MLX:
            counts = self.sae.feature_counts.tolist()
        else:
            counts = list(self.sae.feature_counts)

        # Analyze each feature
        for fid, feature in self.features.items():
            feature.activation_count = int(counts[fid])

            if feature.activation_count < min_activation_count:
                result.dead_features += 1
                continue

            result.active_features += 1

            # Analyze context distribution
            self._analyze_context_distribution(feature)

            # Determine feature type
            self._classify_feature_type(feature)

            # Compute interpretability
            self._compute_interpretability(feature)

            result.features[fid] = feature
            result.type_counts[feature.feature_type] += 1

        # Cluster features
        self.clusters = self._cluster_features(cluster_threshold)
        result.clusters = self.clusters

        # Get top features per type
        result.top_state_features = self._get_top_features(SCFeatureType.STATE)
        result.top_transition_features = self._get_top_features(SCFeatureType.TRANSITION)
        result.top_guard_features = self._get_top_features(SCFeatureType.GUARD)
        result.top_event_features = self._get_top_features(SCFeatureType.EVENT)

        return result

    def _analyze_context_distribution(self, feature: SCFeature):
        """Analyze context distribution for a feature."""
        ctx_counts = self.sae.context_feature_counts
        total = 0
        distribution = {}

        for ctx, feature_counts in ctx_counts.items():
            count = feature_counts.get(feature.feature_id, 0)
            if count > 0:
                distribution[ctx] = count
                total += count

        if total > 0:
            feature.context_distribution = {
                ctx: count / total
                for ctx, count in distribution.items()
            }
            # Primary context is the one with highest activation
            if distribution:
                feature.primary_context = max(distribution.items(), key=lambda x: x[1])[0]

    def _classify_feature_type(self, feature: SCFeature):
        """Classify feature type based on context distribution."""
        if not feature.context_distribution:
            feature.feature_type = SCFeatureType.UNKNOWN
            return

        # Map contexts to feature types
        state_contexts = {
            SemanticContext.STATE_DEF,
            SemanticContext.STATE_LABEL,
            SemanticContext.STATE_TYPE,
            SemanticContext.STATE_CHILDREN,
        }
        transition_contexts = {
            SemanticContext.TRANSITION,
            SemanticContext.TRANS_FROM,
            SemanticContext.TRANS_TO,
        }
        guard_context = {SemanticContext.TRANS_GUARD}
        event_context = {SemanticContext.TRANS_EVENT}
        structural_context = {SemanticContext.STRUCTURAL}

        # Calculate type scores
        state_score = sum(
            feature.context_distribution.get(ctx, 0)
            for ctx in state_contexts
        )
        trans_score = sum(
            feature.context_distribution.get(ctx, 0)
            for ctx in transition_contexts
        )
        guard_score = sum(
            feature.context_distribution.get(ctx, 0)
            for ctx in guard_context
        )
        event_score = sum(
            feature.context_distribution.get(ctx, 0)
            for ctx in event_context
        )
        structural_score = sum(
            feature.context_distribution.get(ctx, 0)
            for ctx in structural_context
        )

        scores = {
            SCFeatureType.STATE: state_score,
            SCFeatureType.TRANSITION: trans_score,
            SCFeatureType.GUARD: guard_score,
            SCFeatureType.EVENT: event_score,
            SCFeatureType.STRUCTURAL: structural_score,
        }

        max_score = max(scores.values())
        if max_score < 0.4:  # No dominant type
            feature.feature_type = SCFeatureType.MIXED
        else:
            feature.feature_type = max(scores.items(), key=lambda x: x[1])[0]

    def _compute_interpretability(self, feature: SCFeature):
        """Compute interpretability score for feature."""
        score = 0.0

        # High activation count
        if feature.activation_count > 100:
            score += 0.3

        # Clear primary context
        if feature.primary_context:
            primary_ratio = feature.context_distribution.get(feature.primary_context, 0)
            score += 0.3 * primary_ratio

        # Non-mixed type
        if feature.feature_type not in (SCFeatureType.MIXED, SCFeatureType.UNKNOWN):
            score += 0.2

        # Has associated tokens
        if feature.associated_tokens:
            score += 0.2

        feature.interpretability = min(1.0, score)

    def _cluster_features(self, threshold: float) -> List[FeatureCluster]:
        """Cluster similar features."""
        clusters = []

        # Group by feature type
        type_groups: Dict[SCFeatureType, List[int]] = defaultdict(list)
        for fid, feature in self.features.items():
            if feature.activation_count > 0:
                type_groups[feature.feature_type].append(fid)

        # Create cluster per type
        for ftype, feature_ids in type_groups.items():
            if len(feature_ids) > 0:
                cluster = FeatureCluster(
                    cluster_id=len(clusters),
                    feature_type=ftype,
                    feature_ids=set(feature_ids),
                    description=f"{ftype.name} features ({len(feature_ids)})"
                )
                clusters.append(cluster)

        return clusters

    def _get_top_features(self, ftype: SCFeatureType, n: int = 10) -> List[int]:
        """Get top features of a given type."""
        candidates = [
            (fid, f.activation_count)
            for fid, f in self.features.items()
            if f.feature_type == ftype
        ]
        sorted_features = sorted(candidates, key=lambda x: -x[1])
        return [fid for fid, _ in sorted_features[:n]]

    def get_feature_for_context(
        self,
        context: SemanticContext,
        n: int = 5
    ) -> List[Tuple[int, float]]:
        """Get top features for a semantic context."""
        candidates = []
        for fid, feature in self.features.items():
            score = feature.context_distribution.get(context, 0)
            if score > 0:
                candidates.append((fid, score))

        return sorted(candidates, key=lambda x: -x[1])[:n]

    def describe_feature(self, feature_id: int) -> str:
        """Get detailed description of a feature."""
        if feature_id not in self.features:
            return f"Feature {feature_id}: not found"

        f = self.features[feature_id]
        lines = [f.describe()]

        if f.context_distribution:
            lines.append("  Context distribution:")
            for ctx, ratio in sorted(f.context_distribution.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"    {ctx.name}: {ratio:.2%}")

        if f.co_occurring_features:
            lines.append("  Co-occurring features:")
            for other_id, score in f.co_occurring_features[:5]:
                lines.append(f"    Feature {other_id}: {score:.2f}")

        return "\n".join(lines)

    def describe_cluster(self, cluster_id: int) -> str:
        """Get description of a feature cluster."""
        if cluster_id >= len(self.clusters):
            return f"Cluster {cluster_id}: not found"

        c = self.clusters[cluster_id]
        lines = [
            f"Cluster {c.cluster_id}: {c.description}",
            f"  Type: {c.feature_type.name}",
            f"  Size: {c.size} features",
            f"  Features: {list(c.feature_ids)[:10]}...",
        ]
        return "\n".join(lines)


def test_analyzer():
    """Test feature analyzer."""
    print("=" * 60)
    print("Testing Feature Analyzer")
    print("=" * 60)

    from .activation_collector import ActivationCollector
    from .sae_trainer import SAETrainer, TrainerConfig

    # Collect activations
    collector = ActivationCollector()
    sc_jsons = [
        '{"root_state": {"label": "__root__", "children": [{"label": "A"}, {"label": "B"}]}, "transitions": [{"from": ["A"], "to": ["B"], "event": "GO", "guard": "ready"}]}',
        '{"root_state": {"label": "__root__", "children": [{"label": "X"}, {"label": "Y"}, {"label": "Z"}]}, "transitions": [{"from": ["X"], "to": ["Y"], "event": "NEXT"}, {"from": ["Y"], "to": ["Z"], "event": "NEXT"}]}',
    ]

    for _ in range(5):
        for js in sc_jsons:
            collector.collect(js)

    cache = collector.cache
    print(f"\n1. Collected {cache.total} activations")

    # Train SAE
    config = TrainerConfig(
        input_dim=64,
        expansion_factor=8,
        k_active=8,
        num_epochs=30,
    )
    trainer = SAETrainer(config)
    sae = trainer.train(cache, verbose=False)
    print("2. SAE trained")

    # Analyze features
    analyzer = FeatureAnalyzer(sae)
    result = analyzer.analyze(min_activation_count=1)

    print(f"\n3. Analysis results:")
    print(result.summary())

    # Show top features
    print("\n4. Feature details:")
    for ftype in [SCFeatureType.STATE, SCFeatureType.TRANSITION, SCFeatureType.EVENT]:
        top = analyzer._get_top_features(ftype, n=3)
        if top:
            print(f"\n  Top {ftype.name} features: {top}")
            for fid in top[:2]:
                print(f"    {analyzer.describe_feature(fid)}")

    # Show clusters
    print("\n5. Clusters:")
    for i, cluster in enumerate(result.clusters[:3]):
        print(f"  {analyzer.describe_cluster(i)}")

    print("\n" + "=" * 60)
    print("Feature analyzer tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_analyzer()
