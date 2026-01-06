"""
Steering-Guided Iteration for TRM Refinement

H×L iteration pattern with steering vectors:
- H (Hierarchy levels): Root → Compound → Leaf
- L (Layer groups): Early → Middle → Late

Each iteration targets specific aspects with appropriate steering.

STEERING DIRECTIONS:
- Structural: JSON syntax, field presence
- Hierarchy: Parent-child, nesting
- Transition: State references, events
- Semantic: Meaning, consistency
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple, Callable
from enum import Enum

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    mx = None
    MLX_AVAILABLE = False


class HierarchyLevel(Enum):
    """Hierarchy levels for H dimension."""
    ROOT = "root"
    COMPOUND = "compound"
    LEAF = "leaf"


class LayerGroup(Enum):
    """Layer groups for L dimension."""
    EARLY = "early"      # L0-7: Structural
    MIDDLE = "middle"    # L8-15: Semantic
    LATE = "late"        # L16-23: Output


class SteeringAspect(Enum):
    """Aspects to steer."""
    STRUCTURAL = "structural"
    HIERARCHY = "hierarchy"
    TRANSITION = "transition"
    SEMANTIC = "semantic"


@dataclass
class SteeringDirection:
    """A steering direction for a specific aspect."""
    aspect: SteeringAspect
    vector: Optional[Any] = None  # Steering vector
    strength: float = 1.0
    layer_group: LayerGroup = LayerGroup.MIDDLE

    def get_layers(self) -> List[int]:
        """Get layer indices for this direction."""
        if self.layer_group == LayerGroup.EARLY:
            return list(range(0, 8))
        elif self.layer_group == LayerGroup.MIDDLE:
            return list(range(8, 16))
        else:  # LATE
            return list(range(16, 24))


@dataclass
class IterationState:
    """State of the H×L iteration."""
    hierarchy_level: HierarchyLevel
    layer_group: LayerGroup
    aspect: SteeringAspect
    iteration: int
    validity_score: float
    changes: List[str] = field(default_factory=list)


@dataclass
class IterationResult:
    """Result of a single iteration."""
    state_before: IterationState
    state_after: IterationState
    chart_before: Dict[str, Any]
    chart_after: Dict[str, Any]
    improved: bool
    duration: float


@dataclass
class GuidedIterationConfig:
    """Configuration for guided iteration."""
    max_h_iterations: int = 3  # Per hierarchy level
    max_l_iterations: int = 4  # Per layer group
    strength_schedule: List[float] = field(
        default_factory=lambda: [1.0, 1.2, 1.5, 1.8]
    )
    early_stop_threshold: float = 0.95
    min_improvement: float = 0.01


class SteeringVectorCache:
    """Cache for computed steering vectors."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}

    def get(self, aspect: SteeringAspect) -> Optional[Any]:
        """Get cached steering vector."""
        return self._cache.get(aspect.value)

    def set(self, aspect: SteeringAspect, vector: Any):
        """Cache a steering vector."""
        self._cache[aspect.value] = vector

    def compute_from_examples(
        self,
        model: Any,
        aspect: SteeringAspect,
        positive: List[str],
        negative: List[str],
        layer: int = 12,
    ) -> Optional[Any]:
        """Compute steering vector from contrastive examples."""
        if not MLX_AVAILABLE or model is None:
            return None

        cached = self.get(aspect)
        if cached is not None:
            return cached

        # Get activations for positive and negative examples
        pos_acts = []
        neg_acts = []

        for ex in positive[:5]:
            if hasattr(model, 'get_activations'):
                acts = model.get_activations(ex, layer)
                if acts is not None:
                    pos_acts.append(acts)

        for ex in negative[:5]:
            if hasattr(model, 'get_activations'):
                acts = model.get_activations(ex, layer)
                if acts is not None:
                    neg_acts.append(acts)

        if not pos_acts or not neg_acts:
            return None

        # Compute difference of means
        pos_mean = mx.mean(mx.stack(pos_acts), axis=0)
        neg_mean = mx.mean(mx.stack(neg_acts), axis=0)
        vector = pos_mean - neg_mean

        self.set(aspect, vector)
        return vector


class AspectAnalyzer:
    """Analyze chart to identify weak aspects."""

    def analyze(self, chart: Dict[str, Any]) -> Dict[SteeringAspect, float]:
        """Return scores for each aspect."""
        return {
            SteeringAspect.STRUCTURAL: self._structural_score(chart),
            SteeringAspect.HIERARCHY: self._hierarchy_score(chart),
            SteeringAspect.TRANSITION: self._transition_score(chart),
            SteeringAspect.SEMANTIC: self._semantic_score(chart),
        }

    def get_weakest(self, chart: Dict[str, Any]) -> SteeringAspect:
        """Get the weakest aspect."""
        scores = self.analyze(chart)
        return min(scores, key=scores.get)

    def get_total_score(self, chart: Dict[str, Any]) -> float:
        """Get total validity score."""
        scores = self.analyze(chart)
        return sum(scores.values()) / len(scores)

    def _structural_score(self, chart: Dict[str, Any]) -> float:
        score = 0.0
        if "root_state" in chart:
            score += 0.4
            root = chart["root_state"]
            if "label" in root:
                score += 0.2
            if "type" in root:
                score += 0.2
            if root.get("children"):
                score += 0.2
        return score

    def _hierarchy_score(self, chart: Dict[str, Any]) -> float:
        root = chart.get("root_state", {})
        children = root.get("children", [])

        if not children:
            return 0.0

        score = 0.5
        has_initial = any(c.get("is_initial") for c in children if isinstance(c, dict))
        if has_initial:
            score += 0.3

        # Check compound states
        for c in children:
            if isinstance(c, dict) and c.get("type") == 2:
                if c.get("children"):
                    score += 0.1
                else:
                    score -= 0.1

        return max(0.0, min(1.0, score))

    def _transition_score(self, chart: Dict[str, Any]) -> float:
        transitions = chart.get("transitions", [])
        if not transitions:
            return 0.5

        states = self._collect_states(chart)
        valid = 0

        for t in transitions:
            if not isinstance(t, dict):
                continue
            refs_ok = all(r in states for r in t.get("from", []) + t.get("to", []))
            has_event = bool(t.get("event"))
            if refs_ok and has_event:
                valid += 1
            elif refs_ok:
                valid += 0.5

        return valid / len(transitions)

    def _semantic_score(self, chart: Dict[str, Any]) -> float:
        root = chart.get("root_state", {})
        children = root.get("children", [])
        score = 0.0

        if len(children) >= 2:
            score += 0.4
        if chart.get("transitions"):
            score += 0.3

        # Check for duplicates
        labels = self._collect_labels(chart)
        if len(labels) == len(set(labels)):
            score += 0.3

        return min(1.0, score)

    def _collect_states(self, chart: Dict[str, Any]) -> Set[str]:
        states = set()
        def collect(s):
            if isinstance(s, dict):
                states.add(s.get("label", ""))
                for c in s.get("children", []):
                    collect(c)
        collect(chart.get("root_state", {}))
        return states

    def _collect_labels(self, chart: Dict[str, Any]) -> List[str]:
        labels = []
        def collect(s):
            if isinstance(s, dict):
                labels.append(s.get("label", ""))
                for c in s.get("children", []):
                    collect(c)
        collect(chart.get("root_state", {}))
        return labels


class SteeringGuidedIterator:
    """
    Iterate with steering guidance following H×L pattern.

    H×L means:
    - Outer loop over hierarchy levels (root → compound → leaf)
    - Inner loop over layer groups (early → middle → late)
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        config: Optional[GuidedIterationConfig] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.config = config or GuidedIterationConfig()
        self.verbose = verbose

        self.analyzer = AspectAnalyzer()
        self.vector_cache = SteeringVectorCache()

        # Aspect to layer group mapping
        self.aspect_layers = {
            SteeringAspect.STRUCTURAL: LayerGroup.EARLY,
            SteeringAspect.HIERARCHY: LayerGroup.EARLY,
            SteeringAspect.TRANSITION: LayerGroup.MIDDLE,
            SteeringAspect.SEMANTIC: LayerGroup.LATE,
        }

    def iterate(
        self,
        chart: Dict[str, Any],
        prompt: str,
    ) -> Tuple[Dict[str, Any], List[IterationResult]]:
        """
        Run H×L guided iteration.

        Returns:
            (refined_chart, iteration_results)
        """
        results = []
        current = chart.copy()
        total_iterations = 0

        hierarchy_levels = [HierarchyLevel.ROOT, HierarchyLevel.COMPOUND, HierarchyLevel.LEAF]

        for h_level in hierarchy_levels:
            for h_iter in range(self.config.max_h_iterations):
                current_score = self.analyzer.get_total_score(current)

                if current_score >= self.config.early_stop_threshold:
                    if self.verbose:
                        print(f"  Early stop at {current_score:.1%}")
                    return current, results

                # Get weakest aspect
                weak_aspect = self.analyzer.get_weakest(current)
                layer_group = self.aspect_layers[weak_aspect]

                # Determine strength from schedule
                strength_idx = min(total_iterations, len(self.config.strength_schedule) - 1)
                strength = self.config.strength_schedule[strength_idx]

                # Create steering direction
                direction = SteeringDirection(
                    aspect=weak_aspect,
                    strength=strength,
                    layer_group=layer_group,
                )

                # Apply iteration
                start = time.time()
                refined, changes = self._apply_iteration(
                    current, prompt, h_level, direction
                )
                duration = time.time() - start

                new_score = self.analyzer.get_total_score(refined)

                state_before = IterationState(
                    hierarchy_level=h_level,
                    layer_group=layer_group,
                    aspect=weak_aspect,
                    iteration=total_iterations,
                    validity_score=current_score,
                )

                state_after = IterationState(
                    hierarchy_level=h_level,
                    layer_group=layer_group,
                    aspect=weak_aspect,
                    iteration=total_iterations,
                    validity_score=new_score,
                    changes=changes,
                )

                improved = new_score > current_score + self.config.min_improvement

                result = IterationResult(
                    state_before=state_before,
                    state_after=state_after,
                    chart_before=current,
                    chart_after=refined,
                    improved=improved,
                    duration=duration,
                )
                results.append(result)

                if self.verbose:
                    status = "+" if improved else "="
                    print(f"  [{status}] H={h_level.value} L={layer_group.value} "
                          f"A={weak_aspect.value}: {current_score:.1%} -> {new_score:.1%}")

                if improved:
                    current = refined

                total_iterations += 1

                if total_iterations >= self.config.max_h_iterations * self.config.max_l_iterations:
                    return current, results

        return current, results

    def _apply_iteration(
        self,
        chart: Dict[str, Any],
        prompt: str,
        h_level: HierarchyLevel,
        direction: SteeringDirection,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Apply a single iteration with steering."""
        if self.model is None:
            return self._mock_iteration(chart, h_level, direction.aspect)

        # Build focused prompt
        focused_prompt = self._build_focused_prompt(chart, prompt, h_level, direction.aspect)

        # Get steering vector
        vector = self.vector_cache.get(direction.aspect)

        # Apply steering hooks
        layers = direction.get_layers()
        hooks = []
        for layer in layers:
            hooks.append((
                f"model.layers.{layer}.mlp",
                lambda m, i, o: o + direction.strength * vector if vector else o
            ))

        if hasattr(self.model, 'run_with_hooks'):
            output = self.model.run_with_hooks(focused_prompt, hooks=hooks)
        else:
            output = self.model.generate(focused_prompt)

        try:
            refined = json.loads(output)
            return refined, [f"refined_{direction.aspect.value}"]
        except json.JSONDecodeError:
            return chart, []

    def _build_focused_prompt(
        self,
        chart: Dict[str, Any],
        original_prompt: str,
        h_level: HierarchyLevel,
        aspect: SteeringAspect,
    ) -> str:
        """Build prompt focused on specific level and aspect."""
        chart_json = json.dumps(chart, indent=2)

        level_focus = {
            HierarchyLevel.ROOT: "Focus on the root state structure.",
            HierarchyLevel.COMPOUND: "Focus on compound states and their children.",
            HierarchyLevel.LEAF: "Focus on leaf states and transitions.",
        }

        aspect_focus = {
            SteeringAspect.STRUCTURAL: "Ensure JSON structure is valid.",
            SteeringAspect.HIERARCHY: "Fix parent-child relationships and initial states.",
            SteeringAspect.TRANSITION: "Fix state references in transitions.",
            SteeringAspect.SEMANTIC: "Fix duplicate labels and ensure consistency.",
        }

        return f"""Original: {original_prompt}

Current chart:
{chart_json}

{level_focus.get(h_level, "")}
{aspect_focus.get(aspect, "")}

Output corrected JSON:"""

    def _mock_iteration(
        self,
        chart: Dict[str, Any],
        h_level: HierarchyLevel,
        aspect: SteeringAspect,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Mock iteration for testing."""
        refined = json.loads(json.dumps(chart))  # Deep copy
        changes = []

        if aspect == SteeringAspect.STRUCTURAL:
            if "root_state" not in refined:
                refined["root_state"] = {"label": "__root__", "type": 2, "children": []}
                changes.append("added_root")
            elif "type" not in refined["root_state"]:
                refined["root_state"]["type"] = 2
                changes.append("added_type")

        elif aspect == SteeringAspect.HIERARCHY:
            children = refined.get("root_state", {}).get("children", [])
            if children and not any(c.get("is_initial") for c in children if isinstance(c, dict)):
                children[0]["is_initial"] = True
                changes.append("marked_initial")

            # Fix empty compound states
            for c in children:
                if isinstance(c, dict) and c.get("type") == 2 and not c.get("children"):
                    c["children"] = [{"label": f"{c.get('label', 'sub')}_child", "type": 1, "is_initial": True}]
                    changes.append("added_compound_child")

        elif aspect == SteeringAspect.TRANSITION:
            states = set()
            def collect(s):
                if isinstance(s, dict):
                    states.add(s.get("label", ""))
                    for ch in s.get("children", []):
                        collect(ch)
            collect(refined.get("root_state", {}))

            for t in refined.get("transitions", []):
                if isinstance(t, dict):
                    # Fix invalid refs
                    for key in ["from", "to"]:
                        t[key] = [r for r in t.get(key, []) if r in states]
                        if not t[key] and states:
                            t[key] = [list(states)[0]]
                            changes.append(f"fixed_{key}")
                    if not t.get("event"):
                        t["event"] = "E"
                        changes.append("added_event")

        elif aspect == SteeringAspect.SEMANTIC:
            # Fix duplicates
            seen = set()
            def fix(s, suffix=0):
                if isinstance(s, dict):
                    label = s.get("label", "")
                    if label in seen:
                        s["label"] = f"{label}_{suffix}"
                        changes.append("fixed_dup")
                    seen.add(s.get("label", ""))
                    for i, ch in enumerate(s.get("children", [])):
                        fix(ch, suffix + i + 1)
            fix(refined.get("root_state", {}))

        return refined, changes


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate steering-guided iteration."""
    print("=" * 60)
    print("Steering-Guided H×L Iteration")
    print("=" * 60)

    # Start with a broken chart
    broken_chart = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "s0", "type": 1},  # Missing is_initial
                {"label": "s1", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["s0"], "to": ["TYPO_s1"]},  # Bad ref, no event
        ]
    }

    print(f"\nStarting chart (broken):")
    print(json.dumps(broken_chart, indent=2))

    iterator = SteeringGuidedIterator(model=None, verbose=True)

    print("\nRunning H×L iteration:")
    refined, results = iterator.iterate(broken_chart, "Generate a state machine")

    print(f"\nRefined chart:")
    print(json.dumps(refined, indent=2))

    print(f"\nResults:")
    print(f"  Iterations: {len(results)}")
    print(f"  Improvements: {sum(1 for r in results if r.improved)}")

    initial_score = iterator.analyzer.get_total_score(broken_chart)
    final_score = iterator.analyzer.get_total_score(refined)
    print(f"  Validity: {initial_score:.1%} -> {final_score:.1%}")

    return refined, results


if __name__ == "__main__":
    demo()
