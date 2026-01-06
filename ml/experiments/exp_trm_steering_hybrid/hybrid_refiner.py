"""
Hybrid Refiner: Combining TRM Refinement with Steering

Two-stage approach:
1. First pass: Steered generation with validity-oriented direction
2. Refinement: TRM H×L iteration with steering-guided corrections

TRM = Test-time Refinement Method
H×L = Hierarchy × Layers iteration pattern

TARGET: 60% → 99% validity
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


class RefinementPhase(Enum):
    """Phases of the hybrid refinement process."""
    STEERED_GENERATION = "steered_generation"
    TRM_STRUCTURAL = "trm_structural"
    TRM_SEMANTIC = "trm_semantic"
    TRM_TRANSITION = "trm_transition"
    FINAL_POLISH = "final_polish"


@dataclass
class SteeringConfig:
    """Configuration for steering during generation."""
    validity_direction: Optional[Any] = None  # Steering vector
    strength: float = 1.0
    layers: List[int] = field(default_factory=lambda: list(range(8, 15)))
    apply_to: str = "mlp"  # "attention", "mlp", or "both"


@dataclass
class TRMConfig:
    """Configuration for TRM refinement."""
    max_iterations: int = 5
    hierarchy_levels: int = 3  # H in H×L
    layer_groups: int = 4      # L in H×L
    early_stop_threshold: float = 0.95
    refinement_strength: float = 1.2


@dataclass
class HybridConfig:
    """Configuration for hybrid refinement."""
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    trm: TRMConfig = field(default_factory=TRMConfig)
    use_steering_in_trm: bool = True
    max_total_iterations: int = 10


@dataclass
class RefinementStep:
    """A single step in the refinement process."""
    phase: RefinementPhase
    iteration: int
    validity_before: float
    validity_after: float
    changes_made: List[str]
    duration: float

    @property
    def improvement(self) -> float:
        return self.validity_after - self.validity_before


@dataclass
class RefinementResult:
    """Result of hybrid refinement."""
    original_chart: Dict[str, Any]
    refined_chart: Dict[str, Any]
    initial_validity: float
    final_validity: float
    steps: List[RefinementStep]
    total_iterations: int
    success: bool

    @property
    def total_improvement(self) -> float:
        return self.final_validity - self.initial_validity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_validity": self.initial_validity,
            "final_validity": self.final_validity,
            "improvement": self.total_improvement,
            "total_iterations": self.total_iterations,
            "success": self.success,
            "phases": [s.phase.value for s in self.steps],
        }


class ValidityChecker:
    """Check statechart validity with fine-grained scores."""

    def check(self, chart: Dict[str, Any]) -> Dict[str, float]:
        """Return validity scores for each aspect."""
        scores = {
            "structural": self._check_structural(chart),
            "hierarchy": self._check_hierarchy(chart),
            "transitions": self._check_transitions(chart),
            "semantic": self._check_semantic(chart),
        }
        scores["total"] = sum(scores.values()) / len(scores)
        return scores

    def _check_structural(self, chart: Dict[str, Any]) -> float:
        """Check structural validity."""
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

    def _check_hierarchy(self, chart: Dict[str, Any]) -> float:
        """Check hierarchy validity."""
        root = chart.get("root_state", {})
        children = root.get("children", [])

        if not children:
            return 0.0

        score = 0.5  # Has children

        # Check for initial state
        has_initial = any(
            c.get("is_initial", False) for c in children
            if isinstance(c, dict)
        )
        if has_initial:
            score += 0.3

        # Check compound states have children
        def check_compound(state):
            if not isinstance(state, dict):
                return 1.0
            if state.get("type") == 2:  # Compound
                if not state.get("children"):
                    return 0.5
            return 1.0

        compound_score = sum(check_compound(c) for c in children) / max(len(children), 1)
        score += 0.2 * compound_score

        return min(1.0, score)

    def _check_transitions(self, chart: Dict[str, Any]) -> float:
        """Check transition validity."""
        transitions = chart.get("transitions", [])
        if not transitions:
            return 0.5  # No transitions is not invalid

        # Collect all state names
        all_states = set()
        def collect(state):
            if isinstance(state, dict):
                all_states.add(state.get("label", ""))
                for c in state.get("children", []):
                    collect(c)
        collect(chart.get("root_state", {}))

        valid_count = 0
        for t in transitions:
            if not isinstance(t, dict):
                continue

            # Check references
            refs_valid = all(
                r in all_states
                for r in t.get("from", []) + t.get("to", [])
            )
            has_event = bool(t.get("event"))

            if refs_valid and has_event:
                valid_count += 1
            elif refs_valid:
                valid_count += 0.5

        return valid_count / len(transitions) if transitions else 0.5

    def _check_semantic(self, chart: Dict[str, Any]) -> float:
        """Check semantic validity."""
        score = 0.0
        root = chart.get("root_state", {})
        children = root.get("children", [])

        # Has at least 2 states (meaningful machine)
        if len(children) >= 2:
            score += 0.4

        # Has transitions
        if chart.get("transitions"):
            score += 0.3

        # No duplicate labels
        labels = []
        def collect_labels(state):
            if isinstance(state, dict):
                labels.append(state.get("label", ""))
                for c in state.get("children", []):
                    collect_labels(c)
        collect_labels(root)

        if len(labels) == len(set(labels)):
            score += 0.3

        return min(1.0, score)


class SteeringGenerator:
    """Generate statecharts with steering."""

    def __init__(
        self,
        model: Optional[Any] = None,
        config: Optional[SteeringConfig] = None,
    ):
        self.model = model
        self.config = config or SteeringConfig()

    def generate(self, prompt: str) -> str:
        """Generate with steering applied."""
        if self.model is None:
            return self._mock_generate(prompt)

        # Apply steering hooks
        hooks = self._create_hooks()

        if hasattr(self.model, 'run_with_hooks'):
            return self.model.run_with_hooks(prompt, hooks=hooks)
        else:
            return self.model.generate(prompt)

    def _create_hooks(self) -> List[Tuple[str, Callable]]:
        """Create steering hooks for target layers."""
        hooks = []
        if self.config.validity_direction is None:
            return hooks

        for layer in self.config.layers:
            if self.config.apply_to in ("mlp", "both"):
                hooks.append((
                    f"model.layers.{layer}.mlp",
                    self._steering_hook
                ))
            if self.config.apply_to in ("attention", "both"):
                hooks.append((
                    f"model.layers.{layer}.self_attn",
                    self._steering_hook
                ))

        return hooks

    def _steering_hook(self, module, inputs, outputs):
        """Add steering direction to outputs."""
        if MLX_AVAILABLE and self.config.validity_direction is not None:
            direction = mx.broadcast_to(
                self.config.validity_direction,
                outputs.shape
            )
            return outputs + self.config.strength * direction
        return outputs

    def _mock_generate(self, prompt: str) -> str:
        """Mock generation for testing."""
        # Simulate ~70% validity output
        return json.dumps({
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "idle", "type": 1, "is_initial": True},
                    {"label": "active", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["idle"], "to": ["active"], "event": "START"},
            ]
        })


class TRMRefiner:
    """
    Test-time Refinement Method for statecharts.

    Uses H×L iteration pattern:
    - H (Hierarchy): Refine at different hierarchy levels
    - L (Layers): Use different layer groups for different aspects
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        config: Optional[TRMConfig] = None,
        steering_config: Optional[SteeringConfig] = None,
    ):
        self.model = model
        self.config = config or TRMConfig()
        self.steering_config = steering_config
        self.checker = ValidityChecker()

    def refine(
        self,
        chart: Dict[str, Any],
        prompt: str,
    ) -> Tuple[Dict[str, Any], List[RefinementStep]]:
        """
        Refine chart using H×L iteration.

        Returns:
            (refined_chart, steps)
        """
        steps = []
        current = chart.copy()

        for iteration in range(self.config.max_iterations):
            scores = self.checker.check(current)

            if scores["total"] >= self.config.early_stop_threshold:
                break

            # Identify weakest aspect
            aspects = ["structural", "hierarchy", "transitions", "semantic"]
            weakest = min(aspects, key=lambda a: scores[a])

            # Apply targeted refinement
            start = time.time()
            refined, changes = self._refine_aspect(current, weakest, prompt)
            duration = time.time() - start

            new_scores = self.checker.check(refined)

            step = RefinementStep(
                phase=self._phase_for_aspect(weakest),
                iteration=iteration,
                validity_before=scores["total"],
                validity_after=new_scores["total"],
                changes_made=changes,
                duration=duration,
            )
            steps.append(step)

            if new_scores["total"] > scores["total"]:
                current = refined

        return current, steps

    def _phase_for_aspect(self, aspect: str) -> RefinementPhase:
        """Map aspect to refinement phase."""
        mapping = {
            "structural": RefinementPhase.TRM_STRUCTURAL,
            "hierarchy": RefinementPhase.TRM_SEMANTIC,
            "transitions": RefinementPhase.TRM_TRANSITION,
            "semantic": RefinementPhase.TRM_SEMANTIC,
        }
        return mapping.get(aspect, RefinementPhase.TRM_STRUCTURAL)

    def _refine_aspect(
        self,
        chart: Dict[str, Any],
        aspect: str,
        prompt: str,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Refine a specific aspect of the chart."""
        if self.model is None:
            return self._mock_refine(chart, aspect)

        # Build refinement prompt
        refine_prompt = self._build_refine_prompt(chart, aspect, prompt)

        # Generate with steering if configured
        if self.steering_config:
            generator = SteeringGenerator(self.model, self.steering_config)
            output = generator.generate(refine_prompt)
        else:
            output = self.model.generate(refine_prompt)

        try:
            refined = json.loads(output)
            return refined, [f"refined_{aspect}"]
        except json.JSONDecodeError:
            return chart, []

    def _build_refine_prompt(
        self,
        chart: Dict[str, Any],
        aspect: str,
        original_prompt: str,
    ) -> str:
        """Build prompt for aspect refinement."""
        chart_json = json.dumps(chart, indent=2)

        aspect_instructions = {
            "structural": "Fix any structural issues. Ensure root_state has label, type, and children.",
            "hierarchy": "Fix hierarchy issues. Ensure compound states have children and initial states are marked.",
            "transitions": "Fix transition issues. Ensure all from/to references exist and events are specified.",
            "semantic": "Fix semantic issues. Ensure no duplicate labels and machine is meaningful.",
        }

        return f"""Original request: {original_prompt}

Current statechart (needs refinement):
{chart_json}

Issue: {aspect} validity is low.
Instruction: {aspect_instructions.get(aspect, "Fix issues.")}

Output the corrected statechart JSON only:"""

    def _mock_refine(
        self,
        chart: Dict[str, Any],
        aspect: str,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Mock refinement for testing."""
        refined = json.loads(json.dumps(chart))  # Deep copy
        changes = []

        if aspect == "structural":
            if "root_state" not in refined:
                refined["root_state"] = {"label": "__root__", "type": 2, "children": []}
                changes.append("added_root_state")

        elif aspect == "hierarchy":
            children = refined.get("root_state", {}).get("children", [])
            if children and not any(c.get("is_initial") for c in children):
                children[0]["is_initial"] = True
                changes.append("marked_initial_state")

        elif aspect == "transitions":
            # Collect valid states
            states = set()
            def collect(s):
                if isinstance(s, dict):
                    states.add(s.get("label", ""))
                    for c in s.get("children", []):
                        collect(c)
            collect(refined.get("root_state", {}))

            # Fix invalid refs
            for t in refined.get("transitions", []):
                for key in ["from", "to"]:
                    t[key] = [r for r in t.get(key, []) if r in states]
                    if not t[key] and states:
                        t[key] = [list(states)[0]]
                        changes.append(f"fixed_{key}_ref")

                if not t.get("event"):
                    t["event"] = "EVENT"
                    changes.append("added_event")

        elif aspect == "semantic":
            # Fix duplicates
            seen = set()
            def fix_dupes(state, suffix=0):
                if isinstance(state, dict):
                    label = state.get("label", "")
                    if label in seen:
                        state["label"] = f"{label}_{suffix}"
                        changes.append(f"fixed_duplicate_{label}")
                    seen.add(state.get("label", ""))
                    for i, c in enumerate(state.get("children", [])):
                        fix_dupes(c, suffix + i + 1)
            fix_dupes(refined.get("root_state", {}))

        return refined, changes


class HybridRefiner:
    """
    Combines steered generation with TRM refinement.

    Two-stage approach:
    1. Steered generation: Initial output with validity steering
    2. TRM refinement: Iterative H×L improvement
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        config: Optional[HybridConfig] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.config = config or HybridConfig()
        self.verbose = verbose

        self.generator = SteeringGenerator(model, self.config.steering)
        self.refiner = TRMRefiner(
            model,
            self.config.trm,
            self.config.steering if self.config.use_steering_in_trm else None,
        )
        self.checker = ValidityChecker()

    def refine(self, prompt: str) -> RefinementResult:
        """
        Generate and refine a statechart.

        Args:
            prompt: Generation prompt

        Returns:
            RefinementResult with refined chart
        """
        all_steps = []

        # Phase 1: Steered generation
        start = time.time()
        output = self.generator.generate(prompt)
        gen_duration = time.time() - start

        try:
            initial_chart = json.loads(output)
        except json.JSONDecodeError:
            initial_chart = {"root_state": {"label": "__root__", "type": 2, "children": []}}

        initial_scores = self.checker.check(initial_chart)

        if self.verbose:
            print(f"Initial validity: {initial_scores['total']:.1%}")

        all_steps.append(RefinementStep(
            phase=RefinementPhase.STEERED_GENERATION,
            iteration=0,
            validity_before=0.0,
            validity_after=initial_scores["total"],
            changes_made=["initial_generation"],
            duration=gen_duration,
        ))

        # Phase 2: TRM refinement
        if initial_scores["total"] < self.config.trm.early_stop_threshold:
            refined_chart, trm_steps = self.refiner.refine(initial_chart, prompt)
            all_steps.extend(trm_steps)

            if self.verbose:
                for step in trm_steps:
                    print(f"  {step.phase.value}: {step.validity_before:.1%} -> {step.validity_after:.1%}")
        else:
            refined_chart = initial_chart

        final_scores = self.checker.check(refined_chart)

        if self.verbose:
            print(f"Final validity: {final_scores['total']:.1%}")

        success = final_scores["total"] >= 0.90

        return RefinementResult(
            original_chart=initial_chart,
            refined_chart=refined_chart,
            initial_validity=initial_scores["total"],
            final_validity=final_scores["total"],
            steps=all_steps,
            total_iterations=len(all_steps),
            success=success,
        )


# =============================================================================
# Convenience functions
# =============================================================================

def create_hybrid_refiner(
    model: Optional[Any] = None,
    steering_strength: float = 1.0,
    max_iterations: int = 5,
) -> HybridRefiner:
    """Create a hybrid refiner with custom settings."""
    config = HybridConfig(
        steering=SteeringConfig(strength=steering_strength),
        trm=TRMConfig(max_iterations=max_iterations),
    )
    return HybridRefiner(model, config)


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate hybrid refinement."""
    print("=" * 60)
    print("Hybrid Refiner: TRM + Steering")
    print("=" * 60)

    refiner = HybridRefiner(model=None, verbose=True)

    prompt = "Generate a login state machine:"
    print(f"\nPrompt: {prompt}\n")

    result = refiner.refine(prompt)

    print(f"\nResult:")
    print(f"  Initial: {result.initial_validity:.1%}")
    print(f"  Final: {result.final_validity:.1%}")
    print(f"  Improvement: {result.total_improvement:+.1%}")
    print(f"  Iterations: {result.total_iterations}")
    print(f"  Success: {result.success}")

    return result


if __name__ == "__main__":
    demo()
