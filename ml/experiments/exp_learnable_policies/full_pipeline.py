"""
Full Statechart Extraction Pipeline

Combines three approaches:
1. SAE State Discovery - Find states from neural activations
2. LLM Guard Proposals - Generate candidate guards from natural language
3. Evolutionary Refinement - Optimize guards against test data

This is the most complete pipeline for automatic statechart extraction.
"""

import mlx.core as mx
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import defaultdict
import random
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp_sae_statechart.sae_state_module import TopKSAE, SAEConfig
from exp_guard_synthesis.guard_synthesizer import (
    GuardSynthesizer, GuardGenome, Expr, Var, Const, BinOp, UnaryOp, ExprType
)


# =============================================================================
# STATECHART PROTO OUTPUT
# =============================================================================

def to_sc_proto_json(states: List[Dict], transitions: List[Dict]) -> Dict:
    """
    Convert discovered statechart to sc proto JSON format.

    This format is compatible with the sc tool and can be validated/visualized.
    """
    # Build state children
    children = []
    for i, state in enumerate(states):
        children.append({
            "label": state.get("label", f"S{i}"),
            "type": 1,  # STATE_TYPE_BASIC
            "is_initial": i == 0,
        })

    # Build transitions
    proto_transitions = []
    for trans in transitions:
        proto_transitions.append({
            "from": [trans.get("from_label", f"S{trans['from']}")],
            "to": [trans.get("to_label", f"S{trans['to']}")],
            "event": trans.get("event", ""),
            "guard": {
                "expression": trans.get("guard", "true")
            } if trans.get("guard") else None,
        })

    return {
        "root_state": {
            "label": "__root__",
            "type": 2,  # STATE_TYPE_OR
            "children": children,
        },
        "transitions": proto_transitions,
    }


# =============================================================================
# SIMPLIFIED STATE DISCOVERY (without full SAE)
# =============================================================================

@dataclass
class DiscoveredState:
    """A state discovered from feature patterns."""
    id: int
    features: Set[int]
    frequency: int = 0
    label: str = ""
    contexts: List[Dict] = field(default_factory=list)

    def __hash__(self):
        return self.id


class SimpleStateDiscoverer:
    """
    Discover states from context features using simple clustering.

    This is a simplified version that works directly on context variables
    rather than requiring SAE encoding.
    """

    def __init__(self, feature_names: List[str], n_bins: int = 3):
        self.feature_names = feature_names
        self.n_bins = n_bins
        self.states: Dict[tuple, DiscoveredState] = {}
        self.state_counter = 0

    def discretize_context(self, context: Dict) -> tuple:
        """Convert continuous context to discrete state key."""
        key = []
        for name in self.feature_names:
            val = context.get(name, 0)
            if isinstance(val, bool):
                key.append(1 if val else 0)
            else:
                # Discretize to bins
                bin_idx = min(self.n_bins - 1, max(0, int(val * self.n_bins / 100)))
                key.append(bin_idx)
        return tuple(key)

    def get_or_create_state(self, context: Dict) -> DiscoveredState:
        """Get existing state or create new one."""
        key = self.discretize_context(context)

        if key not in self.states:
            state = DiscoveredState(
                id=self.state_counter,
                features=set(range(len(key))),  # All features define state
                label=f"S{self.state_counter}",
            )
            self.states[key] = state
            self.state_counter += 1

        state = self.states[key]
        state.frequency += 1
        state.contexts.append(context)
        return state


# =============================================================================
# GUARD SYNTHESIS WITH LLM SEEDING
# =============================================================================

class LLMSeededGuardSynthesizer:
    """
    Guard synthesis that seeds evolution with LLM-style templates.
    """

    def __init__(self, variables: List[str]):
        self.variables = variables
        self.constants = [0.0, 0.5, 1.0, 10.0, 30.0, 50.0, 100.0, True, False]
        self.synthesizer = GuardSynthesizer(
            variables=variables,
            constants=self.constants,
            max_depth=3
        )

    def generate_template_candidates(self, from_state: DiscoveredState,
                                    to_state: DiscoveredState) -> List[GuardGenome]:
        """Generate template-based candidates based on state differences."""
        candidates = []

        # Analyze what's different between states
        from_ctx = from_state.contexts[-1] if from_state.contexts else {}
        to_ctx = to_state.contexts[-1] if to_state.contexts else {}

        # Generate candidates based on changing variables
        for var in self.variables:
            from_val = from_ctx.get(var, 0)
            to_val = to_ctx.get(var, 0)

            if from_val != to_val:
                # This variable changed - create guards around it

                # Simple threshold guards
                if isinstance(to_val, (int, float)):
                    # var > threshold
                    expr = BinOp(
                        op='>',
                        left=Var(var, ExprType.INT),
                        right=Const(from_val, ExprType.INT),
                        expr_type=ExprType.BOOL
                    )
                    candidates.append(GuardGenome(expr=expr))

                    # var < threshold
                    expr = BinOp(
                        op='<',
                        left=Var(var, ExprType.INT),
                        right=Const(to_val, ExprType.INT),
                        expr_type=ExprType.BOOL
                    )
                    candidates.append(GuardGenome(expr=expr))

                if isinstance(to_val, bool):
                    # Boolean changed - guard on the new value
                    if to_val:
                        expr = Var(var, ExprType.BOOL)
                    else:
                        expr = UnaryOp(op='not', operand=Var(var, ExprType.BOOL),
                                      expr_type=ExprType.BOOL)
                    candidates.append(GuardGenome(expr=expr))

        return candidates

    def synthesize_guard(self, positive: List[Dict], negative: List[Dict],
                        from_state: DiscoveredState = None,
                        to_state: DiscoveredState = None,
                        n_generations: int = 20) -> GuardGenome:
        """Synthesize guard using template seeding + evolution."""

        # Start with template candidates if we have state info
        initial_pop = []
        if from_state and to_state:
            initial_pop = self.generate_template_candidates(from_state, to_state)

        # Add random candidates to fill population
        while len(initial_pop) < 30:
            initial_pop.append(GuardGenome(expr=self.synthesizer.random_expr()))

        # Run evolution
        best = self.synthesizer.evolve(
            positive_examples=positive,
            negative_examples=negative,
            population_size=30,
            n_generations=n_generations,
            verbose=False
        )

        return best


# =============================================================================
# FULL PIPELINE
# =============================================================================

@dataclass
class ExtractedStatechart:
    """Complete extracted statechart."""
    states: List[Dict]
    transitions: List[Dict]
    proto_json: Dict

    def to_mermaid(self) -> str:
        """Generate Mermaid diagram."""
        lines = ["stateDiagram-v2"]

        # Add states
        for state in self.states:
            label = state.get("label", "S?")
            freq = state.get("frequency", 0)
            lines.append(f"    {label}: {label} (n={freq})")

        # Add transitions
        for trans in self.transitions:
            from_label = trans.get("from_label", f"S{trans['from']}")
            to_label = trans.get("to_label", f"S{trans['to']}")
            guard = trans.get("guard", "")
            if guard and guard != "true":
                guard_short = guard[:30] + "..." if len(guard) > 30 else guard
                lines.append(f"    {from_label} --> {to_label}: [{guard_short}]")
            else:
                lines.append(f"    {from_label} --> {to_label}")

        return "\n".join(lines)


class FullStatechartPipeline:
    """
    Complete pipeline for statechart extraction.

    1. Process observations to discover states
    2. Track transitions between states
    3. Synthesize guards for each transition
    4. Output complete statechart
    """

    def __init__(self, feature_names: List[str], n_bins: int = 5):
        self.state_discoverer = SimpleStateDiscoverer(feature_names, n_bins)
        self.guard_synthesizer = LLMSeededGuardSynthesizer(feature_names)

        self.current_state: Optional[DiscoveredState] = None
        self.transitions: Dict[Tuple[int, int], List[Tuple[Dict, Dict]]] = defaultdict(list)
        self.observation_count = 0

    def process(self, context: Dict) -> DiscoveredState:
        """Process an observation."""
        self.observation_count += 1

        state = self.state_discoverer.get_or_create_state(context)

        # Track transition
        if self.current_state is not None and self.current_state.id != state.id:
            key = (self.current_state.id, state.id)
            self.transitions[key].append((
                self.current_state.contexts[-1] if self.current_state.contexts else {},
                context
            ))

        self.current_state = state
        return state

    def extract(self, min_frequency: int = 3,
                min_transitions: int = 2) -> ExtractedStatechart:
        """Extract complete statechart."""

        # Filter states by frequency
        valid_states = {
            k: v for k, v in self.state_discoverer.states.items()
            if v.frequency >= min_frequency
        }

        # Build state list
        states = []
        state_id_map = {}  # old_id -> new_idx

        for idx, (key, state) in enumerate(sorted(valid_states.items(),
                                                   key=lambda x: -x[1].frequency)):
            state_id_map[state.id] = idx
            states.append({
                "id": idx,
                "label": f"S{idx}",
                "frequency": state.frequency,
                "original_id": state.id,
            })

        # Build transitions with guards
        transitions = []

        for (from_id, to_id), examples in self.transitions.items():
            # Skip if either state was filtered out
            if from_id not in state_id_map or to_id not in state_id_map:
                continue

            # Skip if not enough transitions
            if len(examples) < min_transitions:
                continue

            # Get states
            from_state = None
            to_state = None
            for state in valid_states.values():
                if state.id == from_id:
                    from_state = state
                if state.id == to_id:
                    to_state = state

            # Build positive examples (transitions that fired)
            positive = [ex[1] for ex in examples]  # Use destination context

            # Build negative examples (other transitions from same source that didn't fire)
            negative = []
            for (other_from, other_to), other_examples in self.transitions.items():
                if other_from == from_id and other_to != to_id:
                    negative.extend([ex[1] for ex in other_examples])

            # Synthesize guard
            guard = self.guard_synthesizer.synthesize_guard(
                positive=positive,
                negative=negative if negative else [],
                from_state=from_state,
                to_state=to_state,
                n_generations=15
            )

            guard_str = guard.expr.to_string() if guard.expr else "true"

            transitions.append({
                "from": state_id_map[from_id],
                "to": state_id_map[to_id],
                "from_label": f"S{state_id_map[from_id]}",
                "to_label": f"S{state_id_map[to_id]}",
                "guard": guard_str,
                "fitness": guard.fitness,
                "count": len(examples),
            })

        # Sort transitions by count
        transitions.sort(key=lambda t: -t["count"])

        # Generate proto JSON
        proto_json = to_sc_proto_json(states, transitions)

        return ExtractedStatechart(
            states=states,
            transitions=transitions,
            proto_json=proto_json
        )


# =============================================================================
# DEMO: GAME SIMULATION
# =============================================================================

class GameSimulator:
    """Simulate a game with clear state transitions for testing."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.health = 100.0
        self.score = 0.0
        self.level = 1
        self.is_powered_up = False
        self.is_invincible = False
        self.combo = 0

    def step(self, action: int) -> Dict:
        """
        Actions:
        0: idle
        1: attack (+score, maybe combo)
        2: take_damage (-health)
        3: heal (+health)
        4: power_up (toggle)
        5: level_up (+level)
        """

        if action == 1:  # attack
            self.score += 10
            if random.random() < 0.3:
                self.combo += 1
            else:
                self.combo = 0

        elif action == 2:  # take damage
            if not self.is_invincible:
                self.health = max(0, self.health - random.uniform(10, 30))

        elif action == 3:  # heal
            self.health = min(100, self.health + random.uniform(10, 20))

        elif action == 4:  # power up
            self.is_powered_up = not self.is_powered_up
            if self.is_powered_up:
                self.is_invincible = random.random() < 0.3

        elif action == 5:  # level up
            if self.score >= self.level * 50:
                self.level += 1
                self.score = 0

        return self.get_context()

    def get_context(self) -> Dict:
        return {
            "health": self.health,
            "score": self.score,
            "level": float(self.level),
            "is_powered_up": self.is_powered_up,
            "is_invincible": self.is_invincible,
            "combo": float(self.combo),
            "low_health": self.health < 30,
            "high_score": self.score > 50,
        }


def demo_full_pipeline():
    """Demonstrate the full statechart extraction pipeline."""

    print("=" * 70)
    print("FULL STATECHART EXTRACTION PIPELINE DEMO")
    print("=" * 70)
    print()
    print("Components:")
    print("  1. State Discovery (discretized context)")
    print("  2. LLM-Seeded Guard Synthesis")
    print("  3. Evolutionary Refinement")
    print("  4. SC Proto JSON Output")
    print()

    # Create pipeline
    feature_names = ["health", "score", "level", "is_powered_up",
                    "is_invincible", "combo", "low_health", "high_score"]
    pipeline = FullStatechartPipeline(feature_names, n_bins=3)

    # Simulate game
    print("Simulating game...")
    game = GameSimulator()

    n_episodes = 30
    steps_per_episode = 40

    for ep in range(n_episodes):
        game.reset()
        ctx = game.get_context()
        pipeline.process(ctx)

        for _ in range(steps_per_episode):
            # Biased action selection to create interesting patterns
            if game.health < 30:
                action = 3  # heal when low
            elif game.score > 50:
                action = random.choices([5, 1], weights=[0.7, 0.3])[0]  # try level up
            elif not game.is_powered_up and random.random() < 0.2:
                action = 4  # sometimes power up
            else:
                action = random.choices([1, 2, 0], weights=[0.5, 0.3, 0.2])[0]

            ctx = game.step(action)
            pipeline.process(ctx)

    print(f"  Episodes: {n_episodes}")
    print(f"  Total observations: {pipeline.observation_count}")
    print(f"  Unique states found: {len(pipeline.state_discoverer.states)}")
    print(f"  Transition types: {len(pipeline.transitions)}")
    print()

    # Extract statechart
    print("Extracting statechart...")
    chart = pipeline.extract(min_frequency=5, min_transitions=3)

    print()
    print("=" * 70)
    print("EXTRACTED STATECHART")
    print("=" * 70)
    print()

    print(f"States: {len(chart.states)}")
    for state in chart.states[:8]:
        print(f"  {state['label']}: frequency={state['frequency']}")

    print()
    print(f"Transitions: {len(chart.transitions)}")
    for trans in chart.transitions[:10]:
        guard_preview = trans['guard'][:40] + "..." if len(trans['guard']) > 40 else trans['guard']
        print(f"  {trans['from_label']} → {trans['to_label']}: "
              f"[{guard_preview}] (n={trans['count']}, fit={trans['fitness']:.2f})")

    print()
    print("=" * 70)
    print("MERMAID DIAGRAM")
    print("=" * 70)
    print(chart.to_mermaid())

    print()
    print("=" * 70)
    print("SC PROTO JSON (excerpt)")
    print("=" * 70)
    proto = chart.proto_json
    print(f"Root: {proto['root_state']['label']}")
    print(f"States: {len(proto['root_state']['children'])}")
    print(f"Transitions: {len(proto['transitions'])}")
    print()
    print("First transition:")
    if proto['transitions']:
        print(json.dumps(proto['transitions'][0], indent=2))

    return chart


if __name__ == "__main__":
    demo_full_pipeline()
