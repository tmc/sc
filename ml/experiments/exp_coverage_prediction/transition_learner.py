"""
Transition Learning for Statechart Discovery

Learn transitions, guards, and actions from execution traces.

Components:
1. Transition Discovery - Find which states transition to which
2. Guard Synthesis - Learn boolean conditions for transitions
3. Action Inference - Learn state modifications on transitions
4. Event Detection - Identify triggering events

Integrates with exp_guard_synthesis for guard expression learning.
"""

import ast
import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from collections import defaultdict
from enum import Enum, auto
import random

from .state_discovery import DiscoveredState, StateType


@dataclass
class TransitionObservation:
    """An observed transition between states."""
    from_state: str
    to_state: str
    input_value: Any
    input_repr: str
    context_before: Dict[str, Any]
    context_after: Dict[str, Any]
    line_before: int
    line_after: int


@dataclass
class LearnedGuard:
    """A learned guard condition."""
    expression: str              # Boolean expression as string
    variables: List[str]         # Variables used in guard
    true_examples: int = 0       # Examples where guard was true
    false_examples: int = 0      # Examples where guard was false
    precision: float = 0.0       # TP / (TP + FP)
    recall: float = 0.0          # TP / (TP + FN)
    f1: float = 0.0

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate guard in given context."""
        try:
            return eval(self.expression, {"__builtins__": {}}, context)
        except:
            return True  # Default to true if can't evaluate


@dataclass
class LearnedAction:
    """A learned action (state modification)."""
    target: str                  # Variable being modified
    operation: str               # 'set', 'increment', 'append', etc.
    expression: Optional[str]    # Value expression
    examples: int = 0


@dataclass
class LearnedTransition:
    """A complete learned transition."""
    id: str
    from_state: str
    to_state: str
    event: Optional[str] = None
    guard: Optional[LearnedGuard] = None
    actions: List[LearnedAction] = field(default_factory=list)
    observation_count: int = 0
    confidence: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'from': self.from_state,
            'to': self.to_state,
            'event': self.event,
            'guard': self.guard.expression if self.guard else None,
            'actions': [{'target': a.target, 'op': a.operation} for a in self.actions],
            'count': self.observation_count,
            'confidence': self.confidence,
        }


class GuardSynthesizer:
    """
    Synthesize guard expressions from positive/negative examples.

    Uses a template-based approach with evolutionary refinement.
    """

    # Guard templates in order of complexity
    TEMPLATES = [
        # Simple comparisons
        "{var} > {const}",
        "{var} < {const}",
        "{var} == {const}",
        "{var} != {const}",
        "{var} >= {const}",
        "{var} <= {const}",
        # Boolean checks
        "{var}",
        "not {var}",
        # Collection checks
        "len({var}) > {const}",
        "len({var}) == {const}",
        "{var} in {collection}",
        "{const} in {var}",
        # Compound
        "{var1} > {var2}",
        "{var1} == {var2}",
        "{var1} and {var2}",
        "{var1} or {var2}",
        "{var} > {const1} and {var} < {const2}",
    ]

    def __init__(self, variables: List[str], constants: List[Any] = None):
        self.variables = variables
        self.constants = constants or [0, 1, -1, True, False, None, "", []]

    def synthesize(
        self,
        positive_examples: List[Dict[str, Any]],
        negative_examples: List[Dict[str, Any]],
        max_attempts: int = 100,
    ) -> Optional[LearnedGuard]:
        """
        Synthesize a guard that accepts positive and rejects negative examples.
        """
        if not positive_examples:
            return None

        # Extract common variables
        all_vars = set()
        for ex in positive_examples + negative_examples:
            all_vars.update(ex.keys())
        vars_to_use = [v for v in self.variables if v in all_vars]

        if not vars_to_use:
            vars_to_use = list(all_vars)[:5]

        # Extract constants from examples (only hashable values)
        example_constants = set()
        for c in self.constants:
            try:
                example_constants.add(c)
            except TypeError:
                pass  # Skip unhashable values
        for ex in positive_examples:
            for v in ex.values():
                if isinstance(v, (int, float, bool, str)) and v not in example_constants:
                    example_constants.add(v)

        best_guard = None
        best_f1 = 0.0

        # Try templates
        for _ in range(max_attempts):
            template = random.choice(self.TEMPLATES)
            guard_expr = self._instantiate_template(
                template, vars_to_use, list(example_constants)
            )

            if guard_expr:
                guard = self._evaluate_guard(
                    guard_expr, positive_examples, negative_examples
                )
                if guard and guard.f1 > best_f1:
                    best_f1 = guard.f1
                    best_guard = guard

                if best_f1 >= 0.95:
                    break

        return best_guard

    def _instantiate_template(
        self,
        template: str,
        variables: List[str],
        constants: List[Any],
    ) -> Optional[str]:
        """Fill in a template with concrete values."""
        try:
            result = template

            # Fill in variables
            if "{var}" in result:
                result = result.replace("{var}", random.choice(variables))
            if "{var1}" in result:
                result = result.replace("{var1}", random.choice(variables))
            if "{var2}" in result:
                v2 = random.choice(variables)
                result = result.replace("{var2}", v2)

            # Fill in constants
            if "{const}" in result:
                c = random.choice([c for c in constants if isinstance(c, (int, float))] or [0])
                result = result.replace("{const}", repr(c))
            if "{const1}" in result:
                c = random.choice([c for c in constants if isinstance(c, (int, float))] or [0])
                result = result.replace("{const1}", repr(c))
            if "{const2}" in result:
                c = random.choice([c for c in constants if isinstance(c, (int, float))] or [0])
                result = result.replace("{const2}", repr(c))
            if "{collection}" in result:
                c = random.choice([c for c in constants if isinstance(c, (list, tuple, set))] or [[]])
                result = result.replace("{collection}", repr(c))

            # Validate syntax
            ast.parse(result, mode='eval')
            return result

        except:
            return None

    def _evaluate_guard(
        self,
        expression: str,
        positive: List[Dict],
        negative: List[Dict],
    ) -> Optional[LearnedGuard]:
        """Evaluate a guard expression on examples."""
        tp, fp, tn, fn = 0, 0, 0, 0

        # Extract variable names from expression
        try:
            tree = ast.parse(expression, mode='eval')
            variables = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
        except:
            variables = []

        for ctx in positive:
            try:
                result = eval(expression, {"__builtins__": {}, "len": len}, ctx)
                if result:
                    tp += 1
                else:
                    fn += 1
            except:
                fn += 1

        for ctx in negative:
            try:
                result = eval(expression, {"__builtins__": {}, "len": len}, ctx)
                if result:
                    fp += 1
                else:
                    tn += 1
            except:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return LearnedGuard(
            expression=expression,
            variables=variables,
            true_examples=tp,
            false_examples=tn,
            precision=precision,
            recall=recall,
            f1=f1,
        )


class ActionInferrer:
    """
    Infer actions from before/after state pairs.

    Detects what changed between states and generates action expressions.
    """

    def infer_actions(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> List[LearnedAction]:
        """Infer actions from state diff."""
        actions = []

        # Find changed variables
        all_vars = set(before.keys()) | set(after.keys())

        for var in all_vars:
            before_val = before.get(var)
            after_val = after.get(var)

            if before_val != after_val:
                action = self._infer_single_action(var, before_val, after_val)
                if action:
                    actions.append(action)

        return actions

    def _infer_single_action(
        self,
        var: str,
        before: Any,
        after: Any,
    ) -> Optional[LearnedAction]:
        """Infer action for a single variable change."""

        # Boolean toggle
        if isinstance(before, bool) and isinstance(after, bool):
            if after == (not before):
                return LearnedAction(var, 'toggle', None)
            return LearnedAction(var, 'set', repr(after))

        # Integer operations
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            diff = after - before
            if diff > 0:
                if diff == 1:
                    return LearnedAction(var, 'increment', None)
                return LearnedAction(var, 'increment', repr(diff))
            elif diff < 0:
                if diff == -1:
                    return LearnedAction(var, 'decrement', None)
                return LearnedAction(var, 'decrement', repr(-diff))
            return None  # No change

        # List operations
        if isinstance(before, list) and isinstance(after, list):
            if len(after) > len(before) and after[:len(before)] == before:
                # Items appended
                new_items = after[len(before):]
                if len(new_items) == 1:
                    return LearnedAction(var, 'append', repr(new_items[0]))
                return LearnedAction(var, 'extend', repr(new_items))
            elif len(after) < len(before) and before[:len(after)] == after:
                # Items removed
                return LearnedAction(var, 'truncate', repr(len(after)))

        # Generic set
        return LearnedAction(var, 'set', repr(after))


class TransitionLearner:
    """
    Learn complete transitions from execution observations.

    Combines:
    - Transition discovery (which states connect)
    - Guard synthesis (conditions for transitions)
    - Action inference (what changes on transition)
    """

    def __init__(self):
        self.observations: List[TransitionObservation] = []
        self.transitions: Dict[Tuple[str, str], LearnedTransition] = {}
        self.guard_synth = None  # Created when we know variables
        self.action_inferrer = ActionInferrer()

    def add_observation(self, obs: TransitionObservation):
        """Add a transition observation."""
        self.observations.append(obs)

    def add_state_sequence(
        self,
        states: List[str],
        input_value: Any,
        input_repr: str,
        contexts: Optional[List[Dict[str, Any]]] = None,
    ):
        """Add a sequence of state visits."""
        if not contexts:
            contexts = [{}] * len(states)

        for i in range(len(states) - 1):
            obs = TransitionObservation(
                from_state=states[i],
                to_state=states[i + 1],
                input_value=input_value,
                input_repr=input_repr,
                context_before=contexts[i],
                context_after=contexts[i + 1],
                line_before=0,
                line_after=0,
            )
            self.add_observation(obs)

    def learn_transitions(
        self,
        min_observations: int = 2,
        synthesize_guards: bool = True,
    ) -> Dict[str, LearnedTransition]:
        """
        Learn transitions from accumulated observations.
        """
        # Group observations by (from, to) pair
        grouped: Dict[Tuple[str, str], List[TransitionObservation]] = defaultdict(list)
        for obs in self.observations:
            key = (obs.from_state, obs.to_state)
            grouped[key].append(obs)

        # Collect all variables for guard synthesis
        all_vars = set()
        for obs in self.observations:
            all_vars.update(obs.context_before.keys())
        self.guard_synth = GuardSynthesizer(list(all_vars))

        # Learn each transition
        for (from_state, to_state), obs_list in grouped.items():
            if len(obs_list) < min_observations:
                continue

            trans_id = f"t_{from_state}_{to_state}"

            # Learn guard if there are both taken and not-taken cases
            guard = None
            if synthesize_guards:
                # Positive examples: contexts where transition was taken
                positive = [obs.context_before for obs in obs_list]

                # Negative examples: need observations where from_state didn't go to to_state
                negative = []
                for other_key, other_obs in grouped.items():
                    if other_key[0] == from_state and other_key[1] != to_state:
                        negative.extend(obs.context_before for obs in other_obs)

                if positive and negative:
                    guard = self.guard_synth.synthesize(positive, negative)

            # Infer actions from first observation
            actions = []
            if obs_list[0].context_before and obs_list[0].context_after:
                actions = self.action_inferrer.infer_actions(
                    obs_list[0].context_before,
                    obs_list[0].context_after,
                )

            # Create learned transition
            transition = LearnedTransition(
                id=trans_id,
                from_state=from_state,
                to_state=to_state,
                event=None,  # Could infer from input patterns
                guard=guard,
                actions=actions,
                observation_count=len(obs_list),
                confidence=min(1.0, len(obs_list) / 10.0),
            )

            self.transitions[(from_state, to_state)] = transition

        return {t.id: t for t in self.transitions.values()}

    def to_statechart_transitions(self) -> List[Dict]:
        """Convert learned transitions to statechart proto format."""
        result = []
        for trans in self.transitions.values():
            t = {
                'from': [trans.from_state],
                'to': [trans.to_state],
            }
            if trans.event:
                t['event'] = trans.event
            if trans.guard:
                t['guard'] = {'expression': trans.guard.expression}
            if trans.actions:
                t['actions'] = [
                    {'target': a.target, 'operation': a.operation, 'expression': a.expression}
                    for a in trans.actions
                ]
            result.append(t)
        return result


class EventDetector:
    """
    Detect events that trigger transitions.

    Events can be:
    - Input values/patterns
    - Variable changes
    - External signals
    """

    def __init__(self):
        self.event_patterns: Dict[str, List[Any]] = defaultdict(list)

    def analyze_inputs(
        self,
        observations: List[TransitionObservation],
    ) -> Dict[Tuple[str, str], str]:
        """
        Analyze inputs to detect event patterns for transitions.

        Returns mapping of (from, to) -> event name
        """
        # Group by transition
        grouped: Dict[Tuple[str, str], List[Any]] = defaultdict(list)
        for obs in observations:
            key = (obs.from_state, obs.to_state)
            grouped[key].append(obs.input_value)

        events = {}
        for (from_s, to_s), inputs in grouped.items():
            event = self._detect_event_pattern(inputs)
            if event:
                events[(from_s, to_s)] = event

        return events

    def _detect_event_pattern(self, inputs: List[Any]) -> Optional[str]:
        """Detect a common pattern in inputs."""
        if not inputs:
            return None

        # Check for consistent type
        types = set(type(i).__name__ for i in inputs)
        if len(types) == 1:
            t = types.pop()

            # Check for consistent value
            if len(set(repr(i) for i in inputs)) == 1:
                return f"INPUT_{inputs[0]}"

            # Check for value range
            if t in ('int', 'float'):
                min_val = min(inputs)
                max_val = max(inputs)
                if min_val > 0:
                    return "POSITIVE_INPUT"
                elif max_val < 0:
                    return "NEGATIVE_INPUT"
                elif min_val <= 0 <= max_val:
                    return "ANY_INPUT"

            # Check for empty/non-empty
            if t in ('list', 'str'):
                if all(len(i) == 0 for i in inputs):
                    return "EMPTY_INPUT"
                elif all(len(i) > 0 for i in inputs):
                    return "NON_EMPTY_INPUT"

        return None


def demo():
    """Demonstrate transition learning."""
    print("=" * 60)
    print("TRANSITION LEARNER DEMO")
    print("=" * 60)

    learner = TransitionLearner()

    # Simulate observations from a simple program
    # Program: if x > 0: return "positive" else: return "non-positive"

    # Observations where x > 0 (goes to positive branch)
    for x in [1, 5, 10, 100]:
        learner.add_observation(TransitionObservation(
            from_state="entry",
            to_state="branch",
            input_value=x,
            input_repr=str(x),
            context_before={'x': x},
            context_after={'x': x},
            line_before=1,
            line_after=2,
        ))
        learner.add_observation(TransitionObservation(
            from_state="branch",
            to_state="positive_return",
            input_value=x,
            input_repr=str(x),
            context_before={'x': x},
            context_after={'x': x, 'result': 'positive'},
            line_before=2,
            line_after=3,
        ))

    # Observations where x <= 0 (goes to negative branch)
    for x in [-5, -1, 0]:
        learner.add_observation(TransitionObservation(
            from_state="entry",
            to_state="branch",
            input_value=x,
            input_repr=str(x),
            context_before={'x': x},
            context_after={'x': x},
            line_before=1,
            line_after=2,
        ))
        learner.add_observation(TransitionObservation(
            from_state="branch",
            to_state="negative_return",
            input_value=x,
            input_repr=str(x),
            context_before={'x': x},
            context_after={'x': x, 'result': 'non-positive'},
            line_before=2,
            line_after=4,
        ))

    print(f"\nAdded {len(learner.observations)} observations")

    # Learn transitions
    transitions = learner.learn_transitions()

    print(f"\nLearned {len(transitions)} transitions:")
    for trans in transitions.values():
        print(f"\n  {trans.from_state} -> {trans.to_state}")
        print(f"    Observations: {trans.observation_count}")
        if trans.guard:
            print(f"    Guard: {trans.guard.expression} (F1={trans.guard.f1:.2f})")
        if trans.actions:
            print(f"    Actions: {[f'{a.target}={a.operation}' for a in trans.actions]}")

    # Detect events
    detector = EventDetector()
    events = detector.analyze_inputs(learner.observations)
    print(f"\nDetected events: {events}")

    return learner


if __name__ == "__main__":
    demo()
