"""
Causal Learner: Learn causal relationships between actions and guards.

This module goes beyond correlation to learn CAUSAL relationships:
- Action A CAUSES guard G to become enabled (not just correlated with it)
- The causal mechanism goes through context variable X

We use interventional reasoning: if we intervene on action A, does guard G change?
This distinguishes causation from mere correlation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Any, Optional, Callable
from collections import defaultdict
import random
import copy


@dataclass
class CausalEdge:
    """
    A causal edge from action to guard, mediated by variables.

    A -> X -> G means:
      Action A modifies variable X
      Guard G depends on variable X
      Therefore A causally affects G through X
    """
    action: str
    guard: str
    mediating_variables: Set[str]
    strength: float  # Causal strength (0-1)
    confidence: float  # Confidence in this edge (0-1)

    # Evidence counts
    interventions: int = 0  # Number of times we intervened
    expected_changes: int = 0  # Times we expected guard to change
    actual_changes: int = 0  # Times guard actually changed

    @property
    def accuracy(self) -> float:
        """Accuracy of causal predictions."""
        if self.interventions == 0:
            return 0.0
        return self.actual_changes / self.interventions

    def __hash__(self):
        return hash((self.action, self.guard))

    def __eq__(self, other):
        if not isinstance(other, CausalEdge):
            return False
        return self.action == other.action and self.guard == other.guard


@dataclass
class CausalGraph:
    """
    Graph of causal relationships between actions and guards.
    """
    edges: Dict[Tuple[str, str], CausalEdge] = field(default_factory=dict)

    # Variable-level causality
    action_to_vars: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    vars_to_guards: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_edge(self, edge: CausalEdge):
        """Add a causal edge."""
        key = (edge.action, edge.guard)
        self.edges[key] = edge

        for var in edge.mediating_variables:
            self.action_to_vars[edge.action].add(var)
            self.vars_to_guards[var].add(edge.guard)

    def get_edge(self, action: str, guard: str) -> Optional[CausalEdge]:
        """Get causal edge if exists."""
        return self.edges.get((action, guard))

    def get_causal_strength(self, action: str, guard: str) -> float:
        """Get causal strength between action and guard."""
        edge = self.get_edge(action, guard)
        return edge.strength if edge else 0.0

    def get_all_effects(self, action: str) -> List[CausalEdge]:
        """Get all causal effects of an action."""
        return [e for (a, _), e in self.edges.items() if a == action]

    def get_all_causes(self, guard: str) -> List[CausalEdge]:
        """Get all actions that causally affect a guard."""
        return [e for (_, g), e in self.edges.items() if g == guard]

    def print_graph(self):
        """Print the causal graph."""
        print("\n" + "=" * 60)
        print("CAUSAL GRAPH")
        print("=" * 60)

        # Group by action
        by_action = defaultdict(list)
        for (action, guard), edge in self.edges.items():
            by_action[action].append((guard, edge))

        for action, effects in sorted(by_action.items()):
            print(f"\n{action}:")
            for guard, edge in sorted(effects, key=lambda x: -x[1].strength):
                vars_str = ", ".join(edge.mediating_variables)
                print(f"  -> {guard} (strength={edge.strength:.3f}, via {vars_str})")


class CausalLearner:
    """
    Learn causal relationships between actions and guards.

    Uses three approaches:
    1. Observational: Track co-occurrences
    2. Interventional: Perform actions and observe effects
    3. Counterfactual: Ask "what if we hadn't done action A?"
    """

    def __init__(
        self,
        actions: List[str],
        guards: List[str],
        variables: List[str],
        action_executors: Dict[str, Callable[[Dict], Dict]] = None,
        guard_evaluators: Dict[str, Callable[[Dict], bool]] = None,
    ):
        self.actions = actions
        self.guards = guards
        self.variables = variables
        self.action_executors = action_executors or {}
        self.guard_evaluators = guard_evaluators or {}

        self.causal_graph = CausalGraph()

        # Observation history
        self.observations: List[Dict] = []

        # Statistics
        self.action_var_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.var_guard_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.action_guard_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.action_total: Dict[str, int] = defaultdict(int)

    def register_action(self, name: str, executor: Callable[[Dict], Dict]):
        """Register an action executor."""
        self.action_executors[name] = executor

    def register_guard(self, name: str, evaluator: Callable[[Dict], bool]):
        """Register a guard evaluator."""
        self.guard_evaluators[name] = evaluator

    def observe(
        self,
        action: str,
        context_before: Dict[str, Any],
        context_after: Dict[str, Any],
    ):
        """
        Record an observation of action execution.
        """
        # Track which variables changed
        changed_vars = set()
        for var in set(context_before.keys()) | set(context_after.keys()):
            if context_before.get(var) != context_after.get(var):
                changed_vars.add(var)

        # Track which guards changed
        changed_guards = set()
        guards_before = {}
        guards_after = {}
        for guard, evaluator in self.guard_evaluators.items():
            try:
                before = evaluator(context_before)
                after = evaluator(context_after)
                guards_before[guard] = before
                guards_after[guard] = after
                if before != after:
                    changed_guards.add(guard)
            except Exception:
                pass

        # Update statistics
        self.action_total[action] += 1

        for var in changed_vars:
            self.action_var_counts[action][var] += 1

        for guard in changed_guards:
            self.action_guard_counts[action][guard] += 1
            for var in changed_vars:
                self.var_guard_counts[var][guard] += 1

        # Store observation
        self.observations.append({
            "action": action,
            "context_before": context_before.copy(),
            "context_after": context_after.copy(),
            "changed_vars": changed_vars,
            "changed_guards": changed_guards,
            "guards_before": guards_before,
            "guards_after": guards_after,
        })

    def intervene(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform an intervention (execute action) and observe effect.

        Returns new context after intervention.
        """
        if action not in self.action_executors:
            return context.copy()

        executor = self.action_executors[action]
        new_context = executor(context.copy())

        # Record observation
        self.observe(action, context, new_context)

        return new_context

    def learn_causal_graph(
        self,
        min_observations: int = 5,
        min_strength: float = 0.3,
    ) -> CausalGraph:
        """
        Learn causal graph from observations.

        Uses conditional probability to estimate causal strength:
        P(guard changes | action) adjusted for P(guard changes | no action)
        """
        self.causal_graph = CausalGraph()

        for action in self.actions:
            action_count = self.action_total[action]
            if action_count < min_observations:
                continue

            # What variables does action modify?
            modified_vars = set()
            for var, count in self.action_var_counts[action].items():
                if count / action_count > 0.3:  # Modified >30% of the time
                    modified_vars.add(var)

            if not modified_vars:
                continue

            # What guards are affected?
            for guard in self.guards:
                guard_change_count = self.action_guard_counts[action].get(guard, 0)
                if guard_change_count == 0:
                    continue

                # Causal strength = P(guard changes | action)
                strength = guard_change_count / action_count

                if strength < min_strength:
                    continue

                # Find mediating variables (intersection of modified vars and guard dependencies)
                mediating = set()
                for var in modified_vars:
                    if self.var_guard_counts[var].get(guard, 0) > 0:
                        mediating.add(var)

                if not mediating:
                    # Direct effect without identified mediator
                    mediating = modified_vars

                edge = CausalEdge(
                    action=action,
                    guard=guard,
                    mediating_variables=mediating,
                    strength=strength,
                    confidence=min(1.0, action_count / 20),  # More observations = more confidence
                    interventions=action_count,
                    actual_changes=guard_change_count,
                )
                self.causal_graph.add_edge(edge)

        return self.causal_graph

    def predict_guard_changes(
        self,
        action: str,
        context: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Predict which guards will change if we perform action.

        Returns dict of guard_name -> probability of change.
        """
        predictions = {}

        for guard in self.guards:
            edge = self.causal_graph.get_edge(action, guard)
            if edge:
                predictions[guard] = edge.strength
            else:
                predictions[guard] = 0.0

        return predictions

    def test_causal_hypothesis(
        self,
        action: str,
        guard: str,
        test_contexts: List[Dict[str, Any]],
    ) -> Tuple[float, float]:
        """
        Test a causal hypothesis by intervention.

        Returns (accuracy, confidence).
        """
        if action not in self.action_executors:
            return 0.0, 0.0

        correct = 0
        total = 0

        for context in test_contexts:
            # Predict
            predictions = self.predict_guard_changes(action, context)
            predicted_change = predictions.get(guard, 0.0) > 0.5

            # Intervene
            guard_before = self.guard_evaluators[guard](context)
            new_context = self.intervene(action, context)
            guard_after = self.guard_evaluators[guard](new_context)

            actual_change = guard_before != guard_after

            if predicted_change == actual_change:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0
        confidence = min(1.0, total / 10)

        return accuracy, confidence

    def discover_hidden_causes(self) -> List[Tuple[str, str, float]]:
        """
        Discover potential hidden causal relationships not yet in graph.

        Returns list of (action, guard, correlation) tuples worth investigating.
        """
        candidates = []

        for action in self.actions:
            action_count = self.action_total[action]
            if action_count < 3:
                continue

            for guard in self.guards:
                # Skip if already in graph with high strength
                edge = self.causal_graph.get_edge(action, guard)
                if edge and edge.strength > 0.5:
                    continue

                # Check correlation in observations
                guard_changes = self.action_guard_counts[action].get(guard, 0)
                correlation = guard_changes / action_count if action_count > 0 else 0.0

                if correlation > 0.2 and (edge is None or edge.strength < correlation):
                    candidates.append((action, guard, correlation))

        return sorted(candidates, key=lambda x: -x[2])

    def evolve_causal_model(
        self,
        n_iterations: int = 50,
        exploration_rate: float = 0.2,
        test_contexts: List[Dict[str, Any]] = None,
    ) -> CausalGraph:
        """
        Iteratively refine causal model through exploration and testing.

        Combines:
        1. Learning from observations
        2. Testing hypotheses through intervention
        3. Exploring new potential relationships
        """
        if test_contexts is None:
            test_contexts = []

        for iteration in range(n_iterations):
            # 1. Learn from current observations
            self.learn_causal_graph()

            # 2. Test existing hypotheses
            for (action, guard), edge in list(self.causal_graph.edges.items()):
                if test_contexts and random.random() < 0.3:
                    acc, conf = self.test_causal_hypothesis(action, guard, test_contexts[:5])
                    # Update edge based on test
                    edge.confidence = (edge.confidence + conf) / 2
                    if acc < 0.3:
                        edge.strength *= 0.9  # Reduce strength if poor accuracy

            # 3. Explore new relationships
            if random.random() < exploration_rate:
                candidates = self.discover_hidden_causes()
                if candidates:
                    action, guard, _ = candidates[0]
                    # Generate test context and intervene
                    if test_contexts:
                        context = random.choice(test_contexts)
                        self.intervene(action, context)

        # Final learning pass
        self.learn_causal_graph()

        return self.causal_graph


def demo():
    """Demonstrate causal learning."""
    print("=" * 60)
    print("CAUSAL LEARNER: Learning Action->Guard Causality")
    print("=" * 60)

    # Define a simple system: counter with threshold guards
    actions = ["increment", "decrement", "reset", "double"]
    guards = ["is_positive", "is_above_10", "is_even", "is_zero"]
    variables = ["counter"]

    # Define action executors
    def increment(ctx):
        ctx = ctx.copy()
        ctx["counter"] = ctx.get("counter", 0) + 1
        return ctx

    def decrement(ctx):
        ctx = ctx.copy()
        ctx["counter"] = ctx.get("counter", 0) - 1
        return ctx

    def reset(ctx):
        ctx = ctx.copy()
        ctx["counter"] = 0
        return ctx

    def double(ctx):
        ctx = ctx.copy()
        ctx["counter"] = ctx.get("counter", 0) * 2
        return ctx

    action_executors = {
        "increment": increment,
        "decrement": decrement,
        "reset": reset,
        "double": double,
    }

    # Define guard evaluators
    def is_positive(ctx):
        return ctx.get("counter", 0) > 0

    def is_above_10(ctx):
        return ctx.get("counter", 0) > 10

    def is_even(ctx):
        return ctx.get("counter", 0) % 2 == 0

    def is_zero(ctx):
        return ctx.get("counter", 0) == 0

    guard_evaluators = {
        "is_positive": is_positive,
        "is_above_10": is_above_10,
        "is_even": is_even,
        "is_zero": is_zero,
    }

    # Create learner
    learner = CausalLearner(
        actions=actions,
        guards=guards,
        variables=variables,
        action_executors=action_executors,
        guard_evaluators=guard_evaluators,
    )

    # Generate observations through interventions
    print("\nGenerating observations through interventions...")
    test_contexts = [{"counter": i} for i in range(-5, 20)]

    for _ in range(100):
        context = random.choice(test_contexts).copy()
        action = random.choice(actions)
        learner.intervene(action, context)

    print(f"Collected {len(learner.observations)} observations")

    # Learn causal graph
    print("\nLearning causal graph...")
    graph = learner.learn_causal_graph()

    # Print results
    graph.print_graph()

    # Test predictions
    print("\n" + "=" * 60)
    print("TESTING CAUSAL PREDICTIONS")
    print("=" * 60)

    test_cases = [
        ("increment", {"counter": 0}),
        ("increment", {"counter": 9}),
        ("reset", {"counter": 15}),
        ("double", {"counter": 5}),
    ]

    for action, context in test_cases:
        predictions = learner.predict_guard_changes(action, context)
        print(f"\n{action} on counter={context['counter']}:")
        for guard, prob in sorted(predictions.items(), key=lambda x: -x[1]):
            if prob > 0.1:
                print(f"  {guard}: {prob:.2f}")

    return learner


if __name__ == "__main__":
    demo()
