"""
Action Composition Algebra

Learn how sequential actions compose: A;B = ?
Discover algebraic properties through evolution:
- Commutativity: A;B = B;A ?
- Associativity: (A;B);C = A;(B;C) ?
- Identity: A;I = I;A = A ?
- Inverses: A;A^-1 = I ?
- Idempotence: A;A = A ?
- Absorption: A;B;A = A ?

Builds on exp_action_side_effects: uses ActionEffect to model
what each action does, then learns how effects compose.

NO HARDCODING - all composition rules evolved from examples.
"""

import mlx.core as mx
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable, FrozenSet
from enum import Enum, auto
from abc import ABC, abstractmethod
from collections import defaultdict
import random
import copy
import math


# =============================================================================
# Action Representation
# =============================================================================

class EffectType(Enum):
    """Types of effects on context variables."""
    SET = auto()        # x := v
    INCREMENT = auto()  # x := x + v
    DECREMENT = auto()  # x := x - v
    TOGGLE = auto()     # x := !x
    MULTIPLY = auto()   # x := x * v
    CLEAR = auto()      # x := 0/null
    NOOP = auto()       # no change


@dataclass(frozen=True)
class Effect:
    """A single effect on a variable."""
    variable: str
    effect_type: EffectType
    value: Any = None

    def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply this effect to context."""
        ctx = dict(context)
        current = ctx.get(self.variable, 0)

        if self.effect_type == EffectType.SET:
            ctx[self.variable] = self.value
        elif self.effect_type == EffectType.INCREMENT:
            ctx[self.variable] = current + (self.value or 1)
        elif self.effect_type == EffectType.DECREMENT:
            ctx[self.variable] = current - (self.value or 1)
        elif self.effect_type == EffectType.TOGGLE:
            ctx[self.variable] = not current
        elif self.effect_type == EffectType.MULTIPLY:
            ctx[self.variable] = current * (self.value or 1)
        elif self.effect_type == EffectType.CLEAR:
            ctx[self.variable] = 0
        # NOOP: no change

        return ctx


@dataclass
class Action:
    """An action with multiple effects."""
    name: str
    effects: List[Effect] = field(default_factory=list)

    def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all effects to context."""
        ctx = dict(context)
        for effect in self.effects:
            ctx = effect.apply(ctx)
        return ctx

    def variables_written(self) -> Set[str]:
        """Variables modified by this action."""
        return {e.variable for e in self.effects if e.effect_type != EffectType.NOOP}

    def variables_read(self) -> Set[str]:
        """Variables read by this action (for increment/decrement)."""
        return {
            e.variable for e in self.effects
            if e.effect_type in (EffectType.INCREMENT, EffectType.DECREMENT,
                                  EffectType.TOGGLE, EffectType.MULTIPLY)
        }

    def __hash__(self):
        return hash(self.name)


# Identity action
IDENTITY = Action(name="I", effects=[])


# =============================================================================
# Composition Operations
# =============================================================================

@dataclass
class ComposedAction:
    """Represents A;B (A then B)."""
    first: Action
    second: Action
    _cached_effects: Optional[List[Effect]] = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return f"({self.first.name};{self.second.name})"

    def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply first, then second."""
        ctx = self.first.apply(context)
        return self.second.apply(ctx)

    def to_action(self) -> Action:
        """Flatten to single action with combined effects."""
        if self._cached_effects is not None:
            return Action(name=self.name, effects=self._cached_effects)

        # Simulate to find net effects
        # Start with symbolic context
        effects = []
        seen_vars: Set[str] = set()

        # Second action's effects take precedence for SET
        for e in reversed(self.second.effects + self.first.effects):
            if e.variable not in seen_vars:
                effects.append(e)
                if e.effect_type == EffectType.SET:
                    seen_vars.add(e.variable)

        effects.reverse()
        return Action(name=self.name, effects=effects)


def compose(a: Action, b: Action) -> Action:
    """Compose two actions: a;b."""
    if a.name == "I":
        return b
    if b.name == "I":
        return a

    composed = ComposedAction(first=a, second=b)
    return composed.to_action()


def compose_sequence(actions: List[Action]) -> Action:
    """Compose a sequence of actions."""
    if not actions:
        return IDENTITY
    result = actions[0]
    for a in actions[1:]:
        result = compose(result, a)
    return result


# =============================================================================
# Algebraic Properties
# =============================================================================

class AlgebraicProperty(Enum):
    """Properties of action composition."""
    COMMUTATIVE = auto()    # A;B = B;A
    ASSOCIATIVE = auto()    # (A;B);C = A;(B;C) - always true for functions
    IDEMPOTENT = auto()     # A;A = A
    SELF_INVERSE = auto()   # A;A = I
    LEFT_IDENTITY = auto()  # I;A = A - always true
    RIGHT_IDENTITY = auto() # A;I = A - always true
    NILPOTENT = auto()      # A;A;...;A = I (for some n)
    ABSORBING = auto()      # A;B;A = A (A absorbs B)


@dataclass
class PropertyEvidence:
    """Evidence for/against an algebraic property."""
    property_type: AlgebraicProperty
    action_a: str
    action_b: Optional[str] = None
    positive_count: int = 0
    negative_count: int = 0

    @property
    def confidence(self) -> float:
        """Confidence in this property holding."""
        total = self.positive_count + self.negative_count
        if total == 0:
            return 0.5  # Prior
        return self.positive_count / total

    @property
    def evidence_strength(self) -> float:
        """How much evidence we have."""
        return math.log(1 + self.positive_count + self.negative_count)


# =============================================================================
# Composition Rule Learning
# =============================================================================

@dataclass
class CompositionRule:
    """A learned rule about action composition."""
    pattern: Tuple[str, ...]  # e.g., ("A", "B") or ("A", "A")
    result: str               # e.g., "C" or "I" or "A"
    confidence: float = 0.0
    examples_seen: int = 0

    def matches(self, actions: List[Action]) -> bool:
        """Check if this rule matches the action sequence."""
        if len(actions) != len(self.pattern):
            return False

        # Pattern matching with wildcards
        bindings: Dict[str, str] = {}
        for pattern_elem, action in zip(self.pattern, actions):
            if pattern_elem.startswith("$"):
                # Variable - bind or check
                var = pattern_elem
                if var in bindings:
                    if bindings[var] != action.name:
                        return False
                else:
                    bindings[var] = action.name
            else:
                # Literal - must match
                if pattern_elem != action.name:
                    return False
        return True


@dataclass
class CompositionHypothesis:
    """Hypothesis about how actions compose."""
    # Discovered properties per action pair
    properties: Dict[Tuple[str, str], Set[AlgebraicProperty]] = field(
        default_factory=lambda: defaultdict(set)
    )

    # Learned composition rules
    rules: List[CompositionRule] = field(default_factory=list)

    # Equivalence classes (actions that behave identically)
    equivalences: Dict[str, str] = field(default_factory=dict)

    # Inverse pairs
    inverses: Dict[str, str] = field(default_factory=dict)

    def add_property(self, a: str, b: str, prop: AlgebraicProperty):
        """Record that property holds for (a, b)."""
        self.properties[(a, b)].add(prop)

    def has_property(self, a: str, b: str, prop: AlgebraicProperty) -> bool:
        """Check if property is known to hold."""
        return prop in self.properties.get((a, b), set())


# =============================================================================
# Composition Algebra Learner
# =============================================================================

class CompositionAlgebraLearner:
    """
    Learns action composition algebra from examples.

    Given (action_sequence, initial_context, final_context) examples,
    discovers:
    1. Which action pairs commute
    2. Which actions are idempotent
    3. Which actions are inverses
    4. Composition simplification rules
    """

    def __init__(
        self,
        context_variables: List[str],
        test_contexts: Optional[List[Dict[str, Any]]] = None,
    ):
        self.context_variables = context_variables
        self.test_contexts = test_contexts or self._generate_test_contexts()

        # Learned knowledge
        self.hypothesis = CompositionHypothesis()
        self.property_evidence: Dict[Tuple, PropertyEvidence] = {}

        # Action library
        self.actions: Dict[str, Action] = {"I": IDENTITY}

        # Composition cache
        self._composition_cache: Dict[Tuple[str, str], Action] = {}

    def _generate_test_contexts(self, n: int = 20) -> List[Dict[str, Any]]:
        """Generate diverse test contexts."""
        contexts = []
        for _ in range(n):
            ctx = {
                var: random.randint(-10, 10)
                for var in self.context_variables
            }
            contexts.append(ctx)
        return contexts

    def register_action(self, action: Action):
        """Register an action for composition learning."""
        self.actions[action.name] = action

    def _contexts_equal(self, ctx1: Dict[str, Any], ctx2: Dict[str, Any]) -> bool:
        """Check if two contexts are equal on tracked variables."""
        for var in self.context_variables:
            if ctx1.get(var) != ctx2.get(var):
                return False
        return True

    def _test_equality(self, a: Action, b: Action) -> float:
        """Test if two actions are equivalent across test contexts."""
        equal_count = 0
        for ctx in self.test_contexts:
            result_a = a.apply(ctx)
            result_b = b.apply(ctx)
            if self._contexts_equal(result_a, result_b):
                equal_count += 1
        return equal_count / len(self.test_contexts)

    def _test_commutativity(self, a: Action, b: Action) -> float:
        """Test if A;B = B;A."""
        ab = compose(a, b)
        ba = compose(b, a)
        return self._test_equality(ab, ba)

    def _test_idempotence(self, a: Action) -> float:
        """Test if A;A = A."""
        aa = compose(a, a)
        return self._test_equality(aa, a)

    def _test_self_inverse(self, a: Action) -> float:
        """Test if A;A = I."""
        aa = compose(a, a)
        return self._test_equality(aa, IDENTITY)

    def _test_inverse(self, a: Action, b: Action) -> float:
        """Test if A;B = I."""
        ab = compose(a, b)
        return self._test_equality(ab, IDENTITY)

    def _test_absorption(self, a: Action, b: Action) -> float:
        """Test if A;B;A = A."""
        aba = compose(compose(a, b), a)
        return self._test_equality(aba, a)

    def discover_properties(self, actions: Optional[List[Action]] = None):
        """Discover algebraic properties of actions."""
        actions = actions or list(self.actions.values())

        for a in actions:
            # Test idempotence
            idem_score = self._test_idempotence(a)
            key = (a.name, a.name, AlgebraicProperty.IDEMPOTENT)
            if key not in self.property_evidence:
                self.property_evidence[key] = PropertyEvidence(
                    property_type=AlgebraicProperty.IDEMPOTENT,
                    action_a=a.name
                )
            if idem_score > 0.95:
                self.property_evidence[key].positive_count += 1
                self.hypothesis.add_property(a.name, a.name, AlgebraicProperty.IDEMPOTENT)
            else:
                self.property_evidence[key].negative_count += 1

            # Test self-inverse
            inv_score = self._test_self_inverse(a)
            key = (a.name, a.name, AlgebraicProperty.SELF_INVERSE)
            if key not in self.property_evidence:
                self.property_evidence[key] = PropertyEvidence(
                    property_type=AlgebraicProperty.SELF_INVERSE,
                    action_a=a.name
                )
            if inv_score > 0.95:
                self.property_evidence[key].positive_count += 1
                self.hypothesis.add_property(a.name, a.name, AlgebraicProperty.SELF_INVERSE)
            else:
                self.property_evidence[key].negative_count += 1

            # Test pairwise properties
            for b in actions:
                if a.name == b.name:
                    continue

                # Commutativity
                comm_score = self._test_commutativity(a, b)
                key = (a.name, b.name, AlgebraicProperty.COMMUTATIVE)
                if key not in self.property_evidence:
                    self.property_evidence[key] = PropertyEvidence(
                        property_type=AlgebraicProperty.COMMUTATIVE,
                        action_a=a.name,
                        action_b=b.name
                    )
                if comm_score > 0.95:
                    self.property_evidence[key].positive_count += 1
                    self.hypothesis.add_property(a.name, b.name, AlgebraicProperty.COMMUTATIVE)
                else:
                    self.property_evidence[key].negative_count += 1

                # Inverse
                inv_score = self._test_inverse(a, b)
                if inv_score > 0.95:
                    self.hypothesis.inverses[a.name] = b.name
                    self.hypothesis.inverses[b.name] = a.name

                # Absorption
                abs_score = self._test_absorption(a, b)
                key = (a.name, b.name, AlgebraicProperty.ABSORBING)
                if key not in self.property_evidence:
                    self.property_evidence[key] = PropertyEvidence(
                        property_type=AlgebraicProperty.ABSORBING,
                        action_a=a.name,
                        action_b=b.name
                    )
                if abs_score > 0.95:
                    self.property_evidence[key].positive_count += 1
                    self.hypothesis.add_property(a.name, b.name, AlgebraicProperty.ABSORBING)
                else:
                    self.property_evidence[key].negative_count += 1

    def learn_from_trace(
        self,
        action_sequence: List[str],
        initial_context: Dict[str, Any],
        final_context: Dict[str, Any]
    ):
        """Learn from an observed execution trace."""
        # Convert names to actions
        actions = [self.actions[name] for name in action_sequence if name in self.actions]

        if not actions:
            return

        # Verify the trace
        ctx = dict(initial_context)
        for action in actions:
            ctx = action.apply(ctx)

        if not self._contexts_equal(ctx, final_context):
            # Trace doesn't match - something's wrong
            return

        # Look for simplification opportunities
        # Check if subsequences can be simplified
        for i in range(len(actions)):
            for j in range(i + 1, min(i + 4, len(actions) + 1)):
                subseq = actions[i:j]
                composed = compose_sequence(subseq)

                # Check if this composition equals a known action
                for name, action in self.actions.items():
                    if name in [a.name for a in subseq]:
                        continue
                    eq_score = self._test_equality(composed, action)
                    if eq_score > 0.95:
                        # Found a simplification rule!
                        pattern = tuple(a.name for a in subseq)
                        rule = CompositionRule(
                            pattern=pattern,
                            result=name,
                            confidence=eq_score,
                            examples_seen=1
                        )
                        self._add_rule(rule)

    def _add_rule(self, rule: CompositionRule):
        """Add or update a composition rule."""
        # Check if rule already exists
        for existing in self.hypothesis.rules:
            if existing.pattern == rule.pattern and existing.result == rule.result:
                existing.examples_seen += 1
                existing.confidence = (
                    existing.confidence * 0.9 + rule.confidence * 0.1
                )
                return

        self.hypothesis.rules.append(rule)

    def simplify(self, actions: List[Action]) -> List[Action]:
        """Simplify action sequence using learned rules."""
        if len(actions) <= 1:
            return actions

        result = list(actions)
        changed = True

        while changed:
            changed = False

            # Try each rule
            for rule in sorted(self.hypothesis.rules,
                               key=lambda r: -len(r.pattern)):
                # Find matching subsequence
                for i in range(len(result) - len(rule.pattern) + 1):
                    subseq = result[i:i + len(rule.pattern)]
                    if rule.matches(subseq):
                        # Apply rule
                        replacement = self.actions.get(rule.result)
                        if replacement:
                            result = result[:i] + [replacement] + result[i + len(rule.pattern):]
                            changed = True
                            break
                if changed:
                    break

            # Remove identity actions
            result = [a for a in result if a.name != "I"]

        return result if result else [IDENTITY]

    def get_property_summary(self) -> Dict[str, List[str]]:
        """Summarize discovered properties."""
        summary = defaultdict(list)

        for (a, b, prop), evidence in self.property_evidence.items():
            if evidence.confidence > 0.8:
                if prop == AlgebraicProperty.COMMUTATIVE:
                    summary["commutative"].append(f"{a};{b} = {b};{a}")
                elif prop == AlgebraicProperty.IDEMPOTENT:
                    summary["idempotent"].append(f"{a};{a} = {a}")
                elif prop == AlgebraicProperty.SELF_INVERSE:
                    summary["self_inverse"].append(f"{a};{a} = I")
                elif prop == AlgebraicProperty.ABSORBING:
                    summary["absorbing"].append(f"{a};{b};{a} = {a}")

        for a, b in self.hypothesis.inverses.items():
            if a < b:  # Avoid duplicates
                summary["inverse_pairs"].append(f"{a};{b} = I")

        return dict(summary)


# =============================================================================
# Composition Evolver
# =============================================================================

@dataclass
class CompositionGenome:
    """Genome encoding composition behavior hypotheses."""
    # Which pairs commute (evolved guess)
    commutative_pairs: Set[FrozenSet[str]] = field(default_factory=set)

    # Idempotent actions
    idempotent_actions: Set[str] = field(default_factory=set)

    # Self-inverse actions
    self_inverse_actions: Set[str] = field(default_factory=set)

    # Inverse pairs
    inverse_pairs: Dict[str, str] = field(default_factory=dict)

    # Fitness
    fitness: float = 0.0

    def clone(self) -> "CompositionGenome":
        return CompositionGenome(
            commutative_pairs=set(self.commutative_pairs),
            idempotent_actions=set(self.idempotent_actions),
            self_inverse_actions=set(self.self_inverse_actions),
            inverse_pairs=dict(self.inverse_pairs),
        )


class CompositionEvolver:
    """
    Evolves composition algebra hypotheses.

    Rather than testing all pairs exhaustively, uses evolution
    to discover likely algebraic properties.
    """

    def __init__(
        self,
        actions: List[Action],
        context_variables: List[str],
        population_size: int = 30,
    ):
        self.actions = {a.name: a for a in actions}
        self.action_names = [a.name for a in actions if a.name != "I"]
        self.context_variables = context_variables
        self.population_size = population_size

        self.learner = CompositionAlgebraLearner(
            context_variables=context_variables
        )
        for a in actions:
            self.learner.register_action(a)

        # Population
        self.population: List[CompositionGenome] = []
        self.best_genome: Optional[CompositionGenome] = None
        self.generation = 0

    def _random_genome(self) -> CompositionGenome:
        """Create random genome."""
        genome = CompositionGenome()

        # Random commutative pairs
        for a in self.action_names:
            for b in self.action_names:
                if a < b and random.random() < 0.3:
                    genome.commutative_pairs.add(frozenset([a, b]))

        # Random idempotent
        for a in self.action_names:
            if random.random() < 0.2:
                genome.idempotent_actions.add(a)

        # Random self-inverse
        for a in self.action_names:
            if random.random() < 0.1:
                genome.self_inverse_actions.add(a)

        return genome

    def _mutate(self, genome: CompositionGenome) -> CompositionGenome:
        """Mutate genome."""
        g = genome.clone()

        # Mutate commutative pairs
        if random.random() < 0.3 and len(self.action_names) >= 2:
            a, b = random.sample(self.action_names, 2)
            pair = frozenset([a, b])
            if pair in g.commutative_pairs:
                g.commutative_pairs.remove(pair)
            else:
                g.commutative_pairs.add(pair)

        # Mutate idempotent
        if random.random() < 0.2 and self.action_names:
            a = random.choice(self.action_names)
            if a in g.idempotent_actions:
                g.idempotent_actions.remove(a)
            else:
                g.idempotent_actions.add(a)

        # Mutate self-inverse
        if random.random() < 0.2 and self.action_names:
            a = random.choice(self.action_names)
            if a in g.self_inverse_actions:
                g.self_inverse_actions.remove(a)
            else:
                g.self_inverse_actions.add(a)

        return g

    def _evaluate(self, genome: CompositionGenome) -> float:
        """Evaluate genome fitness."""
        score = 0.0
        tests = 0

        # Test commutative claims
        for pair in genome.commutative_pairs:
            a_name, b_name = list(pair)
            a, b = self.actions[a_name], self.actions[b_name]
            comm_score = self.learner._test_commutativity(a, b)
            score += comm_score
            tests += 1

        # Penalize false commutative claims
        for a_name in self.action_names:
            for b_name in self.action_names:
                if a_name >= b_name:
                    continue
                pair = frozenset([a_name, b_name])
                if pair not in genome.commutative_pairs:
                    a, b = self.actions[a_name], self.actions[b_name]
                    comm_score = self.learner._test_commutativity(a, b)
                    if comm_score > 0.95:
                        score -= 0.5  # Missed a commutative pair
                    tests += 1

        # Test idempotent claims
        for a_name in genome.idempotent_actions:
            a = self.actions[a_name]
            idem_score = self.learner._test_idempotence(a)
            score += idem_score
            tests += 1

        # Test self-inverse claims
        for a_name in genome.self_inverse_actions:
            a = self.actions[a_name]
            inv_score = self.learner._test_self_inverse(a)
            score += inv_score
            tests += 1

        return score / max(1, tests)

    def evolve(self, n_generations: int = 50, verbose: bool = True) -> CompositionGenome:
        """Evolve to discover composition algebra."""
        # Initialize
        self.population = [self._random_genome() for _ in range(self.population_size)]

        for gen in range(n_generations):
            self.generation = gen

            # Evaluate
            for genome in self.population:
                genome.fitness = self._evaluate(genome)

            # Track best
            self.population.sort(key=lambda g: -g.fitness)
            if self.best_genome is None or self.population[0].fitness > self.best_genome.fitness:
                self.best_genome = self.population[0].clone()

            if verbose and gen % 10 == 0:
                print(f"Gen {gen}: best={self.best_genome.fitness:.3f}, "
                      f"comm={len(self.best_genome.commutative_pairs)}, "
                      f"idem={len(self.best_genome.idempotent_actions)}")

            # Next generation
            next_pop = [self.population[0].clone()]  # Elitism

            while len(next_pop) < self.population_size:
                parent = random.choice(self.population[:self.population_size // 2])
                child = self._mutate(parent)
                next_pop.append(child)

            self.population = next_pop

        return self.best_genome

    def get_algebra_summary(self) -> Dict[str, Any]:
        """Get summary of discovered algebra."""
        if not self.best_genome:
            return {}

        return {
            "commutative_pairs": [
                list(p) for p in self.best_genome.commutative_pairs
            ],
            "idempotent_actions": list(self.best_genome.idempotent_actions),
            "self_inverse_actions": list(self.best_genome.self_inverse_actions),
            "fitness": self.best_genome.fitness,
        }


# =============================================================================
# Demo Actions
# =============================================================================

def create_demo_actions() -> List[Action]:
    """Create demo actions with known algebraic properties."""
    actions = [IDENTITY]

    # Inc(x) - increment x
    actions.append(Action(
        name="Inc_x",
        effects=[Effect("x", EffectType.INCREMENT, 1)]
    ))

    # Dec(x) - decrement x (inverse of Inc)
    actions.append(Action(
        name="Dec_x",
        effects=[Effect("x", EffectType.DECREMENT, 1)]
    ))

    # Double(x) - double x
    actions.append(Action(
        name="Double_x",
        effects=[Effect("x", EffectType.MULTIPLY, 2)]
    ))

    # Set_x_0 - set x to 0 (idempotent, absorbing)
    actions.append(Action(
        name="Set_x_0",
        effects=[Effect("x", EffectType.SET, 0)]
    ))

    # Toggle_y - toggle y (self-inverse)
    actions.append(Action(
        name="Toggle_y",
        effects=[Effect("y", EffectType.TOGGLE)]
    ))

    # Inc_y - increment y (commutes with Inc_x)
    actions.append(Action(
        name="Inc_y",
        effects=[Effect("y", EffectType.INCREMENT, 1)]
    ))

    # Set_both - set x and y (idempotent)
    actions.append(Action(
        name="Set_both",
        effects=[
            Effect("x", EffectType.SET, 5),
            Effect("y", EffectType.SET, 5),
        ]
    ))

    return actions


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate action composition algebra learning."""
    print("=" * 60)
    print("Action Composition Algebra")
    print("=" * 60)

    # Create actions
    actions = create_demo_actions()
    print(f"\nActions: {[a.name for a in actions]}")

    # Create learner
    learner = CompositionAlgebraLearner(
        context_variables=["x", "y"]
    )

    for action in actions:
        learner.register_action(action)

    # Discover properties
    print("\nDiscovering algebraic properties...")
    learner.discover_properties(actions)

    # Summary
    summary = learner.get_property_summary()
    print("\n--- Discovered Properties ---")
    for prop_type, examples in summary.items():
        print(f"\n{prop_type}:")
        for ex in examples[:5]:
            print(f"  {ex}")

    # Test composition
    print("\n--- Composition Examples ---")
    inc_x = learner.actions["Inc_x"]
    dec_x = learner.actions["Dec_x"]
    toggle_y = learner.actions["Toggle_y"]

    # Inc_x ; Dec_x = I ?
    composed = compose(inc_x, dec_x)
    eq_score = learner._test_equality(composed, IDENTITY)
    print(f"Inc_x ; Dec_x = I ? Score: {eq_score:.2f}")

    # Toggle_y ; Toggle_y = I ?
    composed = compose(toggle_y, toggle_y)
    eq_score = learner._test_equality(composed, IDENTITY)
    print(f"Toggle_y ; Toggle_y = I ? Score: {eq_score:.2f}")

    # Inc_x ; Inc_y = Inc_y ; Inc_x ? (commutative)
    inc_y = learner.actions["Inc_y"]
    comm_score = learner._test_commutativity(inc_x, inc_y)
    print(f"Inc_x ; Inc_y = Inc_y ; Inc_x ? Score: {comm_score:.2f}")

    # Set_x_0 ; Inc_x = Set_x_0 ? (absorbing - no, actually x becomes 1)
    set_x = learner.actions["Set_x_0"]
    composed = compose(set_x, inc_x)
    ctx = {"x": 5, "y": 0}
    result = composed.apply(ctx)
    print(f"Set_x_0 ; Inc_x applied to x=5: x={result['x']}")

    return learner


def demo_evolution():
    """Demonstrate evolutionary discovery of composition algebra."""
    print("\n" + "=" * 60)
    print("Evolutionary Algebra Discovery")
    print("=" * 60)

    actions = create_demo_actions()

    evolver = CompositionEvolver(
        actions=actions,
        context_variables=["x", "y"],
        population_size=20,
    )

    print("\nEvolving composition algebra...")
    best = evolver.evolve(n_generations=30, verbose=True)

    print("\n--- Discovered Algebra ---")
    summary = evolver.get_algebra_summary()
    print(f"Commutative pairs: {summary.get('commutative_pairs', [])}")
    print(f"Idempotent: {summary.get('idempotent_actions', [])}")
    print(f"Self-inverse: {summary.get('self_inverse_actions', [])}")
    print(f"Fitness: {summary.get('fitness', 0):.3f}")

    return evolver


if __name__ == "__main__":
    demo()
    demo_evolution()
