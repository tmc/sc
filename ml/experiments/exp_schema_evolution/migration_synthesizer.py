"""
Migration Synthesizer: Generate Safe State Mappings

Learns migration functions from examples:
- old_config → new_config mappings
- Context transformations
- Rollback plans

Based on proto/statecharts/v1/evolution.proto MigrationPlan semantics.

Mapping types (from StateMappingType):
- IDENTITY: Same label in both versions
- RENAME: Label changed, structure preserved
- TO_PARENT: State removed, map to parent's initial
- TO_SIBLING: State removed, map to sibling
- TO_INITIAL: State removed, map to chart initial
- SPLIT: One state splits into multiple
- MERGE: Multiple states merge into one

NO HARDCODING: Migration patterns learned from example pairs.
"""

import random
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from enum import Enum, auto

from .schema_diff import (
    Statechart, State, Transition, Event,
    ChartDiff, StateDiff, ChangeType, BreakingChange,
    MigrationRequirement, SchemaDiffer,
)


# =============================================================================
# MAPPING TYPES (mirrors proto/statecharts/v1/evolution.proto)
# =============================================================================

class StateMappingType(Enum):
    """Type of state mapping in migration."""
    UNSPECIFIED = 0
    IDENTITY = 1      # s ∈ σ₁ ⟹ s ∈ σ₂ (same label)
    RENAME = 2        # s₁ ∈ σ₁ ⟹ s₂ ∈ σ₂ (label changed)
    TO_PARENT = 3     # s removed ⟹ default(parent(s)) ∈ σ₂
    TO_SIBLING = 4    # s removed ⟹ sibling ∈ σ₂
    TO_INITIAL = 5    # s removed ⟹ initial state ∈ σ₂
    SPLIT = 6         # s ∈ σ₁ ⟹ f(context) ∈ σ₂ (conditional)
    MERGE = 7         # {s₁, s₂} ∩ σ₁ ≠ ∅ ⟹ t ∈ σ₂
    CUSTOM = 8        # User-defined mapping
    ERROR = 9         # No valid mapping


class MigrationStrategy(Enum):
    """Migration execution strategy."""
    STOP_THE_WORLD = 1   # Halt, migrate, resume
    ROLLING = 2          # One machine at a time
    BLUE_GREEN = 3       # Run both, switch
    CANARY = 4           # Subset first


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Configuration:
    """Active state configuration."""
    states: Set[str]
    context: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> 'Configuration':
        return Configuration(
            states=self.states.copy(),
            context=copy.deepcopy(self.context),
        )


# =============================================================================
# MAPPING AND TRANSFORMATION
# =============================================================================

@dataclass
class StateMapping:
    """
    Mapping rule for state migration.

    Formal: m = (type, from, to, condition, transform)
    """
    mapping_type: StateMappingType
    from_states: List[str]
    to_states: List[str]
    condition: Optional[str] = None  # CEL expression
    context_transform: Optional['ContextTransformation'] = None
    priority: int = 0

    def applies(self, config: Configuration) -> bool:
        """Check if mapping applies to configuration."""
        if not self.from_states:
            return False

        for state in self.from_states:
            if state in config.states:
                if self.condition:
                    # Evaluate condition (simplified)
                    try:
                        return eval(self.condition, {}, config.context)
                    except:
                        return False
                return True
        return False

    def apply(self, config: Configuration) -> Configuration:
        """Apply mapping to configuration."""
        new_config = config.copy()

        # Remove source states
        for state in self.from_states:
            new_config.states.discard(state)

        # Add target states
        for state in self.to_states:
            new_config.states.add(state)

        # Apply context transformation
        if self.context_transform:
            new_config.context = self.context_transform.apply(new_config.context)

        return new_config


@dataclass
class ContextTransformation:
    """
    Context variable transformation.

    Operations:
    - set: key → expression
    - remove: [keys]
    - rename: old_key → new_key
    """
    set_values: Dict[str, str] = field(default_factory=dict)
    remove_keys: List[str] = field(default_factory=list)
    rename_keys: Dict[str, str] = field(default_factory=dict)

    def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transformation to context."""
        result = context.copy()

        # Apply renames first
        for old_key, new_key in self.rename_keys.items():
            if old_key in result:
                result[new_key] = result.pop(old_key)

        # Remove keys
        for key in self.remove_keys:
            result.pop(key, None)

        # Set values (simplified eval)
        for key, expr in self.set_values.items():
            try:
                result[key] = eval(expr, {}, result)
            except:
                pass

        return result


# =============================================================================
# MIGRATION PLAN
# =============================================================================

@dataclass
class MigrationPlan:
    """
    Complete migration specification.

    Formal: M = (V₁, V₂, strategy, mappings, transform, validations)
    """
    plan_id: str
    from_version: str
    to_version: str
    strategy: MigrationStrategy = MigrationStrategy.STOP_THE_WORLD
    state_mappings: List[StateMapping] = field(default_factory=list)
    context_transform: Optional[ContextTransformation] = None
    validations: List['MigrationValidation'] = field(default_factory=list)
    rollback_mappings: List[StateMapping] = field(default_factory=list)
    fitness: float = 0.0

    def migrate(self, config: Configuration) -> Configuration:
        """
        Apply migration to configuration.

        Returns new configuration in target version.
        """
        result = config.copy()

        # Sort mappings by priority
        sorted_mappings = sorted(
            self.state_mappings,
            key=lambda m: m.priority,
            reverse=True
        )

        # Apply matching mappings
        for mapping in sorted_mappings:
            if mapping.applies(result):
                result = mapping.apply(result)

        # Apply global context transform
        if self.context_transform:
            result.context = self.context_transform.apply(result.context)

        return result

    def rollback(self, config: Configuration) -> Configuration:
        """Apply rollback to configuration."""
        result = config.copy()

        for mapping in self.rollback_mappings:
            if mapping.applies(result):
                result = mapping.apply(result)

        return result

    def copy(self) -> 'MigrationPlan':
        """Deep copy."""
        return MigrationPlan(
            plan_id=self.plan_id,
            from_version=self.from_version,
            to_version=self.to_version,
            strategy=self.strategy,
            state_mappings=[copy.deepcopy(m) for m in self.state_mappings],
            context_transform=copy.deepcopy(self.context_transform),
            validations=[copy.deepcopy(v) for v in self.validations],
            rollback_mappings=[copy.deepcopy(m) for m in self.rollback_mappings],
            fitness=self.fitness,
        )


@dataclass
class MigrationValidation:
    """Post-migration validation check."""
    name: str
    expression: str  # CEL expression
    error_message: str


# =============================================================================
# MIGRATION SYNTHESIZER
# =============================================================================

class MigrationSynthesizer:
    """
    Synthesizes migration plans from examples.

    Learning approach:
    1. Analyze schema diff
    2. Generate candidate mappings for each change
    3. Evolve mapping selection based on example pairs
    4. Validate against safety constraints
    """

    def __init__(
        self,
        population_size: int = 30,
        n_generations: int = 50,
        mutation_rate: float = 0.3,
    ):
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.differ = SchemaDiffer()

    def synthesize_from_diff(
        self,
        diff: ChartDiff,
        old_chart: Statechart,
        new_chart: Statechart,
    ) -> MigrationPlan:
        """
        Synthesize migration plan from schema diff.

        Uses heuristics to generate initial mappings,
        then refines based on constraints.
        """
        plan = MigrationPlan(
            plan_id=f"migrate_{diff.from_version}_to_{diff.to_version}",
            from_version=diff.from_version,
            to_version=diff.to_version,
        )

        new_states = new_chart.get_all_states()

        for change in diff.state_changes:
            mapping = self._synthesize_state_mapping(change, old_chart, new_chart)
            if mapping:
                plan.state_mappings.append(mapping)

        # Generate rollback mappings (inverse)
        plan.rollback_mappings = self._generate_rollback_mappings(plan.state_mappings)

        return plan

    def _synthesize_state_mapping(
        self,
        change: StateDiff,
        old_chart: Statechart,
        new_chart: Statechart,
    ) -> Optional[StateMapping]:
        """Generate mapping for a state change."""
        new_states = new_chart.get_all_states()

        if change.change_type == ChangeType.REMOVED:
            # Find best target for removed state
            old_state = change.old_state

            # Option 1: Map to renamed state
            for other in diff.state_changes if hasattr(self, 'current_diff') else []:
                if other.change_type == ChangeType.RENAMED:
                    if other.state_label == change.state_label:
                        return StateMapping(
                            mapping_type=StateMappingType.RENAME,
                            from_states=[change.state_label],
                            to_states=[other.new_label],
                        )

            # Option 2: Map to parent's initial child
            if old_state and old_state.parent_label:
                parent_label = old_state.parent_label
                if parent_label in new_states:
                    parent = new_states[parent_label]
                    if parent.children:
                        initial_child = next(
                            (c for c in parent.children if c.is_initial),
                            parent.children[0]
                        )
                        return StateMapping(
                            mapping_type=StateMappingType.TO_PARENT,
                            from_states=[change.state_label],
                            to_states=[initial_child.label],
                        )

            # Option 3: Map to chart initial
            if new_chart.root_state and new_chart.root_state.children:
                initial = next(
                    (c for c in new_chart.root_state.children if c.is_initial),
                    new_chart.root_state.children[0]
                )
                return StateMapping(
                    mapping_type=StateMappingType.TO_INITIAL,
                    from_states=[change.state_label],
                    to_states=[initial.label],
                )

        elif change.change_type == ChangeType.RENAMED:
            return StateMapping(
                mapping_type=StateMappingType.RENAME,
                from_states=[change.state_label],
                to_states=[change.new_label],
            )

        elif change.change_type == ChangeType.MOVED:
            # State moved but still exists
            return StateMapping(
                mapping_type=StateMappingType.IDENTITY,
                from_states=[change.state_label],
                to_states=[change.state_label],
            )

        return None

    def _generate_rollback_mappings(
        self,
        forward_mappings: List[StateMapping]
    ) -> List[StateMapping]:
        """Generate inverse mappings for rollback."""
        rollback = []

        for mapping in forward_mappings:
            if mapping.mapping_type == StateMappingType.RENAME:
                rollback.append(StateMapping(
                    mapping_type=StateMappingType.RENAME,
                    from_states=mapping.to_states,
                    to_states=mapping.from_states,
                ))
            elif mapping.mapping_type in [
                StateMappingType.TO_PARENT,
                StateMappingType.TO_SIBLING,
                StateMappingType.TO_INITIAL,
            ]:
                # These can't be directly inverted - need original state
                rollback.append(StateMapping(
                    mapping_type=StateMappingType.ERROR,
                    from_states=mapping.to_states,
                    to_states=mapping.from_states,
                ))

        return rollback

    def evolve_migration(
        self,
        old_chart: Statechart,
        new_chart: Statechart,
        example_pairs: List[Tuple[Configuration, Configuration]],
        verbose: bool = True,
    ) -> MigrationPlan:
        """
        Evolve migration plan from example pairs.

        Args:
            old_chart: Source version
            new_chart: Target version
            example_pairs: (old_config, expected_new_config) pairs
            verbose: Print progress

        Returns:
            Best evolved migration plan
        """
        # Get diff
        diff = self.differ.diff(old_chart, new_chart)

        # Initialize population
        base_plan = self.synthesize_from_diff(diff, old_chart, new_chart)
        population = [base_plan]

        # Add random variations
        for _ in range(self.population_size - 1):
            variant = self._mutate_plan(base_plan, old_chart, new_chart)
            population.append(variant)

        best_ever = None

        for gen in range(self.n_generations):
            # Evaluate fitness
            for plan in population:
                plan.fitness = self._evaluate_fitness(plan, example_pairs)

            # Sort by fitness
            population.sort(key=lambda p: p.fitness, reverse=True)

            # Track best
            if best_ever is None or population[0].fitness > best_ever.fitness:
                best_ever = population[0].copy()

            if verbose and gen % 10 == 0:
                print(f"Gen {gen:3d}: Best fitness = {population[0].fitness:.3f}")

            # Perfect?
            if population[0].fitness >= 0.99:
                break

            # Selection and reproduction
            elite = [p.copy() for p in population[:5]]
            new_pop = elite

            while len(new_pop) < self.population_size:
                parent = random.choice(population[:self.population_size // 2])
                child = self._mutate_plan(parent, old_chart, new_chart)
                new_pop.append(child)

            population = new_pop

        return best_ever

    def _mutate_plan(
        self,
        plan: MigrationPlan,
        old_chart: Statechart,
        new_chart: Statechart,
    ) -> MigrationPlan:
        """Mutate a migration plan."""
        new_plan = plan.copy()
        new_states = new_chart.get_all_states()
        new_state_labels = list(new_states.keys())

        for mapping in new_plan.state_mappings:
            if random.random() < self.mutation_rate:
                # Change target state
                if new_state_labels:
                    mapping.to_states = [random.choice(new_state_labels)]

                # Change mapping type
                if random.random() < 0.3:
                    mapping.mapping_type = random.choice([
                        StateMappingType.TO_PARENT,
                        StateMappingType.TO_SIBLING,
                        StateMappingType.TO_INITIAL,
                    ])

        return new_plan

    def _evaluate_fitness(
        self,
        plan: MigrationPlan,
        example_pairs: List[Tuple[Configuration, Configuration]],
    ) -> float:
        """
        Evaluate migration plan on example pairs.

        Fitness = fraction of examples correctly migrated.
        """
        if not example_pairs:
            return 0.0

        correct = 0
        for old_config, expected_new in example_pairs:
            actual_new = plan.migrate(old_config)

            # Check if states match
            if actual_new.states == expected_new.states:
                correct += 1

        return correct / len(example_pairs)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def synthesize_migration(
    old_chart: Statechart,
    new_chart: Statechart,
) -> MigrationPlan:
    """Quick migration synthesis without examples."""
    synth = MigrationSynthesizer()
    diff = synth.differ.diff(old_chart, new_chart)
    return synth.synthesize_from_diff(diff, old_chart, new_chart)


def evolve_migration(
    old_chart: Statechart,
    new_chart: Statechart,
    examples: List[Tuple[Configuration, Configuration]],
    n_generations: int = 30,
    verbose: bool = True,
) -> MigrationPlan:
    """Evolve migration from examples."""
    synth = MigrationSynthesizer(n_generations=n_generations)
    return synth.evolve_migration(old_chart, new_chart, examples, verbose)


# =============================================================================
# TESTING
# =============================================================================

def test_migration_synthesizer():
    """Test migration synthesis."""
    print("=" * 60)
    print("MIGRATION SYNTHESIZER TEST")
    print("=" * 60)

    # Create old chart
    old_chart = Statechart(
        version="1.0.0",
        root_state=State(
            label="__root__",
            state_type="NORMAL",
            children=[
                State(label="Idle", is_initial=True),
                State(label="Active"),
                State(label="Done"),
            ]
        ),
    )

    # Create new chart (Done removed, Complete added)
    new_chart = Statechart(
        version="2.0.0",
        root_state=State(
            label="__root__",
            state_type="NORMAL",
            children=[
                State(label="Idle", is_initial=True),
                State(label="Active"),
                State(label="Complete"),
            ]
        ),
    )

    # Synthesize migration
    synth = MigrationSynthesizer()
    diff = synth.differ.diff(old_chart, new_chart)
    plan = synth.synthesize_from_diff(diff, old_chart, new_chart)

    print(f"\nMigration: {plan.from_version} → {plan.to_version}")
    print(f"Mappings: {len(plan.state_mappings)}")

    for mapping in plan.state_mappings:
        print(f"  {mapping.mapping_type.name}: {mapping.from_states} → {mapping.to_states}")

    # Test migration
    old_config = Configuration(states={"Done"})
    new_config = plan.migrate(old_config)

    print(f"\nMigration test:")
    print(f"  Old config: {old_config.states}")
    print(f"  New config: {new_config.states}")

    # Test evolution with examples
    print("\n" + "-" * 60)
    print("EVOLUTION TEST")
    print("-" * 60)

    examples = [
        (Configuration(states={"Idle"}), Configuration(states={"Idle"})),
        (Configuration(states={"Active"}), Configuration(states={"Active"})),
        (Configuration(states={"Done"}), Configuration(states={"Complete"})),
    ]

    evolved_plan = synth.evolve_migration(
        old_chart, new_chart, examples,
        verbose=True,
    )

    print(f"\nEvolved plan fitness: {evolved_plan.fitness:.3f}")

    # Test evolved migration
    for old_config, expected in examples:
        actual = evolved_plan.migrate(old_config)
        match = "OK" if actual.states == expected.states else "FAIL"
        print(f"  {old_config.states} → {actual.states} [{match}]")

    print("\n" + "=" * 60)
    print("MIGRATION SYNTHESIZER TEST COMPLETE")
    print("=" * 60)

    return evolved_plan


if __name__ == "__main__":
    test_migration_synthesizer()
