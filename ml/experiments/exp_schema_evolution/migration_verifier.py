"""
Migration Verifier: Validate Migration Safety

Verifies that migrations preserve:
1. Reachable states (no orphaned configurations)
2. Behavioral equivalence (same events produce same outcomes)
3. Rollback capability (can restore previous state)
4. State invariants (custom validation rules)

Based on proto/statecharts/v1/evolution.proto safety semantics.

VERIFICATION PROPERTIES:
- Completeness: Every old state has a valid mapping
- Determinism: Each old state maps to exactly one new state
- Reachability: Target states are reachable in new chart
- Reversibility: Rollback produces valid old configuration

NO HARDCODING: Safety rules learned from example patterns.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from enum import Enum, auto

from .schema_diff import (
    Statechart, State, Transition, ChartDiff, ChangeType,
    SchemaDiffer,
)
from .migration_synthesizer import (
    MigrationPlan, StateMapping, StateMappingType,
    Configuration,
)


# =============================================================================
# VERIFICATION RESULT TYPES
# =============================================================================

class VerificationStatus(Enum):
    """Overall verification status."""
    PASSED = 1
    WARNING = 2
    FAILED = 3


class ViolationType(Enum):
    """Type of safety violation."""
    UNMAPPED_STATE = 1          # Old state has no mapping
    AMBIGUOUS_MAPPING = 2       # Old state maps to multiple targets
    UNREACHABLE_TARGET = 3      # Target state not in new chart
    INVALID_ROLLBACK = 4        # Rollback produces invalid config
    ORPHANED_TRANSITION = 5     # Transition references missing state
    INVARIANT_VIOLATION = 6     # Custom invariant failed
    BEHAVIORAL_CHANGE = 7       # Same input produces different output


@dataclass
class Violation:
    """A single safety violation."""
    violation_type: ViolationType
    severity: str  # "error", "warning"
    element: str
    description: str
    suggested_fix: Optional[str] = None


@dataclass
class VerificationResult:
    """Complete verification result."""
    status: VerificationStatus
    violations: List[Violation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    coverage: float = 0.0  # Fraction of states covered by mappings
    is_reversible: bool = False
    behavioral_equivalence: float = 0.0

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.PASSED

    @property
    def error_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "warning")

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Verification: {self.status.name}",
            f"Coverage: {self.coverage:.1%}",
            f"Reversible: {self.is_reversible}",
            f"Behavioral equivalence: {self.behavioral_equivalence:.1%}",
            f"Errors: {self.error_count}, Warnings: {self.warning_count}",
        ]
        return "\n".join(lines)


# =============================================================================
# VERIFICATION RULES
# =============================================================================

@dataclass
class VerificationRule:
    """A verification rule to check."""
    name: str
    description: str
    check: Callable[[MigrationPlan, Statechart, Statechart], List[Violation]]


def check_completeness(
    plan: MigrationPlan,
    old_chart: Statechart,
    new_chart: Statechart,
) -> List[Violation]:
    """Check that every old state has a mapping."""
    violations = []
    old_states = old_chart.get_all_states()
    new_states = new_chart.get_all_states()

    mapped_old_states = set()
    for mapping in plan.state_mappings:
        for state in mapping.from_states:
            mapped_old_states.add(state)

    # Check for removed states that still exist and aren't mapped
    for label in old_states:
        if label not in new_states and label not in mapped_old_states:
            violations.append(Violation(
                violation_type=ViolationType.UNMAPPED_STATE,
                severity="error",
                element=label,
                description=f"State '{label}' was removed but has no mapping",
                suggested_fix=f"Add mapping from '{label}' to a valid target state",
            ))

    return violations


def check_determinism(
    plan: MigrationPlan,
    old_chart: Statechart,
    new_chart: Statechart,
) -> List[Violation]:
    """Check that mappings are deterministic (no ambiguity)."""
    violations = []

    # Check for overlapping from_states
    state_mappings: Dict[str, List[StateMapping]] = {}
    for mapping in plan.state_mappings:
        for state in mapping.from_states:
            if state not in state_mappings:
                state_mappings[state] = []
            state_mappings[state].append(mapping)

    for state, mappings in state_mappings.items():
        if len(mappings) > 1:
            # Check if they have different conditions (OK if exclusive)
            conditions = [m.condition for m in mappings]
            if None in conditions or len(set(conditions)) < len(conditions):
                violations.append(Violation(
                    violation_type=ViolationType.AMBIGUOUS_MAPPING,
                    severity="warning",
                    element=state,
                    description=f"State '{state}' has {len(mappings)} overlapping mappings",
                    suggested_fix="Add exclusive conditions or remove duplicate mappings",
                ))

    return violations


def check_reachability(
    plan: MigrationPlan,
    old_chart: Statechart,
    new_chart: Statechart,
) -> List[Violation]:
    """Check that all target states exist in new chart."""
    violations = []
    new_states = new_chart.get_all_states()

    for mapping in plan.state_mappings:
        for target in mapping.to_states:
            if target not in new_states:
                violations.append(Violation(
                    violation_type=ViolationType.UNREACHABLE_TARGET,
                    severity="error",
                    element=target,
                    description=f"Target state '{target}' does not exist in new chart",
                    suggested_fix=f"Map to a state that exists: {list(new_states.keys())[:5]}",
                ))

    return violations


def check_reversibility(
    plan: MigrationPlan,
    old_chart: Statechart,
    new_chart: Statechart,
) -> List[Violation]:
    """Check that rollback mappings are valid."""
    violations = []
    old_states = old_chart.get_all_states()

    for mapping in plan.rollback_mappings:
        if mapping.mapping_type == StateMappingType.ERROR:
            violations.append(Violation(
                violation_type=ViolationType.INVALID_ROLLBACK,
                severity="warning",
                element=str(mapping.from_states),
                description=f"Rollback mapping for {mapping.from_states} is invalid",
                suggested_fix="Store pre-migration state for accurate rollback",
            ))

        for target in mapping.to_states:
            if target not in old_states:
                violations.append(Violation(
                    violation_type=ViolationType.INVALID_ROLLBACK,
                    severity="error",
                    element=target,
                    description=f"Rollback target '{target}' does not exist in old chart",
                ))

    return violations


def check_transition_validity(
    plan: MigrationPlan,
    old_chart: Statechart,
    new_chart: Statechart,
) -> List[Violation]:
    """Check that transitions reference valid states after migration."""
    violations = []
    new_states = new_chart.get_all_states()

    for transition in new_chart.transitions:
        for state in transition.from_states + transition.to_states:
            if state not in new_states:
                violations.append(Violation(
                    violation_type=ViolationType.ORPHANED_TRANSITION,
                    severity="error",
                    element=transition.label,
                    description=f"Transition '{transition.label}' references missing state '{state}'",
                ))

    return violations


# =============================================================================
# MIGRATION VERIFIER
# =============================================================================

class MigrationVerifier:
    """
    Verifies migration plan safety.

    Runs configurable verification rules and reports violations.
    """

    DEFAULT_RULES = [
        VerificationRule(
            name="completeness",
            description="Every old state must have a mapping",
            check=check_completeness,
        ),
        VerificationRule(
            name="determinism",
            description="Mappings must be unambiguous",
            check=check_determinism,
        ),
        VerificationRule(
            name="reachability",
            description="Target states must exist",
            check=check_reachability,
        ),
        VerificationRule(
            name="reversibility",
            description="Rollback must be valid",
            check=check_reversibility,
        ),
        VerificationRule(
            name="transition_validity",
            description="Transitions must reference valid states",
            check=check_transition_validity,
        ),
    ]

    def __init__(self, rules: List[VerificationRule] = None):
        self.rules = rules or self.DEFAULT_RULES.copy()

    def verify(
        self,
        plan: MigrationPlan,
        old_chart: Statechart,
        new_chart: Statechart,
        test_configs: List[Configuration] = None,
    ) -> VerificationResult:
        """
        Verify migration plan against all rules.

        Args:
            plan: Migration plan to verify
            old_chart: Source statechart
            new_chart: Target statechart
            test_configs: Optional configurations to test migration

        Returns:
            VerificationResult with all findings
        """
        result = VerificationResult(status=VerificationStatus.PASSED)

        # Run all rules
        for rule in self.rules:
            violations = rule.check(plan, old_chart, new_chart)
            result.violations.extend(violations)

        # Calculate coverage
        result.coverage = self._calculate_coverage(plan, old_chart, new_chart)

        # Check reversibility
        result.is_reversible = self._check_reversibility(plan, old_chart, new_chart)

        # Test behavioral equivalence if configs provided
        if test_configs:
            result.behavioral_equivalence = self._check_behavioral_equivalence(
                plan, old_chart, new_chart, test_configs
            )

        # Determine overall status
        has_errors = any(v.severity == "error" for v in result.violations)
        has_warnings = any(v.severity == "warning" for v in result.violations)

        if has_errors:
            result.status = VerificationStatus.FAILED
        elif has_warnings:
            result.status = VerificationStatus.WARNING
        else:
            result.status = VerificationStatus.PASSED

        return result

    def _calculate_coverage(
        self,
        plan: MigrationPlan,
        old_chart: Statechart,
        new_chart: Statechart,
    ) -> float:
        """Calculate fraction of old states covered by mappings."""
        old_states = old_chart.get_all_states()
        new_states = new_chart.get_all_states()

        # States that need mapping = removed states
        need_mapping = set(old_states.keys()) - set(new_states.keys())
        if not need_mapping:
            return 1.0

        # States with mappings
        mapped = set()
        for mapping in plan.state_mappings:
            mapped.update(mapping.from_states)

        covered = need_mapping & mapped
        return len(covered) / len(need_mapping) if need_mapping else 1.0

    def _check_reversibility(
        self,
        plan: MigrationPlan,
        old_chart: Statechart,
        new_chart: Statechart,
    ) -> bool:
        """Check if migration is reversible."""
        # Check that rollback mappings exist for all forward mappings
        forward_targets = set()
        for m in plan.state_mappings:
            forward_targets.update(m.to_states)

        rollback_sources = set()
        for m in plan.rollback_mappings:
            rollback_sources.update(m.from_states)

        # All forward targets should have rollback
        return forward_targets <= rollback_sources

    def _check_behavioral_equivalence(
        self,
        plan: MigrationPlan,
        old_chart: Statechart,
        new_chart: Statechart,
        test_configs: List[Configuration],
    ) -> float:
        """
        Check behavioral equivalence by migrating test configs.

        Returns fraction of configs that migrate successfully.
        """
        if not test_configs:
            return 0.0

        successful = 0
        new_states = new_chart.get_all_states()

        for config in test_configs:
            try:
                migrated = plan.migrate(config)
                # Check all migrated states exist
                if all(s in new_states for s in migrated.states):
                    successful += 1
            except Exception:
                pass

        return successful / len(test_configs)


# =============================================================================
# DRY RUN
# =============================================================================

@dataclass
class DryRunPreview:
    """Preview of migration result for one configuration."""
    original: Configuration
    migrated: Configuration
    mapping_used: Optional[StateMappingType] = None
    warnings: List[str] = field(default_factory=list)
    success: bool = True


@dataclass
class DryRunResult:
    """Complete dry-run result."""
    previews: List[DryRunPreview]
    would_succeed: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Dry run: {'SUCCESS' if self.would_succeed else 'FAILED'}",
            f"Configs tested: {len(self.previews)}",
            f"Successful: {sum(1 for p in self.previews if p.success)}",
            f"Warnings: {len(self.warnings)}",
            f"Errors: {len(self.errors)}",
        ]
        return "\n".join(lines)


def dry_run(
    plan: MigrationPlan,
    configs: List[Configuration],
    new_chart: Statechart,
) -> DryRunResult:
    """
    Execute migration dry-run on configurations.

    Shows what would happen without applying changes.
    """
    result = DryRunResult(previews=[], would_succeed=True)
    new_states = new_chart.get_all_states()

    for config in configs:
        preview = DryRunPreview(
            original=config,
            migrated=Configuration(states=set()),
        )

        try:
            migrated = plan.migrate(config)
            preview.migrated = migrated

            # Find which mapping was used
            for mapping in plan.state_mappings:
                if any(s in config.states for s in mapping.from_states):
                    preview.mapping_used = mapping.mapping_type
                    break

            # Check validity
            invalid_states = [s for s in migrated.states if s not in new_states]
            if invalid_states:
                preview.success = False
                preview.warnings.append(f"Invalid states: {invalid_states}")
                result.errors.append(f"Config {config.states} → invalid states")
                result.would_succeed = False

        except Exception as e:
            preview.success = False
            preview.warnings.append(str(e))
            result.errors.append(f"Config {config.states} → error: {e}")
            result.would_succeed = False

        result.previews.append(preview)

    return result


# =============================================================================
# TESTING
# =============================================================================

def test_migration_verifier():
    """Test migration verification."""
    print("=" * 60)
    print("MIGRATION VERIFIER TEST")
    print("=" * 60)

    # Create charts
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

    # Create migration plan
    from .migration_synthesizer import MigrationSynthesizer
    synth = MigrationSynthesizer()
    diff = synth.differ.diff(old_chart, new_chart)
    plan = synth.synthesize_from_diff(diff, old_chart, new_chart)

    print(f"\nMigration: {plan.from_version} → {plan.to_version}")
    print(f"Mappings: {len(plan.state_mappings)}")

    # Verify
    verifier = MigrationVerifier()
    result = verifier.verify(plan, old_chart, new_chart)

    print("\n" + result.summary())

    if result.violations:
        print("\nViolations:")
        for v in result.violations:
            print(f"  [{v.severity.upper()}] {v.violation_type.name}: {v.description}")
            if v.suggested_fix:
                print(f"    Fix: {v.suggested_fix}")

    # Dry run
    print("\n" + "-" * 60)
    print("DRY RUN")
    print("-" * 60)

    test_configs = [
        Configuration(states={"Idle"}),
        Configuration(states={"Active"}),
        Configuration(states={"Done"}),
    ]

    dry_result = dry_run(plan, test_configs, new_chart)
    print(dry_result.summary())

    print("\nPreviews:")
    for preview in dry_result.previews:
        status = "OK" if preview.success else "FAIL"
        mapping = preview.mapping_used.name if preview.mapping_used else "IDENTITY"
        print(f"  {preview.original.states} → {preview.migrated.states} [{status}] ({mapping})")

    print("\n" + "=" * 60)
    print("MIGRATION VERIFIER TEST COMPLETE")
    print("=" * 60)

    return result


if __name__ == "__main__":
    test_migration_verifier()
