"""
Context Tracker: Track context mutations and guard evaluations.

This module provides infrastructure for observing how actions modify context
and how those modifications affect guard evaluations.

The key insight is that we need to OBSERVE the relationships before we can
LEARN them - this module handles the observation collection.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Any, Optional, Callable, Tuple
from collections import defaultdict
import copy


@dataclass
class ContextDiff:
    """
    Difference between two context states.

    Captures what changed, how it changed, and the delta.
    """
    variable: str
    old_value: Any
    new_value: Any

    @property
    def changed(self) -> bool:
        return self.old_value != self.new_value

    @property
    def delta(self) -> Any:
        """Compute delta if numeric."""
        if isinstance(self.old_value, (int, float)) and isinstance(self.new_value, (int, float)):
            return self.new_value - self.old_value
        return None

    @property
    def was_set(self) -> bool:
        """Variable was set (didn't exist before)."""
        return self.old_value is None and self.new_value is not None

    @property
    def was_cleared(self) -> bool:
        """Variable was cleared (exists before, not after)."""
        return self.old_value is not None and self.new_value is None

    @property
    def was_toggled(self) -> bool:
        """Boolean value was toggled."""
        if isinstance(self.old_value, bool) and isinstance(self.new_value, bool):
            return self.old_value != self.new_value
        return False

    @property
    def was_incremented(self) -> bool:
        """Numeric value was incremented."""
        delta = self.delta
        return delta is not None and delta > 0

    @property
    def was_decremented(self) -> bool:
        """Numeric value was decremented."""
        delta = self.delta
        return delta is not None and delta < 0

    def __str__(self) -> str:
        if self.was_toggled:
            return f"{self.variable}: {self.old_value} -> {self.new_value} (toggled)"
        elif self.was_incremented:
            return f"{self.variable}: {self.old_value} -> {self.new_value} (+{self.delta})"
        elif self.was_decremented:
            return f"{self.variable}: {self.old_value} -> {self.new_value} ({self.delta})"
        elif self.was_set:
            return f"{self.variable}: None -> {self.new_value} (set)"
        elif self.was_cleared:
            return f"{self.variable}: {self.old_value} -> None (cleared)"
        else:
            return f"{self.variable}: {self.old_value} -> {self.new_value}"


@dataclass
class GuardEvaluation:
    """
    Evaluation of a guard in a context.
    """
    guard_name: str
    result: bool
    context_snapshot: Dict[str, Any]
    variables_accessed: Set[str] = field(default_factory=set)


@dataclass
class ActionExecution:
    """
    Record of an action execution.
    """
    action_name: str
    context_before: Dict[str, Any]
    context_after: Dict[str, Any]
    diffs: List[ContextDiff] = field(default_factory=list)
    guards_before: Dict[str, bool] = field(default_factory=dict)
    guards_after: Dict[str, bool] = field(default_factory=dict)

    def __post_init__(self):
        if not self.diffs:
            self._compute_diffs()

    def _compute_diffs(self):
        """Compute diffs between before and after contexts."""
        all_vars = set(self.context_before.keys()) | set(self.context_after.keys())
        for var in all_vars:
            old_val = self.context_before.get(var)
            new_val = self.context_after.get(var)
            diff = ContextDiff(variable=var, old_value=old_val, new_value=new_val)
            if diff.changed:
                self.diffs.append(diff)

    @property
    def modified_variables(self) -> Set[str]:
        """Variables that were modified by this action."""
        return {d.variable for d in self.diffs}

    @property
    def enabled_guards(self) -> Set[str]:
        """Guards that became enabled (False -> True)."""
        enabled = set()
        for guard, after in self.guards_after.items():
            before = self.guards_before.get(guard, False)
            if not before and after:
                enabled.add(guard)
        return enabled

    @property
    def disabled_guards(self) -> Set[str]:
        """Guards that became disabled (True -> False)."""
        disabled = set()
        for guard, after in self.guards_after.items():
            before = self.guards_before.get(guard, True)
            if before and not after:
                disabled.add(guard)
        return disabled


class ContextTracker:
    """
    Track context mutations across action executions.

    Provides observation infrastructure for learning action->guard dependencies.
    """

    def __init__(
        self,
        guard_evaluators: Dict[str, Callable[[Dict[str, Any]], bool]] = None,
    ):
        """
        Initialize tracker.

        Args:
            guard_evaluators: Dict mapping guard_name -> evaluation function
        """
        self.guard_evaluators = guard_evaluators or {}
        self.executions: List[ActionExecution] = []

        # Statistics
        self.action_counts: Dict[str, int] = defaultdict(int)
        self.variable_modification_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.guard_change_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def register_guard(self, name: str, evaluator: Callable[[Dict[str, Any]], bool]):
        """Register a guard evaluator."""
        self.guard_evaluators[name] = evaluator

    def _evaluate_all_guards(self, context: Dict[str, Any]) -> Dict[str, bool]:
        """Evaluate all registered guards in context."""
        results = {}
        for name, evaluator in self.guard_evaluators.items():
            try:
                results[name] = evaluator(context)
            except Exception:
                results[name] = False
        return results

    def track_action(
        self,
        action_name: str,
        context_before: Dict[str, Any],
        context_after: Dict[str, Any],
    ) -> ActionExecution:
        """
        Track an action execution.

        Args:
            action_name: Name of the action
            context_before: Context state before action
            context_after: Context state after action

        Returns:
            ActionExecution record
        """
        # Evaluate guards before and after
        guards_before = self._evaluate_all_guards(context_before)
        guards_after = self._evaluate_all_guards(context_after)

        execution = ActionExecution(
            action_name=action_name,
            context_before=copy.deepcopy(context_before),
            context_after=copy.deepcopy(context_after),
            guards_before=guards_before,
            guards_after=guards_after,
        )

        self.executions.append(execution)

        # Update statistics
        self.action_counts[action_name] += 1
        for var in execution.modified_variables:
            self.variable_modification_counts[action_name][var] += 1
        for guard in execution.enabled_guards | execution.disabled_guards:
            self.guard_change_counts[action_name][guard] += 1

        return execution

    def track_action_fn(
        self,
        action_name: str,
        action_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Tuple[ActionExecution, Dict[str, Any]]:
        """
        Track an action by executing it.

        Args:
            action_name: Name of the action
            action_fn: Function that takes context and returns modified context
            context: Current context

        Returns:
            (ActionExecution record, new context)
        """
        context_before = copy.deepcopy(context)
        context_after = action_fn(copy.deepcopy(context))

        execution = self.track_action(action_name, context_before, context_after)
        return execution, context_after

    def get_action_variable_correlations(self) -> Dict[str, Dict[str, float]]:
        """
        Get correlation between actions and variables they modify.

        Returns:
            Dict mapping action_name -> {variable: modification_frequency}
        """
        correlations = {}
        for action, var_counts in self.variable_modification_counts.items():
            total = self.action_counts[action]
            correlations[action] = {
                var: count / total if total > 0 else 0.0
                for var, count in var_counts.items()
            }
        return correlations

    def get_action_guard_correlations(self) -> Dict[str, Dict[str, float]]:
        """
        Get correlation between actions and guards they affect.

        Returns:
            Dict mapping action_name -> {guard: change_frequency}
        """
        correlations = {}
        for action, guard_counts in self.guard_change_counts.items():
            total = self.action_counts[action]
            correlations[action] = {
                guard: count / total if total > 0 else 0.0
                for guard, count in guard_counts.items()
            }
        return correlations

    def get_variable_guard_correlations(self) -> Dict[str, Set[str]]:
        """
        Get correlation between variables and guards.

        Based on: when variable X changes, which guards change?

        Returns:
            Dict mapping variable -> set of correlated guards
        """
        var_guard_corr: Dict[str, Set[str]] = defaultdict(set)

        for execution in self.executions:
            changed_guards = execution.enabled_guards | execution.disabled_guards
            for var in execution.modified_variables:
                var_guard_corr[var].update(changed_guards)

        return dict(var_guard_corr)

    def infer_guard_dependencies(self) -> Dict[str, Set[str]]:
        """
        Infer which variables each guard depends on.

        Based on: when guard changes, which variables also changed?

        Returns:
            Dict mapping guard -> set of variables it likely depends on
        """
        guard_var_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        guard_change_total: Dict[str, int] = defaultdict(int)

        for execution in self.executions:
            changed_guards = execution.enabled_guards | execution.disabled_guards
            for guard in changed_guards:
                guard_change_total[guard] += 1
                for var in execution.modified_variables:
                    guard_var_counts[guard][var] += 1

        # Filter to variables that change >50% of the time guard changes
        guard_deps = {}
        for guard, var_counts in guard_var_counts.items():
            total = guard_change_total[guard]
            deps = set()
            for var, count in var_counts.items():
                if total > 0 and count / total > 0.5:
                    deps.add(var)
            guard_deps[guard] = deps

        return guard_deps

    def get_observations_for_evolver(self) -> List["ActionObservation"]:
        """
        Convert tracked executions to observations for ActionEvolver.

        Returns:
            List of ActionObservation objects
        """
        from .action_evolver import ActionObservation

        observations = []
        for execution in self.executions:
            obs = ActionObservation(
                action_name=execution.action_name,
                context_before=execution.context_before,
                context_after=execution.context_after,
                guards_before=execution.guards_before,
                guards_after=execution.guards_after,
            )
            observations.append(obs)

        return observations

    def print_statistics(self):
        """Print tracking statistics."""
        print("\n" + "=" * 60)
        print("CONTEXT TRACKER STATISTICS")
        print("=" * 60)

        print(f"\nTotal executions tracked: {len(self.executions)}")

        print("\nAction counts:")
        for action, count in sorted(self.action_counts.items()):
            print(f"  {action}: {count}")

        print("\nAction -> Variable correlations:")
        corr = self.get_action_variable_correlations()
        for action, var_freq in corr.items():
            if var_freq:
                print(f"  {action}:")
                for var, freq in sorted(var_freq.items(), key=lambda x: -x[1])[:5]:
                    print(f"    -> {var}: {freq:.2f}")

        print("\nInferred guard dependencies:")
        deps = self.infer_guard_dependencies()
        for guard, vars in deps.items():
            if vars:
                print(f"  {guard} depends on: {', '.join(vars)}")

        print("=" * 60)


class InstrumentedContext:
    """
    A context wrapper that tracks all variable accesses and modifications.

    Use this to automatically detect which variables guards actually read.
    """

    def __init__(self, initial_data: Dict[str, Any] = None):
        self._data = initial_data or {}
        self._read_vars: Set[str] = set()
        self._written_vars: Set[str] = set()
        self._tracking = True

    def __getitem__(self, key: str) -> Any:
        if self._tracking:
            self._read_vars.add(key)
        return self._data.get(key)

    def __setitem__(self, key: str, value: Any):
        if self._tracking:
            self._written_vars.add(key)
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        if self._tracking:
            self._read_vars.add(key)
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        if self._tracking:
            self._read_vars.add(key)
        return self._data.get(key, default)

    def get_read_vars(self) -> Set[str]:
        """Get variables that were read."""
        return self._read_vars.copy()

    def get_written_vars(self) -> Set[str]:
        """Get variables that were written."""
        return self._written_vars.copy()

    def reset_tracking(self):
        """Reset tracking state."""
        self._read_vars = set()
        self._written_vars = set()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to regular dict."""
        return self._data.copy()


def demo():
    """Demonstrate context tracking."""
    print("=" * 60)
    print("CONTEXT TRACKER DEMO")
    print("=" * 60)

    # Define guards
    def is_positive(ctx: Dict) -> bool:
        return ctx.get("value", 0) > 0

    def is_even(ctx: Dict) -> bool:
        return ctx.get("value", 0) % 2 == 0

    def is_ready(ctx: Dict) -> bool:
        return ctx.get("initialized", False) and ctx.get("value", 0) >= 0

    # Create tracker
    tracker = ContextTracker(guard_evaluators={
        "is_positive": is_positive,
        "is_even": is_even,
        "is_ready": is_ready,
    })

    # Define actions
    def increment_action(ctx: Dict) -> Dict:
        ctx = ctx.copy()
        ctx["value"] = ctx.get("value", 0) + 1
        return ctx

    def initialize_action(ctx: Dict) -> Dict:
        ctx = ctx.copy()
        ctx["initialized"] = True
        ctx["value"] = 0
        return ctx

    def double_action(ctx: Dict) -> Dict:
        ctx = ctx.copy()
        ctx["value"] = ctx.get("value", 0) * 2
        return ctx

    # Execute and track actions
    context = {"value": -1, "initialized": False}

    print("\nInitial context:", context)

    execution, context = tracker.track_action_fn("initialize", initialize_action, context)
    print(f"\nAfter initialize: {context}")
    print(f"  Modified: {execution.modified_variables}")
    print(f"  Enabled guards: {execution.enabled_guards}")

    for _ in range(5):
        execution, context = tracker.track_action_fn("increment", increment_action, context)

    print(f"\nAfter 5 increments: {context}")

    execution, context = tracker.track_action_fn("double", double_action, context)
    print(f"\nAfter double: {context}")

    # Print statistics
    tracker.print_statistics()

    return tracker


if __name__ == "__main__":
    demo()
