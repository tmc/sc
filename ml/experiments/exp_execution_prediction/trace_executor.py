"""
Trace Executor: Run statecharts to completion with context tracking.

Executes a statechart with given events, tracking:
- State configuration at each step
- Context values after each action
- Guard evaluations
- Terminal state detection
"""

import json
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from grammars.dynamic_constrained_sampler import SCDefinition, SCExecutor, Configuration
from .starlark_eval import eval_guard, eval_action


@dataclass
class ExecutionStep:
    """Single step in execution trace."""
    step_number: int
    event: str
    source_config: Set[str]
    target_config: Set[str]
    context_before: Dict[str, Any]
    context_after: Dict[str, Any]
    guards_evaluated: List[Tuple[str, bool]]  # (expression, result)
    actions_executed: List[str]


@dataclass
class ExecutionResult:
    """Complete execution result."""
    success: bool
    initial_config: Set[str]
    final_config: Set[str]
    initial_context: Dict[str, Any]
    final_context: Dict[str, Any]
    steps: List[ExecutionStep]
    is_terminal: bool
    error: Optional[str] = None

    @property
    def path_length(self) -> int:
        return len(self.steps)

    @property
    def states_visited(self) -> Set[str]:
        visited = set(self.initial_config)
        for step in self.steps:
            visited.update(step.target_config)
        return visited


class TraceExecutor:
    """Executes statecharts with full context tracking."""

    def __init__(self, sc_json: Dict):
        self.sc_json = sc_json
        self.sc_def = SCDefinition.from_json(sc_json)
        self.executor = SCExecutor(self.sc_def)

        # Extract action/guard info from transitions
        self._parse_transitions()

    def _parse_transitions(self):
        """Parse transitions to extract actions and guards."""
        self.transition_guards = {}  # (from, to, event) -> guard_expr
        self.transition_actions = {}  # (from, to, event) -> [action_exprs]

        for t in self.sc_json.get("transitions", []):
            from_states = tuple(t.get("from", []))
            to_states = tuple(t.get("to", []))
            event = t.get("event", "")

            key = (from_states, to_states, event)

            # Guard
            guard = t.get("guard")
            if guard:
                if isinstance(guard, dict):
                    self.transition_guards[key] = guard.get("expression", "")
                else:
                    self.transition_guards[key] = guard

            # Actions
            actions = t.get("actions", [])
            action_exprs = []
            for a in actions:
                if isinstance(a, dict):
                    action_exprs.append(a.get("expression", ""))
                else:
                    action_exprs.append(a)
            if action_exprs:
                self.transition_actions[key] = action_exprs

    def execute(
        self,
        events: List[str],
        initial_context: Optional[Dict[str, Any]] = None,
        max_steps: int = 100,
    ) -> ExecutionResult:
        """
        Execute statechart with given events.

        Args:
            events: Sequence of events to process
            initial_context: Starting context values
            max_steps: Maximum steps before stopping

        Returns:
            ExecutionResult with full trace
        """
        # Initialize
        config = self.executor.initial_config()
        context = initial_context.copy() if initial_context else {}

        # Get initial context from SC if not provided
        if not context and "initial_context" in self.sc_json:
            context = self.sc_json["initial_context"].copy()

        steps = []
        event_idx = 0

        try:
            while event_idx < len(events) and len(steps) < max_steps:
                event = events[event_idx]
                source_config = config.active_states.copy()
                context_before = context.copy()

                # Find matching transition
                new_config, guards_eval, actions_exec = self._step_with_tracking(
                    config, event, context
                )

                if new_config is None:
                    # No valid transition - event ignored
                    event_idx += 1
                    continue

                # Execute actions to update context
                context_after = context.copy()
                for action_expr in actions_exec:
                    context_after = eval_action(action_expr, context_after)

                # Record step
                step = ExecutionStep(
                    step_number=len(steps),
                    event=event,
                    source_config=source_config,
                    target_config=new_config.active_states.copy(),
                    context_before=context_before,
                    context_after=context_after,
                    guards_evaluated=guards_eval,
                    actions_executed=actions_exec,
                )
                steps.append(step)

                # Update state
                config = new_config
                context = context_after
                event_idx += 1

                # Check terminal
                if self._is_terminal(config):
                    break

            return ExecutionResult(
                success=True,
                initial_config=self.executor.initial_config().active_states,
                final_config=config.active_states,
                initial_context=initial_context or {},
                final_context=context,
                steps=steps,
                is_terminal=self._is_terminal(config),
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                initial_config=self.executor.initial_config().active_states,
                final_config=config.active_states if config else set(),
                initial_context=initial_context or {},
                final_context=context,
                steps=steps,
                is_terminal=False,
                error=str(e),
            )

    def _step_with_tracking(
        self,
        config: Configuration,
        event: str,
        context: Dict[str, Any],
    ) -> Tuple[Optional[Configuration], List[Tuple[str, bool]], List[str]]:
        """
        Execute one step with guard/action tracking.

        Returns: (new_config, guards_evaluated, actions_to_execute)
        """
        guards_evaluated = []
        actions_to_execute = []

        # Find transitions that match source and event
        for t in self.sc_json.get("transitions", []):
            from_states = set(t.get("from", []))
            to_states = t.get("to", [])
            t_event = t.get("event", "")

            if t_event != event:
                continue

            if not from_states.intersection(config.active_states):
                continue

            # Evaluate guard if present
            guard = t.get("guard")
            guard_result = True
            if guard:
                guard_expr = guard.get("expression", "") if isinstance(guard, dict) else guard
                guard_result = eval_guard(guard_expr, context)
                guards_evaluated.append((guard_expr, guard_result))

            if not guard_result:
                continue

            # Found matching transition
            # Collect actions
            actions = t.get("actions", [])
            for a in actions:
                expr = a.get("expression", "") if isinstance(a, dict) else a
                if expr:
                    actions_to_execute.append(expr)

            # Use base executor for state transition
            new_config = self.executor.step(config, event)
            return new_config, guards_evaluated, actions_to_execute

        return None, guards_evaluated, []

    def _is_terminal(self, config: Configuration) -> bool:
        """Check if configuration is terminal (all leaf states are final)."""
        for state_label in config.active_states:
            state_info = self.sc_def.states.get(state_label, {})
            if state_info.get("is_final"):
                return True
        return False


def execute_to_completion(
    sc_json: Dict,
    events: List[str],
    initial_context: Optional[Dict[str, Any]] = None,
) -> ExecutionResult:
    """Convenience function to execute SC to completion."""
    executor = TraceExecutor(sc_json)
    return executor.execute(events, initial_context)


if __name__ == "__main__":
    from .sc_templates import CounterMachine, BranchingMachine

    print("Trace Executor Test")
    print("=" * 60)

    # Test Counter
    print("\nCounter Machine:")
    counter = CounterMachine(target=3)
    events = ["BEGIN", "TICK", "TICK", "TICK", "TICK"]
    result = execute_to_completion(counter.to_json(), events, counter.initial_context)

    print(f"  Events: {events}")
    print(f"  Final config: {result.final_config}")
    print(f"  Final context: {result.final_context}")
    print(f"  Path length: {result.path_length}")
    print(f"  Is terminal: {result.is_terminal}")

    # Test Branching
    print("\nBranching Machine (score=75):")
    branching = BranchingMachine(threshold=50)
    branching.initial_context["score"] = 75
    events = ["EVALUATE", "CONTINUE", "FINISH"]
    result = execute_to_completion(branching.to_json(), events, branching.initial_context)

    print(f"  Events: {events}")
    print(f"  Final config: {result.final_config}")
    print(f"  Final context: {result.final_context}")
    print(f"  Path: {' -> '.join(str(s.target_config) for s in result.steps)}")
