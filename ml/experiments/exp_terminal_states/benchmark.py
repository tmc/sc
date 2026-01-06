"""
Terminal State Benchmark - Test completion propagation semantics.

Benchmarks:
1. Completion detection accuracy
2. Completion chain length/depth
3. Parallel region synchronization
4. Edge cases: nested parallels, multiple finals

Tests verify that is_final semantics are correctly implemented:
- Final states have no outgoing transitions
- Parallel completes when ALL regions reach final
- Completion transitions fire automatically
"""

import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

from .terminal_state import State, StateType, Configuration, Transition, FinalStateValidator
from .completion_detector import CompletionDetector, CompletionEvent
from .parent_trigger import ParentTriggerEngine


@dataclass
class BenchmarkResult:
    """Results from a single benchmark test."""
    name: str
    passed: bool
    details: Dict[str, any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    results: List[BenchmarkResult]
    total_time: float

    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def pass_rate(self) -> float:
        return self.passed() / len(self.results) if self.results else 0.0


class StatechartGenerator:
    """Generate test statecharts with various structures."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.counter = 0

    def _unique_label(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def simple_parallel(self, n_regions: int = 2) -> Tuple[State, List[Transition]]:
        """Generate simple parallel state with n regions."""
        regions = []
        transitions = []

        for i in range(n_regions):
            active = State(
                label=self._unique_label(f"R{i}_Active"),
                is_initial=True,
            )
            final = State(
                label=self._unique_label(f"R{i}_Final"),
                is_final=True,
            )
            region = State(
                label=self._unique_label(f"Region{i}"),
                state_type=StateType.NORMAL,
                children=[active, final],
            )
            regions.append(region)
            transitions.append(Transition(
                source=[active.label],
                target=[final.label],
                event=f"done{i}",
            ))

        parallel = State(
            label=self._unique_label("Parallel"),
            state_type=StateType.PARALLEL,
            children=regions,
            is_initial=True,
        )

        done = State(label=self._unique_label("AllDone"), is_final=True)

        root = State(
            label=self._unique_label("Root"),
            state_type=StateType.NORMAL,
            children=[parallel, done],
        )

        # Completion transition
        transitions.append(Transition(
            source=[parallel.label],
            target=[done.label],
            event="",  # τ
        ))

        return root, transitions

    def nested_parallel(self, depth: int = 2) -> Tuple[State, List[Transition]]:
        """Generate nested parallel states."""
        transitions = []

        def build_level(d: int) -> State:
            if d == 0:
                # Leaf: simple OR state with final
                active = State(
                    label=self._unique_label("Leaf_Active"),
                    is_initial=True,
                )
                final = State(
                    label=self._unique_label("Leaf_Final"),
                    is_final=True,
                )
                transitions.append(Transition(
                    source=[active.label],
                    target=[final.label],
                    event=self._unique_label("done"),
                ))
                return State(
                    label=self._unique_label("LeafRegion"),
                    state_type=StateType.NORMAL,
                    children=[active, final],
                )
            else:
                # Parallel with sub-regions
                r1 = build_level(d - 1)
                r2 = build_level(d - 1)
                return State(
                    label=self._unique_label(f"Parallel_d{d}"),
                    state_type=StateType.PARALLEL,
                    children=[r1, r2],
                )

        root_parallel = build_level(depth)
        root_parallel.is_initial = True

        done = State(label=self._unique_label("Done"), is_final=True)

        root = State(
            label=self._unique_label("Root"),
            state_type=StateType.NORMAL,
            children=[root_parallel, done],
        )

        transitions.append(Transition(
            source=[root_parallel.label],
            target=[done.label],
            event="",
        ))

        return root, transitions

    def multiple_finals(self, n_finals: int = 3) -> Tuple[State, List[Transition]]:
        """Generate region with multiple possible final states."""
        children = []
        transitions = []

        active = State(
            label=self._unique_label("Active"),
            is_initial=True,
        )
        children.append(active)

        for i in range(n_finals):
            final = State(
                label=self._unique_label(f"Final{i}"),
                is_final=True,
            )
            children.append(final)
            transitions.append(Transition(
                source=[active.label],
                target=[final.label],
                event=f"end{i}",
            ))

        root = State(
            label=self._unique_label("Root"),
            state_type=StateType.NORMAL,
            children=children,
        )

        return root, transitions


class CompletionDetectionTests:
    """Test completion detection accuracy."""

    def __init__(self):
        self.gen = StatechartGenerator(seed=42)

    def test_simple_parallel_completion(self) -> BenchmarkResult:
        """Test: Parallel completes when all regions final."""
        root, transitions = self.gen.simple_parallel(n_regions=3)
        detector = CompletionDetector(root)

        # Get all final states
        finals = []
        for child in root.children[0].children:  # parallel's children
            for state in child.children:
                if state.is_final:
                    finals.append(state.label)

        # Progressively complete regions
        parallel_label = root.children[0].label
        active_states = {root.label, parallel_label}

        # Start with all regions active
        for region in root.children[0].children:
            active_states.add(region.label)
            for child in region.children:
                if child.is_initial:
                    active_states.add(child.label)

        # Complete one at a time
        completed_count = 0
        for i, region in enumerate(root.children[0].children):
            # Move region to final
            for child in region.children:
                if child.is_initial:
                    active_states.discard(child.label)
                if child.is_final:
                    active_states.add(child.label)

            config = Configuration(active_states=active_states.copy())
            events = detector.update(config)
            completed_count += 1

            is_complete = detector.is_parallel_complete(parallel_label)
            expected_complete = (completed_count == len(root.children[0].children))

            if is_complete != expected_complete:
                return BenchmarkResult(
                    name="simple_parallel_completion",
                    passed=False,
                    error=f"After {completed_count} regions: expected complete={expected_complete}, got {is_complete}",
                )

        return BenchmarkResult(
            name="simple_parallel_completion",
            passed=True,
            details={"n_regions": 3, "completed": completed_count},
        )

    def test_partial_completion(self) -> BenchmarkResult:
        """Test: Parallel NOT complete when only some regions final."""
        root, transitions = self.gen.simple_parallel(n_regions=4)
        detector = CompletionDetector(root)

        parallel_label = root.children[0].label

        # Complete only 2 of 4 regions
        active_states = {root.label, parallel_label}
        for i, region in enumerate(root.children[0].children):
            active_states.add(region.label)
            for child in region.children:
                if i < 2:  # First 2 regions complete
                    if child.is_final:
                        active_states.add(child.label)
                else:  # Last 2 regions still active
                    if child.is_initial:
                        active_states.add(child.label)

        config = Configuration(active_states=active_states)
        detector.update(config)

        is_complete = detector.is_parallel_complete(parallel_label)
        progress = detector.get_completion_progress(parallel_label)

        if is_complete:
            return BenchmarkResult(
                name="partial_completion",
                passed=False,
                error="Parallel reported complete with only 2/4 regions final",
            )

        return BenchmarkResult(
            name="partial_completion",
            passed=True,
            details={"progress": f"{progress[0]}/{progress[1]}"},
        )

    def test_nested_completion(self) -> BenchmarkResult:
        """Test: Nested parallel completion propagation."""
        root, transitions = self.gen.nested_parallel(depth=2)

        # This creates: Parallel(Parallel(Leaf, Leaf), Parallel(Leaf, Leaf))
        # All 4 leaves must complete for root parallel to complete

        engine = ParentTriggerEngine(root, transitions)
        engine.initialize()

        # Find all done events
        done_events = [t.event for t in transitions if t.event.startswith("done")]

        # Fire all done events
        for event in done_events:
            engine.process_event(event)

        # Check if root's parallel child completed
        status = engine.get_state_status()
        root_done = any("Done" in label and status[label] == "FINAL"
                       for label in status)

        return BenchmarkResult(
            name="nested_completion",
            passed=root_done,
            details={
                "events_fired": len(done_events),
                "reached_final": root_done,
            },
        )


class CompletionTriggerTests:
    """Test completion transition triggering."""

    def __init__(self):
        self.gen = StatechartGenerator(seed=43)

    def test_completion_triggers_transition(self) -> BenchmarkResult:
        """Test: Completion automatically triggers parent transition."""
        root, transitions = self.gen.simple_parallel(n_regions=2)

        engine = ParentTriggerEngine(root, transitions)
        engine.initialize()

        # Find the done events
        done_events = sorted([t.event for t in transitions if t.event.startswith("done")])

        # Fire all done events
        all_executions = []
        for event in done_events:
            execs = engine.process_event(event)
            all_executions.extend(execs)

        # Check that completion transition was fired
        completion_fired = any(
            exec.triggered_by == "completion"
            for exec in all_executions
        )

        # Check final state reached
        status = engine.get_state_status()
        final_reached = any("AllDone" in label or "Done" in label
                          for label, s in status.items() if s == "FINAL")

        return BenchmarkResult(
            name="completion_triggers_transition",
            passed=completion_fired and final_reached,
            details={
                "total_transitions": len(all_executions),
                "completion_transitions": sum(1 for e in all_executions
                                             if e.triggered_by == "completion"),
                "final_reached": final_reached,
            },
        )

    def test_completion_chain(self) -> BenchmarkResult:
        """Test: Completion chains propagate correctly."""
        root, transitions = self.gen.nested_parallel(depth=2)

        engine = ParentTriggerEngine(root, transitions)
        engine.initialize()

        # Fire all done events
        done_events = [t.event for t in transitions if t.event.startswith("done")]
        all_executions = []

        for event in done_events:
            execs = engine.process_event(event)
            all_executions.extend(execs)

        # Count completion transitions
        completion_count = sum(1 for e in all_executions
                              if e.triggered_by == "completion")

        # With depth=2, we have 4 leaves and should have completion chain
        return BenchmarkResult(
            name="completion_chain",
            passed=completion_count >= 1,  # At least the final completion
            details={
                "leaf_events": len(done_events),
                "completion_transitions": completion_count,
                "total_transitions": len(all_executions),
            },
        )


class ValidationTests:
    """Test final state validation."""

    def __init__(self):
        self.gen = StatechartGenerator(seed=44)

    def test_no_outgoing_from_final(self) -> BenchmarkResult:
        """Test: Final states should have no outgoing transitions."""
        root, transitions = self.gen.simple_parallel(n_regions=2)

        # Add invalid transition from final state
        for region in root.children[0].children:
            for child in region.children:
                if child.is_final:
                    # This is invalid!
                    transitions.append(Transition(
                        source=[child.label],
                        target=["SomeState"],
                        event="invalid",
                    ))
                    break
            break

        validator = FinalStateValidator()
        errors = validator.validate_no_outgoing(root, transitions)

        return BenchmarkResult(
            name="no_outgoing_from_final",
            passed=len(errors) > 0,  # Should detect invalid transition
            details={
                "errors_found": len(errors),
            },
        )

    def test_parallel_has_finals(self) -> BenchmarkResult:
        """Test: Each parallel region should have final states."""
        # Create invalid parallel: one region has no final
        active1 = State(label="R1_Active", is_initial=True)
        final1 = State(label="R1_Final", is_final=True)
        region1 = State(
            label="Region1",
            state_type=StateType.NORMAL,
            children=[active1, final1],
        )

        # Region2 has NO final state
        active2 = State(label="R2_Active", is_initial=True)
        working2 = State(label="R2_Working")  # No final!
        region2 = State(
            label="Region2",
            state_type=StateType.NORMAL,
            children=[active2, working2],
        )

        parallel = State(
            label="Parallel",
            state_type=StateType.PARALLEL,
            children=[region1, region2],
        )

        validator = FinalStateValidator()
        errors = validator.validate_parallel_finals(parallel)

        return BenchmarkResult(
            name="parallel_has_finals",
            passed=len(errors) > 0,  # Should detect missing final
            details={
                "errors_found": len(errors),
                "error_messages": errors[:2],
            },
        )


def run_benchmark() -> BenchmarkSuite:
    """Run complete benchmark suite."""
    start_time = time.time()
    results = []

    # Completion detection tests
    detection_tests = CompletionDetectionTests()
    results.append(detection_tests.test_simple_parallel_completion())
    results.append(detection_tests.test_partial_completion())
    results.append(detection_tests.test_nested_completion())

    # Completion trigger tests
    trigger_tests = CompletionTriggerTests()
    results.append(trigger_tests.test_completion_triggers_transition())
    results.append(trigger_tests.test_completion_chain())

    # Validation tests
    validation_tests = ValidationTests()
    results.append(validation_tests.test_no_outgoing_from_final())
    results.append(validation_tests.test_parallel_has_finals())

    total_time = time.time() - start_time

    return BenchmarkSuite(results=results, total_time=total_time)


def demo():
    """Run benchmark demo."""
    print("=" * 60)
    print("TERMINAL STATE BENCHMARK")
    print("=" * 60)

    suite = run_benchmark()

    print(f"\nResults: {suite.passed()}/{len(suite.results)} tests passed")
    print(f"Time: {suite.total_time*1000:.1f}ms")
    print("-" * 60)

    for result in suite.results:
        status = "PASS" if result.passed else "FAIL"
        print(f"\n[{status}] {result.name}")
        if result.error:
            print(f"  Error: {result.error}")
        for key, value in result.details.items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print(f"SUMMARY: {suite.pass_rate()*100:.0f}% pass rate")
    print("=" * 60)

    return suite


if __name__ == "__main__":
    demo()
