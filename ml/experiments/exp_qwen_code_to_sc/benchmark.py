"""
Benchmark - Test structure recovery from code to statechart.

Tests extraction accuracy on various code patterns:
1. Simple switch/case state machines
2. If/else state handling
3. Nested state patterns
4. Event-driven handlers
5. Complex multi-state systems

Target: 90%+ structure recovery (states, transitions, events).
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional

from .code_analyzer import MultiLanguageAnalyzer, CodeLanguage
from .pattern_extractor import RuleBasedExtractor, HybridExtractor, ExtractionResult
from .sc_builder import StatechartBuilder, StatechartValidator, StatechartSerializer


@dataclass
class TestCase:
    """A benchmark test case."""
    name: str
    code: str
    expected_states: Set[str]
    expected_transitions: List[Tuple[str, str]]  # (from, to) pairs
    expected_initial: Optional[str] = None
    expected_final: Set[str] = field(default_factory=set)
    language: CodeLanguage = CodeLanguage.PYTHON


@dataclass
class RecoveryMetrics:
    """Metrics for structure recovery."""
    states_expected: int
    states_found: int
    states_correct: int
    transitions_expected: int
    transitions_found: int
    transitions_correct: int
    initial_correct: bool
    final_correct: bool

    @property
    def state_precision(self) -> float:
        return self.states_correct / self.states_found if self.states_found else 0.0

    @property
    def state_recall(self) -> float:
        return self.states_correct / self.states_expected if self.states_expected else 0.0

    @property
    def transition_precision(self) -> float:
        return self.transitions_correct / self.transitions_found if self.transitions_found else 0.0

    @property
    def transition_recall(self) -> float:
        return self.transitions_correct / self.transitions_expected if self.transitions_expected else 0.0

    @property
    def overall_score(self) -> float:
        """Overall structure recovery score (0-1)."""
        scores = [
            self.state_recall,
            self.transition_recall,
            1.0 if self.initial_correct else 0.0,
        ]
        return sum(scores) / len(scores)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark test."""
    test_case: TestCase
    metrics: RecoveryMetrics
    extraction_time_ms: float
    build_time_ms: float
    validation_errors: List[str]
    mermaid_output: str = ""


@dataclass
class BenchmarkSuite:
    """Complete benchmark results."""
    results: List[BenchmarkResult]
    total_time: float

    @property
    def avg_state_recall(self) -> float:
        return sum(r.metrics.state_recall for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def avg_transition_recall(self) -> float:
        return sum(r.metrics.transition_recall for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def avg_overall(self) -> float:
        return sum(r.metrics.overall_score for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def passing_tests(self) -> int:
        """Tests with 90%+ overall score."""
        return sum(1 for r in self.results if r.metrics.overall_score >= 0.9)


# Test cases for benchmarking
TEST_CASES = [
    TestCase(
        name="simple_traffic_light",
        code='''
def traffic_light():
    state = "red"
    while True:
        if state == "red":
            wait(30)
            state = "green"
        elif state == "green":
            wait(25)
            state = "yellow"
        elif state == "yellow":
            wait(5)
            state = "red"
''',
        expected_states={"red", "green", "yellow"},
        expected_transitions=[("red", "green"), ("green", "yellow"), ("yellow", "red")],
        expected_initial="red",
    ),
    TestCase(
        name="player_state_machine",
        code='''
def handle_player(event):
    state = "idle"
    if state == "idle":
        if event == "move":
            state = "walking"
        elif event == "attack":
            state = "attacking"
    elif state == "walking":
        if event == "stop":
            state = "idle"
        elif event == "attack":
            state = "attacking"
    elif state == "attacking":
        if event == "done":
            state = "idle"
''',
        expected_states={"idle", "walking", "attacking"},
        expected_transitions=[
            ("idle", "walking"),
            ("idle", "attacking"),
            ("walking", "idle"),
            ("walking", "attacking"),
            ("attacking", "idle"),
        ],
        expected_initial="idle",
    ),
    TestCase(
        name="order_workflow",
        code='''
def process_order():
    status = "pending"
    if status == "pending":
        if payment_received():
            status = "paid"
    elif status == "paid":
        if shipped():
            status = "shipped"
    elif status == "shipped":
        if delivered():
            status = "delivered"
    elif status == "delivered":
        pass  # Final state
''',
        expected_states={"pending", "paid", "shipped", "delivered"},
        expected_transitions=[
            ("pending", "paid"),
            ("paid", "shipped"),
            ("shipped", "delivered"),
        ],
        expected_initial="pending",
        expected_final={"delivered"},
    ),
    TestCase(
        name="c_switch_traffic",
        code='''
switch (light_state) {
    case RED:
        if (timer_expired) {
            light_state = GREEN;
            start_timer(25);
        }
        break;
    case GREEN:
        if (timer_expired) {
            light_state = YELLOW;
            start_timer(5);
        }
        break;
    case YELLOW:
        if (timer_expired) {
            light_state = RED;
            start_timer(30);
        }
        break;
}
''',
        expected_states={"RED", "GREEN", "YELLOW"},
        expected_transitions=[("RED", "GREEN"), ("GREEN", "YELLOW"), ("YELLOW", "RED")],
        language=CodeLanguage.C_LIKE,
    ),
    TestCase(
        name="connection_state",
        code='''
def connection_handler():
    state = "disconnected"

    if state == "disconnected":
        if connect_request():
            state = "connecting"
    elif state == "connecting":
        if connected():
            state = "connected"
        elif timeout():
            state = "disconnected"
    elif state == "connected":
        if disconnect_request():
            state = "disconnecting"
        elif error():
            state = "disconnected"
    elif state == "disconnecting":
        if disconnected():
            state = "disconnected"
''',
        expected_states={"disconnected", "connecting", "connected", "disconnecting"},
        expected_transitions=[
            ("disconnected", "connecting"),
            ("connecting", "connected"),
            ("connecting", "disconnected"),
            ("connected", "disconnecting"),
            ("connected", "disconnected"),
            ("disconnecting", "disconnected"),
        ],
        expected_initial="disconnected",
    ),
    TestCase(
        name="media_player",
        code='''
def media_player(event):
    current_state = "stopped"

    if current_state == "stopped":
        if event == "play":
            current_state = "playing"
    elif current_state == "playing":
        if event == "pause":
            current_state = "paused"
        elif event == "stop":
            current_state = "stopped"
    elif current_state == "paused":
        if event == "play":
            current_state = "playing"
        elif event == "stop":
            current_state = "stopped"
''',
        expected_states={"stopped", "playing", "paused"},
        expected_transitions=[
            ("stopped", "playing"),
            ("playing", "paused"),
            ("playing", "stopped"),
            ("paused", "playing"),
            ("paused", "stopped"),
        ],
        expected_initial="stopped",
    ),
    TestCase(
        name="login_flow",
        code='''
def login_handler():
    state = "logged_out"

    if state == "logged_out":
        if credentials_valid():
            state = "logged_in"
        elif needs_2fa():
            state = "awaiting_2fa"
    elif state == "awaiting_2fa":
        if code_valid():
            state = "logged_in"
        elif timeout():
            state = "logged_out"
    elif state == "logged_in":
        if logout():
            state = "logged_out"
        elif session_expired():
            state = "logged_out"
''',
        expected_states={"logged_out", "awaiting_2fa", "logged_in"},
        expected_transitions=[
            ("logged_out", "logged_in"),
            ("logged_out", "awaiting_2fa"),
            ("awaiting_2fa", "logged_in"),
            ("awaiting_2fa", "logged_out"),
            ("logged_in", "logged_out"),
        ],
        expected_initial="logged_out",
    ),
    TestCase(
        name="document_lifecycle",
        code='''
switch (doc_state) {
    case DRAFT:
        if (submit) {
            doc_state = REVIEW;
            notify_reviewer();
        }
        break;
    case REVIEW:
        if (approve) {
            doc_state = APPROVED;
        } else if (reject) {
            doc_state = DRAFT;
        }
        break;
    case APPROVED:
        if (publish) {
            doc_state = PUBLISHED;
        } else if (archive) {
            doc_state = ARCHIVED;
        }
        break;
    case PUBLISHED:
        if (archive) {
            doc_state = ARCHIVED;
        }
        break;
    case ARCHIVED:
        // Final state
        break;
}
''',
        expected_states={"DRAFT", "REVIEW", "APPROVED", "PUBLISHED", "ARCHIVED"},
        expected_transitions=[
            ("DRAFT", "REVIEW"),
            ("REVIEW", "APPROVED"),
            ("REVIEW", "DRAFT"),
            ("APPROVED", "PUBLISHED"),
            ("APPROVED", "ARCHIVED"),
            ("PUBLISHED", "ARCHIVED"),
        ],
        expected_final={"ARCHIVED"},
        language=CodeLanguage.C_LIKE,
    ),
]


class StructureRecoveryBenchmark:
    """Benchmark structure recovery from code to statechart."""

    def __init__(self, use_llm: bool = False):
        """Initialize benchmark.

        Args:
            use_llm: Whether to use LLM extraction (requires mlx_lm).
        """
        self.extractor = HybridExtractor(use_llm=use_llm) if use_llm else RuleBasedExtractor()
        self.builder = StatechartBuilder()
        self.validator = StatechartValidator()

    def run_test(self, test_case: TestCase) -> BenchmarkResult:
        """Run a single benchmark test."""
        # Extract patterns
        start = time.perf_counter()
        result = self.extractor.extract(test_case.code)
        extraction_time = (time.perf_counter() - start) * 1000

        # Build statechart
        start = time.perf_counter()
        sc = self.builder.build(result)
        build_time = (time.perf_counter() - start) * 1000

        # Validate
        errors = self.validator.validate(sc)

        # Calculate metrics
        metrics = self._calculate_metrics(test_case, result, sc)

        # Generate Mermaid output
        mermaid = StatechartSerializer.to_mermaid(sc)

        return BenchmarkResult(
            test_case=test_case,
            metrics=metrics,
            extraction_time_ms=extraction_time,
            build_time_ms=build_time,
            validation_errors=errors,
            mermaid_output=mermaid,
        )

    def _calculate_metrics(
        self,
        test_case: TestCase,
        result: ExtractionResult,
        sc,
    ) -> RecoveryMetrics:
        """Calculate recovery metrics."""
        # States
        found_states = result.combined_states
        expected_states = test_case.expected_states
        correct_states = found_states & expected_states

        # Transitions
        found_transitions = set()
        for t in result.combined_transitions:
            from_state = t.get('from', '*')
            to_state = t.get('to', '')
            if from_state and to_state:
                found_transitions.add((from_state, to_state))

        expected_transitions = set(test_case.expected_transitions)
        correct_transitions = found_transitions & expected_transitions

        # Initial state
        initial_correct = False
        if result.patterns:
            for pattern in result.patterns:
                if pattern.initial_state == test_case.expected_initial:
                    initial_correct = True
                    break

        # Final states
        found_finals = set()
        for pattern in result.patterns:
            found_finals.update(pattern.final_states)
        final_correct = found_finals == test_case.expected_final

        return RecoveryMetrics(
            states_expected=len(expected_states),
            states_found=len(found_states),
            states_correct=len(correct_states),
            transitions_expected=len(expected_transitions),
            transitions_found=len(found_transitions),
            transitions_correct=len(correct_transitions),
            initial_correct=initial_correct,
            final_correct=final_correct,
        )

    def run_suite(self, test_cases: List[TestCase] = None) -> BenchmarkSuite:
        """Run complete benchmark suite."""
        if test_cases is None:
            test_cases = TEST_CASES

        start = time.time()
        results = []

        for test_case in test_cases:
            result = self.run_test(test_case)
            results.append(result)

        total_time = time.time() - start
        return BenchmarkSuite(results=results, total_time=total_time)


def run_benchmark(use_llm: bool = False) -> BenchmarkSuite:
    """Run structure recovery benchmark."""
    benchmark = StructureRecoveryBenchmark(use_llm=use_llm)
    return benchmark.run_suite()


def demo():
    """Run benchmark demo."""
    print("=" * 70)
    print("STRUCTURE RECOVERY BENCHMARK")
    print("=" * 70)

    suite = run_benchmark(use_llm=False)

    print(f"\nResults: {suite.passing_tests}/{len(suite.results)} tests >= 90%")
    print(f"Average State Recall: {suite.avg_state_recall*100:.1f}%")
    print(f"Average Transition Recall: {suite.avg_transition_recall*100:.1f}%")
    print(f"Average Overall Score: {suite.avg_overall*100:.1f}%")
    print(f"Total Time: {suite.total_time*1000:.1f}ms")
    print("-" * 70)

    for result in suite.results:
        score = result.metrics.overall_score
        status = "PASS" if score >= 0.9 else "FAIL"
        print(f"\n[{status}] {result.test_case.name}")
        print(f"  States: {result.metrics.states_correct}/{result.metrics.states_expected} "
              f"(recall: {result.metrics.state_recall*100:.0f}%)")
        print(f"  Transitions: {result.metrics.transitions_correct}/{result.metrics.transitions_expected} "
              f"(recall: {result.metrics.transition_recall*100:.0f}%)")
        print(f"  Initial correct: {result.metrics.initial_correct}")
        print(f"  Overall: {score*100:.0f}%")
        print(f"  Time: {result.extraction_time_ms + result.build_time_ms:.1f}ms")

        if result.validation_errors:
            print(f"  Validation errors: {result.validation_errors}")

    print("\n" + "=" * 70)
    target_met = suite.avg_overall >= 0.9
    print(f"TARGET (90%+): {'ACHIEVED' if target_met else 'NOT MET'} ({suite.avg_overall*100:.1f}%)")
    print("=" * 70)

    return suite


if __name__ == "__main__":
    demo()
