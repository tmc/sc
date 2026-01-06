"""
Test Trace Generator using LLM

Generates event sequences that maximize coverage of a statechart.
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from .coverage_analyzer import CoverageAnalyzer, CoverageReport


@dataclass
class GeneratedTrace:
    """A generated test trace with metadata."""
    events: List[str]
    description: str = ""
    coverage_goal: str = ""  # "state", "transition", "full"
    generation_time_ms: float = 0.0

    # Coverage achieved
    state_coverage: float = 0.0
    transition_coverage: float = 0.0
    states_visited: List[str] = field(default_factory=list)


@dataclass
class TraceGeneratorConfig:
    """Configuration for trace generation."""
    max_trace_length: int = 20
    num_traces: int = 5
    temperature: float = 0.3
    coverage_goal: str = "full"  # "state", "transition", "full"


class TraceGenerator:
    """
    LLM-based test trace generator for statecharts.

    Given a statechart, generates event sequences designed to achieve
    maximum state and transition coverage.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        config: Optional[TraceGeneratorConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or TraceGeneratorConfig()

    def generate_traces(
        self,
        statechart: Dict,
        existing_coverage: Optional[CoverageReport] = None,
    ) -> List[GeneratedTrace]:
        """
        Generate test traces for a statechart.

        Args:
            statechart: Statechart JSON definition
            existing_coverage: Optional existing coverage to improve upon

        Returns:
            List of generated traces
        """
        analyzer = CoverageAnalyzer(statechart)
        traces = []

        # Generate multiple traces
        for i in range(self.config.num_traces):
            # Build prompt based on coverage goal
            prompt = self._build_prompt(statechart, analyzer, existing_coverage)

            # Generate with LLM
            start = time.time()
            events = self._generate_trace(prompt, analyzer)
            gen_time = (time.time() - start) * 1000

            # Evaluate coverage
            report, state_seq = analyzer.execute_trace(events)

            trace = GeneratedTrace(
                events=events,
                description=f"Trace {i+1} targeting {self.config.coverage_goal} coverage",
                coverage_goal=self.config.coverage_goal,
                generation_time_ms=gen_time,
                state_coverage=report.state_coverage,
                transition_coverage=report.transition_coverage,
                states_visited=state_seq,
            )
            traces.append(trace)

            # Update existing coverage for next iteration
            if existing_coverage is None:
                existing_coverage = report
            else:
                existing_coverage.covered_states.update(report.covered_states)
                existing_coverage.covered_transitions.update(report.covered_transitions)
                existing_coverage.uncovered_states -= report.covered_states
                existing_coverage.uncovered_transitions -= report.covered_transitions

        return traces

    def _build_prompt(
        self,
        statechart: Dict,
        analyzer: CoverageAnalyzer,
        existing_coverage: Optional[CoverageReport],
    ) -> str:
        """Build prompt for trace generation."""
        # Format statechart info
        states = list(analyzer.all_states)
        transitions = []
        events_list = set()
        for (f, t, e) in analyzer.all_transitions:
            transitions.append(f"  {f} --{e}--> {t}")
            events_list.add(e)

        trans_str = "\n".join(transitions)
        events_str = ", ".join(sorted(events_list))

        prompt = f"""Statechart:
- States: {', '.join(states)}
- Initial: {analyzer.initial_state}
- Available events: {events_str}
- Transitions:
{trans_str}

Generate a test trace to cover all transitions. Output ONLY a JSON array of events.
Example format: ["EVENT1", "EVENT2", "EVENT3"]

Test trace:"""

        return prompt

    def _generate_trace(self, prompt: str, analyzer: CoverageAnalyzer) -> List[str]:
        """Generate a single trace using LLM."""
        if self.model is None:
            # Fallback: generate trace programmatically
            return self._generate_trace_programmatic(analyzer)

        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=self.config.temperature)
            output = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=200,
                sampler=sampler,
            )

            # Parse events from output
            events = self._parse_events(output, analyzer)
            return events

        except Exception as e:
            print(f"[WARN] LLM generation failed: {e}")
            return self._generate_trace_programmatic(analyzer)

    def _parse_events(self, output: str, analyzer: CoverageAnalyzer) -> List[str]:
        """Parse event names from LLM output."""
        events = []

        # Try to find JSON array
        match = re.search(r'\[([^\]]+)\]', output)
        if match:
            try:
                parsed = json.loads(f"[{match.group(1)}]")
                if isinstance(parsed, list):
                    # Filter to valid events
                    valid_events = set()
                    for (f, t, e) in analyzer.all_transitions:
                        valid_events.add(e)

                    for item in parsed:
                        if isinstance(item, str) and item in valid_events:
                            events.append(item)
                        elif isinstance(item, str):
                            # Try to match partial
                            for ve in valid_events:
                                if ve.lower() == item.lower():
                                    events.append(ve)
                                    break
            except json.JSONDecodeError:
                pass

        # Fallback: look for event names in text
        if not events:
            valid_events = set()
            for (f, t, e) in analyzer.all_transitions:
                valid_events.add(e)

            for event in valid_events:
                if event in output:
                    events.append(event)

        return events[:self.config.max_trace_length]

    def _generate_trace_programmatic(self, analyzer: CoverageAnalyzer) -> List[str]:
        """Generate trace using simple graph traversal (fallback)."""
        events = []
        current = analyzer.initial_state
        visited_transitions = set()

        for _ in range(self.config.max_trace_length):
            # Find unvisited transition from current state
            available = []
            for (f, t, e) in analyzer.all_transitions:
                if f == current:
                    priority = 0 if (f, t, e) not in visited_transitions else 1
                    available.append((priority, f, t, e))

            if not available:
                break

            # Sort by priority (unvisited first)
            available.sort(key=lambda x: x[0])
            _, f, t, e = available[0]

            events.append(e)
            visited_transitions.add((f, t, e))
            current = t

        return events


def generate_test_traces(
    statechart: Dict,
    model=None,
    tokenizer=None,
    num_traces: int = 5,
) -> Tuple[List[GeneratedTrace], float, float]:
    """
    Generate test traces for a statechart.

    Args:
        statechart: Statechart JSON
        model: Optional LLM model
        tokenizer: Optional tokenizer
        num_traces: Number of traces to generate

    Returns:
        (traces, state_coverage, transition_coverage)
    """
    config = TraceGeneratorConfig(num_traces=num_traces)
    generator = TraceGenerator(model, tokenizer, config)
    traces = generator.generate_traces(statechart)

    # Compute combined coverage
    analyzer = CoverageAnalyzer(statechart)
    all_covered_states = set()
    all_covered_transitions = set()

    for trace in traces:
        report, _ = analyzer.execute_trace(trace.events)
        all_covered_states.update(report.covered_states)
        all_covered_transitions.update(report.covered_transitions)

    state_cov = len(all_covered_states) / len(analyzer.all_states) if analyzer.all_states else 0
    trans_cov = len(all_covered_transitions) / len(analyzer.all_transitions) if analyzer.all_transitions else 0

    return traces, state_cov, trans_cov
