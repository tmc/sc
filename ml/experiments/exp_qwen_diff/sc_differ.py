#!/usr/bin/env python3
"""
SC Differ: Generate human-readable statechart diffs using QwenCoder.

Takes two statecharts and produces:
1. Structured change detection
2. Human-readable summary via LLM
3. Impact analysis
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

import mlx.core as mx
from mlx_lm import load, generate

from .change_detector import (
    ChangeDetector, DiffResult, ChangeType, Compatibility,
    StateChange, TransitionChange
)


@dataclass
class DiffConfig:
    """Configuration for diff generation."""
    max_tokens: int = 256
    temperature: float = 0.3
    include_impact: bool = True
    include_migration_hints: bool = True
    verbose: bool = False


@dataclass
class DiffSummary:
    """Complete diff summary."""
    diff_result: DiffResult
    structured_summary: str
    llm_summary: str
    impact_analysis: str = ""
    migration_hints: List[str] = field(default_factory=list)
    generation_time: float = 0.0


class SCDiffer:
    """
    Generates human-readable statechart diffs.

    Combines programmatic detection with LLM summarization.
    """

    SYSTEM_PROMPT = """You are a statechart expert. Given changes between two statecharts, write a clear, concise summary.

Focus on:
1. What changed (states, transitions, events)
2. Impact on existing machines
3. Whether changes are breaking

Be specific and technical. Use bullet points. Keep it under 150 words."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        sc_path: str = "./sc",
    ):
        """Initialize with QwenCoder model."""
        print(f"Loading model: {model_name}")
        self.model, self.tokenizer = load(model_name)
        print(f"Model loaded.")

        self.detector = ChangeDetector(sc_path)

    def diff(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any],
        config: Optional[DiffConfig] = None,
    ) -> DiffSummary:
        """
        Generate complete diff summary.

        Args:
            before: Original statechart
            after: Modified statechart
            config: Generation config

        Returns:
            DiffSummary with structured and LLM summaries
        """
        config = config or DiffConfig()
        start_time = time.time()

        # Detect changes
        diff_result = self.detector.diff(before, after)

        # Generate structured summary
        structured = self._generate_structured_summary(diff_result)

        # Generate LLM summary
        if diff_result.is_empty:
            llm_summary = "No changes detected between the statecharts."
        else:
            llm_summary = self._generate_llm_summary(diff_result, config)

        # Impact analysis
        impact = ""
        if config.include_impact:
            impact = self._analyze_impact(diff_result)

        # Migration hints
        hints = []
        if config.include_migration_hints and diff_result.compatibility == Compatibility.BREAKING:
            hints = self._generate_migration_hints(diff_result)

        return DiffSummary(
            diff_result=diff_result,
            structured_summary=structured,
            llm_summary=llm_summary,
            impact_analysis=impact,
            migration_hints=hints,
            generation_time=time.time() - start_time,
        )

    def _generate_structured_summary(self, diff: DiffResult) -> str:
        """Generate structured text summary."""
        lines = []

        lines.append(f"Summary: {diff.summary()}")
        lines.append(f"Compatibility: {diff.compatibility.name}")
        lines.append("")

        if diff.states_added:
            lines.append("States Added:")
            for change in diff.states_added:
                lines.append(f"  + {change.state_label}")

        if diff.states_removed:
            lines.append("States Removed:")
            for change in diff.states_removed:
                lines.append(f"  - {change.state_label}")

        if diff.states_modified:
            lines.append("States Modified:")
            for change in diff.states_modified:
                lines.append(f"  ~ {change.state_label}: {change.details}")

        if diff.transitions_added:
            lines.append("Transitions Added:")
            for change in diff.transitions_added:
                lines.append(f"  + {change.from_states} -> {change.to_states} [{change.event}]")

        if diff.transitions_removed:
            lines.append("Transitions Removed:")
            for change in diff.transitions_removed:
                lines.append(f"  - {change.from_states} -> {change.to_states} [{change.event}]")

        if diff.breaking_reasons:
            lines.append("")
            lines.append("Breaking Changes:")
            for reason in diff.breaking_reasons:
                lines.append(f"  ! {reason}")

        return "\n".join(lines)

    def _generate_llm_summary(self, diff: DiffResult, config: DiffConfig) -> str:
        """Generate human-readable summary via LLM."""
        # Build prompt
        prompt = self._build_diff_prompt(diff)

        if config.verbose:
            print(f"Prompt length: {len(prompt)} chars")

        # Generate
        response = self._generate(prompt, config)

        # Clean up response
        response = response.strip()
        response = re.sub(r'```.*?```', '', response, flags=re.DOTALL)

        return response

    def _build_diff_prompt(self, diff: DiffResult) -> str:
        """Build prompt for LLM summarization."""
        changes = []

        for change in diff.states_added:
            changes.append(f"- Added state: {change.state_label}")
        for change in diff.states_removed:
            changes.append(f"- Removed state: {change.state_label}")
        for change in diff.states_modified:
            changes.append(f"- Modified state {change.state_label}: {change.details}")
        for change in diff.transitions_added:
            changes.append(f"- Added transition: {change.from_states[0]} -> {change.to_states[0]} on {change.event}")
        for change in diff.transitions_removed:
            changes.append(f"- Removed transition: {change.from_states[0]} -> {change.to_states[0]} on {change.event}")

        prompt = f"""{self.SYSTEM_PROMPT}

## Changes Detected
{chr(10).join(changes)}

## Compatibility
{diff.compatibility.name}

## Summary
Write a clear, human-readable summary of these changes:
"""
        return prompt

    def _generate(self, prompt: str, config: DiffConfig) -> str:
        """Generate response from LLM."""
        input_ids = self.tokenizer.encode(prompt)
        generated_tokens = []

        for step in range(config.max_tokens):
            x = mx.array([input_ids + generated_tokens])
            logits = self.model(x)[:, -1, :]

            if config.temperature > 0:
                logits = logits / config.temperature

            probs = mx.softmax(logits, axis=-1)
            next_token = int(mx.argmax(probs, axis=-1))

            if next_token == self.tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token)

            # Stop at reasonable length
            current = self.tokenizer.decode(generated_tokens)
            if len(current) > 500 or current.count('\n\n') > 3:
                break

        return self.tokenizer.decode(generated_tokens)

    def _analyze_impact(self, diff: DiffResult) -> str:
        """Analyze impact of changes."""
        impacts = []

        if diff.compatibility == Compatibility.COMPATIBLE:
            impacts.append("Low impact - changes are backward compatible.")
        elif diff.compatibility == Compatibility.BREAKING:
            impacts.append("HIGH IMPACT - breaking changes detected!")

        # Specific impacts
        if diff.states_removed:
            impacts.append(f"Machines in removed states ({[c.state_label for c in diff.states_removed]}) need migration.")

        if diff.transitions_removed:
            events = set(c.event for c in diff.transitions_removed)
            impacts.append(f"Events that no longer work: {events}")

        if diff.states_added:
            impacts.append(f"New states available: {[c.state_label for c in diff.states_added]}")

        return " ".join(impacts)

    def _generate_migration_hints(self, diff: DiffResult) -> List[str]:
        """Generate migration hints for breaking changes."""
        hints = []

        for change in diff.states_removed:
            # Suggest mapping to similar state
            if diff.states_added:
                target = diff.states_added[0].state_label
                hints.append(f"Map state '{change.state_label}' to '{target}'")
            else:
                hints.append(f"Handle machines in state '{change.state_label}' before migration")

        for change in diff.transitions_removed:
            if diff.transitions_added:
                new_trans = diff.transitions_added[0]
                hints.append(
                    f"Replace event '{change.event}' with '{new_trans.event}'"
                )

        return hints


def generate_diff(
    before: Dict[str, Any],
    after: Dict[str, Any],
    verbose: bool = False,
) -> DiffSummary:
    """Convenience function to generate diff."""
    differ = SCDiffer()
    config = DiffConfig(verbose=verbose)
    return differ.diff(before, after, config)


def demo():
    """Demo diff generation."""
    print("=" * 60)
    print("SC DIFFER DEMO")
    print("=" * 60)

    differ = SCDiffer()
    config = DiffConfig(verbose=True)

    # Test case: Complex change
    before = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Idle", "is_initial": True},
                {"label": "Loading"},
                {"label": "Ready"},
            ]
        },
        "transitions": [
            {"from": ["Idle"], "to": ["Loading"], "event": "START"},
            {"from": ["Loading"], "to": ["Ready"], "event": "LOADED"},
            {"from": ["Ready"], "to": ["Idle"], "event": "RESET"},
        ]
    }

    after = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Idle", "is_initial": True},
                {"label": "Loading"},
                {"label": "Active"},  # Changed from Ready
                {"label": "Error"},   # New state
            ]
        },
        "transitions": [
            {"from": ["Idle"], "to": ["Loading"], "event": "START"},
            {"from": ["Loading"], "to": ["Active"], "event": "SUCCESS"},  # Changed
            {"from": ["Loading"], "to": ["Error"], "event": "FAILURE"},   # New
            {"from": ["Active"], "to": ["Idle"], "event": "STOP"},        # Changed
            {"from": ["Error"], "to": ["Idle"], "event": "RETRY"},        # New
        ]
    }

    print("\nGenerating diff...")
    summary = differ.diff(before, after, config)

    print("\n" + "=" * 60)
    print("STRUCTURED SUMMARY")
    print("=" * 60)
    print(summary.structured_summary)

    print("\n" + "=" * 60)
    print("LLM SUMMARY")
    print("=" * 60)
    print(summary.llm_summary)

    print("\n" + "=" * 60)
    print("IMPACT ANALYSIS")
    print("=" * 60)
    print(summary.impact_analysis)

    if summary.migration_hints:
        print("\n" + "=" * 60)
        print("MIGRATION HINTS")
        print("=" * 60)
        for hint in summary.migration_hints:
            print(f"  - {hint}")

    print(f"\nGeneration time: {summary.generation_time:.2f}s")

    print("\n" + "=" * 60)
    print("SC DIFFER DEMO COMPLETE")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    demo()
