#!/usr/bin/env python3
"""
Benchmark: Run full SC attention analysis suite.

Combines structure heads, hierarchy attention, and validity diff
analysis to provide comprehensive attention pattern insights.
"""

import sys
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from .structure_heads import StructureHeadDetector, StructureHeadAnalysis
from .hierarchy_attention import HierarchyAttentionAnalyzer, HierarchyAnalysis
from .validity_diff import ValidityDiffAnalyzer, ValidityAnalysis

from utils.mlux_loader import MLUX_AVAILABLE


@dataclass
class BenchmarkResult:
    """Combined benchmark results."""
    structure_analysis: Optional[StructureHeadAnalysis] = None
    hierarchy_analysis: Optional[HierarchyAnalysis] = None
    validity_analyses: Dict[str, ValidityAnalysis] = field(default_factory=dict)
    total_time: float = 0.0
    mlux_available: bool = False

    @property
    def num_structure_heads_found(self) -> int:
        if not self.structure_analysis:
            return 0
        return len([h for h in self.structure_analysis.head_scores
                   if h.structure_score > 0.1])

    @property
    def num_hierarchy_heads_found(self) -> int:
        if not self.hierarchy_analysis:
            return 0
        return len([h for h in self.hierarchy_analysis.head_scores
                   if h.overall_score > 0.1])

    @property
    def avg_validity_accuracy(self) -> float:
        if not self.validity_analyses:
            return 0.0
        return sum(a.classification_accuracy for a in self.validity_analyses.values()) / len(self.validity_analyses)


# Test statecharts
TEST_STATECHARTS = [
    {
        "name": "simple_toggle",
        "sc": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Off", "is_initial": True},
                    {"label": "On"},
                ]
            },
            "transitions": [
                {"from": ["Off"], "to": ["On"], "event": "TOGGLE"},
                {"from": ["On"], "to": ["Off"], "event": "TOGGLE"},
            ]
        }
    },
    {
        "name": "traffic_light",
        "sc": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Red", "is_initial": True},
                    {"label": "Green"},
                    {"label": "Yellow"},
                ]
            },
            "transitions": [
                {"from": ["Red"], "to": ["Green"], "event": "TIMER"},
                {"from": ["Green"], "to": ["Yellow"], "event": "TIMER"},
                {"from": ["Yellow"], "to": ["Red"], "event": "TIMER"},
            ]
        }
    },
    {
        "name": "hierarchical_player",
        "sc": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Stopped", "is_initial": True},
                    {
                        "label": "Playing",
                        "type": 2,
                        "children": [
                            {"label": "Normal", "is_initial": True},
                            {"label": "FastForward"},
                        ]
                    },
                    {"label": "Paused"},
                ]
            },
            "transitions": [
                {"from": ["Stopped"], "to": ["Playing"], "event": "PLAY"},
                {"from": ["Playing"], "to": ["Paused"], "event": "PAUSE"},
                {"from": ["Paused"], "to": ["Playing"], "event": "PLAY"},
                {"from": ["Playing"], "to": ["Stopped"], "event": "STOP"},
            ]
        }
    },
]


class SCAttentionBenchmark:
    """Full SC attention analysis benchmark."""

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    ):
        self.model_name = model_name
        self._structure_detector = None
        self._hierarchy_analyzer = None
        self._validity_analyzer = None

    def _init_analyzers(self):
        """Initialize all analyzers (shares model loading)."""
        print("Initializing analyzers...")
        self._structure_detector = StructureHeadDetector(self.model_name)
        self._hierarchy_analyzer = HierarchyAttentionAnalyzer(self.model_name)
        self._validity_analyzer = ValidityDiffAnalyzer(self.model_name)

    def run_benchmark(
        self,
        statecharts: Optional[List[Dict]] = None,
        verbose: bool = True,
    ) -> BenchmarkResult:
        """
        Run full benchmark suite.

        Args:
            statecharts: List of statecharts to analyze
            verbose: Print progress

        Returns:
            BenchmarkResult with all analyses
        """
        statecharts = statecharts or TEST_STATECHARTS
        result = BenchmarkResult(mlux_available=MLUX_AVAILABLE)

        start_time = time.time()

        if verbose:
            print("=" * 60)
            print("SC ATTENTION ANALYSIS BENCHMARK")
            print("=" * 60)
            print(f"MLUX available: {MLUX_AVAILABLE}")
            print(f"Statecharts to analyze: {len(statecharts)}")

        self._init_analyzers()

        # 1. Structure head analysis
        if verbose:
            print("\n--- Structure Head Analysis ---")

        # Use most complex SC for structure analysis
        complex_sc = max(statecharts, key=lambda s: len(json.dumps(s['sc'])))
        sc_str = json.dumps(complex_sc['sc'])

        result.structure_analysis = self._structure_detector.analyze(sc_str)

        if verbose:
            top3 = result.structure_analysis.top_structure_heads[:3]
            print(f"  Top structure heads: {top3}")

        # 2. Hierarchy attention analysis
        if verbose:
            print("\n--- Hierarchy Attention Analysis ---")

        # Use hierarchical SC
        hier_sc = next((s for s in statecharts if 'hierarchical' in s['name']), statecharts[-1])
        result.hierarchy_analysis = self._hierarchy_analyzer.analyze(hier_sc['sc'])

        if verbose:
            top3 = result.hierarchy_analysis.top_parent_child_heads[:3]
            print(f"  Top parent-child heads: {top3}")

        # 3. Validity difference analysis
        if verbose:
            print("\n--- Validity Difference Analysis ---")

        # Use simple SC for validity analysis
        simple_sc = statecharts[0]['sc']
        result.validity_analyses = self._validity_analyzer.analyze_error_types(simple_sc)

        if verbose:
            for error_type, analysis in result.validity_analyses.items():
                print(f"  {error_type}: accuracy={analysis.classification_accuracy:.1%}")

        result.total_time = time.time() - start_time

        return result

    def print_summary(self, result: BenchmarkResult):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        print(f"\n--- Configuration ---")
        print(f"  MLUX available: {result.mlux_available}")
        print(f"  Total time: {result.total_time:.2f}s")

        print(f"\n--- Structure Analysis ---")
        if result.structure_analysis:
            print(f"  Heads analyzed: {len(result.structure_analysis.head_scores)}")
            print(f"  Structure-aware heads (>0.1): {result.num_structure_heads_found}")
            if result.structure_analysis.top_structure_heads:
                l, h, s = result.structure_analysis.top_structure_heads[0]
                print(f"  Best structure head: L{l}H{h} ({s:.4f})")

        print(f"\n--- Hierarchy Analysis ---")
        if result.hierarchy_analysis:
            print(f"  Hierarchy nodes: {len(result.hierarchy_analysis.hierarchy)}")
            print(f"  Hierarchy-aware heads (>0.1): {result.num_hierarchy_heads_found}")
            print(f"  Avg parent-child attention: {result.hierarchy_analysis.avg_parent_child_attention:.4f}")

        print(f"\n--- Validity Analysis ---")
        print(f"  Error types analyzed: {len(result.validity_analyses)}")
        print(f"  Avg classification accuracy: {result.avg_validity_accuracy:.1%}")
        for error_type, analysis in result.validity_analyses.items():
            print(f"    - {error_type}: {analysis.classification_accuracy:.1%}")

        # Overall assessment
        print(f"\n--- Overall Assessment ---")

        findings = []
        if result.num_structure_heads_found > 10:
            findings.append("Found significant structure-aware attention patterns")
        if result.num_hierarchy_heads_found > 10:
            findings.append("Found hierarchy-tracking attention heads")
        if result.avg_validity_accuracy > 0.7:
            findings.append(f"Attention can distinguish valid/invalid ({result.avg_validity_accuracy:.1%})")

        if findings:
            for f in findings:
                print(f"  [OK] {f}")
        else:
            print("  [--] Limited findings (may need mlux for full analysis)")


def demo():
    """Quick demo with minimal output."""
    print("=" * 60)
    print("SC ATTENTION BENCHMARK DEMO")
    print("=" * 60)

    benchmark = SCAttentionBenchmark()
    result = benchmark.run_benchmark(statecharts=TEST_STATECHARTS[:2])
    benchmark.print_summary(result)

    return result


def run_full_benchmark():
    """Run full benchmark with all test cases."""
    benchmark = SCAttentionBenchmark()
    result = benchmark.run_benchmark(statecharts=TEST_STATECHARTS, verbose=True)
    benchmark.print_summary(result)

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        run_full_benchmark()
    else:
        demo()
