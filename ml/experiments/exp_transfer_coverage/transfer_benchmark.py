"""
Transfer Benchmark - Evaluate coverage topology transfer.

Tests whether structure learned on simple programs transfers to complex ones.
"""

import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

from .topology_transfer import TopologyTransferLearner, AbstractTopology
from .program_families import (
    generate_family_dataset,
    get_transfer_pairs,
    PROGRAM_FAMILIES,
    Complexity,
)
from ..exp_coverage_prediction.dataset import Dataset
from ..exp_coverage_prediction.statechart_evolver import EvolutionConfig


@dataclass
class TransferResult:
    source_family: str
    target_family: str
    source_complexity: str
    target_complexity: str
    source_f1: float
    zeroshot_f1: float
    finetuned_f1: float
    scratch_f1: float
    transfer_benefit: float
    pattern_type: str
    source_states: int
    transfer_time: float

    @property
    def zeroshot_ratio(self) -> float:
        return self.zeroshot_f1 / self.scratch_f1 if self.scratch_f1 > 0 else 0

    @property
    def transfer_successful(self) -> bool:
        return self.transfer_benefit > 0


class TransferBenchmark:
    """Benchmark transfer learning across program families."""

    def __init__(self, evolution_config: Optional[EvolutionConfig] = None):
        self.config = evolution_config or EvolutionConfig(
            population_size=20, n_generations=25, verbose=False
        )
        self.results: List[TransferResult] = []

    def run_transfer(self, source_family: str, target_family: str,
                     verbose: bool = True) -> TransferResult:
        """Run single transfer experiment."""
        start = time.time()

        source_dataset = generate_family_dataset(source_family, n_per_template=15)
        target_dataset = generate_family_dataset(target_family, n_per_template=15)

        if verbose:
            print(f"\n{'='*50}")
            print(f"TRANSFER: {source_family} -> {target_family}")
            print(f"{'='*50}")

        learner = TopologyTransferLearner(source_dataset, target_dataset, self.config)
        transfer_results = learner.full_transfer(verbose=verbose)

        elapsed = time.time() - start

        result = TransferResult(
            source_family=source_family,
            target_family=target_family,
            source_complexity=PROGRAM_FAMILIES[source_family].complexity.name,
            target_complexity=PROGRAM_FAMILIES[target_family].complexity.name,
            source_f1=transfer_results['source_f1'],
            zeroshot_f1=transfer_results['zeroshot_f1'],
            finetuned_f1=transfer_results['finetuned_f1'],
            scratch_f1=transfer_results['scratch_f1'],
            transfer_benefit=transfer_results['transfer_benefit'],
            pattern_type=transfer_results['pattern_type'],
            source_states=transfer_results['source_states'],
            transfer_time=elapsed,
        )

        self.results.append(result)
        return result

    def run_all_pairs(self, verbose: bool = True) -> List[TransferResult]:
        """Run transfer experiments on all recommended pairs."""
        pairs = get_transfer_pairs()

        if verbose:
            print("=" * 60)
            print("TRANSFER COVERAGE BENCHMARK")
            print("=" * 60)
            print(f"Running {len(pairs)} transfer experiments...")

        for source, target in pairs:
            self.run_transfer(source, target, verbose=verbose)

        if verbose:
            self.print_summary()

        return self.results

    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 70)
        print("TRANSFER BENCHMARK SUMMARY")
        print("=" * 70)

        print(f"\n{'Source -> Target':<35} {'Zero':>6} {'Fine':>6} {'Scratch':>7} {'Benefit':>8}")
        print("-" * 70)

        successful = 0
        for r in self.results:
            pair = f"{r.source_family} -> {r.target_family}"
            print(f"{pair:<35} {r.zeroshot_f1:>6.3f} {r.finetuned_f1:>6.3f} "
                  f"{r.scratch_f1:>7.3f} {r.transfer_benefit:>+8.3f}")
            if r.transfer_successful:
                successful += 1

        print("-" * 70)
        avg_benefit = sum(r.transfer_benefit for r in self.results) / len(self.results) if self.results else 0
        avg_zeroshot = sum(r.zeroshot_ratio for r in self.results) / len(self.results) if self.results else 0

        print(f"\nSuccessful transfers: {successful}/{len(self.results)}")
        print(f"Average benefit: {avg_benefit:+.3f}")
        print(f"Average zero-shot ratio: {avg_zeroshot:.1%}")

        if successful > len(self.results) // 2:
            print("\nCONCLUSION: Coverage topology TRANSFERS across programs!")
        else:
            print("\nCONCLUSION: Limited transfer - may need more similar programs.")


def demo():
    """Run transfer benchmark demo."""
    print("=" * 60)
    print("TRANSFER COVERAGE DEMO")
    print("=" * 60)

    benchmark = TransferBenchmark()

    # Run subset for demo
    pairs = get_transfer_pairs()[:3]
    for source, target in pairs:
        benchmark.run_transfer(source, target, verbose=True)

    benchmark.print_summary()
    return benchmark.results


if __name__ == "__main__":
    demo()
