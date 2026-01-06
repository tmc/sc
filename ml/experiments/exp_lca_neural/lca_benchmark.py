"""
LCA Benchmark - Compare approaches for LCA prediction.

Approaches:
1. Supervised neural network
2. Evolutionary lookup
3. Hybrid (supervised + evolution)
4. Baseline: Random guess
5. Baseline: Always predict root

Tests:
- In-distribution accuracy (same hierarchies)
- Generalization (unseen hierarchies)
- Sample efficiency (learning curve)
"""

import time
from dataclasses import dataclass
from typing import List, Dict, Optional

from .lca_dataset import (
    LCAExample,
    Hierarchy,
    LCADatasetGenerator,
)
from .neural_lca import (
    SupervisedLCATrainer,
    EvolutionaryLCAPredictor,
    HybridLCAPredictor,
    LCAModelConfig,
)


@dataclass
class BenchmarkResult:
    approach: str
    train_accuracy: float
    test_accuracy: float
    generalization_accuracy: float
    train_time: float

    def to_dict(self) -> Dict:
        return {
            'approach': self.approach,
            'train_acc': self.train_accuracy,
            'test_acc': self.test_accuracy,
            'gen_acc': self.generalization_accuracy,
            'time': self.train_time,
        }


class RandomBaseline:
    """Baseline: Random LCA prediction."""

    def __init__(self, hierarchies: List[Hierarchy]):
        import random
        self.rng = random.Random(42)
        self.all_states = []
        for h in hierarchies:
            self.all_states.extend(h.nodes.keys())

    def predict(self, source_id: str, target_id: str) -> str:
        return self.rng.choice(self.all_states) if self.all_states else "<unk>"

    def evaluate(self, examples: List[LCAExample]) -> Dict[str, float]:
        correct = sum(1 for ex in examples if self.predict(ex.source_id, ex.target_id) == ex.lca_id)
        return {'accuracy': correct / len(examples) if examples else 0}


class RootBaseline:
    """Baseline: Always predict root as LCA."""

    def __init__(self, hierarchies: List[Hierarchy]):
        self.roots = {h.id: h.root_id for h in hierarchies}
        self.default_root = hierarchies[0].root_id if hierarchies else "root"

    def predict(self, source_id: str, target_id: str, hierarchy_id: str = "") -> str:
        return self.roots.get(hierarchy_id, self.default_root)

    def evaluate(self, examples: List[LCAExample]) -> Dict[str, float]:
        correct = sum(1 for ex in examples
                     if self.roots.get(ex.hierarchy_id, self.default_root) == ex.lca_id)
        return {'accuracy': correct / len(examples) if examples else 0}


class LCABenchmark:
    """Benchmark LCA prediction approaches."""

    def __init__(self, seed: int = 42):
        self.generator = LCADatasetGenerator(seed)
        self.results: List[BenchmarkResult] = []

    def run(
        self,
        n_hierarchies: int = 20,
        examples_per_hierarchy: int = 30,
        verbose: bool = True,
    ) -> List[BenchmarkResult]:
        """Run full benchmark."""
        self.results = []

        if verbose:
            print("=" * 70)
            print("LCA PREDICTION BENCHMARK")
            print("=" * 70)

        # Generate data
        examples, hierarchies = self.generator.generate_dataset(
            n_hierarchies=n_hierarchies,
            examples_per_hierarchy=examples_per_hierarchy,
        )

        # Split: 70% train, 15% test (same hierarchies), 15% generalization (new hierarchies)
        train, gen_test, train_h, gen_h = self.generator.split_by_hierarchy(
            examples, hierarchies, train_ratio=0.7
        )

        # Further split train for in-distribution test
        n_test = len(train) // 5
        test = train[-n_test:]
        train = train[:-n_test]

        if verbose:
            print(f"\nData split:")
            print(f"  Train: {len(train)} examples from {len(train_h)} hierarchies")
            print(f"  Test (in-dist): {len(test)} examples")
            print(f"  Test (gen): {len(gen_test)} examples from {len(gen_h)} hierarchies")

        # 1. Random baseline
        if verbose:
            print("\n[1/5] Random baseline...")
        random_bl = RandomBaseline(hierarchies)
        self.results.append(BenchmarkResult(
            approach="Random",
            train_accuracy=random_bl.evaluate(train)['accuracy'],
            test_accuracy=random_bl.evaluate(test)['accuracy'],
            generalization_accuracy=random_bl.evaluate(gen_test)['accuracy'],
            train_time=0.0,
        ))

        # 2. Root baseline
        if verbose:
            print("[2/5] Root baseline...")
        root_bl = RootBaseline(hierarchies)
        self.results.append(BenchmarkResult(
            approach="AlwaysRoot",
            train_accuracy=root_bl.evaluate(train)['accuracy'],
            test_accuracy=root_bl.evaluate(test)['accuracy'],
            generalization_accuracy=root_bl.evaluate(gen_test)['accuracy'],
            train_time=0.0,
        ))

        # 3. Supervised
        if verbose:
            print("[3/5] Supervised neural network...")
        start = time.time()
        supervised = SupervisedLCATrainer(LCAModelConfig())
        supervised.train(train, train_h, n_epochs=40, verbose=False)
        sup_time = time.time() - start
        self.results.append(BenchmarkResult(
            approach="Supervised",
            train_accuracy=supervised.evaluate(train)['accuracy'],
            test_accuracy=supervised.evaluate(test)['accuracy'],
            generalization_accuracy=supervised.evaluate(gen_test)['accuracy'],
            train_time=sup_time,
        ))

        # 4. Evolutionary
        if verbose:
            print("[4/5] Evolutionary...")
        start = time.time()
        evolutionary = EvolutionaryLCAPredictor()
        evolutionary.evolve(train, train_h, n_generations=40, verbose=False)
        evo_time = time.time() - start
        self.results.append(BenchmarkResult(
            approach="Evolutionary",
            train_accuracy=evolutionary.evaluate(train)['accuracy'],
            test_accuracy=evolutionary.evaluate(test)['accuracy'],
            generalization_accuracy=evolutionary.evaluate(gen_test)['accuracy'],
            train_time=evo_time,
        ))

        # 5. Hybrid
        if verbose:
            print("[5/5] Hybrid...")
        start = time.time()
        hybrid = HybridLCAPredictor()
        hybrid.train(train, train_h, n_supervised_epochs=25, n_evolution_gens=15, verbose=False)
        hyb_time = time.time() - start
        self.results.append(BenchmarkResult(
            approach="Hybrid",
            train_accuracy=hybrid.evaluate(train)['accuracy'],
            test_accuracy=hybrid.evaluate(test)['accuracy'],
            generalization_accuracy=hybrid.evaluate(gen_test)['accuracy'],
            train_time=hyb_time,
        ))

        if verbose:
            self._print_summary()

        return self.results

    def _print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)

        print(f"\n{'Approach':<15} {'Train':>8} {'Test':>8} {'Gen':>8} {'Time':>8}")
        print("-" * 70)

        for r in self.results:
            print(f"{r.approach:<15} {r.train_accuracy:>8.3f} {r.test_accuracy:>8.3f} "
                  f"{r.generalization_accuracy:>8.3f} {r.train_time:>7.1f}s")

        # Find best
        best_test = max(self.results, key=lambda r: r.test_accuracy)
        best_gen = max(self.results, key=lambda r: r.generalization_accuracy)

        print("-" * 70)
        print(f"Best test accuracy: {best_test.approach} ({best_test.test_accuracy:.3f})")
        print(f"Best generalization: {best_gen.approach} ({best_gen.generalization_accuracy:.3f})")

        # Insights
        print("\n" + "-" * 70)
        print("INSIGHTS:")

        supervised = next((r for r in self.results if r.approach == "Supervised"), None)
        evolutionary = next((r for r in self.results if r.approach == "Evolutionary"), None)
        hybrid = next((r for r in self.results if r.approach == "Hybrid"), None)

        if supervised and evolutionary:
            if supervised.generalization_accuracy > evolutionary.generalization_accuracy:
                print("- Supervised generalizes better than evolutionary (learns patterns)")
            else:
                print("- Evolutionary matches/beats supervised on generalization")

        if hybrid and supervised:
            if hybrid.test_accuracy > supervised.test_accuracy:
                print("- Hybrid improves over pure supervised")

        print("=" * 70)

    def run_sample_efficiency(
        self,
        train_sizes: List[int] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Test sample efficiency (learning curves)."""
        if train_sizes is None:
            train_sizes = [50, 100, 200, 400]

        examples, hierarchies = self.generator.generate_dataset(
            n_hierarchies=20, examples_per_hierarchy=40
        )
        train_all, test, train_h, _ = self.generator.split_by_hierarchy(examples, hierarchies)

        results = {'sizes': train_sizes, 'supervised': [], 'evolutionary': [], 'hybrid': []}

        if verbose:
            print("\nSAMPLE EFFICIENCY TEST")
            print("-" * 40)

        for size in train_sizes:
            train = train_all[:size]

            # Supervised
            sup = SupervisedLCATrainer(LCAModelConfig())
            sup.train(train, train_h, n_epochs=30, verbose=False)
            sup_acc = sup.evaluate(test)['accuracy']
            results['supervised'].append(sup_acc)

            # Evolutionary
            evo = EvolutionaryLCAPredictor()
            evo.evolve(train, train_h, n_generations=30, verbose=False)
            evo_acc = evo.evaluate(test)['accuracy']
            results['evolutionary'].append(evo_acc)

            # Hybrid
            hyb = HybridLCAPredictor()
            hyb.train(train, train_h, n_supervised_epochs=20, n_evolution_gens=10, verbose=False)
            hyb_acc = hyb.evaluate(test)['accuracy']
            results['hybrid'].append(hyb_acc)

            if verbose:
                print(f"Size {size:4d}: Sup={sup_acc:.3f}, Evo={evo_acc:.3f}, Hyb={hyb_acc:.3f}")

        return results


def demo():
    """Run LCA benchmark demo."""
    print("=" * 60)
    print("LCA BENCHMARK DEMO")
    print("=" * 60)

    benchmark = LCABenchmark(seed=42)
    results = benchmark.run(n_hierarchies=15, examples_per_hierarchy=25, verbose=True)

    return results


if __name__ == "__main__":
    demo()
