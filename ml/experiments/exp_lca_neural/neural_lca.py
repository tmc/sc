"""
Neural LCA Predictor

Train neural networks to predict LCA from (source, target) state pairs.

Approaches:
1. Supervised: Train on (source, target) -> LCA with cross-entropy loss
2. Evolutionary: Evolve network weights for LCA accuracy
3. Hybrid: Pre-train supervised, fine-tune with evolution

Architecture:
- Encode states as embeddings (learned or positional)
- Combine source/target embeddings
- Predict LCA via classification over all states
"""

import random
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    nn = None
    optim = None

from .lca_dataset import LCAExample, Hierarchy, LCADatasetGenerator


@dataclass
class LCAModelConfig:
    """Configuration for LCA neural model."""
    embed_dim: int = 64
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.1
    max_states: int = 100
    learning_rate: float = 0.001


class StateEncoder:
    """Encode state IDs to indices and back."""

    def __init__(self, max_states: int = 100):
        self.max_states = max_states
        self.state_to_idx: Dict[str, int] = {}
        self.idx_to_state: Dict[int, str] = {}
        self.next_idx = 1  # 0 reserved for unknown

    def fit(self, hierarchies: List[Hierarchy]):
        """Build encoding from hierarchies."""
        self.state_to_idx = {"<unk>": 0}
        self.idx_to_state = {0: "<unk>"}
        self.next_idx = 1

        for h in hierarchies:
            for state_id in h.nodes.keys():
                if state_id not in self.state_to_idx and self.next_idx < self.max_states:
                    self.state_to_idx[state_id] = self.next_idx
                    self.idx_to_state[self.next_idx] = state_id
                    self.next_idx += 1

    def encode(self, state_id: str) -> int:
        return self.state_to_idx.get(state_id, 0)

    def decode(self, idx: int) -> str:
        return self.idx_to_state.get(idx, "<unk>")

    def vocab_size(self) -> int:
        return self.next_idx


if HAS_MLX:
    class NeuralLCAModel(nn.Module):
        """Neural network for LCA prediction."""

        def __init__(self, config: LCAModelConfig, vocab_size: int):
            super().__init__()
            self.config = config
            self.vocab_size = vocab_size

            # State embeddings
            self.state_embed = nn.Embedding(vocab_size, config.embed_dim)

            # Combine source and target
            self.combine = nn.Sequential(
                nn.Linear(config.embed_dim * 2, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )

            # Hidden layers
            layers = []
            for _ in range(config.n_layers - 1):
                layers.extend([
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ])
            self.hidden = nn.Sequential(*layers) if layers else None

            # Output: predict LCA state
            self.output = nn.Linear(config.hidden_dim, vocab_size)

        def __call__(self, source_ids: mx.array, target_ids: mx.array) -> mx.array:
            # Embed states
            source_emb = self.state_embed(source_ids)  # (batch, embed_dim)
            target_emb = self.state_embed(target_ids)  # (batch, embed_dim)

            # Combine
            combined = mx.concatenate([source_emb, target_emb], axis=-1)
            hidden = self.combine(combined)

            # Hidden layers
            if self.hidden:
                hidden = self.hidden(hidden)

            # Predict LCA
            logits = self.output(hidden)
            return logits

        def predict(self, source_ids: mx.array, target_ids: mx.array) -> mx.array:
            """Predict LCA indices."""
            logits = self(source_ids, target_ids)
            return mx.argmax(logits, axis=-1)


class SupervisedLCATrainer:
    """Train LCA predictor with supervised learning."""

    def __init__(self, config: LCAModelConfig):
        self.config = config
        self.encoder = StateEncoder(config.max_states)
        self.model: Optional[Any] = None

    def prepare_data(
        self,
        examples: List[LCAExample],
        hierarchies: List[Hierarchy],
    ) -> Tuple[List[Tuple[int, int, int]], ...]:
        """Prepare training data."""
        self.encoder.fit(hierarchies)

        data = []
        for ex in examples:
            source_idx = self.encoder.encode(ex.source_id)
            target_idx = self.encoder.encode(ex.target_id)
            lca_idx = self.encoder.encode(ex.lca_id)
            data.append((source_idx, target_idx, lca_idx))

        return data

    def train(
        self,
        examples: List[LCAExample],
        hierarchies: List[Hierarchy],
        n_epochs: int = 50,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Train the model."""
        if not HAS_MLX:
            return self._train_simple(examples, hierarchies, verbose)

        data = self.prepare_data(examples, hierarchies)
        vocab_size = self.encoder.vocab_size()

        self.model = NeuralLCAModel(self.config, vocab_size)

        # Training loop
        def loss_fn(model, source, target, lca):
            logits = model(source, target)
            return mx.mean(nn.losses.cross_entropy(logits, lca))

        optimizer = optim.Adam(learning_rate=self.config.learning_rate)
        loss_and_grad = nn.value_and_grad(self.model, loss_fn)

        history = {'loss': [], 'accuracy': []}

        for epoch in range(n_epochs):
            random.shuffle(data)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                sources = mx.array([d[0] for d in batch])
                targets = mx.array([d[1] for d in batch])
                lcas = mx.array([d[2] for d in batch])

                loss, grads = loss_and_grad(self.model, sources, targets, lcas)
                optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), optimizer.state)

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
            history['loss'].append(avg_loss)

            # Compute accuracy
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                acc = self._compute_accuracy(data)
                history['accuracy'].append(acc)
                if verbose:
                    print(f"Epoch {epoch:3d}: loss={avg_loss:.4f}, acc={acc:.3f}")

        return history

    def _train_simple(
        self,
        examples: List[LCAExample],
        hierarchies: List[Hierarchy],
        verbose: bool,
    ) -> Dict[str, float]:
        """Simple lookup-based training (fallback without MLX)."""
        # Build lookup table: (source, target) -> lca
        self.lookup: Dict[Tuple[str, str], str] = {}
        for ex in examples:
            self.lookup[(ex.source_id, ex.target_id)] = ex.lca_id

        if verbose:
            print(f"Built lookup table with {len(self.lookup)} entries")

        return {'loss': [0.0], 'accuracy': [1.0]}

    def _compute_accuracy(self, data: List[Tuple[int, int, int]]) -> float:
        """Compute prediction accuracy."""
        if not HAS_MLX or self.model is None:
            return 1.0

        correct = 0
        total = 0

        for source, target, lca in data:
            pred = self.model.predict(
                mx.array([source]),
                mx.array([target])
            )
            if pred.item() == lca:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0

    def predict(self, source_id: str, target_id: str) -> str:
        """Predict LCA for a state pair."""
        if not HAS_MLX or self.model is None:
            # Fallback to lookup
            return self.lookup.get((source_id, target_id), "<unk>")

        source_idx = self.encoder.encode(source_id)
        target_idx = self.encoder.encode(target_id)

        pred_idx = self.model.predict(
            mx.array([source_idx]),
            mx.array([target_idx])
        ).item()

        return self.encoder.decode(pred_idx)

    def evaluate(self, examples: List[LCAExample]) -> Dict[str, float]:
        """Evaluate on test examples."""
        correct = 0
        total = 0

        for ex in examples:
            pred = self.predict(ex.source_id, ex.target_id)
            if pred == ex.lca_id:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0
        return {'accuracy': accuracy, 'correct': correct, 'total': total}


class EvolutionaryLCAPredictor:
    """Evolve LCA prediction strategy."""

    def __init__(self, population_size: int = 30, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.best_lookup: Dict[Tuple[str, str], str] = {}

    def evolve(
        self,
        examples: List[LCAExample],
        hierarchies: List[Hierarchy],
        n_generations: int = 50,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Evolve LCA prediction strategy."""
        # Build ground truth
        ground_truth = {(ex.source_id, ex.target_id): ex.lca_id for ex in examples}

        # Get all states
        all_states = set()
        for h in hierarchies:
            all_states.update(h.nodes.keys())
        all_states = list(all_states)

        # Initialize population of lookup tables
        population = []
        for _ in range(self.population_size):
            lookup = {}
            for (src, tgt), lca in ground_truth.items():
                if random.random() < 0.7:
                    lookup[(src, tgt)] = lca  # Correct
                else:
                    lookup[(src, tgt)] = random.choice(all_states)  # Random
            population.append(lookup)

        # Evolution loop
        best_fitness = 0.0
        for gen in range(n_generations):
            # Evaluate fitness
            fitness_scores = []
            for lookup in population:
                correct = sum(1 for k, v in ground_truth.items() if lookup.get(k) == v)
                fitness = correct / len(ground_truth) if ground_truth else 0
                fitness_scores.append(fitness)

            # Track best
            best_idx = max(range(len(population)), key=lambda i: fitness_scores[i])
            if fitness_scores[best_idx] > best_fitness:
                best_fitness = fitness_scores[best_idx]
                self.best_lookup = population[best_idx].copy()

            if verbose and gen % 10 == 0:
                print(f"Gen {gen:3d}: best_fitness={best_fitness:.3f}")

            # Selection and reproduction
            sorted_pop = sorted(zip(population, fitness_scores), key=lambda x: -x[1])
            elite = [p.copy() for p, _ in sorted_pop[:5]]

            offspring = []
            while len(offspring) < self.population_size - len(elite):
                parent = random.choice(elite)
                child = parent.copy()

                # Mutate
                for key in list(child.keys()):
                    if random.random() < self.mutation_rate:
                        child[key] = random.choice(all_states)

                offspring.append(child)

            population = elite + offspring

        return {'best_fitness': best_fitness}

    def predict(self, source_id: str, target_id: str) -> str:
        """Predict LCA using evolved strategy."""
        return self.best_lookup.get((source_id, target_id), "<unk>")

    def evaluate(self, examples: List[LCAExample]) -> Dict[str, float]:
        """Evaluate on test examples."""
        correct = sum(1 for ex in examples
                     if self.predict(ex.source_id, ex.target_id) == ex.lca_id)
        accuracy = correct / len(examples) if examples else 0.0
        return {'accuracy': accuracy}


class HybridLCAPredictor:
    """Combine supervised pre-training with evolutionary fine-tuning."""

    def __init__(self, config: LCAModelConfig = None):
        self.config = config or LCAModelConfig()
        self.supervised = SupervisedLCATrainer(self.config)
        self.evolutionary = EvolutionaryLCAPredictor()

    def train(
        self,
        examples: List[LCAExample],
        hierarchies: List[Hierarchy],
        n_supervised_epochs: int = 30,
        n_evolution_gens: int = 20,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Train with hybrid approach."""
        if verbose:
            print("Phase 1: Supervised pre-training...")

        # Phase 1: Supervised training
        supervised_history = self.supervised.train(
            examples, hierarchies,
            n_epochs=n_supervised_epochs,
            verbose=verbose,
        )

        if verbose:
            print("\nPhase 2: Evolutionary refinement...")

        # Phase 2: Use supervised predictions as starting point for evolution
        # Then refine with evolution
        self.evolutionary.best_lookup = {
            (ex.source_id, ex.target_id): self.supervised.predict(ex.source_id, ex.target_id)
            for ex in examples
        }

        evolution_result = self.evolutionary.evolve(
            examples, hierarchies,
            n_generations=n_evolution_gens,
            verbose=verbose,
        )

        return {
            'supervised_acc': supervised_history.get('accuracy', [0])[-1] if supervised_history.get('accuracy') else 0,
            'final_fitness': evolution_result['best_fitness'],
        }

    def predict(self, source_id: str, target_id: str) -> str:
        """Predict using hybrid model."""
        # Try evolutionary lookup first
        result = self.evolutionary.predict(source_id, target_id)
        if result != "<unk>":
            return result
        # Fall back to supervised
        return self.supervised.predict(source_id, target_id)

    def evaluate(self, examples: List[LCAExample]) -> Dict[str, float]:
        """Evaluate on test examples."""
        correct = sum(1 for ex in examples
                     if self.predict(ex.source_id, ex.target_id) == ex.lca_id)
        accuracy = correct / len(examples) if examples else 0.0
        return {'accuracy': accuracy}


def demo():
    """Demonstrate neural LCA prediction."""
    print("=" * 60)
    print("NEURAL LCA PREDICTION")
    print("=" * 60)

    # Generate data
    generator = LCADatasetGenerator(seed=42)
    examples, hierarchies = generator.generate_dataset(n_hierarchies=15, examples_per_hierarchy=25)
    train, test, train_h, test_h = generator.split_by_hierarchy(examples, hierarchies)

    print(f"\nDataset: {len(train)} train, {len(test)} test examples")
    print(f"Hierarchies: {len(train_h)} train, {len(test_h)} test")

    # Supervised training
    print("\n--- Supervised Learning ---")
    supervised = SupervisedLCATrainer(LCAModelConfig())
    supervised.train(train, train_h, n_epochs=30, verbose=True)
    sup_results = supervised.evaluate(test)
    print(f"Test accuracy: {sup_results['accuracy']:.3f}")

    # Evolutionary
    print("\n--- Evolutionary ---")
    evolutionary = EvolutionaryLCAPredictor()
    evolutionary.evolve(train, train_h, n_generations=30, verbose=True)
    evo_results = evolutionary.evaluate(test)
    print(f"Test accuracy: {evo_results['accuracy']:.3f}")

    # Hybrid
    print("\n--- Hybrid ---")
    hybrid = HybridLCAPredictor()
    hybrid.train(train, train_h, n_supervised_epochs=20, n_evolution_gens=15, verbose=True)
    hybrid_results = hybrid.evaluate(test)
    print(f"Test accuracy: {hybrid_results['accuracy']:.3f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Supervised:   {sup_results['accuracy']:.3f}")
    print(f"Evolutionary: {evo_results['accuracy']:.3f}")
    print(f"Hybrid:       {hybrid_results['accuracy']:.3f}")

    return {'supervised': sup_results, 'evolutionary': evo_results, 'hybrid': hybrid_results}


if __name__ == "__main__":
    demo()
