"""
LCA Dataset Generation

Generate (source_state, target_state) -> LCA training examples
from hierarchical statecharts.

LCA (Lowest Common Ancestor) is critical for inter-level transitions:
- Determines exit path from source
- Determines entry path to target
- Affects which entry/exit actions execute
"""

import random
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from enum import Enum, auto


class StateType(Enum):
    BASIC = auto()      # Leaf state
    COMPOSITE = auto()  # Has children (OR-state)
    PARALLEL = auto()   # Concurrent regions (AND-state)
    INITIAL = auto()    # Initial pseudo-state
    FINAL = auto()      # Final pseudo-state


@dataclass
class HierarchyNode:
    """A node in the state hierarchy."""
    id: str
    name: str
    state_type: StateType
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    depth: int = 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent_id is None


@dataclass
class LCAExample:
    """A single (source, target) -> LCA training example."""
    hierarchy_id: str
    source_id: str
    target_id: str
    lca_id: str
    source_depth: int
    target_depth: int
    lca_depth: int
    source_path: List[str]  # Path from root to source
    target_path: List[str]  # Path from root to target
    exit_path: List[str]    # States to exit (source to LCA)
    entry_path: List[str]   # States to enter (LCA to target)

    def to_dict(self) -> Dict:
        return {
            'hierarchy_id': self.hierarchy_id,
            'source': self.source_id,
            'target': self.target_id,
            'lca': self.lca_id,
            'source_depth': self.source_depth,
            'target_depth': self.target_depth,
            'lca_depth': self.lca_depth,
        }


@dataclass
class Hierarchy:
    """A complete state hierarchy."""
    id: str
    nodes: Dict[str, HierarchyNode]
    root_id: str
    max_depth: int

    def get_path_to_root(self, node_id: str) -> List[str]:
        """Get path from node to root (inclusive)."""
        path = []
        current = node_id
        while current is not None:
            path.append(current)
            node = self.nodes.get(current)
            current = node.parent_id if node else None
        return path

    def get_path_from_root(self, node_id: str) -> List[str]:
        """Get path from root to node (inclusive)."""
        return list(reversed(self.get_path_to_root(node_id)))

    def compute_lca(self, source_id: str, target_id: str) -> str:
        """Compute LCA of two states."""
        source_path = set(self.get_path_to_root(source_id))
        target_ancestors = self.get_path_to_root(target_id)

        for ancestor in target_ancestors:
            if ancestor in source_path:
                return ancestor

        return self.root_id

    def get_leaf_states(self) -> List[str]:
        """Get all leaf (basic) states."""
        return [nid for nid, node in self.nodes.items() if node.is_leaf()]

    def get_all_states(self) -> List[str]:
        """Get all state IDs."""
        return list(self.nodes.keys())


class HierarchyGenerator:
    """Generate random hierarchies for training."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.counter = 0

    def generate_balanced(self, depth: int = 3, branching: int = 3) -> Hierarchy:
        """Generate balanced tree hierarchy."""
        nodes = {}
        root_id = "root"

        def create_subtree(parent_id: Optional[str], current_depth: int, prefix: str) -> str:
            node_id = f"{prefix}" if prefix else "root"
            node = HierarchyNode(
                id=node_id,
                name=node_id,
                state_type=StateType.BASIC if current_depth == depth else StateType.COMPOSITE,
                parent_id=parent_id,
                depth=current_depth,
            )
            nodes[node_id] = node

            if current_depth < depth:
                for i in range(branching):
                    child_prefix = f"{prefix}_{i}" if prefix else f"s{i}"
                    child_id = create_subtree(node_id, current_depth + 1, child_prefix)
                    node.children.append(child_id)

            return node_id

        create_subtree(None, 0, "")

        hierarchy_id = f"balanced_d{depth}_b{branching}_{self.counter}"
        self.counter += 1

        return Hierarchy(id=hierarchy_id, nodes=nodes, root_id=root_id, max_depth=depth)

    def generate_unbalanced(self, max_depth: int = 4, max_children: int = 4) -> Hierarchy:
        """Generate unbalanced tree hierarchy."""
        nodes = {}
        root_id = "root"

        def create_subtree(parent_id: Optional[str], current_depth: int, prefix: str) -> str:
            node_id = f"{prefix}" if prefix else "root"

            # Randomly decide if leaf
            is_leaf = current_depth >= max_depth or (current_depth > 1 and self.rng.random() < 0.3)

            node = HierarchyNode(
                id=node_id,
                name=node_id,
                state_type=StateType.BASIC if is_leaf else StateType.COMPOSITE,
                parent_id=parent_id,
                depth=current_depth,
            )
            nodes[node_id] = node

            if not is_leaf:
                n_children = self.rng.randint(1, max_children)
                for i in range(n_children):
                    child_prefix = f"{prefix}_{i}" if prefix else f"s{i}"
                    child_id = create_subtree(node_id, current_depth + 1, child_prefix)
                    node.children.append(child_id)

            return node_id

        create_subtree(None, 0, "")

        hierarchy_id = f"unbalanced_{self.counter}"
        self.counter += 1

        actual_max_depth = max(n.depth for n in nodes.values())
        return Hierarchy(id=hierarchy_id, nodes=nodes, root_id=root_id, max_depth=actual_max_depth)

    def generate_game_like(self) -> Hierarchy:
        """Generate game-like hierarchy (menus, gameplay, pause, etc.)."""
        nodes = {}

        # Root
        nodes["root"] = HierarchyNode(
            id="root", name="Game", state_type=StateType.COMPOSITE, depth=0,
            children=["menu", "gameplay", "pause", "gameover"]
        )

        # Menu subtree
        nodes["menu"] = HierarchyNode(
            id="menu", name="Menu", state_type=StateType.COMPOSITE, parent_id="root", depth=1,
            children=["main_menu", "settings", "credits"]
        )
        for child in ["main_menu", "settings", "credits"]:
            nodes[child] = HierarchyNode(
                id=child, name=child, state_type=StateType.BASIC, parent_id="menu", depth=2
            )

        # Gameplay subtree
        nodes["gameplay"] = HierarchyNode(
            id="gameplay", name="Gameplay", state_type=StateType.COMPOSITE, parent_id="root", depth=1,
            children=["exploring", "combat", "dialogue", "inventory"]
        )
        for child in ["exploring", "combat", "dialogue", "inventory"]:
            nodes[child] = HierarchyNode(
                id=child, name=child, state_type=StateType.BASIC, parent_id="gameplay", depth=2
            )

        # Simple states
        for state in ["pause", "gameover"]:
            nodes[state] = HierarchyNode(
                id=state, name=state, state_type=StateType.BASIC, parent_id="root", depth=1
            )

        hierarchy_id = f"game_{self.counter}"
        self.counter += 1

        return Hierarchy(id=hierarchy_id, nodes=nodes, root_id="root", max_depth=2)


class LCADatasetGenerator:
    """Generate LCA training datasets."""

    def __init__(self, seed: int = 42):
        self.hierarchy_gen = HierarchyGenerator(seed)
        self.rng = random.Random(seed)

    def generate_examples_from_hierarchy(
        self,
        hierarchy: Hierarchy,
        n_examples: int = 50,
        leaf_only: bool = False,
    ) -> List[LCAExample]:
        """Generate LCA examples from a hierarchy."""
        if leaf_only:
            states = hierarchy.get_leaf_states()
        else:
            states = hierarchy.get_all_states()

        if len(states) < 2:
            return []

        examples = []
        for _ in range(n_examples):
            source, target = self.rng.sample(states, 2)
            example = self._create_example(hierarchy, source, target)
            examples.append(example)

        return examples

    def _create_example(self, hierarchy: Hierarchy, source: str, target: str) -> LCAExample:
        """Create a single LCA example."""
        lca = hierarchy.compute_lca(source, target)

        source_path = hierarchy.get_path_from_root(source)
        target_path = hierarchy.get_path_from_root(target)
        lca_path = hierarchy.get_path_from_root(lca)

        # Exit path: source up to (but not including) LCA
        exit_path = list(reversed(source_path[len(lca_path):]))

        # Entry path: LCA children down to target
        entry_path = target_path[len(lca_path):]

        return LCAExample(
            hierarchy_id=hierarchy.id,
            source_id=source,
            target_id=target,
            lca_id=lca,
            source_depth=hierarchy.nodes[source].depth,
            target_depth=hierarchy.nodes[target].depth,
            lca_depth=hierarchy.nodes[lca].depth,
            source_path=source_path,
            target_path=target_path,
            exit_path=exit_path,
            entry_path=entry_path,
        )

    def generate_dataset(
        self,
        n_hierarchies: int = 20,
        examples_per_hierarchy: int = 30,
        include_game_like: bool = True,
    ) -> Tuple[List[LCAExample], List[Hierarchy]]:
        """Generate complete LCA dataset."""
        hierarchies = []
        all_examples = []

        # Balanced hierarchies
        for depth in [2, 3, 4]:
            for branching in [2, 3]:
                h = self.hierarchy_gen.generate_balanced(depth, branching)
                hierarchies.append(h)
                examples = self.generate_examples_from_hierarchy(h, examples_per_hierarchy)
                all_examples.extend(examples)

        # Unbalanced hierarchies
        for _ in range(n_hierarchies // 2):
            h = self.hierarchy_gen.generate_unbalanced()
            hierarchies.append(h)
            examples = self.generate_examples_from_hierarchy(h, examples_per_hierarchy)
            all_examples.extend(examples)

        # Game-like hierarchies
        if include_game_like:
            for _ in range(3):
                h = self.hierarchy_gen.generate_game_like()
                hierarchies.append(h)
                examples = self.generate_examples_from_hierarchy(h, examples_per_hierarchy)
                all_examples.extend(examples)

        return all_examples, hierarchies

    def split_by_hierarchy(
        self,
        examples: List[LCAExample],
        hierarchies: List[Hierarchy],
        train_ratio: float = 0.7,
    ) -> Tuple[List[LCAExample], List[LCAExample], List[Hierarchy], List[Hierarchy]]:
        """Split dataset by hierarchy for generalization testing."""
        n_train = int(len(hierarchies) * train_ratio)
        train_hierarchies = hierarchies[:n_train]
        test_hierarchies = hierarchies[n_train:]

        train_ids = {h.id for h in train_hierarchies}
        test_ids = {h.id for h in test_hierarchies}

        train_examples = [e for e in examples if e.hierarchy_id in train_ids]
        test_examples = [e for e in examples if e.hierarchy_id in test_ids]

        return train_examples, test_examples, train_hierarchies, test_hierarchies


def demo():
    """Demonstrate LCA dataset generation."""
    print("=" * 60)
    print("LCA DATASET GENERATION")
    print("=" * 60)

    generator = LCADatasetGenerator(seed=42)
    examples, hierarchies = generator.generate_dataset(n_hierarchies=10, examples_per_hierarchy=20)

    print(f"\nGenerated {len(examples)} examples from {len(hierarchies)} hierarchies")

    # Show example
    if examples:
        ex = examples[0]
        print(f"\nExample:")
        print(f"  Source: {ex.source_id} (depth={ex.source_depth})")
        print(f"  Target: {ex.target_id} (depth={ex.target_depth})")
        print(f"  LCA: {ex.lca_id} (depth={ex.lca_depth})")
        print(f"  Exit path: {ex.exit_path}")
        print(f"  Entry path: {ex.entry_path}")

    # Split for generalization
    train, test, train_h, test_h = generator.split_by_hierarchy(examples, hierarchies)
    print(f"\nTrain/test split (by hierarchy):")
    print(f"  Train: {len(train)} examples from {len(train_h)} hierarchies")
    print(f"  Test: {len(test)} examples from {len(test_h)} hierarchies")

    return examples, hierarchies


if __name__ == "__main__":
    demo()
