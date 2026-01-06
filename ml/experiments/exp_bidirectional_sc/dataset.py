"""
Bidirectional Dataset: Text ↔ Statechart pairs

Creates training pairs for both directions:
- Forward: description → SC JSON
- Reverse: SC JSON → description
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import random


@dataclass
class BidirectionalPair:
    """A text-SC pair for bidirectional training."""
    task_id: str
    description: str  # Natural language description
    statechart: str   # SC JSON string
    category: str
    difficulty: int


@dataclass
class BidirectionalDataset:
    """Dataset with forward and reverse examples."""
    pairs: List[BidirectionalPair]
    
    def get_forward_examples(self) -> List[Tuple[str, str]]:
        """Get text → SC examples."""
        return [(p.description, p.statechart) for p in self.pairs]
    
    def get_reverse_examples(self) -> List[Tuple[str, str]]:
        """Get SC → text examples."""
        return [(p.statechart, p.description) for p in self.pairs]
    
    def get_mixed_examples(self, shuffle: bool = True) -> List[Tuple[str, str, str]]:
        """Get mixed examples with direction label."""
        examples = []
        for p in self.pairs:
            examples.append((p.description, p.statechart, "forward"))
            examples.append((p.statechart, p.description, "reverse"))
        if shuffle:
            random.shuffle(examples)
        return examples
    
    def split(self, train_ratio: float = 0.8) -> Tuple['BidirectionalDataset', 'BidirectionalDataset']:
        """Split into train and test sets."""
        pairs = list(self.pairs)
        random.shuffle(pairs)
        split_idx = int(len(pairs) * train_ratio)
        return (
            BidirectionalDataset(pairs[:split_idx]),
            BidirectionalDataset(pairs[split_idx:])
        )
    
    def __len__(self) -> int:
        return len(self.pairs)


def create_dataset(benchmark_path: Optional[Path] = None) -> BidirectionalDataset:
    """
    Create bidirectional dataset from benchmark.
    
    Args:
        benchmark_path: Path to exp_sc_generation_benchmark
        
    Returns:
        BidirectionalDataset with all pairs
    """
    if benchmark_path is None:
        benchmark_path = Path(__file__).parent.parent / "exp_sc_generation_benchmark"
    
    # Load tasks
    tasks_path = benchmark_path / "benchmark_tasks.json"
    with open(tasks_path, 'r') as f:
        data = json.load(f)
        tasks = data["tasks"]
    
    # Load gold standards
    gold_dir = benchmark_path / "gold_standards"
    
    pairs = []
    for task in tasks:
        task_id = task["id"]
        gold_path = gold_dir / f"{task_id}.json"
        
        if gold_path.exists():
            with open(gold_path, 'r') as f:
                gold_sc = json.load(f)
            
            pairs.append(BidirectionalPair(
                task_id=task_id,
                description=task["prompt"],
                statechart=json.dumps(gold_sc, indent=2),
                category=task["category"],
                difficulty=task["difficulty"],
            ))
    
    return BidirectionalDataset(pairs)


def format_forward_prompt(description: str) -> str:
    """Format prompt for text → SC generation."""
    return f"""Generate a statechart in JSON format for the following description:

{description}

Output only valid JSON with root_state and transitions."""


def format_reverse_prompt(sc_json: str) -> str:
    """Format prompt for SC → text explanation."""
    return f"""Explain this statechart in natural language:

{sc_json}

Describe the states, transitions, and behavior concisely."""


def demo():
    """Demonstrate dataset creation."""
    print("Creating bidirectional dataset...")
    dataset = create_dataset()
    print(f"Total pairs: {len(dataset)}")
    
    train, test = dataset.split(0.8)
    print(f"Train: {len(train)}, Test: {len(test)}")
    
    # Show example
    if dataset.pairs:
        p = dataset.pairs[0]
        print(f"\nExample pair ({p.task_id}):")
        print(f"Description: {p.description[:100]}...")
        print(f"SC: {p.statechart[:100]}...")
    
    return dataset


if __name__ == "__main__":
    demo()
