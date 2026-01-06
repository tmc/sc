"""
Data loader for Sudoku-Extreme dataset from HuggingFace.

Dataset: sapientinc/sudoku-extreme
Format: CSV with columns (source, question, answer, rating)
- question: 81-char string ('.' = empty, '1'-'9' = filled)
- answer: 81-char string (full solution)
- rating: difficulty rating
"""

import csv
import os
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import mlx.core as mx
import numpy as np

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


@dataclass
class SudokuPuzzle:
    """Single Sudoku puzzle."""
    question: np.ndarray  # [81] int values 0-9 (0 = empty)
    answer: np.ndarray    # [81] int values 1-9
    rating: int           # difficulty rating


def shuffle_sudoku(
    question: np.ndarray,
    answer: np.ndarray,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply valid Sudoku transformations for data augmentation.

    Transformations preserve constraint validity:
    - Digit permutation (1-9 shuffled, 0 unchanged)
    - Transpose
    - Row band permutation (3 bands of 3 rows)
    - Column stack permutation (3 stacks of 3 columns)
    - Row permutation within bands
    - Column permutation within stacks

    Based on TRM's shuffle_sudoku implementation.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Digit mapping: permute 1-9, keep 0 unchanged
    digit_perm = np.concatenate([[0], rng.permutation(np.arange(1, 10))])

    # Transpose flag
    transpose = rng.random() < 0.5

    # Row permutation: shuffle 3 bands, then shuffle rows within each band
    bands = rng.permutation(3)
    row_perm = np.concatenate([
        b * 3 + rng.permutation(3) for b in bands
    ])

    # Column permutation: shuffle 3 stacks, then shuffle columns within each stack
    stacks = rng.permutation(3)
    col_perm = np.concatenate([
        s * 3 + rng.permutation(3) for s in stacks
    ])

    def transform(board: np.ndarray) -> np.ndarray:
        # Reshape to 9x9
        b = board.reshape(9, 9).copy()

        # Transpose
        if transpose:
            b = b.T

        # Apply row/column permutation
        b = b[row_perm][:, col_perm]

        # Apply digit permutation
        b = digit_perm[b]

        return b.flatten()

    return transform(question), transform(answer)


class SudokuExtremeDataset:
    """
    Dataset wrapper for Sudoku-Extreme.

    Supports:
    - Loading train/test splits
    - Data augmentation via shuffle_sudoku
    - Batching with MLX arrays
    """

    def __init__(
        self,
        split: str = "train",
        data_dir: Optional[str] = None,
        min_difficulty: Optional[int] = None,
        max_samples: Optional[int] = None,
        num_augmentations: int = 0,
        seed: int = 42,
    ):
        """
        Args:
            split: "train" or "test"
            data_dir: Directory with cached data (or None to download)
            min_difficulty: Minimum puzzle rating (filter easy puzzles)
            max_samples: Maximum number of base puzzles to load
            num_augmentations: Number of augmented versions per puzzle
            seed: Random seed for augmentation
        """
        self.split = split
        self.min_difficulty = min_difficulty
        self.num_augmentations = num_augmentations
        self.rng = np.random.default_rng(seed)

        # Load data
        self.puzzles = self._load_data(data_dir, max_samples)
        print(f"Loaded {len(self.puzzles)} {split} puzzles")

    def _load_data(
        self,
        data_dir: Optional[str],
        max_samples: Optional[int]
    ) -> list[SudokuPuzzle]:
        """Load puzzles from HuggingFace or local cache."""

        # Try to download from HuggingFace
        if hf_hub_download is not None:
            try:
                csv_path = hf_hub_download(
                    "sapientinc/sudoku-extreme",
                    f"{self.split}.csv",
                    repo_type="dataset"
                )
            except Exception as e:
                print(f"Warning: Could not download from HuggingFace: {e}")
                csv_path = None
        else:
            csv_path = None

        # Fallback to local path
        if csv_path is None and data_dir is not None:
            csv_path = os.path.join(data_dir, f"{self.split}.csv")

        if csv_path is None or not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Could not find Sudoku-Extreme {self.split} data. "
                "Install huggingface_hub or provide data_dir."
            )

        # Parse CSV
        puzzles = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header

            for row in reader:
                if len(row) < 4:
                    continue

                source, question, answer, rating = row[:4]

                # Parse strings to arrays
                q = np.array([
                    0 if c == '.' else int(c)
                    for c in question
                ], dtype=np.int32)

                a = np.array([int(c) for c in answer], dtype=np.int32)
                r = int(rating)

                # Filter by difficulty
                if self.min_difficulty is not None and r < self.min_difficulty:
                    continue

                puzzles.append(SudokuPuzzle(question=q, answer=a, rating=r))

                if max_samples is not None and len(puzzles) >= max_samples:
                    break

        return puzzles

    def __len__(self) -> int:
        """Total samples including augmentations."""
        return len(self.puzzles) * (1 + self.num_augmentations)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a puzzle (with augmentation if applicable).

        Returns:
            question: [81] int array (0 = empty, 1-9 = digit)
            answer: [81] int array (1-9)
        """
        base_idx = idx // (1 + self.num_augmentations)
        aug_idx = idx % (1 + self.num_augmentations)

        puzzle = self.puzzles[base_idx]

        if aug_idx == 0:
            # Original puzzle
            return puzzle.question.copy(), puzzle.answer.copy()
        else:
            # Augmented version
            return shuffle_sudoku(puzzle.question, puzzle.answer, self.rng)

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> Iterator[Tuple[mx.array, mx.array]]:
        """
        Iterate over batches as MLX arrays.

        Yields:
            questions: [B, 81] MLX array
            answers: [B, 81] MLX array
        """
        n = len(self)
        indices = np.arange(n)

        if shuffle:
            self.rng.shuffle(indices)

        for start in range(0, n, batch_size):
            batch_indices = indices[start:start + batch_size]

            questions = []
            answers = []

            for idx in batch_indices:
                q, a = self[idx]
                questions.append(q)
                answers.append(a)

            yield (
                mx.array(np.stack(questions)),
                mx.array(np.stack(answers))
            )


def load_sudoku_extreme(
    split: str = "train",
    batch_size: int = 64,
    **kwargs
) -> SudokuExtremeDataset:
    """Convenience function to load dataset."""
    return SudokuExtremeDataset(split=split, **kwargs)


def test_data_loader():
    """Test the data loader."""
    print("=" * 60)
    print("Testing Sudoku-Extreme Data Loader")
    print("=" * 60)

    # Create dataset (small subset for testing)
    try:
        dataset = SudokuExtremeDataset(
            split="train",
            max_samples=100,
            num_augmentations=2,
            seed=42
        )
    except FileNotFoundError as e:
        print(f"Skipping test: {e}")
        return

    print(f"\n1. Dataset size: {len(dataset)} (100 base + 200 augmented)")

    # Get a sample
    q, a = dataset[0]
    print(f"\n2. Sample puzzle:")
    print(f"   Question shape: {q.shape}")
    print(f"   Answer shape: {a.shape}")
    print(f"   Empty cells: {np.sum(q == 0)}")

    # Visualize
    print(f"\n   Question (9x9):")
    for row in q.reshape(9, 9):
        print("   ", " ".join(str(x) if x > 0 else "." for x in row))

    print(f"\n   Answer (9x9):")
    for row in a.reshape(9, 9):
        print("   ", " ".join(str(x) for x in row))

    # Test augmentation
    print(f"\n3. Testing augmentation...")
    q_aug, a_aug = dataset[1]  # First augmented version
    print(f"   Original empty cells: {np.sum(q == 0)}")
    print(f"   Augmented empty cells: {np.sum(q_aug == 0)}")
    print(f"   Arrays equal: {np.array_equal(q, q_aug)}")

    # Test batching
    print(f"\n4. Testing batch iteration...")
    for i, (batch_q, batch_a) in enumerate(dataset.iter_batches(batch_size=32)):
        print(f"   Batch {i}: questions={batch_q.shape}, answers={batch_a.shape}")
        if i >= 2:
            print("   ...")
            break

    print("\n" + "=" * 60)
    print("Data loader test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_data_loader()
