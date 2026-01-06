"""
Sudoku Data Loader: Load real Sudoku puzzles from HuggingFace.

Dataset: sapientinc/sudoku-extreme (or similar)
Format: CSV with question (81 chars), answer (81 chars), rating
"""

import sys
import random
from typing import Tuple, Optional, List

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

import mlx.core as mx

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')


def parse_sudoku_string(s: str) -> List[int]:
    """
    Parse Sudoku string to list of integers.

    '.' or '0' = empty (0)
    '1'-'9' = given digit
    """
    arr = [0] * 81
    for i, c in enumerate(s[:81]):
        if c in '123456789':
            arr[i] = int(c)
        # '.' or '0' stays as 0
    return arr


def load_sudoku_dataset(
    num_train: int = 1000,
    num_test: int = 200,
    dataset_name: str = "gsurma/sudoku",
    seed: int = 42,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """
    Load Sudoku puzzles from HuggingFace.

    Args:
        num_train: Number of training puzzles
        num_test: Number of test puzzles
        dataset_name: HuggingFace dataset name
        seed: Random seed for shuffling

    Returns:
        (train_puzzles, train_solutions, test_puzzles, test_solutions)
        Each is [N, 81] int32 array
    """
    if not HAS_DATASETS:
        print("Warning: datasets library not available, generating random data")
        return generate_random_sudoku(num_train, num_test, seed)

    try:
        print(f"Loading dataset: {dataset_name}")
        ds = load_dataset(dataset_name, split="train")

        # Shuffle (inefficiently converting to list if shuffle not avail, but ds usually handles it)
        try:
            ds = ds.shuffle(seed=seed)
        except Exception:
            pass

        # Take subset
        total_needed = num_train + num_test
        if len(ds) < total_needed:
            print(f"Warning: Dataset has {len(ds)} samples, need {total_needed}")
            total_needed = len(ds)
            num_train = int(total_needed * 0.8)
            num_test = total_needed - num_train

        # Parse puzzles and solutions
        train_puzzles = []
        train_solutions = []
        test_puzzles = []
        test_solutions = []

        # Detect column names (different datasets use different names)
        sample = ds[0]
        if 'puzzle' in sample:
            puzzle_col, solution_col = 'puzzle', 'solution'
        elif 'quizzes' in sample:
            puzzle_col, solution_col = 'quizzes', 'solutions'
        elif 'question' in sample:
            puzzle_col, solution_col = 'question', 'answer'
        else:
            # Try to find string columns
            puzzle_col = solution_col = None
            for k, v in sample.items():
                if isinstance(v, str) and len(v) >= 81:
                    if puzzle_col is None:
                        puzzle_col = k
                    else:
                        solution_col = k
                        break
            if puzzle_col is None or solution_col is None:
                print(f"Warning: Could not detect columns in {list(sample.keys())}")
                return generate_random_sudoku(num_train, num_test, seed)

        print(f"Using columns: puzzle={puzzle_col}, solution={solution_col}")

        # Manual reservoir sampling or iterating if not shuffled?
        # Assuming shuffled or just taking first N
        
        count = 0
        for item in ds:
            if count >= total_needed:
                break

            puzzle = parse_sudoku_string(item[puzzle_col])
            solution = parse_sudoku_string(item[solution_col])

            # Validate solution has all digits 1-9
            # Pure python check
            valid_sol = True
            for x in solution:
                if x < 1 or x > 9:
                    valid_sol = False
                    break
            
            if valid_sol:
                if len(train_puzzles) < num_train:
                    train_puzzles.append(puzzle)
                    train_solutions.append(solution)
                else:
                    test_puzzles.append(puzzle)
                    test_solutions.append(solution)
                count += 1

        print(f"Loaded {len(train_puzzles)} train, {len(test_puzzles)} test puzzles")

        # Convert to MLX
        return (
            mx.array(train_puzzles, dtype=mx.int32),
            mx.array(train_solutions, dtype=mx.int32),
            mx.array(test_puzzles, dtype=mx.int32),
            mx.array(test_solutions, dtype=mx.int32),
        )

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to random data")
        return generate_random_sudoku(num_train, num_test, seed)


def generate_valid_sudoku_solution() -> List[int]:
    """Generate a valid Sudoku solution using backtracking."""
    grid = [0] * 81

    def get_row(g, r): return g[r*9:(r+1)*9]
    def get_col(g, c): return [g[r*9 + c] for r in range(9)]
    def get_box(g, r, c):
        br, bc = 3 * (r // 3), 3 * (c // 3)
        vals = []
        for i in range(3):
            start = (br + i) * 9 + bc
            vals.extend(g[start:start+3])
        return vals

    def is_valid(g, idx, num):
        row, col = idx // 9, idx % 9
        
        # Row
        if num in g[row*9:(row+1)*9]: return False
        
        # Col
        for r in range(9):
            if g[r*9 + col] == num: return False
            
        # Box
        br, bc = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            start = (br + i) * 9 + bc
            for j in range(3):
                if g[start + j] == num: return False
                
        return True

    def solve(idx):
        if idx == 81:
            return True
        
        row, col = idx // 9, idx % 9
        
        # Shuffle digits for variety
        digits = list(range(1, 10))
        random.shuffle(digits)
        
        for num in digits:
            if is_valid(grid, idx, num):
                grid[idx] = num
                if solve(idx + 1):
                    return True
                grid[idx] = 0
        return False

    solve(0)
    return grid


def generate_random_sudoku(
    num_train: int = 1000,
    num_test: int = 200,
    seed: int = 42,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """
    Generate valid Sudoku puzzles with solutions.

    Uses backtracking to create valid solutions, then masks cells.
    """
    random.seed(seed)
    print(f"Generating {num_train + num_test} valid Sudoku puzzles...")

    train_solutions = []
    train_puzzles = []
    test_solutions = []
    test_puzzles = []

    total = num_train + num_test
    for i in range(total):
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{total}")

        # Generate valid solution
        solution = generate_valid_sudoku_solution()

        # Create puzzle by masking cells
        puzzle = list(solution) # Copy
        num_mask = random.randint(45, 55)  # 45-55 empty cells
        
        indices = list(range(81))
        mask_cells = random.sample(indices, num_mask)
        for idx in mask_cells:
            puzzle[idx] = 0

        if i < num_train:
            train_solutions.append(solution)
            train_puzzles.append(puzzle)
        else:
            test_solutions.append(solution)
            test_puzzles.append(puzzle)

    print(f"Generated {len(train_puzzles)} train, {len(test_puzzles)} test puzzles")

    return (
        mx.array(train_puzzles, dtype=mx.int32),
        mx.array(train_solutions, dtype=mx.int32),
        mx.array(test_puzzles, dtype=mx.int32),
        mx.array(test_solutions, dtype=mx.int32),
    )


def print_sudoku(puzzle: List[int], solution: Optional[List[int]] = None):
    """Pretty print a Sudoku puzzle."""
    
    # Handle mx.array if passed
    if hasattr(puzzle, 'tolist'):
        puzzle = puzzle.tolist()
    if solution is not None and hasattr(solution, 'tolist'):
        solution = solution.tolist()

    def format_row(arr, start):
        row = arr[start:start+9]
        parts = []
        for i in range(3):
            part = ' '.join(str(d) if d > 0 else '.' for d in row[i*3:(i+1)*3])
            parts.append(part)
        return ' | '.join(parts)

    print("Puzzle:")
    for r in range(9):
        print(format_row(puzzle, r * 9))
        if r in [2, 5]:
            print('-' * 21)

    if solution is not None:
        print("\nSolution:")
        for r in range(9):
            print(format_row(solution, r * 9))
            if r in [2, 5]:
                print('-' * 21)


def test_sudoku_data():
    """Test the data loader."""
    print("=" * 60)
    print("Testing Sudoku Data Loader (Pure MLX)")
    print("=" * 60)

    # Try to load random data (faster for test)
    train_p, train_s, test_p, test_s = generate_random_sudoku(
        num_train=10,
        num_test=5,
        seed=42
    )

    print(f"\nTrain puzzles: {train_p.shape}")
    print(f"Train solutions: {train_s.shape}")
    print(f"Test puzzles: {test_p.shape}")
    print(f"Test solutions: {test_s.shape}")

    # Show example
    print("\n" + "=" * 60)
    print("Example puzzle:")
    print("=" * 60)
    # Convert first row to list
    puzzle = train_p[0].tolist()
    solution = train_s[0].tolist()
    print_sudoku(puzzle, solution)

    # Statistics
    # MLX math for statistics
    given_counts = mx.sum(train_p > 0, axis=1)
    print(f"\nGiven cells per puzzle: mean={mx.mean(given_counts).item():.1f}, "
          f"min={mx.min(given_counts).item()}, max={mx.max(given_counts).item()}")

    print("\n" + "=" * 60)
    print("Sudoku data loader test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_sudoku_data()
