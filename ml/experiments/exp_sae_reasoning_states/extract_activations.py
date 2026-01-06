"""
Extract activations from trained Sudoku statechart model.

Hooks into the model at each (H, L) iteration to capture:
1. Board state (soft probabilities for each cell)
2. Context embeddings (board encoder output)
3. Guard values (constraint satisfaction signals)
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class IterationActivation:
    """Activations from a single iteration."""
    h_cycle: int               # Outer cycle index
    l_cycle: int               # Inner cycle index
    board_states: np.ndarray   # [81, 10] soft states
    context: np.ndarray        # [hidden_dim] board context
    guard_values: np.ndarray   # [81, 9] guard values per cell


@dataclass
class PuzzleActivations:
    """All activations for a single puzzle."""
    puzzle_id: int
    question: np.ndarray       # [81] input
    answer: np.ndarray         # [81] target
    iterations: List[IterationActivation]
    final_prediction: np.ndarray  # [81] argmax output
    is_correct: bool


class ActivationDataset:
    """
    Dataset of extracted activations for SAE training.

    Structure:
        activations[puzzle_id][h_cycle][l_cycle] = IterationActivation
    """

    def __init__(self, save_dir: str = "activations"):
        self.save_dir = save_dir
        self.puzzles: List[PuzzleActivations] = []

    def add_puzzle(self, puzzle: PuzzleActivations):
        """Add a puzzle's activations."""
        self.puzzles.append(puzzle)

    def __len__(self) -> int:
        return len(self.puzzles)

    def get_all_activations(
        self,
        h_cycle: Optional[int] = None,
        l_cycle: Optional[int] = None,
        flatten_to_cells: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all activations, optionally filtered by cycle.

        Args:
            h_cycle: Filter to specific H cycle (None = all)
            l_cycle: Filter to specific L cycle (None = all)
            flatten_to_cells: If True, flatten to per-cell activations

        Returns:
            board_states: [N, 10] or [N, 81, 10] soft states
            contexts: [N, hidden_dim] context vectors
            guard_values: [N, 9] or [N, 81, 9] guards
        """
        board_states = []
        contexts = []
        guard_values = []

        for puzzle in self.puzzles:
            for it in puzzle.iterations:
                if h_cycle is not None and it.h_cycle != h_cycle:
                    continue
                if l_cycle is not None and it.l_cycle != l_cycle:
                    continue

                if flatten_to_cells:
                    # Each cell becomes a separate sample
                    for cell in range(81):
                        board_states.append(it.board_states[cell])
                        contexts.append(it.context)
                        guard_values.append(it.guard_values[cell])
                else:
                    board_states.append(it.board_states)
                    contexts.append(it.context)
                    guard_values.append(it.guard_values)

        return (
            np.stack(board_states),
            np.stack(contexts),
            np.stack(guard_values),
        )

    def get_iteration_trajectory(
        self,
        puzzle_idx: int,
    ) -> List[IterationActivation]:
        """Get iteration trajectory for a puzzle."""
        return self.puzzles[puzzle_idx].iterations

    def save(self, filename: str = "activations.npz"):
        """Save to disk."""
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, filename)

        # Collect all data
        all_questions = []
        all_answers = []
        all_predictions = []
        all_correct = []
        all_board_states = []
        all_contexts = []
        all_guards = []
        all_h_cycles = []
        all_l_cycles = []
        puzzle_boundaries = [0]
        iteration_boundaries = [0]

        for puzzle in self.puzzles:
            all_questions.append(puzzle.question)
            all_answers.append(puzzle.answer)
            all_predictions.append(puzzle.final_prediction)
            all_correct.append(puzzle.is_correct)

            for it in puzzle.iterations:
                all_board_states.append(it.board_states)
                all_contexts.append(it.context)
                all_guards.append(it.guard_values)
                all_h_cycles.append(it.h_cycle)
                all_l_cycles.append(it.l_cycle)

            iteration_boundaries.append(len(all_board_states))
            puzzle_boundaries.append(len(all_questions))

        np.savez_compressed(
            path,
            questions=np.stack(all_questions),
            answers=np.stack(all_answers),
            predictions=np.stack(all_predictions),
            correct=np.array(all_correct),
            board_states=np.stack(all_board_states),
            contexts=np.stack(all_contexts),
            guards=np.stack(all_guards),
            h_cycles=np.array(all_h_cycles),
            l_cycles=np.array(all_l_cycles),
            puzzle_boundaries=np.array(puzzle_boundaries),
            iteration_boundaries=np.array(iteration_boundaries),
        )
        print(f"Saved activations to {path}")

    @classmethod
    def load(cls, path: str) -> 'ActivationDataset':
        """Load from disk."""
        data = np.load(path)

        dataset = cls(save_dir=os.path.dirname(path))

        num_puzzles = len(data['questions'])
        it_boundaries = data['iteration_boundaries']

        for p in range(num_puzzles):
            it_start = it_boundaries[p]
            it_end = it_boundaries[p + 1]

            iterations = []
            for i in range(it_start, it_end):
                iterations.append(IterationActivation(
                    h_cycle=int(data['h_cycles'][i]),
                    l_cycle=int(data['l_cycles'][i]),
                    board_states=data['board_states'][i],
                    context=data['contexts'][i],
                    guard_values=data['guards'][i],
                ))

            dataset.add_puzzle(PuzzleActivations(
                puzzle_id=p,
                question=data['questions'][p],
                answer=data['answers'][p],
                iterations=iterations,
                final_prediction=data['predictions'][p],
                is_correct=bool(data['correct'][p]),
            ))

        return dataset


def extract_iteration_activations(
    model,  # SudokuStatechart9x9
    questions: mx.array,  # [B, 81]
    answers: mx.array,    # [B, 81]
    H_cycles: int = 3,
    L_cycles: int = 6,
) -> List[PuzzleActivations]:
    """
    Extract activations from model during forward pass.

    Args:
        model: Trained SudokuStatechart9x9 model
        questions: Input puzzles
        answers: Target solutions
        H_cycles: Number of outer cycles
        L_cycles: Number of inner cycles

    Returns:
        List of PuzzleActivations for each puzzle in batch
    """
    B = questions.shape[0]
    results = []

    # Convert to soft states
    soft_board = model.board_to_soft(questions)

    # Storage for iterations
    iteration_data = [[] for _ in range(B)]

    # Run solve with tracking
    board = soft_board
    for h in range(H_cycles):
        # Get H-level context
        h_context = model.encode_board(board)
        h_context = model.h_context_net(h_context)

        for l in range(L_cycles):
            # Get current context
            context = model.encode_board(board) + h_context

            # Collect guard values for all cells
            all_guards = []
            for cell_id in range(81):
                guards = model.guard(board, cell_id)  # [B, 9]
                all_guards.append(guards)
            guard_matrix = mx.stack(all_guards, axis=1)  # [B, 81, 9]

            # Store activations for each puzzle
            for b in range(B):
                iteration_data[b].append(IterationActivation(
                    h_cycle=h,
                    l_cycle=l,
                    board_states=np.array(board[b].tolist()),
                    context=np.array(context[b].tolist()),
                    guard_values=np.array(guard_matrix[b].tolist()),
                ))

            # Step
            board = model.step(board, context)

    # Get final predictions
    logits = model.output_head(board)
    predictions = mx.argmax(logits, axis=-1)

    # Package results
    for b in range(B):
        is_correct = mx.all(predictions[b] == answers[b]).item()

        results.append(PuzzleActivations(
            puzzle_id=b,
            question=np.array(questions[b].tolist()),
            answer=np.array(answers[b].tolist()),
            iterations=iteration_data[b],
            final_prediction=np.array(predictions[b].tolist()),
            is_correct=is_correct,
        ))

    return results


def extract_from_dataset(
    model,
    dataset,  # SudokuExtremeDataset
    max_puzzles: int = 1000,
    batch_size: int = 32,
    save_dir: str = "activations",
) -> ActivationDataset:
    """
    Extract activations for entire dataset.

    Args:
        model: Trained model
        dataset: Sudoku dataset
        max_puzzles: Maximum puzzles to process
        batch_size: Batch size for extraction
        save_dir: Directory to save activations

    Returns:
        ActivationDataset with all extractions
    """
    activation_dataset = ActivationDataset(save_dir=save_dir)

    puzzles_processed = 0
    for questions, answers in dataset.iter_batches(batch_size, shuffle=False):
        if puzzles_processed >= max_puzzles:
            break

        puzzle_acts = extract_iteration_activations(
            model,
            questions,
            answers,
            H_cycles=model.H_cycles,
            L_cycles=model.L_cycles,
        )

        for act in puzzle_acts:
            if puzzles_processed >= max_puzzles:
                break
            act.puzzle_id = puzzles_processed
            activation_dataset.add_puzzle(act)
            puzzles_processed += 1

        if puzzles_processed % 100 == 0:
            print(f"Processed {puzzles_processed}/{max_puzzles} puzzles")

    print(f"Extraction complete: {len(activation_dataset)} puzzles")
    return activation_dataset


def test_extraction():
    """Test activation extraction."""
    print("=" * 60)
    print("Testing Activation Extraction")
    print("=" * 60)

    # Import model
    from ..exp_trm_sudoku_9x9.sudoku_statechart_9x9 import SudokuStatechart9x9

    # Create model
    model = SudokuStatechart9x9(
        hidden_dim=64,
        H_cycles=2,
        L_cycles=3,
    )

    # Create test data
    mx.random.seed(42)
    B = 4
    questions = mx.random.randint(0, 10, (B, 81))
    answers = mx.random.randint(1, 10, (B, 81))

    print(f"\n1. Extracting activations for {B} puzzles...")

    activations = extract_iteration_activations(
        model,
        questions,
        answers,
        H_cycles=2,
        L_cycles=3,
    )

    print(f"   Got {len(activations)} puzzle activations")

    # Check first puzzle
    act = activations[0]
    print(f"\n2. First puzzle:")
    print(f"   Iterations: {len(act.iterations)}")
    print(f"   Board states shape: {act.iterations[0].board_states.shape}")
    print(f"   Context shape: {act.iterations[0].context.shape}")
    print(f"   Guard values shape: {act.iterations[0].guard_values.shape}")
    print(f"   Is correct: {act.is_correct}")

    # Create dataset
    print("\n3. Creating ActivationDataset...")
    dataset = ActivationDataset(save_dir="/tmp/test_activations")
    for act in activations:
        dataset.add_puzzle(act)

    print(f"   Dataset size: {len(dataset)}")

    # Get all activations
    print("\n4. Getting flattened activations...")
    boards, contexts, guards = dataset.get_all_activations(flatten_to_cells=True)
    print(f"   Board states: {boards.shape}")
    print(f"   Contexts: {contexts.shape}")
    print(f"   Guards: {guards.shape}")

    # Filter by cycle
    print("\n5. Filtering by H=1, L=2...")
    boards_filtered, _, _ = dataset.get_all_activations(h_cycle=1, l_cycle=2)
    print(f"   Filtered board states: {boards_filtered.shape}")

    # Save and load
    print("\n6. Testing save/load...")
    dataset.save("test.npz")
    loaded = ActivationDataset.load("/tmp/test_activations/test.npz")
    print(f"   Loaded {len(loaded)} puzzles")

    print("\n" + "=" * 60)
    print("Extraction test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_extraction()
