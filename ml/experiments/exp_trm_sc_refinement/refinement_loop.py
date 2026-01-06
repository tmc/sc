"""
Refinement Loop: H×L iteration dynamics for statechart refinement.

Implements the TRM-style two-level iteration:
- H_cycles: Outer cycles for global structure refinement
- L_cycles: Inner cycles for local detail refinement
"""

import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from .sc_refiner import SCRefiner, SCRefinerConfig


@dataclass
class RefinementStep:
    """Record of a single refinement step."""
    h_cycle: int
    l_cycle: int
    iteration_idx: int

    # Validity metrics
    validity_score: float
    num_valid_states: int
    num_valid_transitions: int

    # Predictions at this step
    state_type_entropy: float
    initial_confidence: float

    # Changes from previous step
    num_states_changed: int
    num_transitions_changed: int

    def to_dict(self) -> Dict:
        return {
            "h_cycle": self.h_cycle,
            "l_cycle": self.l_cycle,
            "iteration_idx": self.iteration_idx,
            "validity_score": self.validity_score,
            "num_valid_states": self.num_valid_states,
            "num_valid_transitions": self.num_valid_transitions,
            "state_type_entropy": self.state_type_entropy,
            "initial_confidence": self.initial_confidence,
            "num_states_changed": self.num_states_changed,
            "num_transitions_changed": self.num_transitions_changed,
        }


@dataclass
class RefinementResult:
    """Result of a full refinement run."""
    input_validity: float
    output_validity: float
    num_iterations: int
    converged: bool
    convergence_iteration: Optional[int]

    steps: List[RefinementStep] = field(default_factory=list)
    final_predictions: Optional[Dict[str, mx.array]] = None

    @property
    def validity_improvement(self) -> float:
        return self.output_validity - self.input_validity

    def get_validity_trajectory(self) -> np.ndarray:
        return np.array([s.validity_score for s in self.steps])

    def to_dict(self) -> Dict:
        return {
            "input_validity": self.input_validity,
            "output_validity": self.output_validity,
            "validity_improvement": self.validity_improvement,
            "num_iterations": self.num_iterations,
            "converged": self.converged,
            "convergence_iteration": self.convergence_iteration,
            "validity_trajectory": self.get_validity_trajectory().tolist(),
        }


class RefinementLoop:
    """
    Manages the H×L refinement loop.

    Features:
    - Early stopping when validity reaches threshold
    - Tracking of all intermediate steps
    - Support for custom callbacks
    """

    def __init__(
        self,
        model: SCRefiner,
        H_cycles: Optional[int] = None,
        L_cycles: Optional[int] = None,
        validity_threshold: float = 0.95,
        early_stop: bool = True,
    ):
        """
        Initialize refinement loop.

        Args:
            model: SCRefiner model
            H_cycles: Override H cycles (default: model config)
            L_cycles: Override L cycles (default: model config)
            validity_threshold: Stop when validity exceeds this
            early_stop: Enable early stopping
        """
        self.model = model
        self.H_cycles = H_cycles or model.config.H_cycles
        self.L_cycles = L_cycles or model.config.L_cycles
        self.validity_threshold = validity_threshold
        self.early_stop = early_stop

        self.callbacks: List[Callable] = []

    def add_callback(
        self,
        callback: Callable[[int, int, Dict], None],
    ):
        """Add callback for each step: fn(h, l, step_data)."""
        self.callbacks.append(callback)

    def _compute_entropy(self, logits: mx.array) -> float:
        """Compute entropy of predictions."""
        probs = mx.softmax(logits, axis=-1)
        probs = mx.clip(probs, 1e-10, 1.0)
        entropy = -mx.sum(probs * mx.log(probs), axis=-1)
        return float(mx.mean(entropy).tolist())

    def _count_changes(
        self,
        prev_preds: Optional[np.ndarray],
        curr_preds: np.ndarray,
    ) -> int:
        """Count prediction changes from previous step."""
        if prev_preds is None:
            return int(curr_preds.size)
        return int(np.sum(prev_preds != curr_preds))

    def run(
        self,
        state_types: mx.array,
        state_depths: mx.array,
        label_chars: mx.array,
        source_indices: mx.array,
        target_indices: mx.array,
        event_ids: mx.array,
        state_mask: mx.array,
        trans_mask: mx.array,
    ) -> RefinementResult:
        """
        Run the full refinement loop.

        Returns:
            RefinementResult with all steps and final predictions
        """
        steps = []
        prev_state_types = None
        prev_trans_valid = None

        # Initial encoding
        state_emb, trans_emb = self.model.encode(
            state_types, state_depths, label_chars,
            source_indices, target_indices, event_ids,
        )

        # Get initial validity
        initial_validity = float(
            self.model.predict_validity(state_emb)[0].tolist()
        )

        converged = False
        convergence_iter = None
        iteration_idx = 0

        for h in range(self.H_cycles):
            # Compute H-level context
            pooled = mx.mean(state_emb, axis=1)
            h_context = self.model.h_context_net(pooled)

            for l in range(self.L_cycles):
                # L-level refinement step
                state_emb, trans_emb = self.model.step(
                    state_emb, trans_emb, h_context,
                    state_mask, trans_mask,
                )

                # Get predictions
                predictions = self.model.head(state_emb, trans_emb)
                validity = self.model.predict_validity(state_emb)

                # Compute metrics
                curr_state_types = np.array(
                    mx.argmax(predictions["state_type"], axis=-1).tolist()
                )
                curr_trans_valid = np.array(
                    mx.argmax(predictions["trans_valid"], axis=-1).tolist()
                )

                state_entropy = self._compute_entropy(predictions["state_type"])
                initial_conf = float(mx.mean(
                    mx.softmax(predictions["state_initial"], axis=-1)[:, :, 1]
                ).tolist())

                num_valid_states = int(mx.sum(state_mask).tolist())
                num_valid_trans = int(mx.sum(trans_mask).tolist())

                # Record step
                step = RefinementStep(
                    h_cycle=h,
                    l_cycle=l,
                    iteration_idx=iteration_idx,
                    validity_score=float(validity[0].tolist()),
                    num_valid_states=num_valid_states,
                    num_valid_transitions=num_valid_trans,
                    state_type_entropy=state_entropy,
                    initial_confidence=initial_conf,
                    num_states_changed=self._count_changes(
                        prev_state_types, curr_state_types
                    ),
                    num_transitions_changed=self._count_changes(
                        prev_trans_valid, curr_trans_valid
                    ),
                )
                steps.append(step)

                # Callbacks
                for callback in self.callbacks:
                    callback(h, l, step.to_dict())

                # Check convergence
                if self.early_stop:
                    if float(validity[0].tolist()) >= self.validity_threshold:
                        if not converged:
                            converged = True
                            convergence_iter = iteration_idx

                prev_state_types = curr_state_types
                prev_trans_valid = curr_trans_valid
                iteration_idx += 1

        # Final predictions
        final_predictions = self.model.head(state_emb, trans_emb)
        final_predictions["validity"] = self.model.predict_validity(state_emb)

        return RefinementResult(
            input_validity=initial_validity,
            output_validity=float(
                self.model.predict_validity(state_emb)[0].tolist()
            ),
            num_iterations=iteration_idx,
            converged=converged,
            convergence_iteration=convergence_iter,
            steps=steps,
            final_predictions=final_predictions,
        )


def run_refinement(
    model: SCRefiner,
    statechart_data: Dict[str, mx.array],
    H_cycles: Optional[int] = None,
    L_cycles: Optional[int] = None,
    validity_threshold: float = 0.95,
) -> RefinementResult:
    """
    Convenience function to run refinement.

    Args:
        model: SCRefiner model
        statechart_data: Encoded statechart
        H_cycles: Optional override
        L_cycles: Optional override
        validity_threshold: Target validity

    Returns:
        RefinementResult
    """
    loop = RefinementLoop(
        model,
        H_cycles=H_cycles,
        L_cycles=L_cycles,
        validity_threshold=validity_threshold,
    )

    return loop.run(
        state_types=statechart_data["state_types"],
        state_depths=statechart_data["state_depths"],
        label_chars=statechart_data["label_chars"],
        source_indices=statechart_data["source_indices"],
        target_indices=statechart_data["target_indices"],
        event_ids=statechart_data["event_ids"],
        state_mask=statechart_data["state_mask"],
        trans_mask=statechart_data["trans_mask"],
    )


def test_refinement_loop():
    """Test the refinement loop."""
    print("=" * 60)
    print("Testing Refinement Loop")
    print("=" * 60)

    config = SCRefinerConfig(
        hidden_dim=64,
        max_states=16,
        max_transitions=32,
        H_cycles=2,
        L_cycles=3,
    )

    model = SCRefiner(config)

    # Create test data
    B, S, T = 1, 8, 12
    max_label_len = 16

    mx.random.seed(42)

    statechart_data = {
        "state_types": mx.random.randint(0, 4, (B, S)),
        "state_depths": mx.random.randint(0, 4, (B, S)),
        "label_chars": mx.random.randint(0, 128, (B, S, max_label_len)),
        "source_indices": mx.random.randint(0, S, (B, T)),
        "target_indices": mx.random.randint(0, S, (B, T)),
        "event_ids": mx.random.randint(0, 10, (B, T)),
        "state_mask": mx.ones((B, S)),
        "trans_mask": mx.ones((B, T)),
    }

    print("\n1. Creating refinement loop...")
    loop = RefinementLoop(
        model,
        H_cycles=2,
        L_cycles=3,
        validity_threshold=0.95,
        early_stop=True,
    )

    # Add callback
    step_logs = []

    def log_step(h, l, data):
        step_logs.append(f"H={h}, L={l}: validity={data['validity_score']:.3f}")

    loop.add_callback(log_step)

    print("\n2. Running refinement...")
    result = loop.run(**statechart_data)

    print(f"\n3. Results:")
    print(f"   Input validity: {result.input_validity:.3f}")
    print(f"   Output validity: {result.output_validity:.3f}")
    print(f"   Improvement: {result.validity_improvement:.3f}")
    print(f"   Iterations: {result.num_iterations}")
    print(f"   Converged: {result.converged}")

    print(f"\n4. Validity trajectory:")
    for v in result.get_validity_trajectory()[:5]:
        print(f"   {v:.3f}")

    print(f"\n5. Callback logs:")
    for log in step_logs[:5]:
        print(f"   {log}")

    print("\n6. Testing convenience function...")
    result2 = run_refinement(model, statechart_data)
    print(f"   Result type: {type(result2).__name__}")

    print("\n" + "=" * 60)
    print("Refinement loop test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_refinement_loop()
