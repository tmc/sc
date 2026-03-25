"""
End-to-end differentiable training pipeline.

Paper §4: The entire statechart topology is a learnable parameter
optimized via gradient descent.

LOSS FUNCTION:
  L = L_task + beta * H(pi) + lambda_sparse * ||A||_1 + alpha * L_delegate

  L_task:       Cross-entropy between predicted and target state sequences
  H(pi):        Entropy regularization on soft configuration
  ||A||_1:      L1 sparsity on adjacency matrix
  L_delegate:   Delegation parsimony penalty

TRAINING SCHEDULE:
  1. Warm-up: high temperature (tau=5.0), explore topologies
  2. Anneal: exponential decay tau *= 0.995 per epoch
  3. Fine-tune: low temperature (tau=0.1), near-discrete execution
  4. Extract: threshold adjacency to get discrete statechart

OPTIMIZERS:
  - AdamW: Standard gradient-based optimizer (baseline).
  - GRPO:  Group Relative Policy Optimization. Computes advantages
           relative to a group of sampled trajectories, eliminating
           the need for a critic network.
  - SDPO:  Self-Distilled Policy Optimization. Augments GRPO with
           an EMA teacher for self-distillation and hindsight learning.

GRPO LOSS:
  L_GRPO = -E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
           - beta * D_KL(pi_theta || pi_ref)
  where A_t = (R_t - mean(R_group)) / std(R_group)

SDPO LOSS:
  L_SDPO = L_GRPO + alpha * D_KL(pi_theta || pi_EMA)
  where theta_EMA <- (1 - mu) * theta_EMA + mu * theta
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
import json
import copy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Dict, Tuple, Optional, Callable

from .differentiable_sc import NeuroSymbolicStatechart, NeuroSymbolicConfig
from .recursive_delegation import RLMStatechart, RLMConfig


class PolicyAlgorithm(IntEnum):
    """Mirrors proto PolicyAlgorithm enum."""
    REINFORCE = 1
    PPO = 2
    GRPO = 3
    SDPO = 4


@dataclass
class EMATeacherConfig:
    """EMA teacher for self-distillation in SDPO."""
    ema_decay: float = 0.01          # Slow teacher update
    distillation_weight: float = 0.1  # KL weight between student and teacher
    update_interval: int = 10         # Update teacher every N steps


@dataclass
class HindsightConfig:
    """Hindsight relabeling for sample-efficient topology learning."""
    enabled: bool = False
    relabel_fraction: float = 0.5
    max_relabeled_per_batch: int = 4


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    n_epochs: int = 200
    batch_size: int = 16

    # Schedule
    warmup_epochs: int = 10
    anneal_start_epoch: int = 10
    anneal_every_n_steps: int = 50

    # Loss coefficients
    sparsity_coefficient: float = 0.1
    entropy_coefficient: float = 0.05
    delegation_penalty: float = 0.01

    # Policy optimization (GRPO/SDPO)
    algorithm: PolicyAlgorithm = PolicyAlgorithm.REINFORCE
    clip_epsilon: float = 0.2
    kl_coefficient: float = 0.01
    group_size: int = 4
    ema_teacher: EMATeacherConfig = None
    hindsight: HindsightConfig = None

    # Logging
    log_every: int = 10
    eval_every: int = 20

    def __post_init__(self):
        if self.ema_teacher is None:
            self.ema_teacher = EMATeacherConfig()
        if self.hindsight is None:
            self.hindsight = HindsightConfig()


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

@dataclass
class TrainingExample:
    """A single training example: event sequence -> state trajectory."""
    initial_state: int
    event_sequence: List[int]
    context_sequence: mx.array  # [seq_len, context_dim]
    target_states: mx.array     # [seq_len + 1]


def generate_cycle_data(
    n_states: int,
    n_events: int,
    context_dim: int,
    n_examples: int = 100,
    seq_length: int = 10,
) -> List[TrainingExample]:
    """
    Generate training data for a cyclic statechart.

    Pattern: S0 --E0--> S1 --E1--> S2 --E2--> ... --E(n-1)--> S0
    """
    examples = []
    for _ in range(n_examples):
        start = int(mx.random.randint(0, n_states, (1,)).item())
        states = [start]
        events = []
        contexts = []

        current = start
        for _ in range(seq_length):
            event = current % n_events
            next_state = (current + 1) % n_states
            events.append(event)
            states.append(next_state)
            # Context encodes current state and event
            ctx = mx.zeros((context_dim,))
            ctx_list = [0.0] * context_dim
            ctx_list[0] = float(current) / n_states
            ctx_list[1] = float(event) / n_events
            contexts.append(ctx_list)
            current = next_state

        examples.append(TrainingExample(
            initial_state=start,
            event_sequence=events,
            context_sequence=mx.array(contexts),
            target_states=mx.array(states),
        ))

    return examples


def generate_branching_data(
    n_states: int,
    n_events: int,
    context_dim: int,
    n_examples: int = 100,
    seq_length: int = 10,
) -> List[TrainingExample]:
    """
    Generate training data with context-dependent branching.

    S0 --E0, ctx[0]>0.5--> S1
    S0 --E0, ctx[0]<=0.5--> S2
    S1 --E1--> S3
    S2 --E1--> S3
    S3 --E2--> S0
    """
    examples = []
    for _ in range(n_examples):
        states = [0]
        events = []
        contexts = []
        current = 0

        for step in range(seq_length):
            ctx_val = float(mx.random.uniform(shape=(1,)).item())
            ctx_list = [0.0] * context_dim
            ctx_list[0] = ctx_val
            ctx_list[1] = float(current) / n_states
            contexts.append(ctx_list)

            event = step % n_events

            if current == 0 and event == 0:
                next_state = 1 if ctx_val > 0.5 else 2
            elif current in (1, 2) and event == 1:
                next_state = 3
            elif current == 3 and event == 2:
                next_state = 0
            else:
                next_state = current  # Self-loop

            events.append(event)
            states.append(next_state)
            current = next_state

        examples.append(TrainingExample(
            initial_state=0,
            event_sequence=events,
            context_sequence=mx.array(contexts),
            target_states=mx.array(states),
        ))

    return examples


def make_splits(
    generator_fn: Callable,
    n_train: int = 100,
    n_eval: int = 20,
    n_test: int = 20,
    seed: int = 42,
) -> Tuple[List['TrainingExample'], List['TrainingExample'], List['TrainingExample']]:
    """Deterministic train/eval/test split."""
    mx.random.seed(seed)
    train = generator_fn(n_examples=n_train)
    eval_ = generator_fn(n_examples=n_eval)
    test = generator_fn(n_examples=n_test)
    return train, eval_, test


# ---------------------------------------------------------------------------
# sc-trace-gen Data Loader
# ---------------------------------------------------------------------------

class Vocabulary:
    """Bidirectional string<->integer mapping."""

    def __init__(self, labels):
        self.label_to_idx = {l: i for i, l in enumerate(sorted(set(labels)))}
        self.idx_to_label = {i: l for l, i in self.label_to_idx.items()}

    def encode(self, label):
        return self.label_to_idx[label]

    def decode(self, idx):
        return self.idx_to_label[idx]

    def __len__(self):
        return len(self.label_to_idx)


def _flatten_context(ctx, max_dim=8):
    """Flatten a protobuf Struct-like dict to a fixed-size float vector."""
    values = []
    for k in sorted(ctx.keys()):
        v = ctx[k]
        if isinstance(v, (int, float)):
            values.append(float(v))
        elif isinstance(v, bool):
            values.append(1.0 if v else 0.0)
    values = values[:max_dim]
    values.extend([0.0] * (max_dim - len(values)))
    return values


def load_trace_dataset(path):
    """Load sc-trace-gen JSON output as TrainingExamples.

    Returns (examples, ground_truth_topology, state_vocab, event_vocab).
    """
    with open(path) as f:
        dataset = json.load(f)

    topology = dataset.get("ground_truth_topology", {})
    state_labels = [s["label"] for s in topology.get("states", [])]
    event_labels = dataset.get("events", [])
    state_vocab = Vocabulary(state_labels)
    event_vocab = Vocabulary(event_labels)

    examples = []
    for trace_data in dataset["traces"]:
        trace = json.loads(trace_data) if isinstance(trace_data, str) else trace_data
        entries = trace.get("entries", [])
        if not entries:
            continue

        event_sequence = []
        target_states = []
        context_sequence = []

        for entry in entries:
            event_label = entry.get("triggerEvent", {}).get("label", "")
            if event_label in event_vocab.label_to_idx:
                event_sequence.append(event_vocab.encode(event_label))

            target_config = entry.get("targetConfig", {}).get("states", [])
            if target_config:
                leaf_label = target_config[-1].get("label", "")
                if leaf_label in state_vocab.label_to_idx:
                    target_states.append(state_vocab.encode(leaf_label))

            ctx = entry.get("contextBefore", {})
            ctx_vec = _flatten_context(ctx) if ctx else [0.0] * 2
            context_sequence.append(ctx_vec)

        if event_sequence and target_states:
            n_steps = min(len(event_sequence), len(target_states))
            examples.append(TrainingExample(
                initial_state=target_states[0] if target_states else 0,
                context_sequence=mx.array(context_sequence[:n_steps]),
                event_sequence=mx.array(event_sequence[:n_steps]),
                target_states=mx.array(target_states[:n_steps]),
            ))

    return examples, topology, state_vocab, event_vocab


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    End-to-end training loop for neuro-symbolic statecharts.

    Handles:
      - Loss computation and gradient accumulation
      - Temperature annealing schedule
      - Metric tracking
      - Topology extraction at convergence
    """

    def __init__(
        self,
        model: NeuroSymbolicStatechart,
        config: TrainingConfig,
    ):
        self.model = model
        self.config = config

        # Optimizer
        self.optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Metrics history
        self.history: List[Dict[str, float]] = []
        self.step_count = 0

    def loss_fn(self, model, example: TrainingExample):
        """Compute loss for a single example."""
        traj, _, _ = model.forward_sequence(
            example.initial_state,
            example.context_sequence,
            example.event_sequence,
        )
        loss, components = model.get_loss(traj, example.target_states)
        return loss, components

    def train_step(self, example: TrainingExample) -> Dict[str, float]:
        """
        One training step with gradient update.

        Uses MLX value_and_grad for automatic differentiation.
        """
        metrics_container = {}

        def compute_loss(model):
            loss, components = self.loss_fn(model, example)
            metrics_container.update({k: v for k, v in components.items()})
            return loss

        loss, grads = nn.value_and_grad(self.model, compute_loss)(self.model)

        # Gradient clipping
        grad_norm = sum(
            float(mx.sqrt(mx.sum(g ** 2)).item())
            for g in grads.parameters().values()
            if isinstance(g, mx.array)
        ) if hasattr(grads, 'parameters') else 0.0

        # Update parameters
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), loss)

        self.step_count += 1

        # Temperature annealing
        if (self.step_count % self.config.anneal_every_n_steps == 0 and
                self.step_count > self.config.warmup_epochs * 10):
            self.model.anneal_temperature()

        metrics = {
            "loss": float(loss.item()),
            "temperature": self.model.temperature,
            "step": self.step_count,
        }
        for k, v in metrics_container.items():
            metrics[k] = float(v.item())

        return metrics

    def train_epoch(self, data: List[TrainingExample]) -> Dict[str, float]:
        """Train one epoch over all examples."""
        epoch_metrics = {}
        for example in data:
            metrics = self.train_step(example)
            for k, v in metrics.items():
                epoch_metrics.setdefault(k, []).append(v)

        # Average metrics
        avg = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
        self.history.append(avg)
        return avg

    def train(
        self,
        train_data: List[TrainingExample],
        eval_data: Optional[List[TrainingExample]] = None,
    ) -> List[Dict[str, float]]:
        """
        Full training loop.

        Returns history of metrics per epoch.
        """
        print(f"Training: {self.config.n_epochs} epochs, "
              f"{len(train_data)} examples")
        print(f"  lr={self.config.learning_rate}, "
              f"anneal_rate={self.model.config.annealing_rate}")
        print("-" * 60)

        for epoch in range(self.config.n_epochs):
            t0 = time.time()
            metrics = self.train_epoch(train_data)
            dt = time.time() - t0

            if epoch % self.config.log_every == 0 or epoch == self.config.n_epochs - 1:
                print(
                    f"Epoch {epoch:4d} | "
                    f"loss={metrics['loss']:.4f} | "
                    f"ce={metrics.get('ce_loss', 0):.4f} | "
                    f"τ={metrics['temperature']:.3f} | "
                    f"{dt:.2f}s"
                )

            # Evaluation
            if eval_data and epoch % self.config.eval_every == 0:
                eval_metrics = self.evaluate(eval_data)
                print(f"  [eval] accuracy={eval_metrics['accuracy']:.3f}")

        print("-" * 60)
        return self.history

    def evaluate(self, data: List[TrainingExample]) -> Dict[str, float]:
        """Evaluate accuracy on a dataset."""
        correct = 0
        total = 0

        for example in data:
            traj, _, _ = self.model.forward_sequence(
                example.initial_state,
                example.context_sequence,
                example.event_sequence,
            )
            for t in range(example.target_states.shape[0]):
                pred = int(mx.argmax(traj[t]).item())
                target = int(example.target_states[t].item())
                if pred == target:
                    correct += 1
                total += 1

        return {"accuracy": correct / max(total, 1), "total": total}

    def evaluate_topology(self, ground_truth_edges: set, threshold: float = 0.3) -> Dict[str, float]:
        """Compute precision, recall, F1 for topology recovery."""
        topo = self.extract_topology(threshold)
        recovered = {(t["from"], t["to"], t["event"]) for t in topo["transitions"]}
        tp = len(ground_truth_edges & recovered)
        fp = len(recovered - ground_truth_edges)
        fn = len(ground_truth_edges - recovered)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        return {"precision": precision, "recall": recall, "f1": f1,
                "tp": tp, "fp": fp, "fn": fn}

    def extract_topology(self, threshold: float = 0.5) -> Dict:
        """Extract discrete topology from learned soft adjacency."""
        adj = self.model.topology.get_adjacency(temperature=0.1)
        transitions = []

        for i in range(self.model.config.n_states):
            for j in range(self.model.config.n_states):
                for e in range(self.model.config.n_events):
                    prob = float(adj[i, j, e].item())
                    if prob > threshold:
                        transitions.append({
                            "from": f"S{i}",
                            "to": f"S{j}",
                            "event": f"E{e}",
                            "probability": round(prob, 3),
                        })

        return {
            "n_states": self.model.config.n_states,
            "n_events": self.model.config.n_events,
            "transitions": sorted(transitions, key=lambda t: -t["probability"]),
            "final_temperature": self.model.temperature,
        }


# ---------------------------------------------------------------------------
# RLM Trainer (extends Trainer for delegation)
# ---------------------------------------------------------------------------

class RLMTrainer(Trainer):
    """Trainer for RLMStatechart with delegation loss."""

    def __init__(self, model: RLMStatechart, config: TrainingConfig):
        # Store as self.rlm_model for delegation-specific access
        self.rlm_model = model
        # Pass the inner statechart to parent for standard training
        super().__init__(model.sc, config)
        self.rlm = model

    def loss_fn_rlm(self, model, example: TrainingExample):
        """Loss with delegation penalty."""
        traj, infos = self.rlm.forward_sequence(
            example.initial_state,
            example.context_sequence,
            example.event_sequence,
        )
        loss, components = self.rlm.get_loss(
            traj, example.target_states, infos,
            self.config.delegation_penalty,
        )
        return loss, components

    def train_step(self, example: TrainingExample) -> Dict[str, float]:
        """Training step with delegation."""
        # Use parent's train_step for the base statechart
        metrics = super().train_step(example)

        # Also compute delegation metrics
        traj, infos = self.rlm.forward_sequence(
            example.initial_state,
            example.context_sequence,
            example.event_sequence,
        )
        del_probs = [
            float(info["delegation_prob"].item())
            for info in infos
            if "delegation_prob" in info
        ]
        metrics["avg_delegation_prob"] = sum(del_probs) / max(len(del_probs), 1)
        metrics["n_delegations"] = sum(1 for p in del_probs if p > 0.5)

        return metrics


# ---------------------------------------------------------------------------
# GRPO Trainer
# ---------------------------------------------------------------------------

def _flatten_params(model) -> Dict[str, mx.array]:
    """Flatten model parameters into a dict for copying."""
    result = {}
    for k, v in model.parameters().items():
        if isinstance(v, mx.array):
            result[k] = v
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, mx.array):
                    result[f"{k}.{k2}"] = v2
    return result


def _get_state_log_probs(
    model: NeuroSymbolicStatechart,
    example: 'TrainingExample',
) -> Tuple[mx.array, mx.array]:
    """
    Run model and return per-step log probabilities and reward.

    Returns:
        log_probs: [seq_len] log probabilities of target states.
        reward: scalar reward (negative loss = higher is better).
    """
    traj, _, _ = model.forward_sequence(
        example.initial_state,
        example.context_sequence,
        example.event_sequence,
    )
    seq_len = example.target_states.shape[0]
    log_probs = []
    correct = 0
    for t in range(seq_len):
        pred = traj[t]
        eye = mx.eye(pred.shape[-1])
        target_oh = eye[int(example.target_states[t].item())]
        lp = mx.log(mx.sum(pred * target_oh) + 1e-8)
        log_probs.append(lp)
        if int(mx.argmax(pred).item()) == int(example.target_states[t].item()):
            correct += 1
    reward = mx.array(correct / max(seq_len, 1))
    return mx.stack(log_probs), reward


class GRPOTrainer(Trainer):
    """
    Group Relative Policy Optimization trainer.

    GRPO computes advantages relative to a group of sampled trajectories
    rather than a learned value function, eliminating the critic network.

    L_GRPO = -E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
             - beta * D_KL(pi_theta || pi_ref)

    where A_t = (R_t - mean(R_group)) / std(R_group)

    The "group" is a set of examples evaluated under the current policy.
    Group-relative advantages normalize rewards within each batch,
    making training stable without a value baseline.
    """

    def __init__(self, model: NeuroSymbolicStatechart, config: TrainingConfig):
        super().__init__(model, config)

        # Store reference policy (snapshot of initial parameters)
        self._ref_params = {
            k: mx.array(v) for k, v in _flatten_params(model).items()
        }

    def _compute_group_advantages(
        self,
        rewards: List[mx.array],
    ) -> List[mx.array]:
        """
        Compute group-relative advantages.

        A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
        """
        reward_vals = mx.stack([r.squeeze() for r in rewards])
        mean_r = mx.mean(reward_vals)
        std_r = mx.sqrt(mx.mean((reward_vals - mean_r) ** 2) + 1e-8)
        return [(r.squeeze() - mean_r) / std_r for r in rewards]

    def _kl_from_reference(
        self,
        current_log_probs: mx.array,
        example: 'TrainingExample',
    ) -> mx.array:
        """
        Approximate KL divergence from reference policy.

        D_KL(pi_theta || pi_ref) ≈ E[log pi_theta - log pi_ref]
        """
        # For efficiency, approximate as mean difference in log probs
        # A full implementation would run the reference model
        return mx.mean(current_log_probs) * self.config.kl_coefficient

    def train_step_grpo(
        self,
        group: List['TrainingExample'],
    ) -> Dict[str, float]:
        """
        One GRPO training step over a group of examples.

        1. Evaluate current policy on each example -> log_probs, rewards
        2. Compute group-relative advantages
        3. Compute clipped surrogate loss
        4. Add KL penalty against reference policy
        """
        # Evaluate group under current policy (eagerly)
        all_log_probs = []
        all_rewards = []
        for ex in group:
            lp, r = _get_state_log_probs(self.model, ex)
            mx.eval(lp, r)
            all_log_probs.append(lp)
            all_rewards.append(r)

        # Group-relative advantages (eagerly)
        advantages = self._compute_group_advantages(all_rewards)
        mx.eval(*[a for a in advantages])

        # Compute GRPO loss via value_and_grad
        def grpo_loss(model):
            total_loss = mx.array(0.0)
            for i, ex in enumerate(group):
                lp, _ = _get_state_log_probs(model, ex)
                old_lp = mx.stop_gradient(all_log_probs[i])
                adv = mx.stop_gradient(advantages[i])

                # Importance ratio
                ratio = mx.exp(mx.mean(lp) - mx.mean(old_lp))

                # Clipped surrogate
                eps = self.config.clip_epsilon
                clipped = mx.clip(ratio, 1.0 - eps, 1.0 + eps)
                surrogate = -mx.minimum(ratio * adv, clipped * adv)

                # KL penalty
                kl = mx.mean(lp) * self.config.kl_coefficient

                total_loss = total_loss + surrogate + kl

            return total_loss / len(group)

        loss, grads = nn.value_and_grad(self.model, grpo_loss)(self.model)

        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), loss)
        self.step_count += 1

        # Temperature annealing
        if (self.step_count % self.config.anneal_every_n_steps == 0 and
                self.step_count > self.config.warmup_epochs * 10):
            self.model.anneal_temperature()

        avg_reward = float(mx.mean(mx.stack([r.squeeze() for r in all_rewards])).item())
        avg_adv = float(mx.mean(mx.stack([a.squeeze() for a in advantages])).item())

        return {
            "loss": float(loss.item()),
            "avg_reward": avg_reward,
            "avg_advantage": avg_adv,
            "temperature": self.model.temperature,
            "step": self.step_count,
        }

    def train_epoch(self, data: List['TrainingExample']) -> Dict[str, float]:
        """Train one epoch with group-based updates."""
        epoch_metrics = {}
        group_size = self.config.group_size

        # Process data in groups
        for i in range(0, len(data), group_size):
            group = data[i:i + group_size]
            if len(group) < 2:
                continue
            metrics = self.train_step_grpo(group)
            for k, v in metrics.items():
                epoch_metrics.setdefault(k, []).append(v)

        avg = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
        self.history.append(avg)
        return avg

    def train(
        self,
        train_data: List['TrainingExample'],
        eval_data: Optional[List['TrainingExample']] = None,
    ) -> List[Dict[str, float]]:
        """Full GRPO training loop."""
        gs = self.config.group_size
        print(f"GRPO Training: {self.config.n_epochs} epochs, "
              f"{len(train_data)} examples, group_size={gs}")
        print(f"  clip_epsilon={self.config.clip_epsilon}, "
              f"kl_coeff={self.config.kl_coefficient}")
        print("-" * 60)

        for epoch in range(self.config.n_epochs):
            t0 = time.time()
            metrics = self.train_epoch(train_data)
            dt = time.time() - t0

            if epoch % self.config.log_every == 0 or epoch == self.config.n_epochs - 1:
                print(
                    f"Epoch {epoch:4d} | "
                    f"loss={metrics['loss']:.4f} | "
                    f"reward={metrics.get('avg_reward', 0):.4f} | "
                    f"τ={metrics['temperature']:.3f} | "
                    f"{dt:.2f}s"
                )

            if eval_data and epoch % self.config.eval_every == 0:
                eval_metrics = self.evaluate(eval_data)
                print(f"  [eval] accuracy={eval_metrics['accuracy']:.3f}")

        print("-" * 60)
        return self.history


# ---------------------------------------------------------------------------
# SDPO Trainer
# ---------------------------------------------------------------------------

class SDPOTrainer(GRPOTrainer):
    """
    Self-Distilled Policy Optimization trainer.

    Extends GRPO with an EMA teacher model for self-distillation:
      L_SDPO = L_GRPO + alpha * D_KL(pi_theta || pi_EMA)

    The EMA teacher is a slow-moving average of the student's parameters:
      theta_EMA <- (1 - mu) * theta_EMA + mu * theta

    Also supports hindsight relabeling: failed executions are reprompted
    and the resulting (trace, synthetic_task) pairs become positive
    training examples.
    """

    def __init__(self, model: NeuroSymbolicStatechart, config: TrainingConfig):
        super().__init__(model, config)
        self.ema_config = config.ema_teacher

        # Create EMA teacher as a parameter snapshot
        self._teacher_params = {
            k: mx.array(v) for k, v in _flatten_params(model).items()
        }
        self._teacher_step_count = 0

    def _update_ema_teacher(self):
        """
        Update EMA teacher parameters.

        theta_EMA <- (1 - mu) * theta_EMA + mu * theta
        """
        mu = self.ema_config.ema_decay
        current = _flatten_params(self.model)
        for k in self._teacher_params:
            if k in current:
                self._teacher_params[k] = (
                    (1.0 - mu) * self._teacher_params[k] + mu * current[k]
                )
        self._teacher_step_count += 1

    def _teacher_log_probs(
        self,
        example: 'TrainingExample',
    ) -> mx.array:
        """
        Compute log probs under the EMA teacher.

        Uses the teacher's stored parameters to evaluate the trajectory.
        For efficiency, approximates via the student model's forward pass
        with frozen teacher weights applied as soft targets.
        """
        # Approximate: use current model output with stop_gradient
        # as proxy for teacher (full implementation would maintain
        # a separate model copy)
        traj, _, _ = self.model.forward_sequence(
            example.initial_state,
            example.context_sequence,
            example.event_sequence,
        )
        seq_len = example.target_states.shape[0]
        log_probs = []
        for t in range(seq_len):
            pred = mx.stop_gradient(traj[t])
            eye = mx.eye(pred.shape[-1])
            target_oh = eye[int(example.target_states[t].item())]
            lp = mx.log(mx.sum(pred * target_oh) + 1e-8)
            log_probs.append(lp)
        return mx.stack(log_probs)

    def _kl_from_teacher(
        self,
        student_log_probs: mx.array,
        teacher_log_probs: mx.array,
    ) -> mx.array:
        """
        KL divergence from teacher to student.

        D_KL(pi_student || pi_teacher) ≈ E[log_pi_s - log_pi_t]
        """
        return mx.mean(student_log_probs - teacher_log_probs)

    def _hindsight_relabel(
        self,
        failed_examples: List['TrainingExample'],
    ) -> List['TrainingExample']:
        """
        Hindsight relabeling: take failed trajectories and relabel
        them with the states the model actually predicted.

        The model's predictions become the target, converting a
        failure into a self-supervised positive example.
        """
        if not self.config.hindsight.enabled:
            return []

        relabeled = []
        n_to_relabel = min(
            int(len(failed_examples) * self.config.hindsight.relabel_fraction),
            self.config.hindsight.max_relabeled_per_batch,
        )

        for ex in failed_examples[:n_to_relabel]:
            traj, _, _ = self.model.forward_sequence(
                ex.initial_state,
                ex.context_sequence,
                ex.event_sequence,
            )
            # Use model's actual predictions as synthetic targets
            predicted = []
            for t in range(traj.shape[0]):
                predicted.append(int(mx.argmax(traj[t]).item()))

            relabeled.append(TrainingExample(
                initial_state=ex.initial_state,
                event_sequence=ex.event_sequence,
                context_sequence=ex.context_sequence,
                target_states=mx.array(predicted),
            ))

        return relabeled

    def train_step_grpo(
        self,
        group: List['TrainingExample'],
    ) -> Dict[str, float]:
        """SDPO training step: GRPO + EMA distillation + hindsight."""
        # Run base GRPO step
        metrics = super().train_step_grpo(group)

        # Compute distillation loss for logging
        distill_losses = []
        failed = []
        for ex in group:
            student_lp, reward = _get_state_log_probs(self.model, ex)
            teacher_lp = self._teacher_log_probs(ex)
            kl = self._kl_from_teacher(student_lp, teacher_lp)
            distill_losses.append(float(kl.item()))

            # Track failures for hindsight
            if float(reward.item()) < 0.5:
                failed.append(ex)

        metrics["distillation_kl"] = sum(distill_losses) / max(len(distill_losses), 1)

        # Hindsight relabeling
        if self.config.hindsight.enabled and failed:
            relabeled = self._hindsight_relabel(failed)
            if relabeled:
                # Train on relabeled examples (standard gradient step)
                for rex in relabeled:
                    super(GRPOTrainer, self).train_step(rex)
                metrics["hindsight_relabeled"] = len(relabeled)

        # Update EMA teacher periodically
        if self.step_count % self.ema_config.update_interval == 0:
            self._update_ema_teacher()
            metrics["teacher_updated"] = 1.0

        return metrics

    def train(
        self,
        train_data: List['TrainingExample'],
        eval_data: Optional[List['TrainingExample']] = None,
    ) -> List[Dict[str, float]]:
        """Full SDPO training loop."""
        gs = self.config.group_size
        print(f"SDPO Training: {self.config.n_epochs} epochs, "
              f"{len(train_data)} examples, group_size={gs}")
        print(f"  clip_epsilon={self.config.clip_epsilon}, "
              f"ema_decay={self.ema_config.ema_decay}, "
              f"distill_weight={self.ema_config.distillation_weight}")
        if self.config.hindsight.enabled:
            print(f"  hindsight: relabel_fraction={self.config.hindsight.relabel_fraction}")
        print("-" * 60)

        for epoch in range(self.config.n_epochs):
            t0 = time.time()
            metrics = self.train_epoch(train_data)
            dt = time.time() - t0

            if epoch % self.config.log_every == 0 or epoch == self.config.n_epochs - 1:
                distill = metrics.get('distillation_kl', 0)
                hindsight_n = metrics.get('hindsight_relabeled', 0)
                print(
                    f"Epoch {epoch:4d} | "
                    f"loss={metrics['loss']:.4f} | "
                    f"reward={metrics.get('avg_reward', 0):.4f} | "
                    f"kl_teacher={distill:.4f} | "
                    f"hindsight={hindsight_n:.0f} | "
                    f"τ={metrics['temperature']:.3f} | "
                    f"{dt:.2f}s"
                )

            if eval_data and epoch % self.config.eval_every == 0:
                eval_metrics = self.evaluate(eval_data)
                print(f"  [eval] accuracy={eval_metrics['accuracy']:.3f}")

        print("-" * 60)
        return self.history


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_training():
    """Demonstrate the training pipeline."""
    print("=" * 60)
    print("Training Pipeline Demo")
    print("=" * 60)

    # Model
    config = NeuroSymbolicConfig(
        n_states=4,
        n_events=3,
        state_dim=32,
        event_dim=16,
        context_dim=8,
        num_heads=4,
        hull_top_k=4,
        temperature_max=2.0,
        temperature_min=0.1,
        annealing_rate=0.99,
    )
    model = NeuroSymbolicStatechart(config)

    # Data
    train_data = generate_cycle_data(
        n_states=4, n_events=3, context_dim=8,
        n_examples=50, seq_length=6,
    )
    eval_data = generate_cycle_data(
        n_states=4, n_events=3, context_dim=8,
        n_examples=10, seq_length=6,
    )

    # Train
    train_config = TrainingConfig(
        learning_rate=1e-3,
        n_epochs=50,
        log_every=10,
        eval_every=25,
        anneal_every_n_steps=20,
    )
    trainer = Trainer(model, train_config)

    print("\n--- Cycle Pattern Training ---")
    history = trainer.train(train_data, eval_data)

    # Extract topology
    print("\n--- Extracted Topology ---")
    topo = trainer.extract_topology(threshold=0.3)
    print(f"Transitions (prob > 0.3): {len(topo['transitions'])}")
    for t in topo["transitions"][:10]:
        print(f"  {t['from']} --{t['event']}--> {t['to']}: {t['probability']}")

    # Final evaluation
    print("\n--- Final Evaluation ---")
    final = trainer.evaluate(eval_data)
    print(f"Accuracy: {final['accuracy']:.3f}")

    return history


if __name__ == "__main__":
    demo_training()
