"""
SAE-Grown Statecharts: Sparse Autoencoders as Interpretable State Bottleneck

This experiment integrates Sparse Autoencoders (SAEs) with differentiable statecharts
to achieve the mech-interp holy grail: grown statecharts built from monosemantic atoms.

Key Insight (Anthropic 2023-2025):
  SAE features naturally form FSA-like circuits. By using SAE latents AS the state
  representation, we force the grown chart to be built from interpretable atoms
  instead of dense, polysemantic mush.

Approach #2: SAE as State Bottleneck
  Instead of: state_vec = nn.Embedding(n_states, dim)  # dense, opaque
  We use:     state = topk(SAE.encode(hidden))         # sparse, monosemantic

The hierarchy emerges naturally:
  - Some features only fire when others are on (superstate patterns)
  - Feature co-activation clusters = discovered states
  - Transition = feature activation pattern change

Reference:
  - Anthropic "Towards Monosemanticity" (2023)
  - "Scaling Monosemanticity" on Claude 3 Sonnet (2024)
  - "Finite State Automata Inside Transformers" (2025)
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import json


@dataclass
class SAEConfig:
    """Configuration for the SAE state bottleneck."""
    input_dim: int = 256          # LM hidden dimension
    expansion_factor: int = 16     # Overcomplete factor (16x = 4096 latents for 256 dim)
    k_active: int = 32            # TopK active features per step
    dead_threshold: float = 1e-6  # Threshold for dead feature detection
    resample_dead: bool = True    # Resample dead features periodically
    l1_coefficient: float = 1e-3  # Sparsity penalty
    tied_weights: bool = True     # Tie encoder/decoder weights


class TopKSAE(nn.Module):
    """
    TopK Sparse Autoencoder for statechart state representation.

    Unlike ReLU SAEs, TopK guarantees exact sparsity:
      - Always exactly k features active
      - No need to tune L1 coefficient for sparsity level
      - Dead features are clearly identifiable

    The active features ARE the state:
      - Feature 42 active = "inside async function"
      - Feature 1337 active = "awaiting promise"
      - Co-activation {42, 1337, 89} = compound state
    """

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        latent_dim = config.input_dim * config.expansion_factor

        # Encoder: project to overcomplete space
        self.encoder = nn.Linear(config.input_dim, latent_dim)

        # Decoder: reconstruct from sparse code
        self.decoder = nn.Linear(latent_dim, config.input_dim, bias=False)

        if config.tied_weights:
            # Tied weights: decoder = encoder.T (classic autoencoder)
            # In practice we'll sync manually since MLX doesn't auto-tie
            pass

        # Track feature activations for dead feature detection
        self.feature_activations = mx.zeros((latent_dim,))
        self.steps_since_active = mx.zeros((latent_dim,))

    def encode(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Encode input to sparse TopK representation.

        Returns:
            acts: Sparse activation values (batch, latent_dim) with k non-zero
            indices: Which k features are active (batch, k)
        """
        # Project to latent space
        pre_acts = self.encoder(x)  # (batch, latent_dim)

        # TopK selection
        k = self.config.k_active

        # Get top-k indices using argsort (MLX topk only returns values)
        # Sort descending, take first k indices
        sorted_indices = mx.argsort(-pre_acts, axis=-1)  # Descending order
        top_indices = sorted_indices[:, :k]

        # Create sparse activation tensor
        batch_size = x.shape[0]
        latent_dim = pre_acts.shape[-1]

        # Build sparse tensor with ReLU-ed top-k values
        acts_list = []
        for b in range(batch_size):
            row = [0.0] * latent_dim
            for i in range(k):
                idx = int(top_indices[b, i].item())
                val = float(pre_acts[b, idx].item())
                row[idx] = max(0.0, val)  # ReLU
            acts_list.append(row)
        acts = mx.array(acts_list)

        return acts, top_indices

    def decode(self, acts: mx.array) -> mx.array:
        """Decode sparse representation back to input space."""
        return self.decoder(acts)

    def forward(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Full forward pass: encode → decode with residual.

        Returns:
            output: Reconstructed input + residual
            acts: Sparse activations (the STATE)
            indices: Active feature indices
        """
        acts, indices = self.encode(x)
        recon = self.decode(acts)

        # Residual connection: output = reconstruction + original
        # This lets gradient flow even with sparse bottleneck
        output = recon + x

        return output, acts, indices

    def compute_loss(self, x: mx.array) -> Tuple[mx.array, Dict]:
        """
        Compute SAE loss: reconstruction + sparsity.

        Returns:
            loss: Total loss scalar
            metrics: Dict with reconstruction_loss, sparsity_loss, etc.
        """
        acts, indices = self.encode(x)
        recon = self.decode(acts)

        # Reconstruction loss (MSE)
        recon_loss = mx.mean((x - recon) ** 2)

        # Sparsity loss (L1 on activations)
        # With TopK this is mostly for regularization, not sparsity control
        sparsity_loss = self.config.l1_coefficient * mx.mean(mx.abs(acts))

        # Total loss
        loss = recon_loss + sparsity_loss

        metrics = {
            'reconstruction_loss': float(recon_loss),
            'sparsity_loss': float(sparsity_loss),
            'active_features': float(mx.sum(acts > 0) / acts.shape[0]),
            'mean_activation': float(mx.mean(acts[acts > 0])) if mx.any(acts > 0) else 0.0,
        }

        return loss, metrics


class SAEStatechartModule(nn.Module):
    """
    Differentiable statechart with SAE state bottleneck.

    The STATE is literally the top-k active SAE features:
      - Each feature is a monosemantic "micro-state"
      - Co-activation patterns = compound states
      - Feature activation changes = transitions
      - Sustained patterns = stable states (superstates)

    Hierarchy emerges naturally:
      - Some features only fire when others are on
      - These conditional features = substates
      - Unconditional features = superstate markers
    """

    def __init__(self, sae_config: SAEConfig, history_depth: int = 3):
        super().__init__()
        self.sae = TopKSAE(sae_config)
        self.history_depth = history_depth

        latent_dim = sae_config.input_dim * sae_config.expansion_factor

        # History bank for restore (stores SAE activations, not hidden states)
        self.history_bank = None  # Will be (batch, history_depth, latent_dim)

        # Transition predictor: given current features, predict next features
        self.transition_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim),
        )

        # Superstate detector: find feature clusters that co-activate
        self.superstate_detector = nn.Linear(latent_dim, latent_dim // 4)

    def step(self, hidden: mx.array, push_history: bool = False) -> Tuple[mx.array, mx.array]:
        """
        Process one step through the SAE statechart.

        Args:
            hidden: LM hidden state (batch, dim)
            push_history: Whether to save current state to history

        Returns:
            output: Processed hidden state
            state_features: Active SAE features (the interpretable state)
        """
        # Encode to sparse state
        output, acts, indices = self.sae.forward(hidden)

        # Optionally push to history
        if push_history and self.history_bank is not None:
            # Shift history and add new state
            self.history_bank = mx.concatenate([
                self.history_bank[:, 1:, :],
                acts[:, None, :]
            ], axis=1)

        return output, acts

    def restore_from_history(self, depth: int = 1) -> Optional[mx.array]:
        """
        Restore state from history bank.

        Returns the SAE activation pattern from `depth` steps ago.
        This is interpretable: you can see exactly which features were active.
        """
        if self.history_bank is None:
            return None

        idx = self.history_depth - depth
        if idx < 0:
            return None

        return self.history_bank[:, idx, :]

    def extract_discovered_states(self, activation_log: mx.array) -> Dict:
        """
        Extract discovered states from activation history.

        Given a log of SAE activations over a sequence, find:
        1. Stable states (sustained feature patterns)
        2. Transitions (pattern changes)
        3. Hierarchy (feature conditional dependencies)

        Args:
            activation_log: (seq_len, latent_dim) SAE activations over time

        Returns:
            Dict with discovered_states, transitions, hierarchy
        """
        import numpy as np

        # Convert to numpy for analysis (MLX lacks some features)
        activation_np = np.array(activation_log.tolist())
        # Handle case of extra batch dimension
        if len(activation_np.shape) == 3:
            activation_np = activation_np.squeeze(1)  # Remove batch dim

        seq_len = activation_np.shape[0]
        latent_dim = activation_np.shape[1]

        # Binarize: which features are active at each step
        active = (activation_np > 0).astype(np.float32)

        # Find stable patterns (same features active for multiple steps)
        stable_patterns = []
        current_pattern = active[0]
        pattern_start = 0

        for t in range(1, seq_len):
            if not np.allclose(active[t], current_pattern):
                # Pattern changed - record the stable state
                duration = t - pattern_start
                if duration >= 2:  # At least 2 steps = stable
                    features = np.where(current_pattern > 0)[0].tolist()
                    stable_patterns.append({
                        'features': features,
                        'start': pattern_start,
                        'end': t,
                        'duration': duration,
                    })
                current_pattern = active[t]
                pattern_start = t

        # Find transitions (which features toggled)
        transitions = []
        for t in range(1, seq_len):
            diff = active[t] - active[t-1]
            activated = np.where(diff > 0)[0].tolist()
            deactivated = np.where(diff < 0)[0].tolist()
            if activated or deactivated:
                transitions.append({
                    'step': t,
                    'activated': activated,
                    'deactivated': deactivated,
                })

        # Find hierarchy (features that only fire when others are on)
        # A feature is a "substate" if it's only active when some other feature is
        co_occurrence = np.matmul(active.T, active)  # (latent, latent)
        feature_counts = np.sum(active, axis=0)

        hierarchy = {}
        for i in range(latent_dim):
            if feature_counts[i] > 0:
                # Find features that are always active when i is active
                conditional_on = []
                for j in range(latent_dim):
                    if i != j and co_occurrence[i, j] == feature_counts[i]:
                        # j is always on when i is on → i is substate of j
                        conditional_on.append(int(j))
                if conditional_on:
                    hierarchy[int(i)] = conditional_on

        return {
            'stable_states': stable_patterns,
            'transitions': transitions,
            'hierarchy': hierarchy,
            'total_features_used': int(np.sum(np.any(active > 0, axis=0))),
        }


@dataclass
class DiscoveredState:
    """A state discovered from SAE feature patterns."""
    id: str
    features: Set[int]           # Which SAE features define this state
    feature_names: List[str]     # Human-readable feature labels (if known)
    superstate_of: List[str]     # IDs of substates
    substate_of: Optional[str]   # ID of parent superstate
    entry_transitions: List[int] # Steps where this state was entered
    exit_transitions: List[int]  # Steps where this state was exited
    total_duration: int          # Total time spent in this state


class SAEStatechartExtractor:
    """
    Extract an explicit statechart from SAE activation patterns.

    This is the "discovery mode" - we run the SAE-augmented model,
    collect activations, then compile them into a formal statechart.

    The result is a proper Harel statechart with:
    - States (stable feature patterns)
    - Transitions (pattern changes, can be labeled with triggering context)
    - Hierarchy (feature conditional dependencies → superstates)
    - Parallel regions (orthogonal feature clusters)
    """

    def __init__(self, sae_module: SAEStatechartModule):
        self.sae_module = sae_module
        self.activation_log = []
        self.context_log = []  # What input caused each activation

    def record(self, hidden: mx.array, context: Optional[str] = None):
        """Record SAE activations for later analysis."""
        _, acts = self.sae_module.step(hidden)
        self.activation_log.append(acts)
        self.context_log.append(context)

    def extract_statechart(self, min_state_duration: int = 2) -> Dict:
        """
        Extract a formal statechart from recorded activations.

        Returns a JSON-serializable statechart definition compatible
        with the sc proto format.
        """
        if not self.activation_log:
            return {'error': 'No activations recorded'}

        # Stack activations
        acts = mx.stack(self.activation_log, axis=0)  # (seq, latent)

        # Get discovered structure
        structure = self.sae_module.extract_discovered_states(acts)

        # Build statechart proto
        states = []
        transitions = []

        # Create state nodes from stable patterns
        for i, pattern in enumerate(structure['stable_states']):
            if pattern['duration'] >= min_state_duration:
                state_id = f"state_{i}"
                states.append({
                    'label': state_id,
                    'type': 'STATE_TYPE_BASIC',  # Leaf state
                    'features': pattern['features'],
                    'duration': pattern['duration'],
                })

        # Create transitions from pattern changes
        for trans in structure['transitions']:
            transitions.append({
                'step': trans['step'],
                'activated': trans['activated'],
                'deactivated': trans['deactivated'],
                'context': self.context_log[trans['step']] if trans['step'] < len(self.context_log) else None,
            })

        # Build hierarchy from feature dependencies
        hierarchy = {}
        for subfeature, superfeatures in structure['hierarchy'].items():
            hierarchy[f'feature_{subfeature}'] = [f'feature_{f}' for f in superfeatures]

        return {
            'states': states,
            'transitions': transitions,
            'hierarchy': hierarchy,
            'total_features': structure['total_features_used'],
            'metadata': {
                'extraction_method': 'SAE_TopK',
                'k': self.sae_module.sae.config.k_active,
                'expansion': self.sae_module.sae.config.expansion_factor,
            }
        }

    def to_mermaid(self) -> str:
        """Generate Mermaid diagram of extracted statechart."""
        chart = self.extract_statechart()

        lines = ['stateDiagram-v2']

        # Add states
        for state in chart.get('states', []):
            label = state['label']
            features = state['features'][:3]  # First 3 features for label
            lines.append(f"    {label}: {label}\\n(features: {features}...)")

        # Add transitions (simplified - consecutive states)
        states = chart.get('states', [])
        for i in range(len(states) - 1):
            src = states[i]['label']
            dst = states[i + 1]['label']
            lines.append(f"    {src} --> {dst}")

        return '\n'.join(lines)


def demo_sae_statechart():
    """
    Demonstrate SAE-grown statecharts.

    We:
    1. Create a SAE statechart module
    2. Feed it some synthetic "code generation" hidden states
    3. Extract the discovered statechart
    4. Show which features correspond to which states
    """
    print("=" * 60)
    print("SAE-GROWN STATECHART DEMO")
    print("=" * 60)

    # Config
    config = SAEConfig(
        input_dim=64,
        expansion_factor=8,  # 512 latent features
        k_active=8,          # 8 active at a time
    )

    # Create module
    module = SAEStatechartModule(config, history_depth=3)
    extractor = SAEStatechartExtractor(module)

    # Simulate hidden states from "code generation"
    # In reality these would come from an LM
    mx.random.seed(42)

    contexts = [
        "def ",
        "function_name",
        "(",
        "arg1",
        ",",
        "arg2",
        ")",
        ":",
        "\n",
        "    ",
        "return",
        " ",
        "result",
        "\n",
    ]

    print("\nProcessing sequence:")
    for i, ctx in enumerate(contexts):
        # Synthetic hidden state (in reality: LM hidden)
        hidden = mx.random.normal((1, config.input_dim)) * 0.5

        # Add some structure: similar contexts → similar hiddens
        if ctx in ["(", ")", ":"]:
            hidden = hidden + mx.array([[1.0] * 32 + [0.0] * 32])
        elif ctx in ["def ", "return"]:
            hidden = hidden + mx.array([[0.0] * 32 + [1.0] * 32])

        extractor.record(hidden, ctx)
        print(f"  Step {i}: '{ctx}'")

    # Extract statechart
    print("\nExtracting statechart from SAE activations...")
    chart = extractor.extract_statechart(min_state_duration=1)

    print(f"\nDiscovered {len(chart['states'])} stable states")
    print(f"Total features used: {chart['total_features']}")

    print("\nStates:")
    for state in chart['states']:
        print(f"  {state['label']}: features={state['features'][:5]}... duration={state['duration']}")

    print("\nTransitions:")
    for trans in chart['transitions'][:5]:
        print(f"  Step {trans['step']}: +{trans['activated'][:3]} -{trans['deactivated'][:3]}")

    print("\nHierarchy (substates):")
    for sub, supers in list(chart['hierarchy'].items())[:5]:
        print(f"  {sub} is substate of {supers}")

    print("\nMermaid Diagram:")
    print(extractor.to_mermaid())

    return chart


if __name__ == "__main__":
    demo_sae_statechart()
