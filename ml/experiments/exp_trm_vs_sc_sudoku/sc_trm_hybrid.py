"""
Hybrid TRM: SC-TRM + Attention Bias + Gradient Truncation.

Combines the best features from experiments:
1. Constraint Biased Attention (from attention_bias.py): Soft inductive bias for Sudoku pairs.
2. Gradient Truncation (from faithful_trm_v3.py): Only backprop through final H-cycle.
3. H/L Cycle Structure (from faithful_trm_v3.py): Recursive refinement.

Hypothesis: This combination fits strict Sudoku constraints while maintaining stable training.
"""

import sys
# sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import math

def rms_norm(x: mx.array, eps: float = 1e-5) -> mx.array:
    """RMS normalization."""
    variance = mx.mean(x * x, axis=-1, keepdims=True)
    return x * mx.rsqrt(variance + eps)

@dataclass
class SCTRMConfig:
    """Config for Hybrid TRM."""
    # TRM Core
    vocab_size: int = 10
    hidden_size: int = 128
    num_heads: int = 4
    ff_dim: int = 256
    num_layers: int = 2     # Layers per L-level
    H_cycles: int = 3       # Outer cycles
    L_cycles: int = 6       # Inner cycles per H
    dropout: float = 0.1
    rms_norm_eps: float = 1e-5
    
    # Attention Bias
    use_constraint_bias: bool = True
    initial_bias_scale: float = 1.0
    final_bias_scale: float = 5.0
    learnable_bias: bool = True
    
    # Initialization
    init_std: float = 0.02

# Backward compatibility (alias)
HybridTRMConfig = SCTRMConfig

class ConstraintGuard:
    """
    Placeholder for constraint enforcement logic.
    Future work: Use this to hard-mask invalid transitions during inference.
    """
    pass


def build_constraint_affinity_matrix() -> mx.array:
    """Build 81x81 constraint affinity matrix (same row/col/box)."""
    cells = mx.arange(81)
    rows = cells // 9
    cols = cells % 9
    boxes = (rows // 3) * 3 + (cols // 3)

    rows_i = rows[:, None]; rows_j = rows[None, :]
    cols_i = cols[:, None]; cols_j = cols[None, :]
    boxes_i = boxes[:, None]; boxes_j = boxes[None, :]

    same_row = (rows_i == rows_j).astype(mx.float32)
    same_col = (cols_i == cols_j).astype(mx.float32)
    same_box = (boxes_i == boxes_j).astype(mx.float32)

    # Union of constraints
    affinity = mx.maximum(mx.maximum(same_row, same_col), same_box)
    
    # Remove self-connections (diagonal)
    eye = mx.eye(81)
    affinity = affinity * (1 - eye)
    return affinity


class ConstraintBiasedAttention(nn.Module):
    """Multi-head attention with learnable constraint bias."""
    def __init__(self, config: SCTRMConfig, constraint_affinity: mx.array):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.constraint_affinity = constraint_affinity
        
        if config.learnable_bias:
            self.bias_scales = mx.ones((config.num_heads,)) * config.initial_bias_scale
        else:
            self.bias_scales = None

    def __call__(self, x: mx.array, bias_strength: float = 1.0) -> mx.array:
        B, N, D = x.shape
        
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        scale = self.head_dim ** -0.5
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        
        # Add constraint bias
        if self.bias_scales is not None:
             # bias_scales: [H] -> [1, H, 1, 1]
             scales = self.bias_scales[None, :, None, None]
             # affinity: [81, 81] -> [1, 1, 81, 81]
             # bias: [1, H, 81, 81]
             bias = self.constraint_affinity[None, None, :, :] * scales * bias_strength
             scores = scores + bias
             
        attn_weights = mx.softmax(scores, axis=-1)
        out = mx.matmul(attn_weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, D)
        return self.o_proj(out)


class HybridBlock(nn.Module):
    """Transformer block with Constraint Biased Attention."""
    def __init__(self, config: SCTRMConfig, constraint_affinity: mx.array):
        super().__init__()
        self.attn = ConstraintBiasedAttention(config, constraint_affinity)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.ff = nn.Sequential(
            nn.Linear(config.hidden_size, config.ff_dim),
            nn.GELU(),
            nn.Linear(config.ff_dim, config.hidden_size)
        )
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x: mx.array, bias_strength: float = 1.0) -> mx.array:
        h = self.ln1(x)
        h = self.attn(h, bias_strength=bias_strength)
        x = x + self.dropout(h)
        
        h = self.ln2(x)
        h = self.ff(h)
        x = x + self.dropout(h)
        return x


class ReasoningModule(nn.Module):
    """L-level network with stacked HybridBlocks."""
    def __init__(self, config: SCTRMConfig, constraint_affinity: mx.array):
        super().__init__()
        self.layers = [HybridBlock(config, constraint_affinity) for _ in range(config.num_layers)]

    def __call__(self, hidden_states: mx.array, input_injection: mx.array, bias_strength: float = 1.0) -> mx.array:
        x = hidden_states + input_injection
        for layer in self.layers:
            x = layer(x, bias_strength=bias_strength)
        return x


class SCTRMHybrid(nn.Module):
    """
    Hybrid TRM model.
    Combines:
    - Recursive H/L structure
    - Gradient Truncation (stable training)
    - Attention Bias (structure injection)
    - Bias Annealing (soft -> hard constraints)
    """
    def __init__(self, config: Optional[SCTRMConfig] = None):
        super().__init__()
        self.config = config or SCTRMConfig()
        
        # Embeddings
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.embed_pos = nn.Embedding(81, self.config.hidden_size) # Flattened 81 positions
        
        # Constraint Bias Matrix
        self.constraint_affinity = build_constraint_affinity_matrix()
        
        # Shared L-level module
        self.L_level = ReasoningModule(self.config, self.constraint_affinity)
        
        # Initial states (truncated normal approximation via clip)
        self.H_init = mx.clip(
            mx.random.normal(shape=(self.config.hidden_size,)) * self.config.init_std,
            -2 * self.config.init_std, 2 * self.config.init_std
        )
        self.L_init = mx.clip(
            mx.random.normal(shape=(self.config.hidden_size,)) * self.config.init_std,
            -2 * self.config.init_std, 2 * self.config.init_std
        )
        
        self.ln_f = nn.LayerNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def __call__(self, input_ids: mx.array, truncate_grad: bool = True) -> mx.array:
        B, L = input_ids.shape
        
        # Embeddings
        positions = mx.arange(L)
        # Scale embedding by sqrt(d_model) - standard Transformer practice
        # Note: Faithful v3 used it, attention_bias.py didn't. We'll use it.
        embed_scale = math.sqrt(self.config.hidden_size)
        input_embed = (self.embed_tokens(input_ids) + self.embed_pos(positions)) * embed_scale
        
        # Initialize states
        z_H = mx.broadcast_to(self.H_init, (B, L, self.config.hidden_size))
        z_L = mx.broadcast_to(self.L_init, (B, L, self.config.hidden_size))
        
        # Bias annealing schedule
        H_total = self.config.H_cycles
        initial_bias = self.config.initial_bias_scale
        final_bias = self.config.final_bias_scale
        
        # H-Cycle Loop
        for h_step in range(H_total):
            # Calculate current bias strength (linear annealing)
            if H_total > 1:
                bias_strength = initial_bias + (final_bias - initial_bias) * h_step / (H_total - 1)
            else:
                bias_strength = final_bias
            
            # Gradient Truncation: Detach H-state if not the final cycle
            # This is the key "Faithful" trick.
            if truncate_grad and h_step < (H_total - 1):
                z_H = mx.stop_gradient(z_H)
                # Note: We don't detach z_L usually in TRM, but Faithful v3 detaches BOTH.
                # Let's follow Faithful v3 exactly for stability.
                z_L = mx.stop_gradient(z_L)
            
            # Inner L-Cycles
            # Note: Faithful v3 calls L_level for L_cycles, then updates z_H once.
            for _l in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + input_embed, bias_strength=bias_strength)
            
            # Update H state
            z_H = self.L_level(z_H, z_L, bias_strength=bias_strength)
            
        return self.lm_head(self.ln_f(z_H))

    def solve(self, puzzle: mx.array, temperature: float = 0.0) -> dict:
        """Solve puzzle (inference mode)."""
        logits = self(puzzle, truncate_grad=False)
        
        if temperature > 0:
            probs = mx.softmax(logits / temperature, axis=-1)
            predictions = mx.argmax(probs, axis=-1)
        else:
            predictions = mx.argmax(logits, axis=-1)
            
        # Overlay original inputs
        mask = puzzle > 0
        predictions = mx.where(mask, puzzle, predictions)
        return {"predictions": predictions, "logits": logits}

    def loss(self, puzzle: mx.array, solution: mx.array, truncate_grad: bool = True):
        logits = self(puzzle, truncate_grad=truncate_grad)
        
        # Cross Entropy on empty cells only
        empty_mask = (puzzle == 0).astype(mx.float32)
        log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)
        
        # Gather correct class probs
        # solution is inputs (0-9). logits are (10 classes).
        # Wait, usually Sudoku logits are 1-9 or 0-9?
        # Standard configs usually have vocab_size=10 (0 for empty in input, but output should predict 1-9).
        # Let's assume prediction index i correspnods to digit i.
        
        target_indices = solution[:, :, None] # [B, 81, 1]
        correct_log_probs = mx.take_along_axis(log_probs, target_indices, axis=-1).squeeze(-1)
        
        loss = -mx.sum(correct_log_probs * empty_mask) / (mx.sum(empty_mask) + 1e-10)
        
        predictions = mx.argmax(logits, axis=-1)
        cell_acc = mx.sum((predictions == solution).astype(mx.float32) * empty_mask) / (mx.sum(empty_mask) + 1e-10)
        
        return loss, {"cell_accuracy": cell_acc}

# Alias for backward compatibility if needed, but class rename is cleaner
HybridTRM = SCTRMHybrid


def create_model(config: Optional[HybridTRMConfig] = None) -> HybridTRM:
    return HybridTRM(config)

if __name__ == "__main__":
    print("Testing Hybrid TRM (Bias + Grad Truncation)...")
    config = SCTRMConfig()
    model = SCTRMHybrid(config)
    
    # Dummy data
    B = 2
    puzzle = mx.zeros((B, 81), dtype=mx.int32)
    solution = mx.ones((B, 81), dtype=mx.int32)
    
    # Test Forward
    logits = model(puzzle, truncate_grad=True)
    print(f"Forward output shape: {logits.shape}")
    
    # Test Loss
    loss, metrics = model.loss(puzzle, solution, truncate_grad=True)
    print(f"Loss: {loss.item():.4f}, Acc: {metrics['cell_accuracy'].item():.4f}")
    
    # Test Bias Matrix
    print(f"Constraint affinity shape: {model.constraint_affinity.shape}")
    print(f"Non-zero elements: {mx.sum(model.constraint_affinity > 0).item()}")
