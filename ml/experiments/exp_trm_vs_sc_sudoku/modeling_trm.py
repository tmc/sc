import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, List
import math

from .layers import RMSNorm, SwiGLU, CastedLinear, RotaryEmbedding
from .puzzle_embedding import PuzzleEmbedding

@dataclass
class TRMConfig:
    vocab_size: int = 32
    hidden_size: int = 512
    num_heads: int = 8
    max_position_embeddings: int = 2048
    H_cycles: int = 3
    L_cycles: int = 6
    L_layers: int = 2
    expansion: int = 4
    puzzle_emb_ndim: int = 512
    num_puzzle_identifiers: int = 1000  # Default, should be set
    puzzle_emb_len: int = 16
    pos_encodings: str = "rope" # "rope" or "learned"
    head_dim: int = 64
    dropout: float = 0.0
    mlp_t: bool = False
    
    # ACT parameters
    halt_exploration_prob: float = 0.0
    halt_max_steps: int = 16

class Attention(nn.Module):
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)
        
        if config.pos_encodings == "rope":
            self.rope = RotaryEmbedding(config.head_dim, config.max_position_embeddings)
        else:
            self.rope = None

    def __call__(self, x, mask=None):
        B, L, _ = x.shape
        
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        
        # MLX attention expects [B, H, L, D] usually for MultiHeadAttention, 
        # but manual implementation gives us control.
        # Let's align with PyTorch TRM which is [B, H, L, D] usually.
        # Transpose to [B, L, H, D] is standard for MLX layers, but let's check rope.
        # My rope implementation expects x, so let's stick to [B, L, H, D].
        
        if self.rope is not None:
            q, k = self.rope(q, k)
            
        # Scaled Dot Product Attention
        # scores: [B, H, L, L] after transpose
        q = q.transpose(0, 2, 1, 3) # [B, H, L, D]
        k = k.transpose(0, 2, 1, 3) # [B, H, L, D]
        v = v.transpose(0, 2, 1, 3) # [B, H, L, D]
        
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale
        
        if mask is not None:
             # Mask should be broadcastable to [B, H, L, L]
             scores = scores + mask
             
        probs = mx.softmax(scores, axis=-1)
        output = (probs @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        
        return self.o_proj(output)

class TransformerBlock(nn.Module):
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size)
        self.attn = Attention(config)
        self.norm2 = RMSNorm(config.hidden_size)
        self.mlp = SwiGLU(config.hidden_size, config.expansion)
        
    def __call__(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x

class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.embed_scale = math.sqrt(config.hidden_size)
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = PuzzleEmbedding(config.num_puzzle_identifiers, config.puzzle_emb_ndim)
            
        if config.pos_encodings == "learned":
            self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        else:
            self.embed_positions = None
            
        # L_level transformer blocks (Shared)
        self.layers = [TransformerBlock(config) for _ in range(config.L_layers)]
        
        # Heads
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = nn.Linear(config.hidden_size, 2, bias=True)
        self.q_head.bias = mx.full((2,), -5.0) # Init bias to -5 as per PyTorch impl for continue bias
        
    def _apply_L_level(self, z_L, z_H, mask=None):
        # Samsung: z_L = L_level(z_L, z_H + inputs) 
        # Actually it's: input to layers is z_L, but z_H is added or concatenated?
        # Checking faithful implementation:
        # z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        # where L_level takes (h, condition, ...)
        # In PyTorch modeling_trm.py L124:
        # def forward(self, hidden_states, condition=None, ...):
        #   hidden_states = hidden_states + condition
        
        x = z_L + z_H
        for layer in self.layers:
            x = layer(x, mask)
        return x

    def forward_one_step(self, z_L, z_H, input_embeddings, mask=None):
        # One high-level Step consisting of L_cycles updates to z_L
        # followed by one update to z_H
        
        # z_L updates
        for _ in range(self.config.L_cycles):
            # z_L = L_level(z_L, z_H + input_embeddings)
            # Condition is z_H + input_embeddings
            condition = z_H + input_embeddings
            z_L = self._apply_L_level(z_L, condition, mask)
            
        # z_H update (once per H-cycle)
        # z_H = L_level(z_H, z_L)
        # Condition is z_L
        z_H = self._apply_L_level(z_H, z_L, mask)
        
        return z_L, z_H

    def __call__(self, input_ids: mx.array, puzzle_ids: Optional[mx.array] = None, mask=None, training=False):
        B, L = input_ids.shape
        
        # Embeddings
        token_emb = self.embed_tokens(input_ids) * self.embed_scale
        
        # Puzzle Embeddings
        if self.config.puzzle_emb_ndim > 0 and puzzle_ids is not None:
            # [B, puzzle_len, D]
            # puzzle_emb = self.puzzle_emb(puzzle_ids) # [B, D] -> need seq len?
            # Samsung: Puzzle embedding is [B, puzzle_emb_len, D].
            # Actually, puzzle_emb is sparse [num_puzzles, D].
            # input to forward is:
            # puzzle_embedding = self.puzzle_emb(puzzle_indices) # [B, D]
            # puzzle_embedding = puzzle_embedding.unsqueeze(1).expand(-1, puzzle_emb_len, -1)
            # Then concat.
            p_emb_vec = self.puzzle_emb(puzzle_ids) # [B, D]
            p_emb_seq = mx.expand_dims(p_emb_vec, 1) # [B, 1, D]
            p_emb_seq = mx.repeat(p_emb_seq, self.config.puzzle_emb_len, axis=1) # [B, PL, D]
            
            # Concatenate [Puzzle | Tokens]
            input_embeddings = mx.concatenate([p_emb_seq, token_emb], axis=1)
        else:
            input_embeddings = token_emb

        # Position embeddings if learned
        if self.embed_positions is not None:
             seq_len = input_embeddings.shape[1]
             positions = mx.arange(seq_len)
             pos_emb = self.embed_positions(positions)
             input_embeddings = input_embeddings + pos_emb
             
        # Initialize z_H, z_L
        # truncated normal with std=1.0 per faithful v3
        z_H = mx.random.normal(input_embeddings.shape) # Approximate
        z_L = mx.random.normal(input_embeddings.shape) # Approximate
        
        # H_cycles loop with gradient truncation
        # Samsung: H_cycles-1 WITHOUT grad, then 1 with grad
        
        if training:
            # First H-1 cycles without gradient tracking
            # In MLX, we use stop_gradient to detach
            for _ in range(self.config.H_cycles - 1):
                z_L, z_H = self.forward_one_step(z_L, z_H, input_embeddings, mask)
                z_L = mx.stop_gradient(z_L)
                z_H = mx.stop_gradient(z_H)
                
            # Final cycle with gradient
            z_L, z_H = self.forward_one_step(z_L, z_H, input_embeddings, mask)
        else:
            # Inference: just run all cycles
            for _ in range(self.config.H_cycles):
                z_L, z_H = self.forward_one_step(z_L, z_H, input_embeddings, mask)
                
        # Output logic
        # LM Head on z_H
        logits = self.lm_head(z_H)
        
        # If we had puzzle embeddings prepended, slice them off
        if self.config.puzzle_emb_ndim > 0:
            logits = logits[:, self.config.puzzle_emb_len:, :]
            z_final_for_q = z_H[:, self.config.puzzle_emb_len:, :]
        else:
            z_final_for_q = z_H
            
        # Q-Head for halting (on final state)
        # pooled or per token? PyTorch TRM: q_head is applied to sequence.
        # "self.q_head(z_H)" -> [B, L, 2]
        q_logits = self.q_head(z_final_for_q)
        
        return logits, q_logits

class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """
    Wrapper for Adaptive Computation Time (ACT).
    Manages the halting loop.
    """
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(config)
        
    def __call__(self, input_ids, puzzle_ids=None, mask=None, training=False):
        # In full ACT, this runs a while loop checking q_logits.
        # For simple training verification, we often just run the inner model once 
        # if treating it as a fixed-depth unroll, but ACT implies dynamic depth.
        
        # For strict port, we need the logic that loops until 'halt' or max_steps.
        # However, MLX generic graph compilation might prefer static unroll or python loop.
        
        # Returning inner result for now as basic verification step.
        return self.inner(input_ids, puzzle_ids, mask, training=training)
