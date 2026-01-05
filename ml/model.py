
import mlx.core as mx
import mlx.nn as nn
import math

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=4, context_size=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(context_size, d_model) # Simple learned positional encoding
        self.blocks = [
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ]
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.context_size = context_size

    def __call__(self, x):
        B, T = x.shape
        # Positions: [0, 1, ... T-1]
        # range is not yet fully in core in same way, using arange
        pos = mx.arange(0, T, dtype=mx.int32) 
        
        # Token + Pos embeddings
        x = self.embedding(x) + self.pos_encoding(pos)
        
        # Mask for causal attention (lower triangular)
        # mlx.nn.MultiHeadAttention handles causal masking if we pass mask.
        # But for custom block, we usually construct it.
        # Simpler: nn.MultiHeadAttention in MLX likely doesn't assume causal by default without mask.
        # We will generate a mask.
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.att = nn.MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def __call__(self, x, mask=None):
        # Attention
        # nn.MultiHeadAttention expects (q, k, v, mask)
        # For self-attention q=k=v=x
        y = self.att(self.ln1(x), self.ln1(x), self.ln1(x), mask=mask)
        x = x + y
        
        # MLP
        x = x + self.mlp(self.ln2(x))
        return x

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.l1 = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU()
        self.l2 = nn.Linear(4 * d_model, d_model)

    def __call__(self, x):
        return self.l2(self.gelu(self.l1(x)))
