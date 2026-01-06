import mlx.core as mx
import mlx.nn as nn
import math

class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)

class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: int = 4, align_to: int = 256):
        super().__init__()
        # Samsung TRM uses a specific alignment logic:
        # ((int(expansion * hidden_size * 2 / 3) + 255) // 256) * 256
        inter_size = ((int(expansion * hidden_size * 2 / 3) + (align_to - 1)) // align_to) * align_to
        
        self.w12 = nn.Linear(hidden_size, 2 * inter_size, bias=False)
        self.w3 = nn.Linear(inter_size, hidden_size, bias=False)

    def __call__(self, x):
        x12 = self.w12(x)
        x1, x2 = mx.split(x12, 2, axis=-1)
        return self.w3(nn.silu(x1) * x2)

class CastedLinear(nn.Linear):
    """
    MLX linear layer that can behave like the 'CastedLinear' from PyTorch implementation,
    though MLX handles casting implicitly in many cases.
    We mainly want to ensure initialization matches.
    """
    def __init__(self, input_dims, output_dims, bias=True):
        super().__init__(input_dims, output_dims, bias=bias)
    
    # In MLX, we might not need explicit casting logic for forward if using standard dtypes,
    # but we can implement it if strictly needed. For now, standard Linear is usually sufficient.

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Precompute cos/sin tables? 
        # MLX's fast.rope is usually efficient enough to compute on fly or we can cache.
        # We'll use mlx.fast.rope which takes standard rope arguments.

    def __call__(self, q, k, offset=0):
        # q: [B, H, L, D] or [B, L, H, D] - MLX expects [B, L, H, D] usually for Rope?
        # Check mlx.fast.rope signature: (x, dims, theta, scale, offset, axis)
        # dims is head dim.
        
        # We need to construct the standard RoPE application.
        # Assuming q, k structure matches what typical MLX attention expects.
        # In PyTorch TRM: q, k are [B, H, L, D] (batch, heads, seq, head_dim)
        # MLX usually prefers [B, L, H, D].
        
        # We will assume calling code handles valid shaping, but typically we implement the logic here.
        
        # Using mlx.fast.rope
        # x: input tensor
        # dims: dimension of rotary embedding
        # theta: base for frequencies (default 10000)
        # scale: scale for frequencies (default 1.0)
        # offset: position offset
        # axis: axis along which to apply rope (usually spatial/sequence dimension?)
        # Wait, mlx.fast.rope applies positional embedding to x.
        
        q = mx.fast.rope(q, self.dim, base=float(self.base), scale=1.0, traditional=False, offset=offset)
        k = mx.fast.rope(k, self.dim, base=float(self.base), scale=1.0, traditional=False, offset=offset)
        
        return q, k
