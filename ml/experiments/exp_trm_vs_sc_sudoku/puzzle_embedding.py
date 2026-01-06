import mlx.core as mx
import mlx.nn as nn

class PuzzleEmbedding(nn.Module):
    """
    Emulates the behavior of Samsung's CastedSparseEmbedding.
    In MLX (and Metal), sparse tensors aren't fully supported in same way as PyTorch's sparse gradients.
    Since models fit in memory usually, we can use a dense Embedding table but only update indices used in batch.
    MLX's gather/scatter behavior naturally handles this for sparse updates during backprop efficiently.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, sparse: bool = True):
        super().__init__()
        self.weight = mx.zeros((num_embeddings, embedding_dim))
        self.sparse = sparse # Flag to indicate intention, though MLX Embedding is naturally dense storage

    def __call__(self, x):
        return self.weight[x]
    
    def as_embedding_layer(self):
        # Helper to treat this as nn.Embedding compatible just in case
        pass
