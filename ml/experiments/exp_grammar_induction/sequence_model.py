"""
Sequence Model for Go Token Prediction

A Transformer model that predicts the next token in a Go token sequence.
The hidden states are analyzed with SAE to discover syntax contexts.

Key Features:
- Causal (decoder-only) transformer for autoregressive prediction
- Returns hidden states at each position for SAE analysis
- Tracks prediction confidence for grammar rule inference
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math
import random

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    nn = None
    optim = None


@dataclass
class TransformerConfig:
    """Configuration for the syntax transformer."""
    vocab_size: int = 100        # Token type vocabulary size
    hidden_dim: int = 256        # Hidden dimension
    num_layers: int = 4          # Number of transformer layers
    num_heads: int = 8           # Number of attention heads
    ff_dim: int = 1024           # Feed-forward dimension
    max_seq_len: int = 512       # Maximum sequence length
    dropout: float = 0.1         # Dropout rate
    
    # For SAE analysis
    return_hidden_states: bool = True


class PositionalEncoding(nn.Module if HAS_MLX else object):
    """Sinusoidal positional encoding."""

    def __init__(self, dim: int, max_len: int = 5000):
        if HAS_MLX:
            super().__init__()
            self.dim = dim

            # Create position encoding table efficiently using numpy then convert
            import numpy as np
            position = np.arange(max_len)[:, np.newaxis]
            div_term = np.exp(
                np.arange(0, dim, 2) * (-math.log(10000.0) / dim)
            )

            pe = np.zeros((max_len, dim))
            pe[:, 0::2] = np.sin(position * div_term)
            pe[:, 1::2] = np.cos(position * div_term)

            self.pe = mx.array(pe)
        else:
            self.dim = dim

    def __call__(self, x: 'mx.array') -> 'mx.array':
        """Add positional encoding to input."""
        if not HAS_MLX:
            raise RuntimeError("MLX not available")
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]


class MultiHeadAttention(nn.Module if HAS_MLX else object):
    """Multi-head self-attention with causal masking."""
    
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        if HAS_MLX:
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.scale = self.head_dim ** -0.5
            
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.out_proj = nn.Linear(dim, dim)
            self.dropout = nn.Dropout(dropout)
        else:
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.scale = self.head_dim ** -0.5
    
    def __call__(
        self,
        x: 'mx.array',
        mask: Optional['mx.array'] = None,
    ) -> 'mx.array':
        """
        Apply multi-head self-attention.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional causal mask
        
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        if not HAS_MLX:
            raise RuntimeError("MLX not available")
        
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention: (batch, heads, seq, head_dim)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        
        # Compute attention scores
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * self.scale
        
        # Apply causal mask
        if mask is not None:
            scores = scores + mask
        
        # Softmax and dropout
        attn = mx.softmax(scores, axis=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = mx.matmul(attn, v)
        
        # Reshape back
        out = mx.transpose(out, (0, 2, 1, 3))
        out = out.reshape(batch_size, seq_len, -1)
        
        return self.out_proj(out)


class TransformerBlock(nn.Module if HAS_MLX else object):
    """A single transformer block with attention and FFN."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        if HAS_MLX:
            super().__init__()
            self.attn = MultiHeadAttention(dim, num_heads, dropout)
            self.ff = nn.Sequential(
                nn.Linear(dim, ff_dim),
                nn.GELU(),
                nn.Linear(ff_dim, dim),
                nn.Dropout(dropout),
            )
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.dropout = nn.Dropout(dropout)
    
    def __call__(
        self,
        x: 'mx.array',
        mask: Optional['mx.array'] = None,
    ) -> 'mx.array':
        """Forward pass through the block."""
        if not HAS_MLX:
            raise RuntimeError("MLX not available")
        
        # Self-attention with residual
        attn_out = self.attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        # FFN with residual
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        
        return x


class GoSyntaxTransformer(nn.Module if HAS_MLX else object):
    """
    Transformer for Go syntax modeling.
    
    Predicts the next token type given a sequence of token types.
    Returns hidden states for SAE analysis.
    """
    
    def __init__(self, config: TransformerConfig):
        if HAS_MLX:
            super().__init__()
        
        self.config = config
        
        if HAS_MLX:
            # Token embedding
            self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
            
            # Positional encoding
            self.pos_encoding = PositionalEncoding(
                config.hidden_dim, config.max_seq_len
            )
            
            # Transformer blocks
            self.blocks = [
                TransformerBlock(
                    config.hidden_dim,
                    config.num_heads,
                    config.ff_dim,
                    config.dropout,
                )
                for _ in range(config.num_layers)
            ]
            
            # Output projection
            self.output_norm = nn.LayerNorm(config.hidden_dim)
            self.output_proj = nn.Linear(config.hidden_dim, config.vocab_size)
            
            # Dropout
            self.dropout = nn.Dropout(config.dropout)
    
    def _create_causal_mask(self, seq_len: int) -> 'mx.array':
        """Create causal attention mask."""
        if not HAS_MLX:
            raise RuntimeError("MLX not available")
        
        # Create lower triangular mask
        mask = mx.triu(
            mx.full((seq_len, seq_len), float('-inf')),
            k=1,
        )
        return mask
    
    def __call__(
        self,
        token_ids: 'mx.array',
        return_hidden_states: bool = False,
    ) -> Tuple['mx.array', Optional[List['mx.array']]]:
        """
        Forward pass.
        
        Args:
            token_ids: Token type IDs (batch, seq_len)
            return_hidden_states: Whether to return hidden states from each layer
        
        Returns:
            logits: Next token predictions (batch, seq_len, vocab_size)
            hidden_states: Optional list of hidden states per layer
        """
        if not HAS_MLX:
            raise RuntimeError("MLX not available")
        
        batch_size, seq_len = token_ids.shape
        
        # Embed tokens
        x = self.token_embedding(token_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Create causal mask
        mask = self._create_causal_mask(seq_len)
        
        # Process through transformer blocks
        hidden_states = []
        for block in self.blocks:
            x = block(x, mask)
            if return_hidden_states or self.config.return_hidden_states:
                hidden_states.append(x)
        
        # Output projection
        x = self.output_norm(x)
        logits = self.output_proj(x)
        
        if return_hidden_states or self.config.return_hidden_states:
            return logits, hidden_states
        return logits, None
    
    def predict_next(
        self,
        token_ids: 'mx.array',
        temperature: float = 1.0,
    ) -> Tuple[int, 'mx.array']:
        """
        Predict the next token.
        
        Args:
            token_ids: Token sequence (batch, seq_len)
            temperature: Sampling temperature
        
        Returns:
            next_token: Predicted next token ID
            probs: Probability distribution over tokens
        """
        if not HAS_MLX:
            raise RuntimeError("MLX not available")
        
        logits, _ = self(token_ids)
        
        # Get logits for last position
        last_logits = logits[:, -1, :] / temperature
        probs = mx.softmax(last_logits, axis=-1)
        
        # Sample
        next_token = int(mx.argmax(probs, axis=-1)[0])
        
        return next_token, probs
    
    def get_hidden_states(
        self,
        token_ids: 'mx.array',
        layer: int = -1,
    ) -> 'mx.array':
        """
        Get hidden states from a specific layer.

        Args:
            token_ids: Token sequence
            layer: Which layer (-1 for last)

        Returns:
            Hidden states (batch, seq_len, hidden_dim)
        """
        if not HAS_MLX:
            raise RuntimeError("MLX not available")

        _, hidden_states = self(token_ids, return_hidden_states=True)
        return hidden_states[layer]

    def get_attention_weights(
        self,
        token_ids: 'mx.array',
    ) -> List['mx.array']:
        """
        Get attention weights from all layers.

        Args:
            token_ids: Token sequence (batch, seq_len)

        Returns:
            List of attention weights per layer (batch, heads, seq, seq)
        """
        if not HAS_MLX:
            raise RuntimeError("MLX not available")

        batch_size, seq_len = token_ids.shape

        # Embed tokens
        x = self.token_embedding(token_ids)
        x = self.pos_encoding(x)

        # Create causal mask
        mask = self._create_causal_mask(seq_len)

        # Collect attention weights from each block
        attention_weights = []
        for block in self.blocks:
            # Get attention weights from the attention layer
            attn = block.attn

            # Compute Q, K, V
            q = attn.q_proj(x)
            k = attn.k_proj(x)

            # Reshape for multi-head attention
            q = q.reshape(batch_size, seq_len, attn.num_heads, attn.head_dim)
            k = k.reshape(batch_size, seq_len, attn.num_heads, attn.head_dim)

            # Transpose: (batch, heads, seq, head_dim)
            q = mx.transpose(q, (0, 2, 1, 3))
            k = mx.transpose(k, (0, 2, 1, 3))

            # Compute attention scores
            scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * attn.scale

            # Apply mask
            if mask is not None:
                scores = scores + mask

            # Softmax to get attention weights
            weights = mx.softmax(scores, axis=-1)
            attention_weights.append(weights)

            # Continue forward pass for next layer
            x = block(x, mask)

        return attention_weights

    def compute_loss(
        self,
        token_ids: 'mx.array',
    ) -> 'mx.array':
        """
        Compute cross-entropy loss for next-token prediction.
        
        Args:
            token_ids: Token sequence (batch, seq_len)
        
        Returns:
            Loss scalar
        """
        if not HAS_MLX:
            raise RuntimeError("MLX not available")
        
        logits, _ = self(token_ids[:, :-1])  # Predict next token
        targets = token_ids[:, 1:]            # Shift targets
        
        # Reshape for cross-entropy
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        
        # Cross-entropy loss
        log_probs = nn.log_softmax(logits_flat, axis=-1)
        loss = -mx.mean(
            mx.take_along_axis(
                log_probs,
                targets_flat[:, None],
                axis=-1,
            )
        )
        
        return loss


@dataclass
class TrainingState:
    """Training state for the syntax transformer."""
    step: int = 0
    epoch: int = 0
    best_loss: float = float('inf')
    losses: List[float] = field(default_factory=list)


class SyntaxTrainer:
    """
    Trainer for the syntax transformer.
    
    Handles training loop, evaluation, and hidden state collection.
    """
    
    def __init__(
        self,
        model: GoSyntaxTransformer,
        learning_rate: float = 1e-4,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.state = TrainingState()
        
        if HAS_MLX:
            self.optimizer = optim.Adam(learning_rate=learning_rate)
    
    def train_step(
        self,
        batch: 'mx.array',
    ) -> float:
        """
        Execute one training step.
        
        Args:
            batch: Token IDs (batch_size, seq_len)
        
        Returns:
            Loss value
        """
        if not HAS_MLX:
            return 0.0
        
        def loss_fn(model):
            return model.compute_loss(batch)
        
        loss, grads = mx.value_and_grad(loss_fn)(self.model)
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)
        
        loss_val = float(loss)
        self.state.losses.append(loss_val)
        self.state.step += 1
        
        return loss_val
    
    def evaluate(
        self,
        sequences: List['mx.array'],
    ) -> Dict[str, float]:
        """
        Evaluate model on sequences.
        
        Returns:
            Dict with loss, accuracy, perplexity
        """
        if not HAS_MLX:
            return {}
        
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        
        for batch in sequences:
            logits, _ = self.model(batch[:, :-1])
            targets = batch[:, 1:]
            
            # Compute loss
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.reshape(-1, vocab_size)
            targets_flat = targets.reshape(-1)
            
            log_probs = nn.log_softmax(logits_flat, axis=-1)
            loss = -mx.mean(
                mx.take_along_axis(log_probs, targets_flat[:, None], axis=-1)
            )
            total_loss += float(loss) * batch_size * seq_len
            
            # Compute accuracy
            predictions = mx.argmax(logits, axis=-1)
            correct = mx.sum(predictions == targets)
            total_correct += int(correct)
            total_tokens += batch_size * seq_len
        
        avg_loss = total_loss / total_tokens
        accuracy = total_correct / total_tokens
        perplexity = math.exp(avg_loss)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'perplexity': perplexity,
        }
    
    def collect_hidden_states(
        self,
        sequences: List['mx.array'],
        layer: int = -1,
    ) -> List['mx.array']:
        """
        Collect hidden states from sequences.
        
        Used for SAE training.
        
        Args:
            sequences: List of token ID tensors
            layer: Which layer to collect from
        
        Returns:
            List of hidden state tensors
        """
        if not HAS_MLX:
            return []
        
        hidden_states = []
        for batch in sequences:
            states = self.model.get_hidden_states(batch, layer=layer)
            hidden_states.append(states)
        
        return hidden_states


def demo():
    """Demonstrate the syntax transformer."""
    print("=" * 60)
    print("GO SYNTAX TRANSFORMER DEMO")
    print("=" * 60)
    
    if not HAS_MLX:
        print("\nMLX not available. Running in simulation mode.")
        
        # Simulate with random data
        config = TransformerConfig(
            vocab_size=50,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            ff_dim=128,
            max_seq_len=64,
        )
        
        print(f"\nConfig: {config}")
        print("\nWould create transformer with:")
        print(f"  - {config.vocab_size} token types")
        print(f"  - {config.hidden_dim} hidden dimension")
        print(f"  - {config.num_layers} layers")
        print(f"  - {config.num_heads} attention heads")
        
        return
    
    # Create model
    config = TransformerConfig(
        vocab_size=50,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        ff_dim=128,
        max_seq_len=64,
    )
    
    model = GoSyntaxTransformer(config)
    print(f"\nCreated model with config: {config}")
    
    # Create sample input
    batch_size = 4
    seq_len = 16
    
    # Random token IDs
    token_ids = mx.array([
        [random.randint(0, 49) for _ in range(seq_len)]
        for _ in range(batch_size)
    ])
    
    print(f"\nInput shape: {token_ids.shape}")
    
    # Forward pass
    logits, hidden_states = model(token_ids, return_hidden_states=True)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Number of hidden state layers: {len(hidden_states)}")
    print(f"Hidden state shape: {hidden_states[0].shape}")
    
    # Compute loss
    loss = model.compute_loss(token_ids)
    print(f"Loss: {float(loss):.4f}")
    
    # Predict next token
    next_token, probs = model.predict_next(token_ids)
    print(f"Predicted next token: {next_token}")
    print(f"Top 5 probs: {sorted(probs[0].tolist(), reverse=True)[:5]}")
    
    # Training demo
    print("\n" + "=" * 60)
    print("TRAINING DEMO")
    print("=" * 60)
    
    trainer = SyntaxTrainer(model, learning_rate=1e-3)
    
    for step in range(5):
        # Generate random batch
        batch = mx.array([
            [random.randint(0, 49) for _ in range(seq_len)]
            for _ in range(batch_size)
        ])
        
        loss = trainer.train_step(batch)
        print(f"Step {step + 1}: loss = {loss:.4f}")
    
    # Collect hidden states
    hidden = trainer.collect_hidden_states([token_ids])
    print(f"\nCollected hidden states: {len(hidden)} batches")
    print(f"Shape per batch: {hidden[0].shape}")


if __name__ == "__main__":
    demo()
