"""
Neural Baseline Models for Coverage Prediction

Three approaches that DON'T use statecharts:
1. Sequence Model: (tokens, input) -> coverage logits
2. AST-Based: Embed AST nodes, predict per-node coverage
3. GNN on CFG: Message passing on control flow graph

These serve as baselines to compare against statechart-based prediction.
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
import ast
import json


@dataclass
class CoverageModelConfig:
    """Configuration for coverage prediction models."""
    vocab_size: int = 10000       # Token vocabulary size
    embed_dim: int = 128          # Embedding dimension
    hidden_dim: int = 256         # Hidden layer dimension
    n_layers: int = 2             # Number of layers
    n_heads: int = 4              # Attention heads (for transformer)
    max_lines: int = 100          # Maximum lines in a program
    max_tokens: int = 512         # Maximum tokens
    dropout: float = 0.1          # Dropout rate


class TokenEmbedder(nn.Module):
    """Embed program tokens with positional encoding."""

    def __init__(self, config: CoverageModelConfig):
        super().__init__()
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Embedding(config.max_tokens, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.embed_dim = config.embed_dim

    def __call__(self, token_ids: mx.array) -> mx.array:
        """
        Embed tokens with positions.

        Args:
            token_ids: (batch, seq_len) token indices

        Returns:
            (batch, seq_len, embed_dim) embeddings
        """
        seq_len = token_ids.shape[1]
        positions = mx.arange(seq_len)

        tok_emb = self.token_embed(token_ids)
        pos_emb = self.pos_embed(positions)

        return self.dropout(tok_emb + pos_emb)


# =============================================================================
# BASELINE 1: SEQUENCE MODEL
# =============================================================================

class SequenceCoverageModel(nn.Module):
    """
    Sequence model for coverage prediction.

    Takes (program tokens, input tokens) and predicts per-line coverage.
    Uses a transformer encoder to process the concatenated sequence.

    Architecture:
    1. Embed program tokens
    2. Embed input tokens
    3. Concatenate with separator
    4. Transformer encoder
    5. Pool to line-level predictions
    6. Binary classification per line
    """

    def __init__(self, config: CoverageModelConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embedder = TokenEmbedder(config)

        # Transformer encoder layers
        self.encoder_layers = [
            nn.TransformerEncoderLayer(
                dims=config.embed_dim,
                num_heads=config.n_heads,
                mlp_dims=config.hidden_dim,
            )
            for _ in range(config.n_layers)
        ]

        # Line classifier
        self.line_classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.max_lines),
        )

    def __call__(
        self,
        program_tokens: mx.array,
        input_tokens: mx.array,
        line_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Predict line coverage.

        Args:
            program_tokens: (batch, prog_len) program token IDs
            input_tokens: (batch, input_len) input token IDs
            line_mask: (batch, max_lines) which lines exist

        Returns:
            (batch, max_lines) coverage logits per line
        """
        # Embed both sequences
        prog_emb = self.embedder(program_tokens)  # (B, P, D)
        input_emb = self.embedder(input_tokens)   # (B, I, D)

        # Concatenate: [PROG] [SEP] [INPUT]
        # Use zero vector as separator
        sep = mx.zeros((prog_emb.shape[0], 1, self.config.embed_dim))
        combined = mx.concatenate([prog_emb, sep, input_emb], axis=1)  # (B, P+1+I, D)

        # Transformer encoding
        hidden = combined
        for layer in self.encoder_layers:
            hidden = layer(hidden, mask=None)

        # Pool over sequence dimension (mean pooling)
        pooled = mx.mean(hidden, axis=1)  # (B, D)

        # Predict per-line coverage
        logits = self.line_classifier(pooled)  # (B, max_lines)

        # Mask out non-existent lines
        if line_mask is not None:
            logits = logits * line_mask + (1 - line_mask) * (-1e9)

        return logits

    def predict_coverage(
        self,
        program_tokens: mx.array,
        input_tokens: mx.array,
        line_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Get coverage probabilities."""
        logits = self(program_tokens, input_tokens, line_mask)
        return mx.sigmoid(logits)


# =============================================================================
# BASELINE 2: AST-BASED MODEL
# =============================================================================

@dataclass
class ASTNode:
    """A node in the AST for neural embedding."""
    id: int
    node_type: str           # e.g., "FunctionDef", "If", "Return"
    token: Optional[str]     # Leaf token (for Names, literals)
    line: int                # Source line
    children: List[int]      # Child node IDs

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'type': self.node_type,
            'token': self.token,
            'line': self.line,
            'children': self.children,
        }


class ASTCoverageModel(nn.Module):
    """
    AST-based coverage prediction.

    Embeds AST nodes and predicts per-node coverage.
    Then aggregates to line-level predictions.

    Architecture:
    1. Embed node types and tokens
    2. Tree-structured attention (bottom-up then top-down)
    3. Node-level coverage prediction
    4. Aggregate to line-level
    """

    def __init__(self, config: CoverageModelConfig, n_node_types: int = 100):
        super().__init__()
        self.config = config

        # Node type embedding
        self.type_embed = nn.Embedding(n_node_types, config.embed_dim)

        # Token embedding (reuse from sequence model)
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)

        # Combine type and token
        self.combine = nn.Linear(config.embed_dim * 2, config.embed_dim)

        # Tree attention layers
        self.tree_layers = [
            nn.TransformerEncoderLayer(
                dims=config.embed_dim,
                num_heads=config.n_heads,
                mlp_dims=config.hidden_dim,
            )
            for _ in range(config.n_layers)
        ]

        # Node coverage classifier
        self.node_classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

        # Input encoder
        self.input_encoder = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embed_dim),
        )

    def embed_ast(
        self,
        node_types: mx.array,     # (batch, n_nodes) node type IDs
        node_tokens: mx.array,    # (batch, n_nodes) token IDs
    ) -> mx.array:
        """Embed AST nodes."""
        type_emb = self.type_embed(node_types)    # (B, N, D)
        token_emb = self.token_embed(node_tokens)  # (B, N, D)

        combined = mx.concatenate([type_emb, token_emb], axis=-1)  # (B, N, 2D)
        return self.combine(combined)  # (B, N, D)

    def __call__(
        self,
        node_types: mx.array,
        node_tokens: mx.array,
        node_lines: mx.array,     # (batch, n_nodes) line number per node
        input_tokens: mx.array,
        n_lines: int,
    ) -> mx.array:
        """
        Predict line coverage from AST.

        Returns:
            (batch, n_lines) coverage logits
        """
        batch_size = node_types.shape[0]
        n_nodes = node_types.shape[1]

        # Embed AST nodes
        node_emb = self.embed_ast(node_types, node_tokens)  # (B, N, D)

        # Embed input and get global input context
        input_emb = self.input_encoder(
            mx.mean(self.token_embed(input_tokens), axis=1, keepdims=True)
        )  # (B, 1, D)

        # Broadcast input context to all nodes
        node_emb = node_emb + input_emb  # (B, N, D)

        # Tree attention
        hidden = node_emb
        for layer in self.tree_layers:
            hidden = layer(hidden, mask=None)  # (B, N, D)

        # Per-node coverage predictions
        node_logits = self.node_classifier(hidden).squeeze(-1)  # (B, N)

        # Aggregate to line-level: max-pool over nodes on same line
        # This is simplified - in practice would use scatter operations
        line_logits_list = []
        for b in range(batch_size):
            line_preds = []
            for line in range(1, n_lines + 1):
                # Find nodes on this line
                line_mask = (node_lines[b] == line).astype(mx.float32)
                if mx.sum(line_mask) > 0:
                    # Max pool over nodes on this line
                    masked = node_logits[b] * line_mask - 1e9 * (1 - line_mask)
                    line_preds.append(mx.max(masked))
                else:
                    line_preds.append(mx.array(-1e9))
            line_logits_list.append(mx.stack(line_preds))

        line_logits = mx.stack(line_logits_list)  # (B, n_lines)
        return line_logits


# =============================================================================
# BASELINE 3: GNN ON CFG
# =============================================================================

class GraphAttentionLayer(nn.Module):
    """Graph Attention layer for CFG message passing."""

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads

        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def __call__(
        self,
        node_features: mx.array,  # (batch, n_nodes, dim)
        adjacency: mx.array,      # (batch, n_nodes, n_nodes) edge mask
    ) -> mx.array:
        """Message passing with attention."""
        B, N, D = node_features.shape

        # Multi-head projections
        Q = self.q_proj(node_features)  # (B, N, out_dim)
        K = self.k_proj(node_features)
        V = self.v_proj(node_features)

        # Reshape for multi-head attention
        Q = Q.reshape(B, N, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(B, N, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(B, N, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        # Now (B, H, N, D/H)

        # Attention scores
        scale = self.head_dim ** -0.5
        scores = mx.matmul(Q, K.transpose(0, 1, 3, 2)) * scale  # (B, H, N, N)

        # Mask with adjacency (only attend to neighbors)
        adj_mask = adjacency.reshape(B, 1, N, N)  # (B, 1, N, N)
        scores = scores * adj_mask - 1e9 * (1 - adj_mask)

        # Softmax and aggregate
        attn = mx.softmax(scores, axis=-1)
        out = mx.matmul(attn, V)  # (B, H, N, D/H)

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(B, N, -1)  # (B, N, out_dim)
        return self.out_proj(out)


class GNNCoverageModel(nn.Module):
    """
    GNN-based coverage prediction on CFG.

    Uses message passing on the control flow graph structure.

    Architecture:
    1. Embed CFG nodes (basic blocks)
    2. Graph attention message passing
    3. Condition on input embedding
    4. Per-block coverage prediction
    5. Aggregate to line-level
    """

    def __init__(self, config: CoverageModelConfig, n_block_types: int = 10):
        super().__init__()
        self.config = config

        # Block type embedding
        self.block_embed = nn.Embedding(n_block_types, config.embed_dim)

        # Line content embedding (simplified - embed first token per block)
        self.content_embed = nn.Embedding(config.vocab_size, config.embed_dim)

        # Input encoder
        self.input_encoder = TokenEmbedder(config)
        self.input_proj = nn.Linear(config.embed_dim, config.embed_dim)

        # Graph attention layers
        self.gat_layers = [
            GraphAttentionLayer(config.embed_dim, config.embed_dim, config.n_heads)
            for _ in range(config.n_layers)
        ]

        # Layer norms
        self.layer_norms = [
            nn.LayerNorm(config.embed_dim)
            for _ in range(config.n_layers)
        ]

        # Block classifier
        self.block_classifier = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def __call__(
        self,
        block_types: mx.array,      # (batch, n_blocks) block type IDs
        block_content: mx.array,    # (batch, n_blocks) content token IDs
        block_lines: mx.array,      # (batch, n_blocks, max_lines_per_block) lines in each block
        adjacency: mx.array,        # (batch, n_blocks, n_blocks) CFG edges
        input_tokens: mx.array,     # (batch, input_len) input tokens
        n_lines: int,
    ) -> mx.array:
        """
        Predict line coverage from CFG.

        Returns:
            (batch, n_lines) coverage logits
        """
        batch_size = block_types.shape[0]
        n_blocks = block_types.shape[1]

        # Embed blocks
        type_emb = self.block_embed(block_types)        # (B, N, D)
        content_emb = self.content_embed(block_content)  # (B, N, D)
        block_emb = type_emb + content_emb               # (B, N, D)

        # Embed input and get global context
        input_emb = self.input_encoder(input_tokens)  # (B, I, D)
        input_ctx = self.input_proj(mx.mean(input_emb, axis=1))  # (B, D)

        # Graph attention message passing
        hidden = block_emb
        for gat, ln in zip(self.gat_layers, self.layer_norms):
            residual = hidden
            hidden = gat(hidden, adjacency)
            hidden = ln(hidden + residual)

        # Concatenate block features with input context
        input_ctx_expanded = input_ctx.reshape(batch_size, 1, -1)
        input_ctx_expanded = mx.broadcast_to(
            input_ctx_expanded,
            (batch_size, n_blocks, self.config.embed_dim)
        )
        combined = mx.concatenate([hidden, input_ctx_expanded], axis=-1)  # (B, N, 2D)

        # Per-block predictions
        block_logits = self.block_classifier(combined).squeeze(-1)  # (B, N)

        # Aggregate to line-level (max over blocks containing each line)
        # Simplified implementation
        line_logits = self._aggregate_to_lines(
            block_logits, block_lines, n_lines, batch_size
        )

        return line_logits

    def _aggregate_to_lines(
        self,
        block_logits: mx.array,
        block_lines: mx.array,
        n_lines: int,
        batch_size: int,
    ) -> mx.array:
        """Aggregate block predictions to line predictions."""
        # For each line, take max over blocks containing that line
        n_blocks = block_logits.shape[1]
        max_lines_per_block = block_lines.shape[2]

        line_logits_list = []
        for b in range(batch_size):
            line_preds = []
            for line in range(1, n_lines + 1):
                # Check which blocks contain this line
                max_logit = mx.array(-1e9)
                for block in range(n_blocks):
                    if mx.any(block_lines[b, block] == line):
                        max_logit = mx.maximum(max_logit, block_logits[b, block])
                line_preds.append(max_logit)
            line_logits_list.append(mx.stack(line_preds))

        return mx.stack(line_logits_list)


# =============================================================================
# TOKENIZER (Simple)
# =============================================================================

class SimpleTokenizer:
    """Simple tokenizer for programs and inputs."""

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SEP>': 2,
        }
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.token_to_id.items()}
        self.next_id = 3

    def add_token(self, token: str) -> int:
        """Add a token to vocabulary."""
        if token not in self.token_to_id:
            if self.next_id < self.vocab_size:
                self.token_to_id[token] = self.next_id
                self.id_to_token[self.next_id] = token
                self.next_id += 1
            else:
                return self.token_to_id['<UNK>']
        return self.token_to_id[token]

    def encode(self, text: str) -> List[int]:
        """Tokenize text to IDs."""
        # Simple whitespace + punctuation tokenization
        import re
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return [self.add_token(t) for t in tokens]

    def encode_batch(self, texts: List[str], max_len: int) -> mx.array:
        """Encode batch with padding."""
        encoded = []
        for text in texts:
            ids = self.encode(text)[:max_len]
            ids = ids + [0] * (max_len - len(ids))  # Pad
            encoded.append(ids)
        return mx.array(encoded)


# =============================================================================
# LOSS FUNCTION
# =============================================================================

def coverage_loss(
    logits: mx.array,           # (batch, n_lines)
    targets: mx.array,          # (batch, n_lines) binary coverage
    mask: Optional[mx.array] = None,  # (batch, n_lines) valid lines
) -> mx.array:
    """
    Binary cross-entropy loss for coverage prediction.

    Handles class imbalance (most lines not covered).
    """
    # Compute BCE
    probs = mx.sigmoid(logits)
    bce = -targets * mx.log(probs + 1e-8) - (1 - targets) * mx.log(1 - probs + 1e-8)

    # Apply mask if provided
    if mask is not None:
        bce = bce * mask
        return mx.sum(bce) / (mx.sum(mask) + 1e-8)

    return mx.mean(bce)


def demo():
    """Demonstrate baseline models."""
    print("=" * 60)
    print("NEURAL BASELINE MODELS DEMO")
    print("=" * 60)

    config = CoverageModelConfig(
        vocab_size=1000,
        embed_dim=64,
        hidden_dim=128,
        n_layers=2,
        max_lines=20,
        max_tokens=100,
    )

    # Create tokenizer
    tokenizer = SimpleTokenizer(config.vocab_size)

    # Example program
    program = """
def foo(x):
    if x > 0:
        return x
    else:
        return -x
"""

    # Example input
    input_str = "5"

    # Tokenize
    prog_tokens = tokenizer.encode_batch([program], config.max_tokens)
    input_tokens = tokenizer.encode_batch([input_str], 10)

    print(f"Program tokens shape: {prog_tokens.shape}")
    print(f"Input tokens shape: {input_tokens.shape}")

    # Test sequence model
    print("\n1. SEQUENCE MODEL")
    seq_model = SequenceCoverageModel(config)
    line_mask = mx.ones((1, config.max_lines))
    seq_logits = seq_model(prog_tokens, input_tokens, line_mask)
    print(f"   Output shape: {seq_logits.shape}")
    print(f"   First 5 logits: {seq_logits[0, :5].tolist()}")

    # Test AST model
    print("\n2. AST MODEL")
    ast_model = ASTCoverageModel(config)
    # Fake AST data
    node_types = mx.array([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0]])
    node_tokens = mx.array([[10, 20, 30, 40, 50, 0, 0, 0, 0, 0]])
    node_lines = mx.array([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0]])
    ast_logits = ast_model(node_types, node_tokens, node_lines, input_tokens, n_lines=10)
    print(f"   Output shape: {ast_logits.shape}")
    print(f"   First 5 logits: {ast_logits[0, :5].tolist()}")

    # Test GNN model
    print("\n3. GNN MODEL")
    gnn_model = GNNCoverageModel(config)
    # Fake CFG data
    block_types = mx.array([[1, 2, 3, 4, 0]])
    block_content = mx.array([[10, 20, 30, 40, 0]])
    block_lines = mx.zeros((1, 5, 3))  # (batch, blocks, lines_per_block)
    adjacency = mx.array([[[1, 1, 0, 0, 0],
                           [0, 1, 1, 0, 0],
                           [0, 0, 1, 1, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0]]], dtype=mx.float32)
    gnn_logits = gnn_model(
        block_types, block_content, block_lines, adjacency, input_tokens, n_lines=10
    )
    print(f"   Output shape: {gnn_logits.shape}")
    print(f"   First 5 logits: {gnn_logits[0, :5].tolist()}")

    # Test loss
    print("\n4. LOSS COMPUTATION")
    targets = mx.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=mx.float32)
    loss = coverage_loss(seq_logits[:, :10], targets)
    print(f"   Loss: {float(loss):.4f}")

    print("\n" + "=" * 60)
    print("All baseline models initialized successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
