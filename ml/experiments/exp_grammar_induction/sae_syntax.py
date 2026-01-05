"""
SAE for Syntax State Discovery

A TopK Sparse Autoencoder that discovers discrete syntax contexts from
the hidden states of the sequence model.

Key Insight:
    SAE features correspond to grammatical constructs:
    - Feature X active → "inside function body"
    - Feature Y active → "after opening brace"
    - Co-activation patterns → compound syntax states

This module:
    1. Trains SAE on sequence model hidden states
    2. Clusters features into discrete syntax contexts
    3. Maps features to Go spec production rules
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
from enum import Enum
import math
import random

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    nn = None


class GoProductionRule(Enum):
    """Go grammar production rules (simplified from spec)."""
    # Package structure
    PACKAGE_CLAUSE = "PackageClause"
    IMPORT_DECL = "ImportDecl"
    
    # Declarations
    CONST_DECL = "ConstDecl"
    TYPE_DECL = "TypeDecl"
    VAR_DECL = "VarDecl"
    FUNC_DECL = "FunctionDecl"
    METHOD_DECL = "MethodDecl"
    
    # Types
    TYPE_LIT = "TypeLit"
    STRUCT_TYPE = "StructType"
    INTERFACE_TYPE = "InterfaceType"
    MAP_TYPE = "MapType"
    CHANNEL_TYPE = "ChannelType"
    FUNC_TYPE = "FunctionType"
    
    # Statements
    BLOCK = "Block"
    IF_STMT = "IfStmt"
    SWITCH_STMT = "SwitchStmt"
    SELECT_STMT = "SelectStmt"
    FOR_STMT = "ForStmt"
    RETURN_STMT = "ReturnStmt"
    GO_STMT = "GoStmt"
    DEFER_STMT = "DeferStmt"
    
    # Expressions
    EXPRESSION = "Expression"
    CALL_EXPR = "CallExpr"
    SELECTOR_EXPR = "SelectorExpr"
    INDEX_EXPR = "IndexExpr"
    SLICE_EXPR = "SliceExpr"
    COMPOSITE_LIT = "CompositeLit"
    
    # Special contexts
    PARAMS = "Parameters"
    RESULTS = "Results"
    FIELD_LIST = "FieldList"
    
    # Unknown
    UNKNOWN = "Unknown"


@dataclass
class SAEConfig:
    """Configuration for the syntax SAE."""
    input_dim: int = 256          # Hidden dimension from transformer
    expansion_factor: int = 16    # Overcomplete factor
    k_active: int = 32            # TopK active features
    l1_coefficient: float = 1e-3  # Sparsity penalty
    dead_threshold: float = 1e-6  # Dead feature detection
    resample_dead: bool = True    # Resample dead features


@dataclass
class SyntaxFeature:
    """A discovered syntax feature from SAE."""
    feature_id: int
    activation_mean: float = 0.0
    activation_count: int = 0
    associated_tokens: Set[str] = field(default_factory=set)
    associated_rules: Set[GoProductionRule] = field(default_factory=set)
    co_occurring_features: Set[int] = field(default_factory=set)
    description: str = ""
    
    @property
    def is_dead(self) -> bool:
        """Check if feature is dead (never activates)."""
        return self.activation_count == 0
    
    def describe(self) -> str:
        """Generate human-readable description."""
        tokens = ', '.join(list(self.associated_tokens)[:5])
        rules = ', '.join(r.value for r in self.associated_rules)
        return (f"Feature {self.feature_id}: "
                f"tokens=[{tokens}], rules=[{rules}], "
                f"count={self.activation_count}")


class TopKSAE(nn.Module if HAS_MLX else object):
    """
    TopK Sparse Autoencoder for syntax discovery.
    
    Unlike ReLU SAEs, TopK guarantees exactly k features are active,
    making it easier to identify discrete syntax states.
    """
    
    def __init__(self, config: SAEConfig):
        if HAS_MLX:
            super().__init__()
        
        self.config = config
        self.latent_dim = config.input_dim * config.expansion_factor
        
        if HAS_MLX:
            # Encoder: project to overcomplete space
            self.encoder = nn.Linear(config.input_dim, self.latent_dim)
            
            # Decoder: reconstruct from sparse code
            self.decoder = nn.Linear(self.latent_dim, config.input_dim, bias=False)
            
            # Track feature activations
            self._activation_counts = mx.zeros((self.latent_dim,))
    
    def encode(self, x: 'mx.array') -> Tuple['mx.array', 'mx.array']:
        """
        Encode input to sparse TopK representation.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            acts: Sparse activations (batch, latent_dim)
            indices: Active feature indices (batch, k)
        """
        if not HAS_MLX:
            raise RuntimeError("MLX not available")

        # Project to latent space
        pre_acts = self.encoder(x)  # (batch, latent_dim)

        # TopK selection - vectorized implementation
        k = self.config.k_active

        # Get top-k values and indices
        sorted_indices = mx.argsort(-pre_acts, axis=-1)
        top_indices = sorted_indices[:, :k]

        # Create mask for top-k positions
        # We use scatter to place 1s at top_k positions, then multiply by pre_acts
        import numpy as np
        batch_size = x.shape[0]

        # Efficient: gather top-k values, apply ReLU, then scatter back
        # Gather the values at top_indices
        top_values = mx.take_along_axis(pre_acts, top_indices, axis=-1)
        top_values = mx.maximum(top_values, 0.0)  # ReLU

        # Create sparse output using numpy (fast) then convert
        acts_np = np.zeros((batch_size, self.latent_dim), dtype=np.float32)
        top_indices_np = np.array(top_indices.tolist())
        top_values_np = np.array(top_values.tolist())

        for b in range(batch_size):
            acts_np[b, top_indices_np[b]] = top_values_np[b]

        acts = mx.array(acts_np)

        return acts, top_indices
    
    def decode(self, acts: 'mx.array') -> 'mx.array':
        """Decode sparse representation back to input space."""
        if not HAS_MLX:
            raise RuntimeError("MLX not available")
        return self.decoder(acts)
    
    def __call__(self, x: 'mx.array') -> Tuple['mx.array', 'mx.array', 'mx.array']:
        """
        Full forward pass.
        
        Returns:
            reconstruction: Reconstructed input
            acts: Sparse activations
            indices: Active feature indices
        """
        if not HAS_MLX:
            raise RuntimeError("MLX not available")
        
        acts, indices = self.encode(x)
        recon = self.decode(acts)
        return recon, acts, indices
    
    def compute_loss(self, x: 'mx.array') -> Tuple['mx.array', Dict]:
        """
        Compute SAE loss.
        
        Returns:
            loss: Total loss
            metrics: Dict with loss components
        """
        if not HAS_MLX:
            raise RuntimeError("MLX not available")
        
        acts, indices = self.encode(x)
        recon = self.decode(acts)
        
        # Reconstruction loss
        recon_loss = mx.mean((x - recon) ** 2)
        
        # Sparsity loss (L1)
        sparsity_loss = self.config.l1_coefficient * mx.mean(mx.abs(acts))
        
        loss = recon_loss + sparsity_loss
        
        metrics = {
            'reconstruction_loss': float(recon_loss),
            'sparsity_loss': float(sparsity_loss),
            'total_loss': float(loss),
            'active_features': float(mx.sum(acts > 0) / acts.shape[0]),
        }
        
        return loss, metrics


class SyntaxSAE:
    """
    SAE-based syntax state discovery.
    
    Wraps TopKSAE with:
    - Feature-to-syntax mapping
    - Cluster analysis for state discovery
    - Integration with Go production rules
    """
    
    def __init__(self, config: SAEConfig):
        self.config = config
        self.sae = TopKSAE(config) if HAS_MLX else None
        
        # Feature tracking
        self.features: Dict[int, SyntaxFeature] = {
            i: SyntaxFeature(feature_id=i)
            for i in range(config.input_dim * config.expansion_factor)
        }
        
        # Token-feature associations
        self.token_feature_counts: Dict[Tuple[str, int], int] = defaultdict(int)
        
        # Feature co-occurrence
        self.cooccurrence: Dict[Tuple[int, int], int] = defaultdict(int)
        
        # Discovered syntax states (clusters of features)
        self.syntax_states: Dict[str, Set[int]] = {}
    
    def train_step(
        self,
        hidden_states: 'mx.array',
        token_types: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Train SAE on hidden states.
        
        Args:
            hidden_states: Hidden states from transformer (batch, seq, dim)
            token_types: Optional token type names for association learning
        
        Returns:
            Training metrics
        """
        if not HAS_MLX:
            return {'error': 'MLX not available'}
        
        # Flatten batch and sequence dimensions
        batch_size, seq_len, dim = hidden_states.shape
        x = hidden_states.reshape(-1, dim)
        
        # Forward pass
        loss, metrics = self.sae.compute_loss(x)
        
        # Get active features for association tracking
        _, acts, indices = self.sae(x)
        
        # Track feature activations
        for b in range(x.shape[0]):
            active_indices = [int(indices[b, i].item()) for i in range(self.config.k_active)]
            
            # Update activation counts
            for idx in active_indices:
                self.features[idx].activation_count += 1
                self.features[idx].activation_mean = (
                    self.features[idx].activation_mean * 0.99 + 
                    float(acts[b, idx]) * 0.01
                )
            
            # Track co-occurrence
            for i, idx1 in enumerate(active_indices):
                for idx2 in active_indices[i+1:]:
                    key = (min(idx1, idx2), max(idx1, idx2))
                    self.cooccurrence[key] += 1
            
            # Track token associations
            if token_types:
                token_idx = b % len(token_types)
                token = token_types[token_idx]
                for idx in active_indices:
                    self.token_feature_counts[(token, idx)] += 1
                    self.features[idx].associated_tokens.add(token)
        
        return metrics
    
    def discover_states(
        self,
        min_activation_count: int = 100,
        cooccurrence_threshold: float = 0.5,
    ) -> Dict[str, Set[int]]:
        """
        Discover syntax states from feature patterns.
        
        Uses co-occurrence clustering to find groups of features
        that represent syntax states.
        
        Args:
            min_activation_count: Minimum activations for a feature to be considered
            cooccurrence_threshold: Minimum co-occurrence ratio for clustering
        
        Returns:
            Dict mapping state names to feature sets
        """
        # Filter to active features
        active_features = [
            f for f in self.features.values()
            if f.activation_count >= min_activation_count
        ]
        
        if not active_features:
            return {}
        
        # Build co-occurrence matrix
        feature_ids = [f.feature_id for f in active_features]
        id_to_idx = {fid: i for i, fid in enumerate(feature_ids)}
        n = len(feature_ids)
        
        # Simple clustering: group features with high co-occurrence
        clusters: List[Set[int]] = []
        used = set()
        
        for f in active_features:
            if f.feature_id in used:
                continue
            
            # Start new cluster
            cluster = {f.feature_id}
            used.add(f.feature_id)
            
            # Add co-occurring features
            for other in active_features:
                if other.feature_id in used:
                    continue
                
                key = (min(f.feature_id, other.feature_id), 
                       max(f.feature_id, other.feature_id))
                cooccur = self.cooccurrence.get(key, 0)
                
                # Check co-occurrence ratio
                min_count = min(f.activation_count, other.activation_count)
                if min_count > 0 and cooccur / min_count >= cooccurrence_threshold:
                    cluster.add(other.feature_id)
                    used.add(other.feature_id)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        # Name clusters based on associated tokens
        syntax_states = {}
        for i, cluster in enumerate(clusters):
            # Get most common tokens
            token_counts: Dict[str, int] = defaultdict(int)
            for fid in cluster:
                for token in self.features[fid].associated_tokens:
                    token_counts[token] += 1
            
            if token_counts:
                top_token = max(token_counts.items(), key=lambda x: x[1])[0]
                state_name = f"state_{top_token}_{i}"
            else:
                state_name = f"state_{i}"
            
            syntax_states[state_name] = cluster
            
            # Update feature references
            for fid in cluster:
                self.features[fid].co_occurring_features.update(cluster)
        
        self.syntax_states = syntax_states
        return syntax_states
    
    def map_to_production_rules(
        self,
        token_rule_mapping: Dict[str, GoProductionRule],
    ) -> Dict[int, Set[GoProductionRule]]:
        """
        Map features to Go production rules.
        
        Args:
            token_rule_mapping: Map from token types to production rules
        
        Returns:
            Dict mapping feature IDs to associated rules
        """
        feature_rules: Dict[int, Set[GoProductionRule]] = defaultdict(set)
        
        for fid, feature in self.features.items():
            for token in feature.associated_tokens:
                if token in token_rule_mapping:
                    rule = token_rule_mapping[token]
                    feature_rules[fid].add(rule)
                    feature.associated_rules.add(rule)
        
        return dict(feature_rules)
    
    def get_state_for_features(
        self,
        active_features: List[int],
    ) -> Optional[str]:
        """
        Get syntax state for a set of active features.
        
        Args:
            active_features: List of active feature IDs
        
        Returns:
            State name or None
        """
        active_set = set(active_features)
        
        best_match = None
        best_overlap = 0
        
        for state_name, state_features in self.syntax_states.items():
            overlap = len(active_set & state_features)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = state_name
        
        return best_match
    
    def encode_with_state(
        self,
        hidden_states: 'mx.array',
    ) -> List[Tuple[List[int], Optional[str]]]:
        """
        Encode hidden states and return active features with syntax state.
        
        Args:
            hidden_states: Hidden states (batch, dim)
        
        Returns:
            List of (active_features, state_name) tuples
        """
        if not HAS_MLX:
            return []
        
        results = []
        _, acts, indices = self.sae(hidden_states)
        
        for b in range(hidden_states.shape[0]):
            active = [int(indices[b, i].item()) for i in range(self.config.k_active)]
            state = self.get_state_for_features(active)
            results.append((active, state))
        
        return results
    
    def describe_features(self, top_n: int = 20) -> str:
        """Generate description of top features."""
        lines = ["=== Discovered Syntax Features ===\n"]
        
        # Sort by activation count
        sorted_features = sorted(
            self.features.values(),
            key=lambda f: -f.activation_count,
        )[:top_n]
        
        for f in sorted_features:
            lines.append(f.describe())
        
        lines.append(f"\n=== Discovered Syntax States ===")
        for state_name, features in self.syntax_states.items():
            lines.append(f"\n{state_name}: {len(features)} features")
            for fid in list(features)[:5]:
                tokens = list(self.features[fid].associated_tokens)[:3]
                lines.append(f"  - Feature {fid}: tokens={tokens}")
        
        return '\n'.join(lines)


def demo():
    """Demonstrate SAE syntax discovery."""
    print("=" * 60)
    print("SAE SYNTAX DISCOVERY DEMO")
    print("=" * 60)
    
    if not HAS_MLX:
        print("\nMLX not available. Running in simulation mode.")
        
        config = SAEConfig(
            input_dim=64,
            expansion_factor=8,
            k_active=8,
        )
        
        sae = SyntaxSAE(config)
        
        # Simulate feature associations
        token_types = ['func', 'IDENT', '(', 'IDENT', ')', '{', 'return', 'IDENT', '}']
        
        print(f"\nSimulating with tokens: {token_types}")
        
        for i, token in enumerate(token_types):
            # Simulate feature activations
            active_features = list(range(i * 2, i * 2 + 8))
            for fid in active_features:
                if fid < len(sae.features):
                    sae.features[fid].activation_count += 1
                    sae.features[fid].associated_tokens.add(token)
        
        # Discover states
        states = sae.discover_states(min_activation_count=0)
        print(f"\nDiscovered {len(states)} states")
        
        print("\n" + sae.describe_features(10))
        return
    
    # Create SAE
    config = SAEConfig(
        input_dim=64,
        expansion_factor=8,
        k_active=8,
    )
    
    sae = SyntaxSAE(config)
    print(f"\nCreated SAE with {config.input_dim * config.expansion_factor} features")
    
    # Generate synthetic hidden states
    token_types = [
        'package', 'IDENT', 'import', 'STRING',
        'func', 'IDENT', '(', ')', '{',
        'if', 'IDENT', '>', 'INT', '{',
        'return', 'IDENT',
        '}', '}',
    ]
    
    print(f"\nTraining on {len(token_types)} token positions...")
    
    for epoch in range(10):
        # Generate random hidden states
        hidden = mx.random.normal((1, len(token_types), config.input_dim))
        
        metrics = sae.train_step(hidden, token_types)
        
        if epoch % 3 == 0:
            print(f"Epoch {epoch}: recon={metrics.get('reconstruction_loss', 0):.4f}")
    
    # Discover states
    print("\nDiscovering syntax states...")
    states = sae.discover_states(min_activation_count=1)
    print(f"Found {len(states)} states")
    
    # Map to production rules
    token_rule_map = {
        'func': GoProductionRule.FUNC_DECL,
        'if': GoProductionRule.IF_STMT,
        'return': GoProductionRule.RETURN_STMT,
        '{': GoProductionRule.BLOCK,
        '}': GoProductionRule.BLOCK,
    }
    
    feature_rules = sae.map_to_production_rules(token_rule_map)
    print(f"\nMapped {len(feature_rules)} features to production rules")
    
    # Describe features
    print("\n" + sae.describe_features(10))


if __name__ == "__main__":
    demo()
