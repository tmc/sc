"""
exp_sae_statechart: Sparse Autoencoders as Interpretable State Bottleneck

Integrates SAEs with differentiable statecharts for:
1. Monosemantic state representations
2. Automatic state discovery from activations
3. Hierarchy extraction from feature dependencies
4. Interpretable history/restore mechanisms

Based on Anthropic's monosemanticity work (2023-2025) showing that
SAE features naturally form FSA-like circuits.
"""

from .sae_state_module import (
    SAEConfig,
    TopKSAE,
    SAEStatechartModule,
    SAEStatechartExtractor,
    DiscoveredState,
)
