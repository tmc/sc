"""
exp_attention_head_taxonomy: Classify attention heads by function.

CATEGORIES:
- state_tracking: Attends to state labels, maintains state memory
- transition_following: Connects from/to states, tracks transition flow
- hierarchy_aware: Tracks parent-child nesting, compound states
- syntax_checking: Focuses on JSON structure, brackets, quotes

Uses mlux for activation caching and analysis.
"""

from .head_classifier import (
    HeadFunction,
    HeadProfile,
    TaxonomyResult,
    TokenClassifier,
    HeadAnalyzer,
    HeadTaxonomy,
    TAXONOMY_PROMPTS,
    build_taxonomy,
    demo,
)

__all__ = [
    "HeadFunction",
    "HeadProfile",
    "TaxonomyResult",
    "TokenClassifier",
    "HeadAnalyzer",
    "HeadTaxonomy",
    "TAXONOMY_PROMPTS",
    "build_taxonomy",
    "demo",
]
