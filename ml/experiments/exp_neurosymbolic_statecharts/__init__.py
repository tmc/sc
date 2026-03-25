"""
Neuro-Symbolic Statecharts for Interactive Reasoning.

Integrates Recursive Language Models (RLMs) with differentiable Harel
statecharts and nested hull geometric memory.

Paper components:
  1. hull_memory.py       - O(log n) convex hull KV-cache
  2. differentiable_sc.py - Gumbel-Softmax topology, neural guards, soft adjacency
  3. recursive_delegation.py - RLM recursive sub-agent spawning
  4. constrained_synthesis.py - Grammar-as-statechart decoding
  5. hybrid_memory.py     - Hull + SSM hybrid memory with learned routing
  6. training.py          - End-to-end training (AdamW, GRPO, SDPO)
  7. benchmark.py         - Empirical validation suite
"""
