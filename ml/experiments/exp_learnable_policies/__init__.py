"""
Learnable Policies: Discovering State Machines from Self-Play

This experiment proves that game rules requiring state (Ko, En Passant, Castling,
Draw by Repetition) can be LEARNED as statecharts rather than hand-coded.

Approach:
1. Start with base game rules (no special rule)
2. Self-play generates training data with rule violations
3. Evolution discovers state structure to prevent violations
4. Extract and visualize learned statechart

Games/Rules:
- Go Ko: History state (single capture tracking)
- Chess En Passant: Ephemeral state (one-move capture window)
- Chess Castling: Persistent flags (piece movement tracking)
- Draw by Repetition: Ring buffer (position history)
"""

from .framework import (
    StateGenome,
    PolicyEvolver,
    GameEnvironment,
    EvolutionConfig,
)
