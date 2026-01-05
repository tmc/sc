# Experiment: Transfer Learning for Statecharts
#
# KEY RESEARCH CONTRIBUTION:
# Show that statechart STRUCTURE transfers across games even when rules differ.
#
# Hypothesis: Evolved topology (AND/OR states, hierarchy, history) captures
# game-agnostic patterns that accelerate learning on new games.
#
# Approach:
# 1. Train topology on source game (TicTacToe)
# 2. Extract structure, discard guard parameters
# 3. Adapt board dimensions for target game (Connect4/Othello)
# 4. Fine-tune only guards - topology is frozen
# 5. Measure: steps to 90% accuracy vs training from scratch
#
# Expected Result: Transfer reduces training steps by 2-5x

from .topology_transfer import (
    extract_topology,
    adapt_topology,
    TopologyTransfer,
)

from .guard_finetuning import (
    freeze_topology,
    finetune_guards,
    GuardFinetuner,
)

from .benchmark import (
    run_transfer_benchmark,
    TransferResults,
)
