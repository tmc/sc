"""
exp_history_transfer: Transfer History Strategy Across Domains

Tests if learned composite->history_type mappings transfer between domains.
Trains on game domain, tests zero-shot transfer to UI domain.

KEY HYPOTHESIS:
Structural patterns that determine optimal history type are domain-agnostic:
- Deep nesting (>2 levels) → DEEP history
- Shallow nesting (1-2 levels) → SHALLOW history
- No child retention needed → NONE history

APPROACH:
1. Extract domain-agnostic features from composite states
2. Train neural predictor: features → history_type on games
3. Evaluate zero-shot on UI domain
4. Compare to training from scratch

Builds on:
- exp_deep_history: HistoryType, scenarios
- exp_transfer_learning: domain adaptation concepts
"""

from .history_transfer import (
    # Types
    HistoryType,
    CompositeFeatures,
    DomainState,
    DomainStatechart,

    # Game domain
    create_game_pause_statechart,
    create_rpg_menu_statechart,
    create_fighting_game_statechart,

    # UI domain
    create_wizard_statechart,
    create_settings_panel_statechart,
    create_modal_navigation_statechart,
    create_email_client_statechart,

    # Model
    HistoryPredictor,

    # Dataset
    create_game_dataset,
    create_ui_dataset,
    dataset_to_tensors,

    # Training
    train_predictor,
    evaluate_predictor,

    # Experiment
    TransferResult,
    run_transfer_experiment,
    analyze_transfer_patterns,
)

__all__ = [
    # Types
    "HistoryType",
    "CompositeFeatures",
    "DomainState",
    "DomainStatechart",

    # Game domain
    "create_game_pause_statechart",
    "create_rpg_menu_statechart",
    "create_fighting_game_statechart",

    # UI domain
    "create_wizard_statechart",
    "create_settings_panel_statechart",
    "create_modal_navigation_statechart",
    "create_email_client_statechart",

    # Model
    "HistoryPredictor",

    # Dataset
    "create_game_dataset",
    "create_ui_dataset",
    "dataset_to_tensors",

    # Training
    "train_predictor",
    "evaluate_predictor",

    # Experiment
    "TransferResult",
    "run_transfer_experiment",
    "analyze_transfer_patterns",
]
