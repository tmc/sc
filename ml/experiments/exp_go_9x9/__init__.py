# Experiment: 9x9 Go with Statecharts
#
# Goal: 100% legal play by construction (0 illegal moves)
#
# Statechart Structure:
# Go9x9 (AND)
# ├── Turn (OR): Black | White
# ├── KoState (OR, HISTORY)
# │   ├── NoKo
# │   └── KoForbidden [point]
# ├── Prisoners (AND)
# │   ├── BlackCaptures: int
# │   └── WhiteCaptures: int
# └── Board[81] (context, not states)
#
# Key insight: Ko rule IS a history state. Suicide prevention IS a guard.
