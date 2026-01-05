# Experiment: Ko Rule Evolution
#
# KEY RESEARCH CONTRIBUTION:
# Show that statechart HISTORY states emerge automatically from self-play,
# rather than being hand-coded.
#
# Approach:
# 1. Start with Go rules WITHOUT Ko (just liberty rules)
# 2. Self-play leads to infinite capture-recapture loops
# 3. Fitness penalizes games that don't finish
# 4. Evolution discovers history mechanism to prevent loops
# 5. Extract the learned statechart structure
#
# Result: History states ARE LEARNABLE, not hand-coded!
