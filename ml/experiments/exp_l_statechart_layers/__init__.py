# Experiment L: Statechart Layers as Neural Network Middleware
#
# Explores inserting differentiable statechart layers between standard layers.
# Tests whether structured state dynamics add value as bottlenecks.
#
# Architectures tested:
# - StatechartSandwich: Linear -> Statechart -> Linear
# - StatechartResidual: x + Statechart(x)
# - StatechartStack: Multiple statechart levels
