# Experiment L: Statechart Layers as Neural Network Middleware

## Overview
Wraps differentiable statechart components as nn.Module layers that can be inserted into standard neural network architectures.

## Components
- `statechart_layer.py`: All layer implementations

## Layer Types

### StatechartLayer
- Base layer: input -> state space -> transitions -> output
- Projects input to soft state configuration
- Computes input-dependent guard values
- Applies transitions and blends configurations

### StatechartSandwich
- Linear -> StatechartLayer -> Linear
- Statechart acts as structured bottleneck

### StatechartResidual
- x + scale * StatechartLayer(x)
- Pre-norm residual connection
- Network can optionally use statechart structure

### StatechartStack
- Multiple statechart layers at different granularities
- Multi-scale state representation (e.g., [2, 4, 8] states)

### StatechartGate
- Uses statechart output to gate another pathway
- gate = sigmoid(proj(StatechartLayer(x)))
- output = gate * other_pathway

## Results
All layers pass gradient flow tests:
```
1. StatechartLayer      - Gradient flow: OK
2. StatechartSandwich   - Gradient flow: OK
3. StatechartResidual   - Gradient flow: OK
4. StatechartStack      - Gradient flow: OK
5. StatechartGate       - Gradient flow: OK
```

## Issues Fixed
- `num_transitions` None issue: Used local variables after assignment to ensure int type

## Reproduction Commands
```bash
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml
.venv/bin/python3 experiments/exp_l_statechart_layers/statechart_layer.py
```

## Usage Example
```python
from experiments.exp_l_statechart_layers.statechart_layer import StatechartResidual

# Add statechart processing as residual
layer = StatechartResidual(dim=64, num_states=4)
output = layer(x)  # x + statechart_processed(x)
```
