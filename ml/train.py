
import sys
from mlx_lm import lora, generate, load
from mlx_lm.tuner.utils import apply_lora_layers

# For now, we reuse the mlx-lm library commands via CLI for simplicity
# This script serves as a placeholder/launcher if we want custom loop.
# But `mlx_lm.lora` provides a robust `train` function.

print("Run the following command to train:")
print("python -m mlx_lm.lora --model Qwen/Qwen2.5-0.5B-Instruct --train --data ml/data --iters 600 --batch-size 4 --lora-layers 4")
