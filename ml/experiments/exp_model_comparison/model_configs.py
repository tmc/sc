"""
Model configurations for Qwen-Coder scaling comparison.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for a model variant."""
    name: str
    size: str  # "0.5B", "1.5B", "3B", "7B"
    hf_name: str  # HuggingFace model name
    mlx_name: str  # MLX Community quantized name
    params_billions: float
    context_length: int = 32768
    quantization: str = "4bit"
    memory_gb: float = 0.0  # Estimated memory usage

    @property
    def display_name(self) -> str:
        return f"Qwen2.5-Coder-{self.size}"


# Available Qwen-Coder models
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "0.5B": ModelConfig(
        name="qwen-0.5b",
        size="0.5B",
        hf_name="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        mlx_name="mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
        params_billions=0.5,
        memory_gb=0.5,
    ),
    "1.5B": ModelConfig(
        name="qwen-1.5b",
        size="1.5B",
        hf_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        mlx_name="mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
        params_billions=1.5,
        memory_gb=1.2,
    ),
    "3B": ModelConfig(
        name="qwen-3b",
        size="3B",
        hf_name="Qwen/Qwen2.5-Coder-3B-Instruct",
        mlx_name="mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
        params_billions=3.0,
        memory_gb=2.5,
    ),
    "7B": ModelConfig(
        name="qwen-7b",
        size="7B",
        hf_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        mlx_name="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        params_billions=7.0,
        memory_gb=5.0,
    ),
}


def get_model_config(size: str) -> ModelConfig:
    """Get config for a model size."""
    size = size.upper().replace("B", "") + "B"
    if size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {size}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[size]


def list_available_models() -> List[str]:
    """List available model sizes."""
    return list(MODEL_CONFIGS.keys())


# Test prompts for SC generation
SC_TEST_PROMPTS = [
    # Simple toggle
    {
        "name": "simple_toggle",
        "prompt": "Generate a statechart JSON for a simple on/off toggle switch with states Off and On, transitions TURN_ON and TURN_OFF.",
        "complexity": "simple",
        "expected_states": 2,
        "expected_transitions": 2,
    },
    # Traffic light
    {
        "name": "traffic_light",
        "prompt": "Generate a statechart JSON for a traffic light with states Red, Yellow, Green and cyclic transitions NEXT.",
        "complexity": "simple",
        "expected_states": 3,
        "expected_transitions": 3,
    },
    # Hierarchical
    {
        "name": "hierarchical_player",
        "prompt": "Generate a statechart JSON for a media player with a Playing state that contains substates Normal and FastForward.",
        "complexity": "hierarchical",
        "expected_states": 4,  # root, Playing, Normal, FastForward + Stopped
        "expected_transitions": 4,
    },
    # Parallel regions
    {
        "name": "parallel_keyboard",
        "prompt": "Generate a statechart JSON for a keyboard with parallel regions for CapsLock and NumLock, each with On/Off states.",
        "complexity": "parallel",
        "expected_states": 5,  # root + 2 regions * 2 states
        "expected_transitions": 4,
    },
    # Guards
    {
        "name": "guarded_counter",
        "prompt": "Generate a statechart JSON for a counter with states Counting and Done, transition INCREMENT with guard 'count < max'.",
        "complexity": "guarded",
        "expected_states": 2,
        "expected_transitions": 2,
    },
    # History state
    {
        "name": "history_editor",
        "prompt": "Generate a statechart JSON for a text editor with states Editing (with substates Insert/Overwrite) and Paused, using history to return to last editing mode.",
        "complexity": "history",
        "expected_states": 4,
        "expected_transitions": 3,
    },
    # Complex
    {
        "name": "complex_order",
        "prompt": "Generate a statechart JSON for an order processing system with states: Pending, Confirmed, Processing (with substates Picking, Packing, Shipping), Delivered, Cancelled.",
        "complexity": "complex",
        "expected_states": 7,
        "expected_transitions": 8,
    },
    # Event with data
    {
        "name": "event_data",
        "prompt": "Generate a statechart JSON for a login form with states Idle, Validating, Success, Error. The SUBMIT event should carry username and password data.",
        "complexity": "event_data",
        "expected_states": 4,
        "expected_transitions": 5,
    },
]


# Complexity weights for scoring
COMPLEXITY_WEIGHTS = {
    "simple": 1.0,
    "hierarchical": 1.5,
    "parallel": 2.0,
    "guarded": 1.3,
    "history": 1.8,
    "complex": 2.5,
    "event_data": 1.4,
}
