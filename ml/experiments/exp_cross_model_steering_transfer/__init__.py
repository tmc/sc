"""
exp_cross_model_steering_transfer: Cross-Model Steering Vector Transfer

Tests whether steering vectors computed on smaller models (Qwen-0.5B)
transfer effectively to larger models (Qwen-1.5B) within the same family.

Key questions:
- Do hierarchy/structure steering directions generalize?
- What dimension adaptation is needed?
- What's the optimal alpha for each target model?

Usage:
    from ml.experiments.exp_cross_model_steering_transfer import (
        VectorTransfer,
        run_transfer_benchmark,
    )

    # Load source vectors and transfer
    transfer = VectorTransfer(source_model="0.5B", target_model="1.5B")
    transferred = transfer.adapt_vector(source_vector)

    # Benchmark effectiveness
    results = run_transfer_benchmark()
"""

from .vector_transfer import (
    VectorTransfer,
    SteeringVector,
    TransferMethod,
    ModelConfig,
)

from .benchmark import (
    run_transfer_benchmark,
    TransferBenchmarkConfig,
    TransferResult,
)

__all__ = [
    'VectorTransfer',
    'SteeringVector',
    'TransferMethod',
    'ModelConfig',
    'run_transfer_benchmark',
    'TransferBenchmarkConfig',
    'TransferResult',
]
