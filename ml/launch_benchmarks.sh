#!/bin/bash
# launch_benchmarks.sh
# Orchestrates the Sudoku TRM vs SC benchmarks.

set -e
export PYTHONUNBUFFERED=1

# Base directory
ML_DIR="/Volumes/tmc/go/src/github.com/tmc/sc/ml"
cd "$ML_DIR"

echo "========================================================"
echo "Sudoku TRM vs SC Benchmark Orchestrator"
echo "========================================================"
echo "Date: $(date)"
echo "Mode: $1"

if [ "$1" == "quick" ]; then
    echo "Running QUICK verification benchmark..."
    # Quick run: 1 seed, fewer epochs, small dataset
    .venv/bin/python -m experiments.exp_trm_vs_sc_sudoku.benchmark_runner \
        --quick \
        --output "experiments/exp_trm_vs_sc_sudoku/benchmark_results/quick_$(date +%Y%m%d_%H%M%S)"

elif [ "$1" == "full" ]; then
    echo "Running FULL benchmark suite..."
    # Full run: 3 seeds, 30,50,100 epochs
    .venv/bin/python -m experiments.exp_trm_vs_sc_sudoku.benchmark_runner \
        --epochs "30,50,100" \
        --seeds 3 \
        --output "experiments/exp_trm_vs_sc_sudoku/benchmark_results/full_$(date +%Y%m%d_%H%M%S)"

else
    echo "Usage: ./launch_benchmarks.sh [quick|full]"
    echo ""
    echo "  quick: Run a short verification test (1 seed, 30 epochs)"
    echo "  full:  Run the complete benchmark suite (3 seeds, 30/50/100 epochs)"
    exit 1
fi

echo "========================================================"
echo "Benchmark Complete"
echo "========================================================"
