#!/bin/bash
# launch_benchmarks_parallel.sh
# Launches 5 benchmarks in parallel background processes.

set -e

# Base directory
ML_DIR="/Volumes/tmc/go/src/github.com/tmc/sc/ml"
cd "$ML_DIR"

# Ensure venv
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Output directory for logs
LOG_DIR="experiments/exp_trm_vs_sc_sudoku/logs"
mkdir -p "$LOG_DIR"

echo "========================================================"
echo "Launching Parallel Benchmarks"
echo "Date: $(date)"
echo "Logs: $LOG_DIR"
echo "========================================================"

# Function to launch a benchmark
launch() {
    MODEL=$1
    EPOCHS=100
    echo "Launching $MODEL ($EPOCHS epochs)..."
    nohup python3 -m experiments.exp_trm_vs_sc_sudoku.run_long_benchmark \
        --model "$MODEL" \
        --epochs "$EPOCHS" \
        > "$LOG_DIR/${MODEL}.log" 2>&1 &
    echo "PID $!"
}

# Launch the 5 core models
launch "vanilla"
launch "faithful"
launch "attention"
launch "hierarchical"
launch "hybrid"

echo "========================================================"
echo "All benchmarks launched in background."
echo "Monitor progress with: tail -f $LOG_DIR/*.log"
echo "========================================================"
