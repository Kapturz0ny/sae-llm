#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO_DIR"

MODEL_PATH="$REPO_DIR/models/Qwen3.5-0.8B"
DATA_PATH="$REPO_DIR/datasets/sycophancy/sycophancy_benchmark.json"
RESULTS_DIR="$REPO_DIR/experiments/scenario1/results"

cd "$REPO_DIR"

echo "Starting Sycophancy Benchmark..."
python3 experiments/scenario1/benchmark_syco.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --results_dir "$RESULTS_DIR"