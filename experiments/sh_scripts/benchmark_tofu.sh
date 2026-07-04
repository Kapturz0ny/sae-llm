#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO_DIR"

# use vllm venv

MODEL_PATH="./models/Qwen3.5-0.8B"
JUDGE_PATH="./models/Qwen3.5-0.8B"
DATA_PATH="./datasets/tofu"
RESULTS_DIR="./experiments/scenario2/results"

cd "$REPO_DIR"

echo "Starting TOFU Benchmark (Scenario 2)..."
python3 experiments/scenario2/benchmark_tofu.py \
    --model_path "$MODEL_PATH" \
    --judge_path "$JUDGE_PATH" \
    --data_path "$DATA_PATH" \
    --results_dir "$RESULTS_DIR" \
    --num_samples 1000