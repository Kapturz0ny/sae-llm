#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO_DIR"

MODEL_PATH="$REPO_DIR/models/Qwen3.5-0.8B"
RESULTS_DIR="$REPO_DIR/experiments/scenario1/results"
TASK="sycophancy"

cd "$REPO_DIR"

echo "Starting XAI Analysis for $TASK..."
python3 experiments/train_sae/analyze_feature_xai.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$RESULTS_DIR" \
    --task "$TASK"