#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO_DIR"

MODEL_PATH="$REPO_DIR/models/Qwen3.5-0.8B"
SAE_PATH="$REPO_DIR/saved_models/sae_qwen_l6_exp32/sae_epoch_5.pt"
DATA_PATH="$REPO_DIR/datasets/sycophancy/sycophancy_search.json"
RESULTS_DIR="$REPO_DIR/experiments/scenario1/results"

cd "$REPO_DIR"

echo "Finding Sycophancy Feature in SAE..."
python3 experiments/train_sae/find_feature.py \
    --model_path "$MODEL_PATH" \
    --sae_path "$SAE_PATH" \
    --data_path "$DATA_PATH" \
    --task "sycophancy" \
    --output_dir "$RESULTS_DIR"