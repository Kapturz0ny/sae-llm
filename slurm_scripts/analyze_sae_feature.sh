#!/bin/bash

set -euo pipefail

module purge
module load python/3.12.13 uv/0.11.2 cuda/13.2

source $SCRATCH/venvs/sae-llm-venv/bin/activate

REPO_DIR="$SCRATCH/repos/sae-llm"
export PYTHONPATH="$REPO_DIR"

MODEL_PATH="$SCRATCH/models/Qwen3.5-9B-Base"
LAYER_IDX=31
MODEL_NAME=$(basename "$MODEL_PATH")

ANALYSIS_FILE="$REPO_DIR/features_analysis/analysis/Qwen3.5-9B-Base/sycophancy/f_analysis_Qwen3.5-9B-Base_sycophancy_L$LAYER_IDX.pt"
VECTOR_FILE="$SCRATCH/models/sv/sv_Qwen3.5-9B-Base_sycophancy_idx_7548_L$LAYER_IDX.pt"

OUTPUT_DIR="$REPO_DIR/features_analysis/reports/$MODEL_NAME/L$LAYER_IDX"

cd "$REPO_DIR"

echo "Starting XAI Analysis..."

python experiments/train_sae/analyze_feature_xai.py \
    --model_path "$MODEL_PATH" \
    --analysis_file "$ANALYSIS_FILE" \
    --vector_file "$VECTOR_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --layer_idx $LAYER_IDX

echo "Job completed successfully!"