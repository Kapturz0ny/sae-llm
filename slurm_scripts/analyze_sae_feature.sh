#!/bin/bash

set -euo pipefail

module purge
module load python/3.12.13 uv/0.11.2 cuda/13.2

source $SCRATCH/venvs/sae-llm-venv/bin/activate

REPO_DIR="$SCRATCH/repos/sae-llm"
export PYTHONPATH="$REPO_DIR"

MODEL_PATH="$SCRATCH/models/Qwen3.5-27B"
MODEL_NAME=$(basename "$MODEL_PATH")
LAYER_IDX=31
TASK="sycophancy"
TOP_K=50

ANALYSIS_FILE="$REPO_DIR/experiments/sae_features/features_analysis/analysis/$MODEL_NAME/$TASK/f_analysis_L${LAYER_IDX}_${TOP_K}.pt"
VECTOR_FILE="$SCRATCH/models/sv/$MODEL_NAME/sv_${TASK}_L${LAYER_IDX}_${TOP_K}.pt"

OUTPUT_DIR="$REPO_DIR/experiments/sae_features/features_analysis/reports/${MODEL_NAME}_${TOP_K}/L$LAYER_IDX"

cd "$REPO_DIR"

echo "Starting XAI Analysis..."

python experiments/train_sae/analyze_feature_xai.py \
    --model_path "$MODEL_PATH" \
    --analysis_file "$ANALYSIS_FILE" \
    --vector_file "$VECTOR_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --layer_idx $LAYER_IDX

echo "Job completed successfully!"