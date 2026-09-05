#!/bin/bash


set -euo pipefail

module purge
module load python/3.12.13 uv/0.11.2 cuda/13.2

source $SCRATCH/venvs/sae-llm-venv/bin/activate
export PYTHONHASHSEED=42

REPO_DIR="/scratch/mbagnows/repos/sae-llm"
MODEL_PATH="$SCRATCH/models/Qwen3.5-9B-Base"
MODEL_NAME=$(basename "$MODEL_PATH")

LAYER_IDX=8
TOP_K=50
TASK="sycophancy"

# SAE_PATH="$SCRATCH/models/SAE-Res-Qwen3.5-9B-Base-W64K-L0_$TOP_K/layer$LAYER_IDX.sae.pt" 
DATA_PATH="$SCRATCH/datasets/sycophancy/sycophancy_search.json"
ANALYSIS_DIR="$REPO_DIR/features_analysis/analysis/$MODEL_NAME/$TASK"
VECTOR_DIR="$SCRATCH/models/sv/$MODEL_NAME"

SAE_TYPE="off_the_shelf"

cd "$REPO_DIR"

for LAYER_IDX in {9..31}; do
    echo "=================================================="
    echo " Scanning layer: $LAYER_IDX"
    echo "=================================================="
    
    CURRENT_SAE_PATH="$SCRATCH/models/SAE-Res-Qwen3.5-9B-Base-W64K-L0_$TOP_K/layer${LAYER_IDX}.sae.pt"
    
    python experiments/train_sae/find_feature.py \
        --model_path "$MODEL_PATH" \
        --sae_path "$CURRENT_SAE_PATH" \
        --sae_type "$SAE_TYPE" \
        --data_path "$DATA_PATH" \
        --task "$TASK" \
        --layer_idx $LAYER_IDX \
        --top_k $TOP_K \
        --analysis_dir "$ANALYSIS_DIR" \
        --vector_dir "$VECTOR_DIR" \
        --seed 42
done

echo "Job completed successfully!"
