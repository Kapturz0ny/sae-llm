#!/bin/bash
#SBATCH --job-name=find_feature
#SBATCH --partition=gpu-m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/find_feature-%j.out

set -euo pipefail

module purge
module load python/3.12.13 uv/0.11.2 cuda/13.2

source $SCRATCH/venvs/sae-llm-venv/bin/activate

REPO_DIR="/scratch/mbagnows/repos/sae-llm"
export PYTHONPATH="$REPO_DIR"

MODEL_PATH="$SCRATCH/models/Qwen3.5-27B"
SAE_PATH="$SCRATCH/models/SAE-Res-Qwen3.5-27B-W80K-L0_50/layer32.sae.pt" 

DATA_PATH="$SCRATCH/datasets/sycophancy/sycophancy_search.json"
ANALYSIS_DIR="$REPO_DIR/experiments/scenario1/analysis_results"
VECTOR_DIR="$SCRATCH/models/sv"

LAYER_IDX=32
TOP_K=50
SAE_TYPE="off_the_shelf"

cd "$REPO_DIR"

echo "Finding Sycophancy Feature in SAE (Type: $SAE_TYPE, Layer: $LAYER_IDX)..."
python3 experiments/train_sae/find_feature.py \
    --model_path "$MODEL_PATH" \
    --sae_path "$SAE_PATH" \
    --sae_type "$SAE_TYPE" \
    --data_path "$DATA_PATH" \
    --task "sycophancy" \
    --layer_idx $LAYER_IDX \
    --top_k $TOP_K \
    --analysis_dir "$ANALYSIS_DIR" \
    --vector_dir "$VECTOR_DIR"

echo "Job completed successfully!"
