#!/bin/bash
#SBATCH --job-name=eval_cap
#SBATCH --partition=gpu-m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_cap-%j.out

set -euo pipefail

module purge
module load python/3.12.13 uv/0.11.2 cuda/13.2

source $SCRATCH/venvs/sae-llm-venv/bin/activate

REPO_DIR="$SCRATCH/repos/sae-llm"
export PYTHONPATH="$REPO_DIR"

MODEL_PATH="$SCRATCH/models/Qwen3.5-9B-Base"
RESULTS_DIR="$REPO_DIR/experiments/benchmarks/results_eval"

TASKS="wikitext,hellaswag,arc_challenge"
BATCH_SIZE=16

VECTOR_PATH="$SCRATCH/models/sv/sv_Qwen3.5-9B_sycophancy_idx_1858_L32.pt"
LAYER_IDX=32
MULTIPLIER=-2.0

cd "$REPO_DIR"
echo "Starting Capabilities Evaluation..."

CMD=(
    python experiments/benchmarks/lm_eval.py
    --model_path "$MODEL_PATH"
    --results_dir "$RESULTS_DIR"
    --tasks "$TASKS"
    --batch_size "$BATCH_SIZE"
    --vector_path "$VECTOR_PATH"
    --layer_idx "$LAYER_IDX"
    --multiplier "$MULTIPLIER"
)


echo "Executing: ${CMD[*]}"
"${CMD[@]}"
echo "Job completed successfully!"