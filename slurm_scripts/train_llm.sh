#!/bin/bash
#SBATCH --job-name=train_llm
#SBATCH --partition=gpu-l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_llm-%j.out

set -euo pipefail

module purge
module load python/3.12.13 uv/0.11.2 cuda/13.2

source $SCRATCH/venvs/sae-llm-venv/bin/activate

REPO_DIR="$SCRATCH/repos/sae-llm"
export PYTHONPATH="$REPO_DIR"

export TRITON_PTXAS_PATH=$(which ptxas)
export TRITON_CACHE_DIR=$SCRATCH/triton_cache
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export LOGLEVEL=INFO

# WandB Configuration
export WANDB_PROJECT="sae-llm"
export WANDB_ENTITY="mbagnows-warsaw-university-of-technology"

# Paths
MODEL_PATH="$SCRATCH/models/Qwen3.5-9B-Instruct"
DATASET_PATH="$SCRATCH/datasets/sycophancy/sycophancy_train.json"
OUTPUT_DIR="$SCRATCH/models/finetuned/qwen_9b_rslora"

CONFIG_PATH="$REPO_DIR/experiments/train_llm/config.yaml"
CHAT_TEMPLATE_PATH="$REPO_DIR/experiments/train_llm/chat_template.yaml"
DS_CONFIG="$REPO_DIR/experiments/train_llm/ds_config.json"

# Dynamic Run Name (equals the output model folder name)
RUN_NAME=$(basename "$OUTPUT_DIR")

# Dynamic Batching Calculation
TARGET_TOTAL_BATCH_SIZE=32
PER_DEVICE_BATCH_SIZE=4
$TOTAL_GPUS=$(( $SLURM_NNODES * $SLURM_GPUS_ON_NODE ))
GRAD_ACCUM_STEPS=$(( $TARGET_TOTAL_BATCH_SIZE / ($PER_DEVICE_BATCH_SIZE * $TOTAL_GPUS) ))

cd "$REPO_DIR"

nodes_array=( $( scontrol show hostname $SLURM_NODELIST ) )
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

echo "Master Node IP: $head_node_ip"
echo "Total GPUs: $TOTAL_GPUS | Grad Accum Steps: $GRAD_ACCUM_STEPS"
echo "Run Name: $RUN_NAME"

srun -l torchrun \
    --nnodes $SLURM_NNODES \
    --nproc_per_node $SLURM_GPUS_ON_NODE \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:12345 \
    experiments/train_llm/main.py \
    --config "$CONFIG_PATH" \
    --chat_template "$CHAT_TEMPLATE_PATH" \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --deepspeed "$DS_CONFIG" \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY"

echo "Training completed successfully!"
