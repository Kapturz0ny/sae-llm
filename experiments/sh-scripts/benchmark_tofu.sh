#!/bin/bash

cd "$(dirname "$0")/../scenario2"

MODEL_PATH="../../models/Qwen3.5-0.8B"
JUDGE_PATH="../../models/Qwen3.5-0.8B"
DATA_PATH="../../datasets/tofu"
RESULTS_DIR="./results"
NUM_SAMPLES=1000

echo "Starting TOFU Benchmark (Scenario 2)..."

python benchmark_tofu.py \
    --model_path "$MODEL_PATH" \
    --judge_path "$JUDGE_PATH" \
    --data_path "$DATA_PATH" \
    --results_dir "$RESULTS_DIR" \
    --num_samples $NUM_SAMPLES

echo "Benchmark finished. Results saved in $RESULTS_DIR."