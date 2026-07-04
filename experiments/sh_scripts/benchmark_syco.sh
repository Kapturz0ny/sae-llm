#!/bin/bash

cd "$(dirname "$0")/../scenario1"

MODEL_PATH="../../models/Qwen3.5-0.8B"
DATA_PATH="../../datasets/sycophancy/sycophancy_benchmark.json"
RESULTS_DIR="./results"

echo "Starting Sycophancy Benchmark..."

python benchmark_syco.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --results_dir "$RESULTS_DIR"

echo "Benchmark finished. Results saved in $RESULTS_DIR."