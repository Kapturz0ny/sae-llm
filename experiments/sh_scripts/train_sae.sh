#!/bin/bash

cd "$(dirname "$0")/../train-sae"

echo "Starting SAE Training..."
python3 main.py --config config.yml