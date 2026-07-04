#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO_DIR"

cd "$REPO_DIR"

echo "Starting SAE Training..."
python3 experiments/train_sae/main.py --config experiments/train_sae/config.yml
