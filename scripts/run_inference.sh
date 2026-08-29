#!/usr/bin/env bash
# Usage: ./scripts/run_inference.sh <input_image_dir> <checkpoint_path>
set -e
cd "$(dirname "$0")/.."

INPUT_DIR="${1:?Usage: ./scripts/run_inference.sh <input_image_dir> <checkpoint_path>}"
CHECKPOINT="${2:?Usage: ./scripts/run_inference.sh <input_image_dir> <checkpoint_path>}"

python src/inference.py --input_dir "$INPUT_DIR" --checkpoint "$CHECKPOINT" --out outputs/predictions.json
echo "Predictions written to outputs/predictions.json"
