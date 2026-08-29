#!/usr/bin/env bash
# Runs the full evaluation suite against a trained checkpoint.
# Usage: ./scripts/evaluate.sh outputs/baseline/model_best.pt
set -e
cd "$(dirname "$0")/.."

CHECKPOINT="${1:?Usage: ./scripts/evaluate.sh <checkpoint_path>}"

echo "=== Calibrating confidence scores ==="
python src/eval/calibration.py --checkpoint "$CHECKPOINT"

echo "=== Building robustness table ==="
python src/eval/robustness.py --checkpoint "$CHECKPOINT" --out outputs/robustness_table.csv

echo "=== Running error analysis ==="
python src/eval/error_analysis.py --checkpoint "$CHECKPOINT" --out outputs/error_analysis.md

echo "=== Running shortcut-learning sanity check ==="
python src/eval/shortcut_check.py --checkpoint "$CHECKPOINT"

echo "All evaluation artifacts written to outputs/"
