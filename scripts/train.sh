#!/usr/bin/env bash
# Usage: ./scripts/train.sh [experiment_name]
set -e
cd "$(dirname "$0")/.."

EXPERIMENT="${1:-}"
if [ -n "$EXPERIMENT" ]; then
  python src/models/train.py --config configs/config.yaml --experiment "$EXPERIMENT"
else
  python src/models/train.py --config configs/config.yaml
fi
