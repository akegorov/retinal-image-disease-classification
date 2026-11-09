#!/usr/bin/env bash
# Download ODIR-5K from Kaggle into data/raw/.
# Requires a Kaggle API token.
#
# Run from the repository root:
#   bash scripts/download_data.sh

set -euo pipefail

DATASET="andrewmvd/ocular-disease-recognition-odir5k"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/raw"

if [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    echo "data/raw/ is not empty, the dataset seems to be downloaded already. Clear it to download again."
    exit 0
fi

mkdir -p "$OUT"
uv run kaggle datasets download -d "$DATASET" -p "$OUT" --unzip
ls "$OUT"
