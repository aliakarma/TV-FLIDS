#!/usr/bin/env bash
# scripts/download_nslkdd.sh
# Downloads NSL-KDD dataset files to data/raw/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TRAIN_URL="https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
TEST_URL="https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"
RAW_DIR="data/raw"

mkdir -p "$RAW_DIR"

if command -v curl >/dev/null 2>&1; then
    DL_BIN="curl -L --fail"
elif command -v wget >/dev/null 2>&1; then
    DL_BIN="wget -q -O"
else
    echo "Error: neither curl nor wget is available in PATH."
    exit 1
fi

echo "[1/2] Downloading KDDTrain+.txt ..."
if [ -f "$RAW_DIR/KDDTrain+.txt" ]; then
    echo "  -> Already exists, skipping."
else
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail "$TRAIN_URL" -o "$RAW_DIR/KDDTrain+.txt"
    else
        wget -q -O "$RAW_DIR/KDDTrain+.txt" "$TRAIN_URL"
    fi
    echo "  -> Saved to $RAW_DIR/KDDTrain+.txt"
fi

echo "[2/2] Downloading KDDTest+.txt ..."
if [ -f "$RAW_DIR/KDDTest+.txt" ]; then
    echo "  -> Already exists, skipping."
else
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail "$TEST_URL" -o "$RAW_DIR/KDDTest+.txt"
    else
        wget -q -O "$RAW_DIR/KDDTest+.txt" "$TEST_URL"
    fi
    echo "  -> Saved to $RAW_DIR/KDDTest+.txt"
fi

echo ""
if command -v wc >/dev/null 2>&1; then
    echo "Download complete."
    echo "Train: $(wc -l < "$RAW_DIR/KDDTrain+.txt") rows"
    echo "Test:  $(wc -l < "$RAW_DIR/KDDTest+.txt") rows"
else
    echo "Download complete."
fi
