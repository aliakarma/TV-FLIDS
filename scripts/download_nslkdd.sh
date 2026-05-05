#!/usr/bin/env bash
# scripts/download_nslkdd.sh
# Downloads NSL-KDD dataset files to data/raw/

set -e

TRAIN_URL="https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
TEST_URL="https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"
RAW_DIR="data/raw"

mkdir -p "$RAW_DIR"

echo "[1/2] Downloading KDDTrain+.txt ..."
if [ -f "$RAW_DIR/KDDTrain+.txt" ]; then
    echo "  → Already exists, skipping."
else
    curl -L "$TRAIN_URL" -o "$RAW_DIR/KDDTrain+.txt"
    echo "  → Saved to $RAW_DIR/KDDTrain+.txt"
fi

echo "[2/2] Downloading KDDTest+.txt ..."
if [ -f "$RAW_DIR/KDDTest+.txt" ]; then
    echo "  → Already exists, skipping."
else
    curl -L "$TEST_URL" -o "$RAW_DIR/KDDTest+.txt"
    echo "  → Saved to $RAW_DIR/KDDTest+.txt"
fi

echo ""
echo "Download complete."
echo "Train: $(wc -l < $RAW_DIR/KDDTrain+.txt) rows"
echo "Test:  $(wc -l < $RAW_DIR/KDDTest+.txt) rows"
