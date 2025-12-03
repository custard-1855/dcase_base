#!/bin/bash

# 単一のcheckpointをテストする簡易スクリプト
# 使用方法: ./run_test_single.sh <checkpoint_path>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <checkpoint_path>"
    echo "Example: $0 ./wandb/new_runs/run-xxxxx/files/checkpoints/best.ckpt"
    exit 1
fi

CKPT_PATH="$1"

if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CKPT_PATH"
    exit 1
fi

echo "=========================================="
echo "Testing checkpoint: $CKPT_PATH"
echo "Time: $(date)"
echo "=========================================="
echo ""

# テスト実行 (引数はcheckpointのパスのみでOK)
uv run train_pretrained.py --test_from_checkpoint "$CKPT_PATH"
