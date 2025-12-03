#!/bin/bash

# 実験設定
BASE_WANDB_DIR="50/"
MIXSTYLE_TYPE="resMix"
ATTN_TYPE="default"  # 基本的なattentionタイプを使用
ATTN_DEEPEN=2        # デフォルトの深さ

# ログディレクトリの作成
LOG_DIR="logs/cmt"
mkdir -p ${LOG_DIR}

# タイムスタンプ
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=========================================="
echo "Accuracy Improvement Experiments"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

ckpt=()
for f in wandb/run-*; do
    file_date="${f:4:8}"
    if [[ "$file_date" -lt 20251202 ]]; then
        mv "$f" old_logs/
    fi



################################################################################
# 実験4: CMT + SEBBs
################################################################################
uv run train_pretrained.py \
    --wandb_dir ${BASE_WANDB_DIR}/CMT_frame-0.5_warm-up-50 + SEBBs \
    --test_from_checkpoint \
    --cmt \
    --phi_frame 0.5 \
    --warmup_epochs 50 \
    --sebbs \
    2>&1 | tee ${LOG_DIR}/${TIMESTAMP}.log
