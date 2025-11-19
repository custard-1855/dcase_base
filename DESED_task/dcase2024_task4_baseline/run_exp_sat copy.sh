#!/bin/bash

# 実験設定
MIXSTYLE_TYPE="resMix"
BASE_WANDB_DIR="sat/150epoch"

# ログディレクトリの作成
LOG_DIR="logs/sat/150epoch"
mkdir -p ${LOG_DIR}

# タイムスタンプ
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=========================================="
echo "Accuracy Improvement Experiments"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

################################################################################
# 実験1: sat
################################################################################
echo "[1/2] Running: sat"
uv run train_pretrained.py \
    --wandb_dir ${BASE_WANDB_DIR}/cutmix_clip_fix \
    --sat \
    2>&1 | tee ${LOG_DIR}/normal${TIMESTAMP}.log

echo ""

echo "[2/2] Running: sat"
uv run train_pretrained.py \
    --wandb_dir ${BASE_WANDB_DIR}/cutmix_beta_param \
    --sat \
    --cutmix_beta 0.5 \
    2>&1 | tee ${LOG_DIR}/normal${TIMESTAMP}.log

echo ""

