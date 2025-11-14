#!/bin/bash

# 実験設定
BASE_WANDB_DIR="cmt"

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

### 全体としては300回す
### 7.5h

################################################################################
# 実験1: 75epoch
################################################################################
echo "[1/3] Running: sat"
uv run train_pretrained.py \
    --wandb_dir ${BASE_WANDB_DIR}/75 \
    --cmt \
    --warmup_epochs 75 \
    2>&1 | tee ${LOG_DIR}/75${TIMESTAMP}.log

echo ""

################################################################################
# 実験3: 100epoch
################################################################################
echo "[2/3] Running: sat + MixStyle"
uv run train_pretrained.py \
    --cmt \
    --warmup_epochs 100 \
    --wandb_dir ${BASE_WANDB_DIR}/100 \
    2>&1 | tee ${LOG_DIR}/100_${TIMESTAMP}.log

echo ""

################################################################################
# 実験3: 125epoch
################################################################################
echo "[3/3] Running: sat + cSEBBs"
uv run train_pretrained.py \
    --cmt \
    --warmup_epochs 125 \
    --wandb_dir ${BASE_WANDB_DIR}/125 \
    2>&1 | tee ${LOG_DIR}/125_${TIMESTAMP}.log

echo ""


