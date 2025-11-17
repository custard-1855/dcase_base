#!/bin/bash

# 実験設定
BASE_WANDB_DIR="cmt/labeled/300"

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
# 実験1: unlabeled 0
################################################################################
echo "[1/4] Running: sat"
uv run train_pretrained.py \
    --wandb_dir ${BASE_WANDB_DIR} \
    --cmt \
    2>&1 | tee ${LOG_DIR}${TIMESTAMP}.log

echo ""


################################################################################
# 実験2: unlabeled 50
################################################################################
echo "[2/4] Running: sat"
uv run train_pretrained.py \
    --wandb_dir ${BASE_WANDB_DIR}50 \
    --cmt \
    --warmup_epochs 50 \
    2>&1 | tee ${LOG_DIR}${TIMESTAMP}.log

echo ""



################################################################################
# 実験1: 75epoch
################################################################################
echo "[3/4] Running: sat"
uv run train_pretrained.py \
    --wandb_dir ${BASE_WANDB_DIR}75 \
    --cmt \
    --warmup_epochs 75 \
    2>&1 | tee ${LOG_DIR}/75${TIMESTAMP}.log

echo ""

################################################################################
# 実験3: 100epoch
################################################################################
echo "[4/4] Running: sat + MixStyle"
uv run train_pretrained.py \
    --cmt \
    --warmup_epochs 100 \
    --wandb_dir ${BASE_WANDB_DIR}100 \
    2>&1 | tee ${LOG_DIR}/100_${TIMESTAMP}.log

echo ""

