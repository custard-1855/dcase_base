#!/bin/bash

# 実験設定
MIXSTYLE_TYPE="resMix"
BASE_WANDB_DIR="sat/1120_50epoch"

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
uv run train_pretrained.py \
    --wandb_dir ${BASE_WANDB_DIR}/cutmix_clip_fix \
    --sat \
    --strong_augment_type cutmix \
    2>&1 | tee ${LOG_DIR}/normal${TIMESTAMP}.log


# 別の強拡張
uv run train_pretrained.py \
    --wandb_dir ${BASE_WANDB_DIR}/frame_shift_time_mask \
    --sat \
    --strong_augment_type frame_shift_time_mask \
    --strong_augment_prob 1.0 \
    --frame_shift_std 90 \
    --time_mask_max 5 \
    --time_mask_prob 0.5
    2>&1 | tee ${LOG_DIR}/normal${TIMESTAMP}.log

# 強拡張なし
uv run train_pretrained.py \
    --wandb_dir ${BASE_WANDB_DIR}/none_strong \
    --sat \
    --strong_augment_type none \
    2>&1 | tee ${LOG_DIR}/normal${TIMESTAMP}.log



################################################################################
# 実験2: normal
################################################################################
uv run train_pretrained.py \
    --wandb_dir ${BASE_WANDB_DIR}/normal \
    2>&1 | tee ${LOG_DIR}/normal_${TIMESTAMP}.log



