#!/bin/bash

# 実験設定
BASE_WANDB_DIR="cmt/150"
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


################################################################################
# 実験1: MixStyle + SEBBs warm-up 0
################################################################################
echo "[1/4] Running: sat"
uv run train_pretrained.py \
    --wandb_dir ${BASE_WANDB_DIR}/MixStyle_SEBBS \
    --attn_type ${ATTN_TYPE} \
    --attn_deepen ${ATTN_DEEPEN} \
    --mixstyle_type ${MIXSTYLE_TYPE} \
    --sebbs \
    2>&1 | tee ${LOG_DIR}/75${TIMESTAMP}.log

echo ""


################################################################################
# 実験6: sat frame shift + time mask
################################################################################
echo "[1/4] Running: sat"
uv run train_pretrained.py \
    --wandb_dir sat/150/frame-shift-60_time-mask-30 \
    --sat \
    --strong_augment_type frame_shift_time_mask \
    --frame_shift_std 60 \
    --time_mask_max 30 \
    2>&1 | tee ${LOG_DIR}/only_sat${TIMESTAMP}.log

echo ""

