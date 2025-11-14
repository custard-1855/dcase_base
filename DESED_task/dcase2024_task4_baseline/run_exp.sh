#!/bin/bash

# 実験設定
MIXSTYLE_TYPE="resMix"
BASE_WANDB_DIR="sat/150epoch/"
ATTN_TYPE="default"  # 基本的なattentionタイプを使用
ATTN_DEEPEN=2        # デフォルトの深さ

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
echo "[1/4] Running: sat"
uv run train_pretrained.py \
    --wandb_dir ${BASE_WANDB_DIR}/sat \
    --sat \
    2>&1 | tee ${LOG_DIR}/normal${TIMESTAMP}.log

echo ""

################################################################################
# 実験3: sat + MixStyle
################################################################################
echo "[2/4] Running: sat + MixStyle"
uv run train_pretrained.py \
    --attn_type ${ATTN_TYPE} \
    --attn_deepen ${ATTN_DEEPEN} \
    --mixstyle_type ${MIXSTYLE_TYPE} \
    --sat \
    --wandb_dir ${BASE_WANDB_DIR}/MixStyle_sat \
    2>&1 | tee ${LOG_DIR}/mixstyle_sat_${TIMESTAMP}.log

echo ""

################################################################################
# 実験2: sat + cSEBBs
################################################################################
echo "[3/4] Running: sat + cSEBBs"
uv run train_pretrained.py \
    --sebbs \
    --sat \
    --wandb_dir ${BASE_WANDB_DIR}/sat_csebbs \
    2>&1 | tee ${LOG_DIR}/sat_csebbs_${TIMESTAMP}.log

echo ""


################################################################################
# 実験3: sat + MixStyle + cSEBBs
################################################################################
echo "[4/4] Running: sat + MixStyle + cSEBBs"
uv run train_pretrained.py \
    --attn_type ${ATTN_TYPE} \
    --attn_deepen ${ATTN_DEEPEN} \
    --mixstyle_type ${MIXSTYLE_TYPE} \
    --sebbs \
    --sat \
    --wandb_dir ${BASE_WANDB_DIR}/sat_MixStyle_csebbs \
    2>&1 | tee ${LOG_DIR}/sat_mixstyle_csebbs_${TIMESTAMP}.log

echo ""

