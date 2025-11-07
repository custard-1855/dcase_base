#!/bin/bash

# 実験設定
BASE_WANDB_DIR="label_acc"
MIXSTYLE_TYPE="resMix"
ATTN_TYPE="default"  # 基本的なattentionタイプを使用
ATTN_DEEPEN=2        # デフォルトの深さ

# ログディレクトリの作成
LOG_DIR="logs/label_acc"
mkdir -p ${LOG_DIR}

# タイムスタンプ
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# cSEBBsは後処理で,後から評価時に確認できるので,一旦は不要

################################################################################
# 実験1: normal
################################################################################
echo "[1/3] Running: normal"
uv run train_pretrained.py \
    --wandb_dir ${BASE_WANDB_DIR}/normal \
    2>&1 | tee ${LOG_DIR}/normal_${TIMESTAMP}.log

echo ""

################################################################################
# 実験2: CMTあり
################################################################################
echo "[2/3] Running: CMT"
uv run train_pretrained.py \
    --cmt \
    --wandb_dir ${BASE_WANDB_DIR}/cmt \
    2>&1 | tee ${LOG_DIR}/cmt_${TIMESTAMP}.log

echo ""

################################################################################
# 実験3: MixStyle
################################################################################
echo "[3/4] Running: MixStyle"
uv run train_pretrained.py \
    --attn_type ${ATTN_TYPE} \
    --attn_deepen ${ATTN_DEEPEN} \
    --mixstyle_type ${MIXSTYLE_TYPE} \
    --wandb_dir ${BASE_WANDB_DIR}/mixstyle_csebbs \
    2>&1 | tee ${LOG_DIR}/mixstyle_${TIMESTAMP}.log

################################################################################
# 実験3: MixStyle + CMT
################################################################################
echo "[4/4] Running: MixStyle + CMT"
uv run train_pretrained.py \
    --attn_type ${ATTN_TYPE} \
    --attn_deepen ${ATTN_DEEPEN} \
    --mixstyle_type ${MIXSTYLE_TYPE} \
    --cmt \
    --wandb_dir ${BASE_WANDB_DIR}/mixstyle_cmt \
    2>&1 | tee ${LOG_DIR}/mixstyle_cmt_${TIMESTAMP}.log

echo "Completed: MixStyle + CMT"
echo ""

################################################################################
# 完了
################################################################################
echo "=========================================="
echo "All Experiments Completed!"
echo "End Time: $(date)"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Logs: ${LOG_DIR}"
echo "  - Wandb: ${BASE_WANDB_DIR}"
echo ""
echo "To view logs:"
echo "  tail -f ${LOG_DIR}/*.log"
echo ""
