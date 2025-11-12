#!/bin/bash

# 実験設定
MIXSTYLE_TYPE="resMix"
BASE_WANDB_DIR="sebbs/fix_"
ATTN_TYPE="default"  # 基本的なattentionタイプを使用
ATTN_DEEPEN=2        # デフォルトの深さ

# ログディレクトリの作成
LOG_DIR="logs/accuracy_improvement_experiments"
mkdir -p ${LOG_DIR}

# タイムスタンプ
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=========================================="
echo "Accuracy Improvement Experiments"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

################################################################################
# 実験1: Normal
################################################################################
echo "[1/3] Running: normal"
uv run train_pretrained.py \
    --wandb_dir ${BASE_WANDB_DIR}/normal \
    2>&1 | tee ${LOG_DIR}/normal${TIMESTAMP}.log

echo ""


################################################################################
# 実験2: cSEBBs
################################################################################
echo "[2/3] Running: cSEBBs"
uv run train_pretrained.py \
    --sebbs \
    --wandb_dir ${BASE_WANDB_DIR}/csebbs \
    2>&1 | tee ${LOG_DIR}/csebbs_${TIMESTAMP}.log

echo ""


################################################################################
# 実験3: MixStyle + cSEBBs
################################################################################
echo "[3/3] Running: MixStyle + cSEBBs"
uv run train_pretrained.py \
    --attn_type ${ATTN_TYPE} \
    --attn_deepen ${ATTN_DEEPEN} \
    --mixstyle_type ${MIXSTYLE_TYPE} \
    --sebbs \
    --wandb_dir ${BASE_WANDB_DIR}/MixStyle_csebbs \
    2>&1 | tee ${LOG_DIR}/mixstyle_csebbs_${TIMESTAMP}.log

echo ""

