#!/bin/bash

################################################################################
# Attention Pattern Experiments
#
# このスクリプトは、異なるattentionパターンで実験を実行します
# 実装されているattention types:
#   - default: 浅い2層CNN
#   - residual_deep: 残差接続で深いネットワーク
#   - multiscale: マルチスケール畳み込み
#   - se_deep: SE-Block組み込み深層ネットワーク
#   - dilated_deep: Dilated Convolution
################################################################################

# 実験設定
MIXSTYLE_TYPE="resMix"
BASE_WANDB_DIR="attention_experiments"

# ログディレクトリの作成
LOG_DIR="logs/attention_experiments"
mkdir -p ${LOG_DIR}

# タイムスタンプ
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=========================================="
echo "Attention Pattern Experiments"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

################################################################################
# 実験1: デフォルトAttention (ベースライン)
################################################################################
echo "[1/9] Running: default attention (depth=2)"
uv run train_pretrained.py \
    --attn_type default \
    --attn_deepen 2 \
    --mixstyle_type ${MIXSTYLE_TYPE} \
    --wandb_dir ${BASE_WANDB_DIR}/default_d2 \
    2>&1 | tee ${LOG_DIR}/default_d2_${TIMESTAMP}.log

echo "Completed: default (depth=2)"
echo ""

################################################################################
# 実験2-3: Residual Deep Attention (異なる深さ)
################################################################################
echo "[2/9] Running: residual_deep attention (depth=2)"
uv run train_pretrained.py \
    --attn_type residual_deep \
    --attn_deepen 2 \
    --mixstyle_type ${MIXSTYLE_TYPE} \
    --wandb_dir ${BASE_WANDB_DIR}/residual_deep_d2 \
    2>&1 | tee ${LOG_DIR}/residual_deep_d2_${TIMESTAMP}.log

echo "Completed: residual_deep (depth=2)"
echo ""

echo "[3/9] Running: residual_deep attention (depth=4)"
uv run train_pretrained.py \
    --attn_type residual_deep \
    --attn_deepen 4 \
    --mixstyle_type ${MIXSTYLE_TYPE} \
    --wandb_dir ${BASE_WANDB_DIR}/residual_deep_d4 \
    2>&1 | tee ${LOG_DIR}/residual_deep_d4_${TIMESTAMP}.log

echo "Completed: residual_deep (depth=4)"
echo ""

################################################################################
# 実験4-5: Multi-Scale Attention (異なる深さ)
################################################################################
echo "[4/9] Running: multiscale attention (depth=2)"
uv run train_pretrained.py \
    --attn_type multiscale \
    --attn_deepen 2 \
    --mixstyle_type ${MIXSTYLE_TYPE} \
    --wandb_dir ${BASE_WANDB_DIR}/multiscale_d2 \
    2>&1 | tee ${LOG_DIR}/multiscale_d2_${TIMESTAMP}.log

echo "Completed: multiscale (depth=2)"
echo ""

echo "[5/9] Running: multiscale attention (depth=3)"
uv run train_pretrained.py \
    --attn_type multiscale \
    --attn_deepen 3 \
    --mixstyle_type ${MIXSTYLE_TYPE} \
    --wandb_dir ${BASE_WANDB_DIR}/multiscale_d3 \
    2>&1 | tee ${LOG_DIR}/multiscale_d3_${TIMESTAMP}.log

echo "Completed: multiscale (depth=3)"
echo ""

################################################################################
# 実験6-7: SE-Deep Attention (異なる深さ)
################################################################################
echo "[6/9] Running: se_deep attention (depth=2)"
uv run train_pretrained.py \
    --attn_type se_deep \
    --attn_deepen 2 \
    --mixstyle_type ${MIXSTYLE_TYPE} \
    --wandb_dir ${BASE_WANDB_DIR}/se_deep_d2 \
    2>&1 | tee ${LOG_DIR}/se_deep_d2_${TIMESTAMP}.log

echo "Completed: se_deep (depth=2)"
echo ""

echo "[7/9] Running: se_deep attention (depth=3)"
uv run train_pretrained.py \
    --attn_type se_deep \
    --attn_deepen 3 \
    --mixstyle_type ${MIXSTYLE_TYPE} \
    --wandb_dir ${BASE_WANDB_DIR}/se_deep_d3 \
    2>&1 | tee ${LOG_DIR}/se_deep_d3_${TIMESTAMP}.log

echo "Completed: se_deep (depth=3)"
echo ""

################################################################################
# 実験8-9: Dilated Deep Attention (異なる深さ)
################################################################################
echo "[8/9] Running: dilated_deep attention (depth=3)"
uv run train_pretrained.py \
    --attn_type dilated_deep \
    --attn_deepen 3 \
    --mixstyle_type ${MIXSTYLE_TYPE} \
    --wandb_dir ${BASE_WANDB_DIR}/dilated_deep_d3 \
    2>&1 | tee ${LOG_DIR}/dilated_deep_d3_${TIMESTAMP}.log

echo "Completed: dilated_deep (depth=3)"
echo ""

echo "[9/9] Running: dilated_deep attention (depth=4)"
uv run train_pretrained.py \
    --attn_type dilated_deep \
    --attn_deepen 4 \
    --mixstyle_type ${MIXSTYLE_TYPE} \
    --wandb_dir ${BASE_WANDB_DIR}/dilated_deep_d4 \
    2>&1 | tee ${LOG_DIR}/dilated_deep_d4_${TIMESTAMP}.log

echo "Completed: dilated_deep (depth=4)"
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
