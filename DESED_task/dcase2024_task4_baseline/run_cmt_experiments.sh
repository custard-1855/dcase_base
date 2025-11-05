#!/bin/bash

################################################################################
# CMT改善策の比較実験
#
# このスクリプトは、CMTの異なる改善策の組み合わせで実験を実行します
# 検証する組み合わせ:
#   1. scale のみ (信頼度重みの正規化)
#   2. warmup_epochs のみ (段階的導入: 50エポック後にCMT適用)
#   3. scale + warmup_epochs (両方適用)
################################################################################

# 実験設定
BASE_WANDB_DIR="cmt_experiments"

# ログディレクトリの作成
LOG_DIR="logs/cmt_experiments"
mkdir -p ${LOG_DIR}

# タイムスタンプ
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=========================================="
echo "CMT Improvement Experiments"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

################################################################################
# 実験1: scale のみ (信頼度重みの正規化)
################################################################################
echo "[1/3] Running: CMT with scale normalization only"
uv run train_pretrained.py \
    --cmt \
    --scale \
    --wandb_dir ${BASE_WANDB_DIR}/scale_only \
    2>&1 | tee ${LOG_DIR}/scale_only_${TIMESTAMP}.log

echo "Completed: scale only"
echo ""

################################################################################
# 実験2: warmup_epochs のみ (段階的導入)
################################################################################
echo "[2/3] Running: CMT with warmup epochs only"
uv run train_pretrained.py \
    --cmt \
    --warmup_epochs 50 \
    --wandb_dir ${BASE_WANDB_DIR}/warmup_only \
    2>&1 | tee ${LOG_DIR}/warmup_only_${TIMESTAMP}.log

echo "Completed: warmup_epochs only"
echo ""

################################################################################
# 実験3: scale + warmup_epochs (両方適用)
################################################################################
echo "[3/3] Running: CMT with scale + warmup_epochs"
uv run train_pretrained.py \
    --cmt \
    --scale \
    --warmup_epochs 50 \
    --wandb_dir ${BASE_WANDB_DIR}/scale_warmup \
    2>&1 | tee ${LOG_DIR}/scale_warmup_${TIMESTAMP}.log

echo "Completed: scale + warmup_epochs"
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
