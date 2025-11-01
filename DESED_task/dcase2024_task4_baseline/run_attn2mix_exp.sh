#!/bin/bash

################################################################################
# 検証対象:
#   1. MixStyle + CMT
#   2. CMT
#   3. cSEBBs
################################################################################

# 実験設定
MIXSTYLE_TYPE="resMix"
BASE_WANDB_DIR="Solo_acc__improvement_experiments"
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
# 実験1: MixStyle + CMT
################################################################################
echo "[1/3] Running: MixStyle + CMT"
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
# 実験2: CMT
################################################################################
echo "[2/3] Running: CMT"
uv run train_pretrained.py \
    --cmt \
    --wandb_dir ${BASE_WANDB_DIR}/mixstyle_cmt \
    2>&1 | tee ${LOG_DIR}/mixstyle_cmt_${TIMESTAMP}.log

echo "Completed: CMT"
echo ""

################################################################################
# 実験3: cSEBBs
################################################################################
echo "[3/3] Running: cSEBBs"
uv run train_pretrained.py \
    --sebbs \
    --wandb_dir ${BASE_WANDB_DIR}/cmt_csebbs \
    2>&1 | tee ${LOG_DIR}/cmt_csebbs_${TIMESTAMP}.log

echo "Completed: cSEBBs"
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
