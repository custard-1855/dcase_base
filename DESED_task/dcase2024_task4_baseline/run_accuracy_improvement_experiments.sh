#!/bin/bash

################################################################################
# 精度向上手法の影響検証実験
#
# このスクリプトは、異なる精度向上手法の組み合わせで実験を実行します
# 検証する組み合わせ:
#   1. MixStyle + cSEBBs
#   2. MixStyle + CMT
#   3. CMT + cSEBBs
#   4. All (MixStyle + CMT + cSEBBs)
################################################################################

# 実験設定
MIXSTYLE_TYPE="resMix"
BASE_WANDB_DIR="re_acc_improve_exp"
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

# ################################################################################
# # 実験1: MixStyle + cSEBBs
# ################################################################################
# echo "[1/4] Running: MixStyle + cSEBBs"
# uv run train_pretrained.py \
#     --attn_type ${ATTN_TYPE} \
#     --attn_deepen ${ATTN_DEEPEN} \
#     --mixstyle_type ${MIXSTYLE_TYPE} \
#     --sebbs \
#     --wandb_dir ${BASE_WANDB_DIR}/mixstyle_csebbs \
#     2>&1 | tee ${LOG_DIR}/mixstyle_csebbs_${TIMESTAMP}.log

# echo "Completed: MixStyle + cSEBBs"
# echo ""


################################################################################
# 実験4: All (MixStyle + CMT + cSEBBs)
################################################################################
echo "[4/4] Running: All (MixStyle + CMT + cSEBBs)"
uv run train_pretrained.py \
    --attn_type ${ATTN_TYPE} \
    --attn_deepen ${ATTN_DEEPEN} \
    --mixstyle_type ${MIXSTYLE_TYPE} \
    --cmt \
    --warmup_epochs 50 \
    --sebbs \
    --wandb_dir ${BASE_WANDB_DIR}/all_techniques \
    2>&1 | tee ${LOG_DIR}/all_techniques_${TIMESTAMP}.log

echo "Completed: All (MixStyle + CMT + cSEBBs)"
echo ""


################################################################################
# 実験2: MixStyle + CMT
################################################################################
echo "[2/4] Running: MixStyle + CMT"
uv run train_pretrained.py \
    --attn_type ${ATTN_TYPE} \
    --attn_deepen ${ATTN_DEEPEN} \
    --mixstyle_type ${MIXSTYLE_TYPE} \
    --cmt \
    --warmup_epochs 50 \
    --wandb_dir ${BASE_WANDB_DIR}/mixstyle_cmt \
    2>&1 | tee ${LOG_DIR}/mixstyle_cmt_${TIMESTAMP}.log

echo "Completed: MixStyle + CMT"
echo ""

################################################################################
# 実験3: CMT + cSEBBs
################################################################################
echo "[3/4] Running: CMT + cSEBBs"
uv run train_pretrained.py \
    --attn_type ${ATTN_TYPE} \
    --attn_deepen ${ATTN_DEEPEN} \
    --cmt \
    --warmup_epochs 50 \
    --sebbs \
    --wandb_dir ${BASE_WANDB_DIR}/cmt_csebbs \
    2>&1 | tee ${LOG_DIR}/cmt_csebbs_${TIMESTAMP}.log

echo "Completed: CMT + cSEBBs"
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
