#!/bin/bash

# ================================================================================
# Ablation Study: Set A (Training Methods) × Set B (Regularization Methods)
# ================================================================================
# Usage:
#   ./run_ablation_a_times_b.sh [OPTIONS]
#
# Options:
#   --category CATEGORY      実験カテゴリ (default: ablation)
#   --method METHOD          手法名 (default: a_times_b)
#   --base-dir DIR           ベースディレクトリ (default: experiments)
#
# Examples:
#   # デフォルト設定で実行
#   ./run_ablation_a_times_b.sh
#
#   # カスタムカテゴリで実行
#   ./run_ablation_a_times_b.sh --category optimization --method my_experiment
# ================================================================================

# デフォルト設定
CATEGORY="ablation"
METHOD="a_times_b"
BASE_DIR="experiments"

# 引数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --category)
            CATEGORY="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ログディレクトリ
LOG_DIR="logs/${CATEGORY}_${METHOD}"
mkdir -p ${LOG_DIR}

# タイムスタンプ
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ================================================================================
# Set A: normal (MeanTeacher) × Set B
# ================================================================================
echo "=========================================="
echo "Set A: normal (MeanTeacher)"
echo "=========================================="
echo ""

# 実験1: normal + MixStyle
echo "[1/9] Running: normal + MixStyle"
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant normal_mixstyle \
    --base_dir ${BASE_DIR} \
    --mixstyle_type "freqAttn" \
    --attn_type "default" \
    2>&1 | tee ${LOG_DIR}/normal_mixstyle_${TIMESTAMP}.log

echo ""
echo "---"
echo ""

# 実験2: normal + SEBBs
echo "[2/9] Running: normal + SEBBs"
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant normal_sebbs \
    --base_dir ${BASE_DIR} \
    --sebbs \
    2>&1 | tee ${LOG_DIR}/normal_sebbs_${TIMESTAMP}.log

echo ""
echo "---"
echo ""

# 実験3: normal + MixStyle + SEBBs
echo "[3/9] Running: normal + MixStyle + SEBBs"
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant normal_mixstyle_sebbs \
    --base_dir ${BASE_DIR} \
    --mixstyle_type "freqAttn" \
    --attn_type "default" \
    --sebbs \
    2>&1 | tee ${LOG_DIR}/normal_mixstyle_sebbs_${TIMESTAMP}.log

echo ""
echo "=========================================="
echo ""

# ================================================================================
# Set A: CMT × Set B
# ================================================================================
echo "=========================================="
echo "Set A: CMT"
echo "=========================================="
echo ""

# 実験4: CMT + MixStyle
echo "[4/9] Running: CMT + MixStyle"
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant cmt_mixstyle \
    --base_dir ${BASE_DIR} \
    --cmt \
    --mixstyle_type "freqAttn" \
    --attn_type "default" \
    2>&1 | tee ${LOG_DIR}/cmt_mixstyle_${TIMESTAMP}.log

echo ""
echo "---"
echo ""

# 実験5: CMT + SEBBs
echo "[5/9] Running: CMT + SEBBs"
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant cmt_sebbs \
    --base_dir ${BASE_DIR} \
    --cmt \
    --sebbs \
    2>&1 | tee ${LOG_DIR}/cmt_sebbs_${TIMESTAMP}.log

echo ""
echo "---"
echo ""

# 実験6: CMT + MixStyle + SEBBs
echo "[6/9] Running: CMT + MixStyle + SEBBs"
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant cmt_mixstyle_sebbs \
    --base_dir ${BASE_DIR} \
    --cmt \
    --mixstyle_type "freqAttn" \
    --attn_type "default" \
    --sebbs \
    2>&1 | tee ${LOG_DIR}/cmt_mixstyle_sebbs_${TIMESTAMP}.log

echo ""
echo "=========================================="
echo ""

# ================================================================================
# Set A: CMT + neg + pos_neg_scale × Set B
# ================================================================================
echo "=========================================="
echo "Set A: CMT + neg + pos_neg_scale"
echo "=========================================="
echo ""

# 実験7: CMT + neg + pos_neg_scale + MixStyle
echo "[7/9] Running: CMT + neg + pos_neg_scale + MixStyle"
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant cmt_neg_scale_mixstyle \
    --base_dir ${BASE_DIR} \
    --cmt \
    --use_neg_sample \
    --pos_neg_scale \
    --mixstyle_type "freqAttn" \
    --attn_type "default" \
    2>&1 | tee ${LOG_DIR}/cmt_neg_scale_mixstyle_${TIMESTAMP}.log

echo ""
echo "---"
echo ""

# 実験8: CMT + neg + pos_neg_scale + SEBBs
echo "[8/9] Running: CMT + neg + pos_neg_scale + SEBBs"
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant cmt_neg_scale_sebbs \
    --base_dir ${BASE_DIR} \
    --cmt \
    --use_neg_sample \
    --pos_neg_scale \
    --sebbs \
    2>&1 | tee ${LOG_DIR}/cmt_neg_scale_sebbs_${TIMESTAMP}.log

echo ""
echo "---"
echo ""

# 実験9: CMT + neg + pos_neg_scale + MixStyle + SEBBs
echo "[9/9] Running: CMT + neg + pos_neg_scale + MixStyle + SEBBs"
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant cmt_neg_scale_mixstyle_sebbs \
    --base_dir ${BASE_DIR} \
    --cmt \
    --use_neg_sample \
    --pos_neg_scale \
    --mixstyle_type "freqAttn" \
    --attn_type "default" \
    --sebbs \
    2>&1 | tee ${LOG_DIR}/cmt_neg_scale_mixstyle_sebbs_${TIMESTAMP}.log

echo ""
echo "=========================================="
echo "All Experiments Completed"
echo "End Time: $(date)"
echo "=========================================="
echo ""
echo "Results are saved in:"
echo "  - Experiments: ${BASE_DIR}/train/${CATEGORY}/${METHOD}/"
echo "  - Logs: ${LOG_DIR}/"
echo ""
