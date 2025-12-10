#!/bin/bash

# ================================================================================
# CMT Negative Sampling: アブレーション実験スクリプト（新方式）
# ================================================================================
# Usage:
#   ./run_cmt_ablation_threshold_analysis.sh [OPTIONS]
#
# Options:
#   --category CATEGORY      実験カテゴリ (default: ablation)
#   --method METHOD          手法名 (default: cmt_negative_sampling)
#   --base-dir DIR           ベースディレクトリ (default: experiments)
#
# Examples:
#   # デフォルト設定で実行
#   ./run_cmt_ablation_threshold_analysis.sh
#
#   # カスタムカテゴリで実行
#   ./run_cmt_ablation_threshold_analysis.sh --category optimization
# ================================================================================

# デフォルト設定
CATEGORY="ablation"
METHOD="cmt_neg_fix_200"
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

echo "=========================================="
echo "CMT Negative Sampling Experiments (新方式)"
echo "Category: ${CATEGORY}"
echo "Method: ${METHOD}"
echo "Base Directory: ${BASE_DIR}"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# ================================================================================
# アブレーション実験
# ================================================================================
echo "=========================================="
echo "Ablation Study"
echo "=========================================="
echo ""


# 実験1: CMT + negative sampling
echo "[1/6] Running: CMT with negative sampling"
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant neg \
    --base_dir ${BASE_DIR} \
    --cmt \
    --use_neg_sample \
    2>&1 | tee ${LOG_DIR}/neg_${TIMESTAMP}.log


# 実験2: Baseline (CMT無効)
echo "[2/6] Running: Baseline (No CMT)"
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant baseline \
    --base_dir ${BASE_DIR} \
    2>&1 | tee ${LOG_DIR}/baseline_${TIMESTAMP}.log

echo ""
echo "---"
echo ""

# 実験3: CMT (negative sampling無し)
echo "[3/6] Running: CMT without negative sampling"
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant cmt_no_neg \
    --base_dir ${BASE_DIR} \
    --cmt \
    2>&1 | tee ${LOG_DIR}/no_neg_${TIMESTAMP}.log

echo ""
echo "---"
echo ""


# 実験4: CMT + negative sampling + non_zero_scale
echo "[4/6] Running: CMT with negative sampling"
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant neg_non_zero \
    --base_dir ${BASE_DIR} \
    --cmt \
    --use_neg_sample \
    --non_zero_scale \
    2>&1 | tee ${LOG_DIR}/neg_non-zero${TIMESTAMP}.log


# 実験5: CMT + negative sampling + pos_neg_scale
echo "[5/6] Running: CMT with negative sampling"
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant neg_pos_neg \
    --base_dir ${BASE_DIR} \
    --cmt \
    --use_neg_sample \
    --pos_neg_scale \
    2>&1 | tee ${LOG_DIR}/neg_pos-neg${TIMESTAMP}.log


# 実験6: CMT + negative sampling + non_zero_scale + pos_neg_scale
echo "[6/6] Running: CMT with negative sampling + double scale"
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant neg_non-zero_pos-neg \
    --base_dir ${BASE_DIR} \
    --cmt \
    --use_neg_sample \
    --non_zero_scale \
    --pos_neg_scale \
    2>&1 | tee ${LOG_DIR}/neg_non-zero_pos-neg${TIMESTAMP}.log



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
