#!/bin/bash

# ================================================================================
# CMT Negative Sampling: アブレーション実験 & 閾値感度分析
# ================================================================================

# 実験設定
BASE_WANDB_DIR="ablation/cmt_negative_sampling"
LOG_DIR="logs/cmt_ablation"
mkdir -p ${LOG_DIR}

# タイムスタンプ
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=========================================="
echo "CMT Negative Sampling Experiments"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# ================================================================================
# Part 1: アブレーション実験
# ================================================================================
echo "=========================================="
echo "Part 1: Ablation Study"
echo "=========================================="
echo ""

# 実験1: Baseline (CMT無効)
echo "[1/3] Running: Baseline (No CMT)"
uv run train_pretrained.py \
    --wandb_dir ${BASE_WANDB_DIR}/ablation/baseline \
    --use_wandb \
    2>&1 | tee ${LOG_DIR}/ablation_baseline_${TIMESTAMP}.log

echo ""
echo "---"
echo ""

# 実験2: CMT (negative sampling無し)
echo "[2/3] Running: CMT without negative sampling"
uv run train_pretrained.py \
    --wandb_dir ${BASE_WANDB_DIR}/ablation/cmt_no_neg \
    --use_wandb \
    --cmt \
    2>&1 | tee ${LOG_DIR}/ablation_cmt_no_neg_${TIMESTAMP}.log

echo ""
echo "---"
echo ""

# 実験3: CMT + negative sampling
echo "[3/3] Running: CMT with negative sampling"
uv run train_pretrained.py \
    --wandb_dir ${BASE_WANDB_DIR}/ablation/cmt_with_neg \
    --use_wandb \
    --cmt \
    --use_neg_sample \
    2>&1 | tee ${LOG_DIR}/ablation_cmt_with_neg_${TIMESTAMP}.log

echo ""
echo "=========================================="
echo "Part 1: Ablation Study Completed"
echo "=========================================="
echo ""

# ================================================================================
# 実験完了
# ================================================================================
echo "=========================================="
echo "All Experiments Completed"
echo "End Time: $(date)"
echo "=========================================="
echo ""
echo "Results are logged in:"
echo "  - WandB: ${BASE_WANDB_DIR}"
echo "  - Logs: ${LOG_DIR}"
echo ""
