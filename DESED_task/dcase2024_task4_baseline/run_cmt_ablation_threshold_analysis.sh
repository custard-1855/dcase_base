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
    2>&1 | tee ${LOG_DIR}/ablation_baseline_${TIMESTAMP}.log

echo ""
echo "---"
echo ""

# 実験2: CMT (negative sampling無し)
echo "[2/3] Running: CMT without negative sampling"
uv run train_pretrained.py \
    --wandb_dir ${BASE_WANDB_DIR}/ablation/cmt_no_neg \
    --cmt \
    2>&1 | tee ${LOG_DIR}/ablation_cmt_no_neg_${TIMESTAMP}.log

echo ""
echo "---"
echo ""

# 実験3: CMT + negative sampling
echo "[3/3] Running: CMT with negative sampling"
uv run train_pretrained.py \
    --wandb_dir ${BASE_WANDB_DIR}/ablation/cmt_with_neg \
    --cmt \
    --use_neg_sample \
    2>&1 | tee ${LOG_DIR}/ablation_cmt_with_neg_${TIMESTAMP}.log

echo ""
echo "=========================================="
echo "Part 1: Ablation Study Completed"
echo "=========================================="
echo ""


# ================================================================================
# Part 4: 閾値感度分析 - phi_neg & phi_pos の組み合わせ
# ================================================================================
echo "=========================================="
echo "Part 4: Threshold Sensitivity Analysis - phi_neg & phi_pos"
echo "=========================================="
echo ""

# phi_neg と phi_pos の組み合わせ
# 形式: "phi_neg phi_pos"
PHI_NEG_POS_COMBINATIONS=(
    "0.2 0.8"
    "0.3 0.7"
    "0.4 0.6"
    "0.5 0.5"
)

for i in "${!PHI_NEG_POS_COMBINATIONS[@]}"; do
    combination=${PHI_NEG_POS_COMBINATIONS[$i]}
    phi_neg=$(echo $combination | cut -d' ' -f1)
    phi_pos=$(echo $combination | cut -d' ' -f2)

    echo "[$((i+1))/${#PHI_NEG_POS_COMBINATIONS[@]}] Running: phi_neg=${phi_neg}, phi_pos=${phi_pos}"

    uv run train_pretrained.py \
        --wandb_dir ${BASE_WANDB_DIR}/sensitivity/phi_neg_${phi_neg}_pos_${phi_pos} \
        --cmt \
        --use_neg_sample \
        --phi_frame 0.5 \
        --phi_clip 0.5 \
        --phi_neg ${phi_neg} \
        --phi_pos ${phi_pos} \
        2>&1 | tee ${LOG_DIR}/sensitivity_phi_neg_${phi_neg}_pos_${phi_pos}_${TIMESTAMP}.log

    echo ""
    echo "---"
    echo ""
done

echo "=========================================="
echo "Part 4: phi_neg & phi_pos Sensitivity Analysis Completed"
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
