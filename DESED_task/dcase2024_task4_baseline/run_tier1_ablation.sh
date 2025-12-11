#!/bin/bash

# 実験設定
CATEGORY="ablation"
METHOD="mixstyle_tier1_1"
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

#SEEDS=(42 123 456)

# ログディレクトリ
LOG_DIR="logs/${CATEGORY}_${METHOD}"
mkdir -p ${LOG_DIR}

# タイムスタンプ
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=========================================="
echo "Tier 1: MixStyle Ablation Study"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# ################################################################################
# # B0: MixStyleのみ (ベースライン)
# ################################################################################
# echo "[B0] Running: MixStyle Only (Baseline)"
# uv run train_pretrained.py \
#     --use_wandb \
#     --category ${CATEGORY} \
#     --method ${METHOD} \
#     --variant baseline \
#     --base_dir ${BASE_DIR} \
#     --mixstyle_type "resMix" \
#     2>&1 | tee ${LOG_DIR}/B0_${TIMESTAMP}.log

################################################################################
# P1-1: Freq Attn (現在の実装) - linear, mixed, CNN
################################################################################
echo "[P1-1] Running: Freq Attn (Current) - linear, mixed, CNN"
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant CNN \
    --base_dir ${BASE_DIR} \
    --mixstyle_type "freqAttn" \
    --attn_type "default" \
    --attn_deepen 2 \
    2>&1 | tee ${LOG_DIR}/P1-1_${TIMESTAMP}.log


################################################################################
# P2-1: Freq Transformer - linear, mixed, Transformer (RQ4: CNN vs Transformer)
################################################################################
echo "[P2-1] Running: Freq Transformer - linear, mixed, Transformer (L=1, H=4)"
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant transformer \
    --base_dir ${BASE_DIR} \
    --mixstyle_type "freqTransformer" \
    --n_layers 1 \
    --n_heads 4 \
    --ff_dim 256 \
    --mixstyle_dropout 0.1 \
    2>&1 | tee ${LOG_DIR}/P2-1_${TIMESTAMP}.log

################################################################################
# 実験完了
################################################################################
echo "=========================================="
echo "All Tier 1 experiments completed!"
echo "End Time: $(date)"
echo "Total runs: 21 (7 experiments × 3 seeds)"
echo "Logs saved to: ${LOG_DIR}"
echo "=========================================="
