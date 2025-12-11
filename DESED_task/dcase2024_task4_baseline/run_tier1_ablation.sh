#!/bin/bash

################################################################################
# Tier 1: MixStyle Ablation Study - 必須の比較実験
#
# 本スクリプトは、ablation_study_design.mdのTier 1実験を実行します。
# 全7実験 × 3シード = 21実行
#
# Experiments:
#   B0: MixStyleのみ (ベースライン)
#   P1-1: Freq Attn (現在の実装) - linear, mixed, CNN
#   P1-2: Freq Attn - residual, mixed, CNN (RQ2: Blend type効果)
#   P1-3: Freq Attn - linear, content, CNN (RQ3: Attn input効果)
#   P1-4: Freq Attn - linear, dual_stream, CNN (RQ3: Attn input効果)
#   P2-1: Freq Transformer - linear, mixed, Transformer (RQ4: CNN vs Transformer)
#   P2-2: Cross Attention - linear, -, Cross-Attn (RQ4: CNN vs Transformer)
################################################################################

# 実験設定
CATEGORY="ablation"
METHOD="mixstyle_tier1_150"
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

################################################################################
# B0: MixStyleのみ (ベースライン)
################################################################################
echo "[B0] Running: MixStyle Only (Baseline)"
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant baseline \
    --base_dir ${BASE_DIR} \
    --mixstyle_type "resMix" \
    --attn_type "default" \
    2>&1 | tee ${LOG_DIR}/B0_seed${seed}_${TIMESTAMP}.log

################################################################################
# P1-1: Freq Attn (現在の実装) - linear, mixed, CNN
################################################################################
echo "[P1-1] Running: Freq Attn (Current) - linear, mixed, CNN"
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant linear_mixed_CNN \
    --base_dir ${BASE_DIR} \
    --mixstyle_type "freqAttn" \
    --attn_type "default" \
    --attn_deepen 2 \
    --blend_type "linear" \
    --attn_input "mixed" \
    2>&1 | tee ${LOG_DIR}/P1-1_seed${seed}_${TIMESTAMP}.log

################################################################################
# P1-2: Freq Attn - residual, mixed, CNN (RQ2: Blend type効果)
################################################################################
echo "[P1-2] Running: Freq Attn - residual, mixed, CNN"
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant residual_mixed_CNN \
    --base_dir ${BASE_DIR} \
    --mixstyle_type "freqAttn" \
    --attn_type "default" \
    --attn_deepen 2 \
    --blend_type "residual" \
    --attn_input "mixed" \
    2>&1 | tee ${LOG_DIR}/P1-2_seed${seed}_${TIMESTAMP}.log

################################################################################
# P1-3: Freq Attn - linear, content, CNN (RQ3: Attn input効果)
################################################################################
echo "[P1-3] Running: Freq Attn - linear, content, CNN"
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant linear_content_CNN \
    --base_dir ${BASE_DIR} \
    --mixstyle_type "freqAttn" \
    --attn_type "default" \
    --attn_deepen 2 \
    --blend_type "linear" \
    --attn_input "content" \
    2>&1 | tee ${LOG_DIR}/P1-3_seed${seed}_${TIMESTAMP}.log

################################################################################
# P1-4: Freq Attn - linear, dual_stream, CNN (RQ3: Attn input効果)
################################################################################
echo "[P1-4] Running: Freq Attn - linear, dual_stream, CNN"
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant linear_dual_stream_CNN \
    --base_dir ${BASE_DIR} \
    --mixstyle_type "freqAttn" \
    --attn_type "default" \
    --attn_deepen 2 \
    --blend_type "linear" \
    --attn_input "dual_stream" \
    2>&1 | tee ${LOG_DIR}/P1-4_seed${seed}_${TIMESTAMP}.log

################################################################################
# P2-1: Freq Transformer - linear, mixed, Transformer (RQ4: CNN vs Transformer)
################################################################################
echo "[P2-1] Running: Freq Transformer - linear, mixed, Transformer (L=1, H=4)"
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant linear_mixed_transformer \
    --base_dir ${BASE_DIR} \
    --mixstyle_type "freqTransformer" \
    --blend_type "linear" \
    --attn_input "mixed" \
    --n_layers 1 \
    --n_heads 4 \
    --ff_dim 256 \
    --mixstyle_dropout 0.1 \
    2>&1 | tee ${LOG_DIR}/P2-1_seed${seed}_${TIMESTAMP}.log

################################################################################
# P2-2: Cross Attention - linear, -, Cross-Attn (RQ4: CNN vs Transformer)
################################################################################
echo "[P2-2] Running: Cross Attention (L=1, H=4)"
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant linear_cross-attn \
    --base_dir ${BASE_DIR} \
    --mixstyle_type "crossAttn" \
    --blend_type "linear" \
    --n_layers 1 \
    --n_heads 4 \
    --ff_dim 256 \
    --mixstyle_dropout 0.1 \
    2>&1 | tee ${LOG_DIR}/P2-2_seed${seed}_${TIMESTAMP}.log

################################################################################
# 実験完了
################################################################################
echo "=========================================="
echo "All Tier 1 experiments completed!"
echo "End Time: $(date)"
echo "Total runs: 21 (7 experiments × 3 seeds)"
echo "Logs saved to: ${LOG_DIR}"
echo "=========================================="
