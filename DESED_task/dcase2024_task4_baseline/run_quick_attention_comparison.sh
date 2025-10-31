#!/bin/bash

################################################################################
# Quick Attention Comparison
#
# 各attentionパターンを浅い層で素早く比較するスクリプト
# 全ての実験を短時間で実行して、どのパターンが有望かを確認します
################################################################################

# 実験設定
MIXSTYLE_TYPE="resMix"
BASE_WANDB_DIR="quick_attention_comparison"
DEPTH=2  # 全て深さ2で統一

# ログディレクトリの作成
LOG_DIR="logs/quick_attention_comparison"
mkdir -p ${LOG_DIR}

# タイムスタンプ
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=========================================="
echo "Quick Attention Comparison"
echo "Start Time: $(date)"
echo "Depth: ${DEPTH} (shallow for quick comparison)"
echo "=========================================="
echo ""

# 実験するattentionタイプ
ATTENTION_TYPES=("default" "residual_deep" "multiscale" "se_deep" "dilated_deep")

# カウンター
TOTAL=${#ATTENTION_TYPES[@]}
CURRENT=0

# 各attentionタイプで実験
for ATTN_TYPE in "${ATTENTION_TYPES[@]}"
do
    CURRENT=$((CURRENT + 1))
    echo "=========================================="
    echo "[${CURRENT}/${TOTAL}] Running: ${ATTN_TYPE} (depth=${DEPTH})"
    echo "=========================================="

    uv run train_pretrained.py \
        --attn_type ${ATTN_TYPE} \
        --attn_deepen ${DEPTH} \
        --mixstyle_type ${MIXSTYLE_TYPE} \
        --wandb_dir ${BASE_WANDB_DIR}/${ATTN_TYPE}_d${DEPTH} \
        2>&1 | tee ${LOG_DIR}/${ATTN_TYPE}_d${DEPTH}_${TIMESTAMP}.log

    EXIT_CODE=$?

    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "✓ Completed: ${ATTN_TYPE}"
    else
        echo "✗ Failed: ${ATTN_TYPE} (exit code: ${EXIT_CODE})"
    fi

    echo ""
done

################################################################################
# 完了
################################################################################
echo "=========================================="
echo "Quick Comparison Completed!"
echo "End Time: $(date)"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Logs: ${LOG_DIR}"
echo "  - Wandb: ${BASE_WANDB_DIR}"
echo ""
echo "次のステップ:"
echo "1. WandBまたはログで結果を確認"
echo "2. 最も有望なattentionパターンを選択"
echo "3. run_single_attention.sh でそのパターンの深さを変えて実験"
echo ""
echo "例:"
echo "  ./run_single_attention.sh residual_deep 4"
echo ""
