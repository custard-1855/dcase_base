#!/bin/bash

# 実験設定
CATEGORY="ablation"
METHOD="mixstyle_150"
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

################################################################################
# 1: CNNあり
################################################################################
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant CNN \
    --base_dir ${BASE_DIR} \
    --mixstyle_type "freqAttn" \
    --attn_type "default" \
    2>&1 | tee ${LOG_DIR}/1_${TIMESTAMP}.log

################################################################################
# 2: CNNなし, MixStyleのみ
################################################################################
uv run train_pretrained.py \
    --use_wandb \
    --category ${CATEGORY} \
    --method ${METHOD} \
    --variant nomal \
    --base_dir ${BASE_DIR} \
    --mixstyle_type "only_mix" \
    --attn_type "default" \
    2>&1 | tee ${LOG_DIR}/2_${TIMESTAMP}.log

