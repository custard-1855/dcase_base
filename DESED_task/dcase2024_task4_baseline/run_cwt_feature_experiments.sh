#!/bin/bash

################################################################################
# CWT特徴量の影響検証実験
#
# このスクリプトは、異なるCWT特徴量の組み合わせで実験を実行します
# 検証する組み合わせ:
#   1. scalogram (スカログラムのみ)
#   2. scalogram+logmel (スカログラム + LogMel)
#   3. real+imag (実部+虚部)
#   4. real+imag+logmel (実部+虚部 + LogMel)
#
# 各実験は10エポックで実行し、実装を確認します
################################################################################

# 実験設定
BASE_WANDB_DIR="cwt_feature_experiments"
N_EPOCHS=10
ATTN_TYPE="default"
ATTN_DEEPEN=2
MIXSTYLE_TYPE="disabled"

# ログディレクトリの作成
LOG_DIR="logs/cwt_feature_experiments"
mkdir -p ${LOG_DIR}

# タイムスタンプ
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=========================================="
echo "CWT Feature Experiments (10 epochs)"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

################################################################################
# 実験1: scalogram (スカログラムのみ)
################################################################################
echo "[1/4] Running: scalogram only"
echo "  use_imaginary: False"
echo "  combine_with_logmel: False"
uv run train_pretrained.py \
    --attn_type ${ATTN_TYPE} \
    --attn_deepen ${ATTN_DEEPEN} \
    --mixstyle_type ${MIXSTYLE_TYPE} \
    --use_wavelet \
    --wandb_dir ${BASE_WANDB_DIR}/scalogram \
    2>&1 | tee ${LOG_DIR}/scalogram_${TIMESTAMP}.log

echo "Completed: scalogram"
echo ""

################################################################################
# 実験2: scalogram+logmel (スカログラム + LogMel)
################################################################################
echo "[2/4] Running: scalogram + LogMel"
echo "  use_imaginary: False"
echo "  combine_with_logmel: True"
uv run train_pretrained.py \
    --attn_type ${ATTN_TYPE} \
    --attn_deepen ${ATTN_DEEPEN} \
    --mixstyle_type ${MIXSTYLE_TYPE} \
    --combine_with_logmel \
    --use_wavelet \
    --wandb_dir ${BASE_WANDB_DIR}/scalogram_logmel \
    2>&1 | tee ${LOG_DIR}/scalogram_logmel_${TIMESTAMP}.log

echo "Completed: scalogram + LogMel"
echo ""

################################################################################
# 実験3: real+imag (実部+虚部)
################################################################################
echo "[3/4] Running: real + imaginary"
echo "  use_imaginary: True"
echo "  combine_with_logmel: False"
uv run train_pretrained.py \
    --attn_type ${ATTN_TYPE} \
    --attn_deepen ${ATTN_DEEPEN} \
    --mixstyle_type ${MIXSTYLE_TYPE} \
    --use_wavelet \
    --use_imaginary \
    --wandb_dir ${BASE_WANDB_DIR}/real_imag \
    2>&1 | tee ${LOG_DIR}/real_imag_${TIMESTAMP}.log

echo "Completed: real + imaginary"
echo ""

################################################################################
# 実験4: real+imag+logmel (実部+虚部 + LogMel)
################################################################################
echo "[4/4] Running: real + imaginary + LogMel"
echo "  use_imaginary: True"
echo "  combine_with_logmel: True"
uv run train_pretrained.py \
    --attn_type ${ATTN_TYPE} \
    --attn_deepen ${ATTN_DEEPEN} \
    --mixstyle_type ${MIXSTYLE_TYPE} \
    --use_wavelet \
    --use_imaginary \
    --combine_with_logmel \
    --wandb_dir ${BASE_WANDB_DIR}/real_imag_logmel \
    2>&1 | tee ${LOG_DIR}/real_imag_logmel_${TIMESTAMP}.log

echo "Completed: real + imaginary + LogMel"
echo ""

################################################################################
# 完了
################################################################################
echo "=========================================="
echo "All CWT Feature Experiments Completed!"
echo "End Time: $(date)"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Logs: ${LOG_DIR}"
echo "  - Wandb: ${BASE_WANDB_DIR}"
echo ""
echo "Experiment Summary:"
echo "  1. scalogram           : CWT amplitude only"
echo "  2. scalogram+logmel    : CWT amplitude + LogMel"
echo "  3. real+imag           : CWT real + imaginary parts"
echo "  4. real+imag+logmel    : CWT real + imaginary + LogMel"
echo ""
echo "To view logs:"
echo "  tail -f ${LOG_DIR}/*_${TIMESTAMP}.log"
echo ""
