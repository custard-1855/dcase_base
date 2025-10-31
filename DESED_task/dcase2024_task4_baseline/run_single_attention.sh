#!/bin/bash

################################################################################
# Single Attention Experiment Runner
#
# 使い方:
#   ./run_single_attention.sh <attn_type> <depth> [mixstyle_type] [additional_args]
#
# 例:
#   ./run_single_attention.sh residual_deep 4
#   ./run_single_attention.sh multiscale 3 resMix
#   ./run_single_attention.sh se_deep 2 resMix --debug
################################################################################

# 引数チェック
if [ $# -lt 2 ]; then
    echo "Usage: $0 <attn_type> <depth> [mixstyle_type] [additional_args]"
    echo ""
    echo "Available attn_types:"
    echo "  - default          : 浅い2層CNN"
    echo "  - residual_deep    : 残差接続で深いネットワーク"
    echo "  - multiscale       : マルチスケール畳み込み"
    echo "  - se_deep          : SE-Block組み込み深層ネットワーク"
    echo "  - dilated_deep     : Dilated Convolution"
    echo ""
    echo "Examples:"
    echo "  $0 residual_deep 4"
    echo "  $0 multiscale 3 resMix"
    echo "  $0 se_deep 2 resMix --debug"
    exit 1
fi

# パラメータ取得
ATTN_TYPE=$1
DEPTH=$2
MIXSTYLE_TYPE=${3:-resMix}  # デフォルトはresMix
shift 3 2>/dev/null || shift 2  # 残りの引数を取得
ADDITIONAL_ARGS="$@"

# 実験名の生成
EXP_NAME="${ATTN_TYPE}_d${DEPTH}"
WANDB_DIR="attention_experiments/${EXP_NAME}"
LOG_DIR="logs/attention_experiments"
mkdir -p ${LOG_DIR}

# タイムスタンプ
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${EXP_NAME}_${TIMESTAMP}.log"

echo "=========================================="
echo "Running Attention Experiment"
echo "=========================================="
echo "  Attention Type   : ${ATTN_TYPE}"
echo "  Depth            : ${DEPTH}"
echo "  MixStyle Type    : ${MIXSTYLE_TYPE}"
echo "  WandB Directory  : ${WANDB_DIR}"
echo "  Log File         : ${LOG_FILE}"
echo "  Additional Args  : ${ADDITIONAL_ARGS}"
echo "  Start Time       : $(date)"
echo "=========================================="
echo ""

# 実験実行
uv run train_pretrained.py \
    --attn_type ${ATTN_TYPE} \
    --attn_deepen ${DEPTH} \
    --mixstyle_type ${MIXSTYLE_TYPE} \
    --wandb_dir ${WANDB_DIR} \
    ${ADDITIONAL_ARGS} \
    2>&1 | tee ${LOG_FILE}

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Experiment Completed!"
echo "=========================================="
echo "  Exit Code        : ${EXIT_CODE}"
echo "  End Time         : $(date)"
echo "  Log saved to     : ${LOG_FILE}"
echo "=========================================="

exit ${EXIT_CODE}
