#!/bin/bash

# GMMパラメータのOptuna最適化スクリプト
# 
# 使用方法:
#   bash run_optuna_gmm.sh [GPU番号]
#
# 例:
#   bash run_optuna_gmm.sh 0        # GPU 0で実行
#   bash run_optuna_gmm.sh 0,1      # GPU 0と1で並列実行

# デフォルト設定
GPU=${1:-0}
CONF_FILE="./confs/optuna_gmm.yaml"
LOG_DIR="./exp/optuna_gmm"
N_JOBS=1  # 使用するGPU数

# GPU設定
export CUDA_VISIBLE_DEVICES=$GPU

# GPU数を自動検出
if [[ $GPU == *","* ]]; then
    N_JOBS=$(echo $GPU | tr ',' '\n' | wc -l | tr -d ' ')
fi

echo "========================================"
echo "Optuna GMM Parameter Tuning"
echo "========================================"
echo "Config file:    $CONF_FILE"
echo "Log directory:  $LOG_DIR"
echo "GPUs:           $GPU"
echo "Number of jobs: $N_JOBS"
echo "========================================"

# Optuna実行
uv run optuna_pretrained.py \
    --conf_file "$CONF_FILE" \
    --log_dir "$LOG_DIR" \
    --n_jobs "$N_JOBS" \
    --tune_gmm

echo ""
echo "========================================"
echo "Optimization completed!"
echo "========================================"
echo "Results saved in: $LOG_DIR"
echo "Database: $LOG_DIR/optuna-sed.db"
echo "Log file: $LOG_DIR/optuna-sed.log"
echo ""
echo "To view results with Optuna Dashboard:"
echo "  optuna-dashboard sqlite:///$LOG_DIR/optuna-sed.db"
echo "========================================"
