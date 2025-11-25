#!/bin/bash

# GMMパラメータのグリッドサーチ実験用スクリプト
# 複数のパラメータ組み合わせを自動的に実行

# 基本設定
GPUS=1
CONF_FILE="./confs/pretrained.yaml"
BASE_LOG_DIR="./exp/gmm_grid_search"

# テストするパラメータの組み合わせ
# 各配列に追加することで新しいパラメータ値をテスト可能

# n_init の候補値
N_INIT_VALUES=(1 3 5)

# max_iter の候補値
MAX_ITER_VALUES=(30 50 100)

# reg_covar の候補値
REG_COVAR_VALUES=(1e-6 5e-4 1e-3)

# tol の候補値
TOL_VALUES=(1e-3 1e-2 1e-6)

echo "========================================"
echo "GMM Grid Search Experiment"
echo "========================================"
echo "n_init values:     ${N_INIT_VALUES[@]}"
echo "max_iter values:   ${MAX_ITER_VALUES[@]}"
echo "reg_covar values:  ${REG_COVAR_VALUES[@]}"
echo "tol values:        ${TOL_VALUES[@]}"
echo "========================================"

# 実験カウンター
TOTAL_EXPERIMENTS=0
COMPLETED_EXPERIMENTS=0

# 総実験数を計算
TOTAL_EXPERIMENTS=$((${#N_INIT_VALUES[@]} * ${#MAX_ITER_VALUES[@]} * ${#REG_COVAR_VALUES[@]} * ${#TOL_VALUES[@]}))
echo "Total experiments to run: ${TOTAL_EXPERIMENTS}"
echo "========================================"

# グリッドサーチ実行
for N_INIT in "${N_INIT_VALUES[@]}"; do
    for MAX_ITER in "${MAX_ITER_VALUES[@]}"; do
        for REG_COVAR in "${REG_COVAR_VALUES[@]}"; do
            for TOL in "${TOL_VALUES[@]}"; do
                COMPLETED_EXPERIMENTS=$((COMPLETED_EXPERIMENTS + 1))
                
                echo ""
                echo "========================================"
                echo "Experiment ${COMPLETED_EXPERIMENTS}/${TOTAL_EXPERIMENTS}"
                echo "========================================"
                
                # run_gmm_experiment.shを呼び出し
                bash run_gmm_experiment.sh \
                    --n_init "${N_INIT}" \
                    --max_iter "${MAX_ITER}" \
                    --reg_covar "${REG_COVAR}" \
                    --tol "${TOL}" \
                    --gpus "${GPUS}" \
                    --conf_file "${CONF_FILE}" \
                    --log_dir "${BASE_LOG_DIR}"
                
                # 実験間の待機時間（オプション）
                # sleep 5
            done
        done
    done
done

echo ""
echo "========================================"
echo "All experiments completed!"
echo "Results are in: ${BASE_LOG_DIR}"
echo "========================================"
