#!/bin/bash

# wandb/new_runs内の全てのcheckpointに対してテストを実行するスクリプト

# 設定
NEW_RUNS_DIR="./wandb/new_runs"
EVALED_RUNS_DIR="./wandb/evaled_runs"
LOG_DIR="logs/test_new_runs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ログディレクトリの作成
mkdir -p ${LOG_DIR}
mkdir -p ${EVALED_RUNS_DIR}

echo "=========================================="
echo "Testing checkpoints from ${NEW_RUNS_DIR}"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# wandb/new_runs内の全てのcheckpointファイルを検索
CHECKPOINT_FILES=$(find ${NEW_RUNS_DIR} -type f -path "run-*/files/checkpoints/*.ckpt" | sort)

# チェックポイントが見つからない場合
if [ -z "$CHECKPOINT_FILES" ]; then
    echo "Warning: No checkpoint files found in ${NEW_RUNS_DIR}"
    echo "Skipping tests."
    echo ""
    echo "=========================================="
    echo "No tests to run"
    echo "End Time: $(date)"
    echo "=========================================="
    exit 0
fi

# 見つかったチェックポイントの数を表示
CHECKPOINT_COUNT=$(echo "$CHECKPOINT_FILES" | wc -l | tr -d ' ')
echo "Found ${CHECKPOINT_COUNT} checkpoint(s)"
echo ""

# 処理済みrunディレクトリを記録する配列
declare -a PROCESSED_RUNS

# 各チェックポイントに対してテストを実行
COUNTER=1
for CKPT_PATH in $CHECKPOINT_FILES; do
    echo "=========================================="
    echo "Testing checkpoint ${COUNTER}/${CHECKPOINT_COUNT}"
    echo "Checkpoint: ${CKPT_PATH}"
    echo "Time: $(date)"
    echo "=========================================="
    
    # run名とcheckpoint名を抽出
    RUN_NAME=$(echo ${CKPT_PATH} | grep -oP 'run-[^/]+' || echo ${CKPT_PATH} | sed -n 's/.*\/\(run-[^\/]*\)\/.*/\1/p')
    RUN_DIR="${NEW_RUNS_DIR}/${RUN_NAME}"
    CKPT_NAME=$(basename ${CKPT_PATH} .ckpt)
    
    # ログファイル名
    LOG_FILE="${LOG_DIR}/${TIMESTAMP}_${RUN_NAME}_${CKPT_NAME}.log"
    
    echo "Log file: ${LOG_FILE}"
    echo ""
    
    # テスト実行
    uv run train_pretrained.py \
        --test_from_checkpoint ${CKPT_PATH} \
        2>&1 | tee ${LOG_FILE}
    
    # 実行結果を確認
    if [ $? -eq 0 ]; then
        echo "✓ Test completed successfully for ${CKPT_NAME}"
        # 処理済みrunディレクトリを記録
        if [[ ! " ${PROCESSED_RUNS[@]} " =~ " ${RUN_DIR} " ]]; then
            PROCESSED_RUNS+=("${RUN_DIR}")
        fi
    else
        echo "✗ Test failed for ${CKPT_NAME}"
    fi
    
    echo ""
    COUNTER=$((COUNTER + 1))
done

echo ""
echo "=========================================="
echo "Moving evaluated runs to ${EVALED_RUNS_DIR}"
echo "=========================================="

# 処理済みrunディレクトリをevaled_runsに移動
for RUN_DIR in "${PROCESSED_RUNS[@]}"; do
    if [ -d "${RUN_DIR}" ]; then
        RUN_NAME=$(basename ${RUN_DIR})
        echo "Moving ${RUN_NAME}..."
        mv "${RUN_DIR}" "${EVALED_RUNS_DIR}/"
        if [ $? -eq 0 ]; then
            echo "✓ Moved ${RUN_NAME} to evaled_runs"
        else
            echo "✗ Failed to move ${RUN_NAME}"
        fi
    fi
done

echo ""
echo "=========================================="
echo "All tests completed"
echo "End Time: $(date)"
echo "Logs saved in: ${LOG_DIR}"
echo "Evaluated runs moved to: ${EVALED_RUNS_DIR}"
echo "=========================================="
