#!/bin/bash

################################################################################
# UMAP可視化自動実行スクリプト
#
# mixstyle_only_150の2つのチェックポイント（CNNとTransformer）から
# 特徴量を抽出し、UMAP可視化を行います。
#
# 使用法:
#   ./run_umap_visualization.sh
#
# オプション:
#   --category CATEGORY      実験カテゴリ (default: ablation)
#   --method METHOD          メソッド名 (default: mixstyle_only_150)
#   --base-dir BASE_DIR      実験ベースディレクトリ (default: experiments)
#   --output-dir OUTPUT_DIR  出力ディレクトリ (default: visualizations/umap)
#   --n-components N         UMAP次元数 2 or 3 (default: 2)
#   --feature-type TYPE      特徴量タイプ student or teacher (default: student)
#   --device DEVICE          使用デバイス cpu or cuda (default: cuda)
################################################################################

set -e  # エラーが発生したら終了

# デフォルト設定
CATEGORY="ablation"
METHOD="mixstyle_only_150"
BASE_DIR="experiments"
OUTPUT_BASE="visualizations/umap"
N_COMPONENTS=2
FEATURE_TYPE="teacher"
DEVICE="cuda"
CONFIG="confs/pretrained.yaml"
BATCH_SIZE=32

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
        --output-dir)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --n-components)
            N_COMPONENTS="$2"
            shift 2
            ;;
        --feature-type)
            FEATURE_TYPE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --category CATEGORY      実験カテゴリ (default: ablation)"
            echo "  --method METHOD          メソッド名 (default: mixstyle_only_150)"
            echo "  --base-dir BASE_DIR      実験ベースディレクトリ (default: experiments)"
            echo "  --output-dir OUTPUT_DIR  出力ディレクトリ (default: visualizations/umap)"
            echo "  --n-components N         UMAP次元数 2 or 3 (default: 2)"
            echo "  --feature-type TYPE      特徴量タイプ student or teacher (default: student)"
            echo "  --device DEVICE          使用デバイス cpu or cuda (default: cuda)"
            echo "  --config CONFIG          設定ファイル (default: confs/pretrained.yaml)"
            echo "  --batch-size SIZE        バッチサイズ (default: 32)"
            return 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            return 1
            ;;
    esac
done

# ディレクトリ設定
EXP_DIR="${BASE_DIR}/train/${CATEGORY}/${METHOD}"
FEATURES_DIR="${OUTPUT_BASE}/features_${METHOD}"
UMAP_DIR="${OUTPUT_BASE}/umap_${METHOD}_${N_COMPONENTS}d"

# タイムスタンプ
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=========================================="
echo "UMAP可視化パイプライン"
echo "Start Time: $(date)"
echo "=========================================="
echo ""
echo "設定:"
echo "  実験ディレクトリ: ${EXP_DIR}"
echo "  特徴量出力: ${FEATURES_DIR}"
echo "  UMAP出力: ${UMAP_DIR}"
echo "  UMAP次元数: ${N_COMPONENTS}"
echo "  特徴量タイプ: ${FEATURE_TYPE}"
echo "  デバイス: ${DEVICE}"
echo ""

################################################################################
# Step 1: チェックポイントの確認
################################################################################
echo "[Step 1/3] チェックポイント確認中..."
echo ""

# 2つのバリアントを指定
VARIANTS=("CNN" "baseline")
CHECKPOINTS=()
MODEL_NAMES=()

for VARIANT in "${VARIANTS[@]}"; do
    VARIANT_DIR="${EXP_DIR}/${VARIANT}"

    if [ ! -d "${VARIANT_DIR}" ]; then
        echo "警告: ${VARIANT_DIR} が見つかりません"
        continue
    fi

    # last.ckptを使用（best.ckptがあればそちらを優先）
    if [ -f "${VARIANT_DIR}/best.ckpt" ]; then
        CHECKPOINT="${VARIANT_DIR}/best.ckpt"
        echo "✓ ${VARIANT}: ${CHECKPOINT} (best)"
    elif [ -f "${VARIANT_DIR}/last.ckpt" ]; then
        CHECKPOINT="${VARIANT_DIR}/last.ckpt"
        echo "✓ ${VARIANT}: ${CHECKPOINT} (last)"
    else
        echo "警告: ${VARIANT_DIR} にチェックポイントが見つかりません"
        continue
    fi

    CHECKPOINTS+=("${CHECKPOINT}")
    MODEL_NAMES+=("${METHOD}_${VARIANT}")
done

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo ""
    echo "エラー: 有効なチェックポイントが見つかりませんでした"
    echo ""
    echo "期待されるパス:"
    for VARIANT in "${VARIANTS[@]}"; do
        echo "  - ${EXP_DIR}/${VARIANT}/best.ckpt"
        echo "  - ${EXP_DIR}/${VARIANT}/last.ckpt"
    done
    echo ""
    return 1
fi

echo ""
echo "見つかったチェックポイント: ${#CHECKPOINTS[@]}個"
echo ""

################################################################################
# Step 2: 特徴量抽出
################################################################################
echo "[Step 2/3] 特徴量抽出中..."
echo ""

# 特徴量ディレクトリの作成
mkdir -p "${FEATURES_DIR}"

# 各モデルのディレクトリを作成
FEATURE_DIRS=()
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    MODEL_FEATURE_DIR="${FEATURES_DIR}/${MODEL_NAME}"
    mkdir -p "${MODEL_FEATURE_DIR}"
    FEATURE_DIRS+=("${MODEL_FEATURE_DIR}")
done

# 特徴量抽出を並列実行
for i in "${!CHECKPOINTS[@]}"; do
    CHECKPOINT="${CHECKPOINTS[$i]}"
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MODEL_FEATURE_DIR="${FEATURE_DIRS[$i]}"

    echo "抽出中: ${MODEL_NAME}"

    cd DESED_task/dcase2024_task4_baseline

    uv run python visualize/extract_features.py \
        --checkpoints "${CHECKPOINT}" \
        --config "${CONFIG}" \
        --output_dir "${MODEL_FEATURE_DIR}" \
        --model_names "${MODEL_NAME}" \
        --datasets validation maestro_validation \
        --batch_size ${BATCH_SIZE} \
        --device ${DEVICE} \
        2>&1 | tee "${MODEL_FEATURE_DIR}/extract_${TIMESTAMP}.log"

    cd ../..

    echo ""
done

echo "特徴量抽出完了！"
echo ""

################################################################################
# Step 3: UMAP可視化
################################################################################
echo "[Step 3/3] UMAP可視化中..."
echo ""

# 出力ディレクトリの作成
mkdir -p "${UMAP_DIR}"

# UMAP可視化実行
cd DESED_task/dcase2024_task4_baseline

uv run python visualize/visualize_umap.py \
    --input_dirs "${FEATURE_DIRS[@]}" \
    --output_dir "${UMAP_DIR}" \
    --feature_type ${FEATURE_TYPE} \
    --n_components ${N_COMPONENTS} \
    --n_neighbors 15 \
    --min_dist 0.1 \
    --metric euclidean \
    --random_state 42 \
    --verbose \
    2>&1 | tee "${UMAP_DIR}/umap_${TIMESTAMP}.log"

cd ../..

echo ""
echo "UMAP可視化完了！"
echo ""

################################################################################
# Step 4: 完了メッセージ
################################################################################
echo "=========================================="
echo "パイプライン完了！"
echo "End Time: $(date)"
echo "=========================================="
echo ""
echo "出力ファイル:"
echo "  特徴量: ${FEATURES_DIR}"
echo "  UMAP可視化: ${UMAP_DIR}"
echo ""
echo "生成された可視化:"
ls -lh "${UMAP_DIR}"/*.png 2>/dev/null || echo "  (PNGファイルが見つかりません)"
echo ""
echo "ログファイル:"
for MODEL_FEATURE_DIR in "${FEATURE_DIRS[@]}"; do
    ls -lh "${MODEL_FEATURE_DIR}"/*.log 2>/dev/null
done
ls -lh "${UMAP_DIR}"/*.log 2>/dev/null
echo ""
echo "=========================================="
