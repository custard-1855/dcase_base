#!/bin/bash
# 可視化スクリプト実行シェル
#
# 使用法:
#   bash run_visualizations.sh [phase]
#
# phase:
#   1 - Phase 1のみ実行（UMAP, Reliability, Report）
#   2 - Phase 2のみ実行（Grad-CAM）
#   all - すべて実行（デフォルト）

set -e  # エラーで停止

# デフォルト設定
PHASE="${1:-all}"
INPUT_DIRS="inference_outputs/baseline inference_outputs/cmt_normal inference_outputs/cmt_neg"
OUTPUT_DIR="visualization_outputs"

# Phase 1: 基本可視化
run_phase1() {
    echo "=========================================="
    echo "Phase 1: 基本可視化"
    echo "=========================================="

    # 1. UMAP可視化
    echo ""
    echo "[1/3] UMAP可視化を実行中..."
    uv run visualize_umap.py \
        --input_dirs ${INPUT_DIRS} \
        --output_dir ${OUTPUT_DIR}/umap \
        --feature_type both

    # 2. Reliability Diagram
    echo ""
    echo "[2/3] Reliability Diagramを実行中..."
    uv run visualize_reliability.py \
        --input_dirs ${INPUT_DIRS} \
        --output_dir ${OUTPUT_DIR}/reliability \
        --pred_type both

    # 3. 統合レポート
    echo ""
    echo "[3/3] 統合レポートを生成中..."
    uv run generate_analysis_report.py \
        --input_dirs ${INPUT_DIRS} \
        --output ${OUTPUT_DIR}/analysis_report.md

    echo ""
    echo "Phase 1 完了！"
    echo "出力ディレクトリ: ${OUTPUT_DIR}"
}

# Phase 2: Grad-CAM分析
run_phase2() {
    echo "=========================================="
    echo "Phase 2: Grad-CAM分析"
    echo "=========================================="

    # チェックポイントパスを取得（metadata.jsonから）
    echo ""
    echo "チェックポイントパスを取得中..."

    BASELINE_CKPT=$(uv run python -c "import json; print(json.load(open('inference_outputs/baseline/inference_metadata.json'))['checkpoint'])")
    CMT_NORMAL_CKPT=$(uv run python -c "import json; print(json.load(open('inference_outputs/cmt_normal/inference_metadata.json'))['checkpoint'])")

    echo "  Baseline: ${BASELINE_CKPT}"
    echo "  CMT Normal: ${CMT_NORMAL_CKPT}"

    # チェックポイントの存在確認
    if [ ! -f "${BASELINE_CKPT}" ]; then
        echo "エラー: ベースラインのチェックポイントが見つかりません: ${BASELINE_CKPT}"
        exit 1
    fi

    if [ ! -f "${CMT_NORMAL_CKPT}" ]; then
        echo "エラー: CMT Normalのチェックポイントが見つかりません: ${CMT_NORMAL_CKPT}"
        exit 1
    fi

    # Grad-CAM実行
    echo ""
    echo "Grad-CAMを実行中..."
    uv run visualize_gradcam.py \
        --input_dirs inference_outputs/baseline inference_outputs/cmt_normal \
        --checkpoints "${BASELINE_CKPT}" "${CMT_NORMAL_CKPT}" \
        --config confs/pretrained.yaml \
        --output_dir ${OUTPUT_DIR}/gradcam \
        --n_samples 10 \
        --pred_type student

    echo ""
    echo "Phase 2 完了！"
    echo "出力ディレクトリ: ${OUTPUT_DIR}/gradcam"
}

# メイン処理
echo "=========================================="
echo "可視化スクリプト実行"
echo "=========================================="
echo "Phase: ${PHASE}"
echo "入力ディレクトリ: ${INPUT_DIRS}"
echo "出力ディレクトリ: ${OUTPUT_DIR}"
echo ""

case "${PHASE}" in
    1)
        run_phase1
        ;;
    2)
        run_phase2
        ;;
    all)
        run_phase1
        echo ""
        echo ""
        read -p "Phase 2 (Grad-CAM) を実行しますか？ [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            run_phase2
        else
            echo "Phase 2 をスキップしました。"
        fi
        ;;
    *)
        echo "エラー: 不正なphase指定: ${PHASE}"
        echo "使用法: bash run_visualizations.sh [1|2|all]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "すべての処理が完了しました！"
echo "=========================================="
echo ""
echo "結果を確認:"
echo "  - UMAP: ${OUTPUT_DIR}/umap/"
echo "  - Reliability: ${OUTPUT_DIR}/reliability/"
echo "  - レポート: ${OUTPUT_DIR}/analysis_report.md"
if [ "${PHASE}" == "all" ] || [ "${PHASE}" == "2" ]; then
    echo "  - Grad-CAM: ${OUTPUT_DIR}/gradcam/"
fi
