#!/bin/bash

# リアルタイムSEDシステム起動スクリプト

# スクリプトのディレクトリに移動
cd "$(dirname "$0")"

# 依存関係のチェック
# echo "依存関係をチェック中..."
# python -c "import sounddevice" 2>/dev/null
# if [ $? -ne 0 ]; then
#     echo "sounddeviceがインストールされていません"
#     echo "インストールしますか? (y/n)"
#     read -r response
#     if [ "$response" = "y" ]; then
#         pip install sounddevice matplotlib
#     else
#         echo "インストールをキャンセルしました"
#         exit 1
#     fi
# fi

# デフォルトの設定
CONFIG="DESED_task/dcase2024_task4_baseline/confs/pretrained.yaml"
DEVICE="cpu"
SAVE_DIR="realtime_results"

# ヘルプメッセージ
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "使用方法: $0 [OPTIONS]"
    echo ""
    echo "オプション:"
    echo "  -c, --checkpoint PATH    モデルチェックポイントのパス"
    echo "  -d, --device DEVICE      使用デバイス (cpu または cuda, デフォルト: cpu)"
    echo "  -s, --save-dir DIR       結果保存ディレクトリ (デフォルト: realtime_results)"
    echo "  --continuous             継続モード"
    echo "  -h, --help               このヘルプを表示"
    echo ""
    echo "例:"
    echo "  $0                                          # 基本的な使い方"
    echo "  $0 --checkpoint model.ckpt                  # チェックポイント指定"
    echo "  $0 --continuous                             # 継続モード"
    echo "  $0 --checkpoint model.ckpt --continuous     # 両方指定"
    exit 0
fi

# コマンドライン引数の解析
CHECKPOINT=""
CONTINUOUS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -s|--save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --continuous)
            CONTINUOUS="--continuous"
            shift
            ;;
        *)
            echo "不明なオプション: $1"
            echo "ヘルプを表示するには -h または --help を使用してください"
            exit 1
            ;;
    esac
done

# リアルタイムSEDを実行
echo ""
echo "============================================"
echo "リアルタイム音響イベント検出システムを起動"
echo "============================================"
echo ""

CMD="python realtime_sed.py --config $CONFIG --device $DEVICE --save-dir $SAVE_DIR"

if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
    echo "チェックポイント: $CHECKPOINT"
else
    echo "チェックポイント: なし（ランダム初期化）"
fi

echo "デバイス: $DEVICE"
echo "保存ディレクトリ: $SAVE_DIR"

if [ -n "$CONTINUOUS" ]; then
    CMD="$CMD --continuous"
    echo "モード: 継続"
else
    echo "モード: 1回のみ"
fi

echo ""
echo "実行コマンド: $CMD"
echo ""

# 実行
eval $CMD
