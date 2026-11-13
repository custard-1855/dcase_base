#!/bin/bash

# ================================================================================
# アブレーション実験のランチャー: 実験セットを 1 つ選び, その構成を上から順に学習する
# ================================================================================
# Usage:
#   ./run_ablation.sh SET [OPTIONS]
#
# SET:
#   mixstyle     MixStyle の入れ方 2 構成                    (旧 run_mixstyle.sh)
#   cmt          CMT の負例サンプリング 4 構成                (旧 run_cmt_ablation.sh)
#   a_times_b    学習手法 (A) × 正則化・後処理 (B) の 9 構成   (旧 run_ablation_a_times_b.sh)
#
# Options:
#   --category CATEGORY      実験カテゴリ (default: ablation)
#   --method METHOD          手法名 (default: セットごと. mixstyle_150 / cmt_neg_fix_150 / a_times_b)
#   --base-dir DIR           ベースディレクトリ (default: experiments)
#   --list                   構成名と引数を表示して終了
#   --dry-run                実行するコマンドを表示するだけ
#
# Examples:
#   ./run_ablation.sh a_times_b
#   ./run_ablation.sh cmt --category optimization --method my_experiment
#   ./run_ablation.sh mixstyle --list
#
# 出力:
#   ${BASE_DIR}/train/${CATEGORY}/${METHOD}/${構成名}/    学習結果
#   logs/${CATEGORY}_${METHOD}/                           標準出力のログ
# ================================================================================

cd "$(dirname "$0")" || exit 1

# 構成 (実行順). add 構成名 ログ名の接頭辞 引数
MIXSTYLE="--mixstyle_type freqAttn --attn_type default"
NAMES=()
declare -A LOGS FLAGS
add() {
    NAMES+=("$1")
    LOGS[$1]="$2"
    FLAGS[$1]="$3"
}

SET="$1"
case ${SET} in
    mixstyle)
        METHOD="mixstyle_150"
        add CNN "1_" "${MIXSTYLE}"
        add nomal "2_" "--mixstyle_type only_mix --attn_type default"
        ;;
    cmt)
        METHOD="cmt_neg_fix_150"
        add baseline "baseline_" ""
        add cmt_no_neg "no_neg_" "--cmt"
        add neg "neg_" "--cmt --use_neg_sample"
        add neg_pos_neg "neg_pos-neg" "--cmt --use_neg_sample --pos_neg_scale"
        ;;
    a_times_b)
        METHOD="a_times_b"
        # Set A: normal (MeanTeacher)
        add normal_mixstyle "normal_mixstyle_" "${MIXSTYLE}"
        add normal_sebbs "normal_sebbs_" "--sebbs"
        add normal_mixstyle_sebbs "normal_mixstyle_sebbs_" "${MIXSTYLE} --sebbs"
        # Set A: CMT
        add cmt_mixstyle "cmt_mixstyle_" "--cmt ${MIXSTYLE}"
        add cmt_sebbs "cmt_sebbs_" "--cmt --sebbs"
        add cmt_mixstyle_sebbs "cmt_mixstyle_sebbs_" "--cmt ${MIXSTYLE} --sebbs"
        # Set A: CMT + neg + pos_neg_scale
        CMT_SCALE="--cmt --use_neg_sample --pos_neg_scale"
        add cmt_neg_scale_mixstyle "cmt_neg_scale_mixstyle_" "${CMT_SCALE} ${MIXSTYLE}"
        add cmt_neg_scale_sebbs "cmt_neg_scale_sebbs_" "${CMT_SCALE} --sebbs"
        add cmt_neg_scale_mixstyle_sebbs "cmt_neg_scale_mixstyle_sebbs_" "${CMT_SCALE} ${MIXSTYLE} --sebbs"
        ;;
    *)
        echo "Usage: $0 {mixstyle|cmt|a_times_b} [OPTIONS]"
        exit 1
        ;;
esac
shift

# デフォルト設定
CATEGORY="ablation"
BASE_DIR="experiments"
DRY_RUN=0

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
        --list)
            for name in "${NAMES[@]}"; do
                printf '%-30s %s\n' "${name}" "${FLAGS[$name]}"
            done
            exit 0
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ログディレクトリ
LOG_DIR="logs/${CATEGORY}_${METHOD}"
if [[ ${DRY_RUN} -eq 0 ]]; then
    mkdir -p "${LOG_DIR}"
fi

# タイムスタンプ
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

i=0
for name in "${NAMES[@]}"; do
    i=$((i + 1))
    # shellcheck disable=SC2206  # FLAGS は空白区切りの引数列
    cmd=(uv run train_pretrained.py --use_wandb --category "${CATEGORY}" --method "${METHOD}"
        --variant "${name}" --base_dir "${BASE_DIR}" ${FLAGS[$name]})

    echo "[${i}/${#NAMES[@]}] Running: ${name}"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        echo "${cmd[*]}"
        continue
    fi
    "${cmd[@]}" 2>&1 | tee "${LOG_DIR}/${LOGS[$name]}${TIMESTAMP}.log"

    echo ""
    echo "---"
    echo ""
done
if [[ ${DRY_RUN} -eq 1 ]]; then
    exit 0
fi

echo "=========================================="
echo "All Experiments Completed"
echo "End Time: $(date)"
echo "=========================================="
echo ""
echo "Results are saved in:"
echo "  - Experiments: ${BASE_DIR}/train/${CATEGORY}/${METHOD}/"
echo "  - Logs: ${LOG_DIR}/"
echo ""
