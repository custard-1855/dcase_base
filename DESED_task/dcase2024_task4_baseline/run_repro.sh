#!/bin/bash

# ================================================================================
# 再現学習キュー: 論文 (ieice2025) の表の構成を上から順に 1 本ずつ学習する
# ================================================================================
# Usage:
#   ./run_repro.sh [OPTIONS] [構成名 ...] [-- train_pretrained.py への追加引数]
#
# Options:
#   --out DIR        出力先 (default: exp/repro)
#   --confs FILE     設定ファイル (default: ./confs/pretrained.yaml)
#   --gpus N         train_pretrained.py の --gpus ("1": GPU, "0": CPU) (default: 1)
#   --wandb          wandb を有効にする (既定は無効)
#   --rotate         出力先が既にあれば ${OUT}.N (N は空いている最小の番号) に退避してから実行する
#   --list           構成名と引数を表示して終了
#   --dry-run        実行するコマンドを表示するだけ
#
# Examples:
#   ./run_repro.sh                         # 全構成を順に実行
#   ./run_repro.sh mt cmt                  # 指定した構成だけ
#   ./run_repro.sh mt -- --fast_dev_run    # 短い実行で経路を確認
#   ./run_repro.sh --rotate                # 前回の結果を残したまま, 全構成をもう一度流す
#
# 出力:
#   ${OUT}/${構成名}/version_N/            TensorBoard, ckpt, test_metrics.json
#   ${OUT}/${構成名}/train_*.log           標準出力のログ
#   ${OUT}/queue.tsv                       構成ごとの commit, 開始・終了時刻, 所要秒, 終了コード
#
# test_metrics.json が既にある構成は飛ばすので, 途中で止まっても同じコマンドで再開できる.
# (--rotate を付けると毎回退避するので, 再開するときは --rotate を外す.)
# 失敗した構成があっても次へ進み, 最後に終了コード 1 を返す.
# ================================================================================

set -o pipefail
cd "$(dirname "$0")" || exit 1

# 構成 (実行順). MixStyle の引数は run_ablation_a_times_b.sh に合わせている.
MIXSTYLE="--mixstyle_type freqAttn --attn_type default"
NAMES=()
declare -A FLAGS
add() {
    NAMES+=("$1")
    FLAGS[$1]="$2"
}
add mt ""
add cmt_scale_mixstyle "--cmt --use_neg_sample --pos_neg_scale ${MIXSTYLE}"
add cmt "--cmt"
# 残り 4 行 (時間があれば)
add mt_mixstyle "${MIXSTYLE}"
add cmt_mixstyle "--cmt ${MIXSTYLE}"
add cmt_neg "--cmt --use_neg_sample"
add cmt_scale "--cmt --use_neg_sample --pos_neg_scale"

# デフォルト設定
OUT="exp/repro"
CONFS="./confs/pretrained.yaml"
GPUS="1"
USE_WANDB=0
ROTATE=0
DRY_RUN=0
SELECTED=()
EXTRA_ARGS=()

# 引数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --out)
            OUT="$2"
            shift 2
            ;;
        --confs)
            CONFS="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --wandb)
            USE_WANDB=1
            shift
            ;;
        --rotate)
            ROTATE=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --list)
            for name in "${NAMES[@]}"; do
                printf '%-20s %s\n' "${name}" "${FLAGS[$name]}"
            done
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            if [[ -z "${FLAGS[$1]+x}" ]]; then
                echo "Unknown config: $1 (--list で一覧)"
                exit 1
            fi
            SELECTED+=("$1")
            shift
            ;;
    esac
done
if [[ ${#SELECTED[@]} -eq 0 ]]; then
    SELECTED=("${NAMES[@]}")
fi

COMMIT=$(git rev-parse --short HEAD)
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    COMMIT="${COMMIT}-dirty"
fi

# 前回の結果を退避する (同じ構成を流し直して, 実行ごとの差を見るとき用)
OUT="${OUT%/}"
if [[ ${ROTATE} -eq 1 && -e "${OUT}" ]]; then
    n=1
    while [[ -e "${OUT}.${n}" ]]; do
        n=$((n + 1))
    done
    echo "[rotate] ${OUT} -> ${OUT}.${n}"
    if [[ ${DRY_RUN} -eq 0 ]]; then
        mv "${OUT}" "${OUT}.${n}" || exit 1
    fi
fi

if [[ ${DRY_RUN} -eq 0 ]]; then
    mkdir -p "${OUT}"
fi

# yaml の wandb.use_wandb は既定で True で, CLI からは無効にできない.
# 無効で流すときは use_wandb だけ False にした設定の写しを使う.
if [[ ${USE_WANDB} -eq 0 ]]; then
    RUN_CONFS="${OUT}/$(basename "${CONFS%.*}").nowandb.yaml"
    if [[ ${DRY_RUN} -eq 0 ]]; then
        uv run python - "${CONFS}" "${RUN_CONFS}" <<'EOF' || exit 1
import sys

import yaml

with open(sys.argv[1]) as f:
    config = yaml.safe_load(f)
config["wandb"]["use_wandb"] = False
with open(sys.argv[2], "w") as f:
    yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
EOF
    fi
else
    RUN_CONFS="${CONFS}"
fi

FAILED=()
for name in "${SELECTED[@]}"; do
    run_dir="${OUT}/${name}"
    if compgen -G "${run_dir}/version_*/test_metrics.json" > /dev/null; then
        echo "[skip] ${name}: $(compgen -G "${run_dir}/version_*/test_metrics.json" | tail -n 1) がある"
        continue
    fi

    # shellcheck disable=SC2206  # FLAGS は空白区切りの引数列
    cmd=(uv run train_pretrained.py --confs "${RUN_CONFS}" --log_dir "${run_dir}" --gpus "${GPUS}" ${FLAGS[$name]})
    if [[ ${USE_WANDB} -eq 1 ]]; then
        cmd+=(--use_wandb --category repro --method ieice2025 --variant "${name}")
    fi
    cmd+=("${EXTRA_ARGS[@]}")

    echo "=========================================="
    echo "[run] ${name} (commit ${COMMIT})"
    echo "${cmd[*]}"
    echo "=========================================="
    if [[ ${DRY_RUN} -eq 1 ]]; then
        continue
    fi

    mkdir -p "${run_dir}"
    start=$(date +%s)
    "${cmd[@]}" 2>&1 | tee "${run_dir}/train_$(date -d "@${start}" +%Y%m%d_%H%M%S).log"
    status=$?
    end=$(date +%s)

    if [[ ! -f "${OUT}/queue.tsv" ]]; then
        printf 'name\tcommit\tstart\tend\tseconds\texit_code\n' > "${OUT}/queue.tsv"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${name}" "${COMMIT}" \
        "$(date -d "@${start}" +%Y-%m-%dT%H:%M:%S)" "$(date -d "@${end}" +%Y-%m-%dT%H:%M:%S)" \
        "$((end - start))" "${status}" >> "${OUT}/queue.tsv"

    if [[ ${status} -ne 0 ]]; then
        echo "[fail] ${name}: exit ${status}"
        FAILED+=("${name}")
    fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed: ${FAILED[*]}"
    exit 1
fi
