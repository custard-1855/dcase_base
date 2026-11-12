#!/bin/bash

# ================================================================================
# 再現学習のベスト ckpt を再評価する (学習なし, 1 構成あたり約 3 分)
# ================================================================================
# Usage:
#   ./run_repro_eval.sh [OPTIONS] 構成名 [構成名 ...] [-- train_pretrained.py への追加引数]
#
# Options:
#   --sebbs          cSEBBs の後処理を掛けて評価する (出力先が変わる. 下記)
#   --src DIR        再現学習の出力 (default: exp/repro)
#   --out DIR        出力先 (default: exp/repro_eval. --sebbs のときは無視)
#   --confs FILE     設定ファイル (default: ${SRC}/pretrained.nowandb.yaml = 学習時の設定)
#   --gpus N         train_pretrained.py の --gpus ("1": GPU, "0": CPU) (default: 1)
#   --dry-run        実行するコマンドを表示するだけ
#
# Examples:
#   ./run_repro_eval.sh mt cmt_scale_mixstyle   # 整形などの後に, 値が変わっていないことを確認
#   ./run_repro_eval.sh --sebbs mt              # ckpt に cSEBBs を掛ける
#
# 出力:
#   ${OUT}/${構成名}/version_N/            test_metrics.json, 予測スコア (--sebbs なし)
#   ${SRC}/${構成名}_sebbs/version_N/      同上 (--sebbs あり)
#   上記の親ディレクトリ/test_*.log        標準出力のログ
#
# --test_from_checkpoint は ckpt に保存された設定を使わず, 渡した yaml と CLI で評価する (issue #14).
# そのため構成のフラグを run_repro.sh --list から引いて渡す.
# また --log_dir も効かず, 結果は ./lightning_logs/version_N に出るので, 終了後に出力先へ移す.
# 複数同時に実行すると移す対象を取り違えるので, 1 本ずつ流すこと.
# ================================================================================

set -o pipefail
cd "$(dirname "$0")" || exit 1

SRC="exp/repro"
OUT="exp/repro_eval"
CONFS=""
GPUS="1"
SEBBS=0
DRY_RUN=0
SELECTED=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --sebbs)
            SEBBS=1
            shift
            ;;
        --src)
            SRC="$2"
            shift 2
            ;;
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
        --dry-run)
            DRY_RUN=1
            shift
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
            SELECTED+=("$1")
            shift
            ;;
    esac
done
if [[ ${#SELECTED[@]} -eq 0 ]]; then
    echo "構成名を 1 つ以上指定する (./run_repro.sh --list で一覧)"
    exit 1
fi
if [[ -z "${CONFS}" ]]; then
    CONFS="${SRC}/pretrained.nowandb.yaml"
fi
if [[ ! -f "${CONFS}" ]]; then
    echo "設定ファイルが無い: ${CONFS} (--confs で指定)"
    exit 1
fi

# 構成のフラグは run_repro.sh が正本
declare -A FLAGS
while read -r name flags; do
    FLAGS[$name]="${flags}"
done < <(./run_repro.sh --list)

FAILED=()
for name in "${SELECTED[@]}"; do
    if [[ -z "${FLAGS[$name]+x}" ]]; then
        echo "Unknown config: ${name} (./run_repro.sh --list で一覧)"
        exit 1
    fi

    # ベスト ckpt (last.ckpt ではない). version が複数あれば最新
    ckpt=$(compgen -G "${SRC}/${name}/version_*/epoch=*.ckpt" | sort -V | tail -n 1)
    if [[ -z "${ckpt}" ]]; then
        echo "[fail] ${name}: ${SRC}/${name}/version_*/epoch=*.ckpt が無い"
        FAILED+=("${name}")
        continue
    fi

    if [[ ${SEBBS} -eq 1 ]]; then
        dest="${SRC}/${name}_sebbs"
    else
        dest="${OUT}/${name}"
    fi

    # shellcheck disable=SC2206  # FLAGS は空白区切りの引数列
    cmd=(uv run train_pretrained.py --confs "${CONFS}" --gpus "${GPUS}" ${FLAGS[$name]}
        --test_from_checkpoint "${ckpt}")
    if [[ ${SEBBS} -eq 1 ]]; then
        cmd+=(--sebbs)
    fi
    cmd+=("${EXTRA_ARGS[@]}")

    echo "=========================================="
    echo "[eval] ${name} -> ${dest}"
    echo "${cmd[*]}"
    echo "=========================================="
    if [[ ${DRY_RUN} -eq 1 ]]; then
        continue
    fi

    mkdir -p "${dest}"
    before=$(compgen -G "lightning_logs/version_*" | sort)
    "${cmd[@]}" 2>&1 | tee "${dest}/test_$(date +%Y%m%d_%H%M%S).log"
    status=$?
    if [[ ${status} -ne 0 ]]; then
        echo "[fail] ${name}: exit ${status}"
        FAILED+=("${name}")
        continue
    fi

    # この実行で増えた lightning_logs/version_N を出力先へ移す
    created=$(comm -13 <(echo "${before}") <(compgen -G "lightning_logs/version_*" | sort))
    if [[ -z "${created}" || $(echo "${created}" | wc -l) -ne 1 || ! -f "${created}/test_metrics.json" ]]; then
        echo "[fail] ${name}: 結果のディレクトリを特定できない (lightning_logs/ を確認)"
        FAILED+=("${name}")
        continue
    fi
    version=$(compgen -G "${dest}/version_*" | wc -l)
    mv "${created}" "${dest}/version_${version}"
    echo "[done] ${name}: ${dest}/version_${version}/test_metrics.json"
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed: ${FAILED[*]}"
    exit 1
fi
