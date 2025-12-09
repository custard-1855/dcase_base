#!/bin/bash
################################################################################
# Kiro Spec 自動実装スクリプト
#
# 機能: .kiro/specs/ 内の有効な仕様に対して、未完了タスクがなくなるまで
#       /impl コマンドをループ実行し、各タスク完了後にコミットする
#
# 前提条件:
#   - .kiro/specs/ に archives 以外のディレクトリが1つだけ存在
#   - spec.json の ready_for_implementation が true
#   - tasks.md が存在
#
# 使い方:
#   .kiro/auto_impl.sh
################################################################################

set -euo pipefail

# ============================================================================
# 設定
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SPECS_DIR="${PROJECT_DIR}/.kiro/specs"
MAX_ITERATIONS=100

# 検出されたフィーチャー名（グローバル）
FEATURE_NAME=""
FEATURE_DIR=""

# ============================================================================
# シグナルハンドラー（Ctrl+C 対応）
# ============================================================================

cleanup() {
    echo ""
    echo "Interrupted by user (Ctrl+C)"
    exit 130
}

trap cleanup SIGINT SIGTERM

# ============================================================================
# エラー終了
# ============================================================================

error_exit() {
    echo "ERROR: $1"
    exit 1
}

# ============================================================================
# フィーチャー検出: archives 以外のディレクトリを検出
# ============================================================================

detect_active_feature() {
    local dirs=()

    # .kiro/specs/ 内のディレクトリを取得（archives を除外）
    while IFS= read -r -d '' dir; do
        local name=$(basename "$dir")
        if [ "$name" != "archives" ]; then
            dirs+=("$name")
        fi
    done < <(find "$SPECS_DIR" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)

    # ディレクトリ数をチェック
    if [ ${#dirs[@]} -eq 0 ]; then
        error_exit "No active feature found in $SPECS_DIR (excluding archives)"
    fi

    if [ ${#dirs[@]} -gt 1 ]; then
        error_exit "Multiple features found: ${dirs[*]}. Expected exactly one (excluding archives)"
    fi

    FEATURE_NAME="${dirs[0]}"
    FEATURE_DIR="${SPECS_DIR}/${FEATURE_NAME}"

    echo "Detected feature: $FEATURE_NAME"
}

# ============================================================================
# 仕様バリデーション: spec.json の ready_for_implementation チェック
# ============================================================================

validate_spec() {
    local spec_file="${FEATURE_DIR}/spec.json"

    if [ ! -f "$spec_file" ]; then
        error_exit "spec.json not found at $spec_file"
    fi

    # jq があれば使用、なければ grep でフォールバック
    local ready="false"
    if command -v jq &> /dev/null; then
        ready=$(jq -r '.ready_for_implementation // false' "$spec_file")
    else
        if grep -q '"ready_for_implementation"[[:space:]]*:[[:space:]]*true' "$spec_file"; then
            ready="true"
        fi
    fi

    if [ "$ready" != "true" ]; then
        error_exit "Feature '$FEATURE_NAME' is not ready for implementation (ready_for_implementation != true)"
    fi

    echo "Spec validation passed: ready_for_implementation = true"
}

# ============================================================================
# タスクファイル存在チェック
# ============================================================================

validate_tasks_file() {
    local tasks_file="${FEATURE_DIR}/tasks.md"

    if [ ! -f "$tasks_file" ]; then
        error_exit "tasks.md not found at $tasks_file"
    fi

    echo "Tasks file found: $tasks_file"
}

# ============================================================================
# 未完了タスクチェック
# ============================================================================

has_pending_tasks() {
    local tasks_file="${FEATURE_DIR}/tasks.md"

    # "- [ ]" で始まる行をカウント（未完了タスク）
    local count=$(grep -c '^- \[ \]' "$tasks_file" 2>/dev/null || echo "0")

    if [ "$count" -gt 0 ]; then
        return 0  # 未完了タスクあり
    else
        return 1  # 未完了タスクなし
    fi
}

# ============================================================================
# 未完了タスク数を取得
# ============================================================================

get_pending_task_count() {
    local tasks_file="${FEATURE_DIR}/tasks.md"
    grep -c '^- \[ \]' "$tasks_file" 2>/dev/null || echo "0"
}

# ============================================================================
# /impl コマンド実行
# ============================================================================

run_impl() {
    local iteration=$1

    echo ""
    echo "=== Iteration $iteration: Running /impl ==="

    # claude コマンド実行
    if ! claude -p '/impl' --dangerously-skip-permissions; then
        echo "ERROR: /impl failed at iteration $iteration"
        return 1
    fi

    echo "/impl completed successfully"
    return 0
}

# ============================================================================
# コミット実行（テスト・Lint・型チェック確認後）
# ============================================================================

run_commit() {
    local iteration=$1

    echo ""
    echo "=== Iteration $iteration: Running commit check ==="

    # テスト・Lint・型チェック確認後にコミット
    if ! claude -p 'テストやLint、型チェックでエラーが出ないことを確認したら未コミットの内容をコミットする' --dangerously-skip-permissions; then
        echo "ERROR: Commit check failed at iteration $iteration"
        return 1
    fi

    echo "Commit completed successfully"
    return 0
}

# ============================================================================
# メインワークフロー
# ============================================================================

run_workflow() {
    echo "Starting auto-impl workflow"
    echo ""

    # 前提条件チェック
    detect_active_feature
    validate_spec
    validate_tasks_file

    local iteration=0
    local initial_count=$(get_pending_task_count)

    echo ""
    echo "Initial pending tasks: $initial_count"

    # メインループ
    while has_pending_tasks; do
        iteration=$((iteration + 1))

        # 無限ループ防止
        if [ $iteration -gt $MAX_ITERATIONS ]; then
            error_exit "Maximum iterations ($MAX_ITERATIONS) reached. Stopping."
        fi

        local remaining=$(get_pending_task_count)
        echo ""
        echo "=========================================="
        echo "Iteration $iteration (remaining: $remaining tasks)"
        echo "=========================================="

        # /impl 実行
        if ! run_impl "$iteration"; then
            error_exit "/impl failed. Manual intervention required."
        fi

        # コミット実行
        if ! run_commit "$iteration"; then
            error_exit "Commit failed. Manual intervention required."
        fi

        # API制限対策で少し待機
        sleep 3
    done

    echo ""
    echo "=========================================="
    echo "All tasks completed!"
    echo "Total iterations: $iteration"
    echo "=========================================="
}

# ============================================================================
# メイン処理
# ============================================================================

main() {
    run_workflow && exit 0 || exit 1
}

main "$@"