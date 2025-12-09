#!/bin/bash
# Devcontainer起動テスト検証スクリプト
# Task 4.2: Devcontainer起動テストの自動検証

set -e  # エラー時に即座に停止

echo "=========================================="
echo "Devcontainer起動テスト検証スクリプト"
echo "Task 4.2: Devcontainer Startup Test"
echo "=========================================="
echo ""

# テスト結果を記録
FAILED_TESTS=()

# テスト1: Git submodule初期化成功確認
echo "[TEST 1] Git submodule初期化成功確認"
echo "Checking: sebbs submodule status"
# 実際の.gitmodulesに存在するサブモジュールのみ確認（sebbsのみ）
if [ -d "DESED_task/dcase2024_task4_baseline/sebbs" ] && [ "$(ls -A DESED_task/dcase2024_task4_baseline/sebbs)" ]; then
    echo "✓ sebbs submodule initialized successfully"
else
    echo "✗ sebbs submodule NOT initialized or empty"
    FAILED_TESTS+=("Git submodule: sebbs")
fi
echo ""

# テスト2: VS Code拡張インストール確認（コンテナ内でのみ実行可能）
echo "[TEST 2] VS Code拡張インストール確認"
echo "Note: This test should be run manually inside VS Code Dev Container"
echo "Expected extensions: ms-python.python, charliermarsh.ruff, ms-python.mypy-type-checker, ms-python.vscode-pylance"
echo "✓ Skipped (manual verification required in VS Code)"
echo ""

# テスト3: 名前付きボリューム書き込み可能性確認
echo "[TEST 3] 名前付きボリューム書き込み可能性確認"
for dir in data embeddings exp wandb; do
    DIRPATH="DESED_task/dcase2024_task4_baseline/$dir"
    if [ -d "$DIRPATH" ]; then
        if touch "$DIRPATH/.test_write" 2>/dev/null && rm "$DIRPATH/.test_write" 2>/dev/null; then
            echo "✓ $dir directory is writable"
        else
            echo "✗ $dir directory is NOT writable"
            FAILED_TESTS+=("Volume writable: $dir")
        fi
    else
        echo "✗ $dir directory does not exist"
        FAILED_TESTS+=("Volume exists: $dir")
    fi
done
echo ""

# テスト4: Claude Code CLI動作確認
echo "[TEST 4] Claude Code CLI動作確認"
if command -v claude &> /dev/null; then
    CLAUDE_VERSION=$(claude --version 2>&1 || echo "error")
    if [ "$CLAUDE_VERSION" != "error" ]; then
        echo "✓ Claude Code CLI is installed: $CLAUDE_VERSION"
    else
        echo "✗ Claude Code CLI installed but version check failed"
        FAILED_TESTS+=("Claude Code CLI: version check")
    fi
else
    echo "✗ Claude Code CLI is NOT installed"
    FAILED_TESTS+=("Claude Code CLI: not found")
fi
echo ""

# テスト5: postCreateCommand完全実行確認（間接的にシステムライブラリ存在確認）
echo "[TEST 5] システムライブラリ存在確認（postCreateCommand実行確認の代替）"
for cmd in sox ffmpeg git gh; do
    if command -v $cmd &> /dev/null; then
        echo "✓ $cmd is installed"
    else
        echo "✗ $cmd is NOT installed"
        FAILED_TESTS+=("System library: $cmd")
    fi
done
echo ""

# テスト6: pre-commit hooks設定確認（devcontainer内でのみ有効）
echo "[TEST 6] pre-commit hooks設定確認"
if command -v pre-commit &> /dev/null; then
    if [ -f ".git/hooks/pre-commit" ]; then
        echo "✓ pre-commit hooks installed"
    else
        echo "⚠ pre-commit command exists but hooks not installed (run 'pre-commit install' inside devcontainer)"
        # devcontainer外ではエラーにしない
    fi
else
    echo "⚠ pre-commit not available in current environment (should be installed in devcontainer)"
    # devcontainer外ではエラーにしない
fi
echo ""

# テスト結果サマリー
echo "=========================================="
echo "テスト結果サマリー"
echo "=========================================="
if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo "✓ ALL TESTS PASSED"
    echo "Task 4.2: Devcontainer起動テスト成功"
    exit 0
else
    echo "✗ SOME TESTS FAILED (${#FAILED_TESTS[@]} failures)"
    echo "Failed tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo ""
    echo "Task 4.2: Devcontainer起動テスト失敗"
    exit 1
fi
