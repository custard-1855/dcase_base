# Devcontainer起動テスト (Task 4.2)

## 概要

Task 4.2「Devcontainer起動テスト」の自動検証スクリプトと手動検証手順を記載します。

## 自動テスト

### 実行方法

```bash
bash tests/test_devcontainer_startup.sh
```

### 検証項目（自動）

1. **Git submodule初期化成功確認**
   - sebbsサブモジュールが初期化され、内容が展開されていることを確認

2. **名前付きボリューム書き込み可能性確認**
   - data/, embeddings/, exp/, wandb/ディレクトリへの書き込み権限を確認

3. **Claude Code CLI動作確認**
   - Claude Code CLIがインストールされ、バージョン確認可能であることを確認

4. **システムライブラリ存在確認**
   - sox, ffmpeg, git, ghがインストールされていることを確認（postCreateCommand実行確認の代替）

## 手動テスト（Devcontainer内でのみ実行可能）

以下の検証項目は、VS Code Dev Container内でのみ実行可能です。

### 1. VS Code拡張インストール確認

VS Code Dev Container起動後、以下を確認してください:

1. VS Codeの拡張タブを開く
2. 以下の拡張がインストール済みであることを確認:
   - `ms-python.python` (Python)
   - `charliermarsh.ruff` (Ruff)
   - `ms-python.mypy-type-checker` (MyPy Type Checker)
   - `ms-python.vscode-pylance` (Pylance)

### 2. pre-commit hooks設定確認

VS Code Dev Container内のターミナルで以下を実行:

```bash
# pre-commitコマンドが利用可能であることを確認
pre-commit --version

# pre-commit hooksがインストールされていることを確認
ls -la .git/hooks/pre-commit
```

### 3. postCreateCommand完全実行確認

VS Code Dev Container起動時のターミナル出力で、以下のメッセージが表示されていることを確認:

```
postCreateCommand completed
```

### 4. 保存時自動フォーマット動作確認

VS Code Dev Container内で以下を確認:

1. Pythonファイルを開く（例: `DESED_task/dcase2024_task4_baseline/local/sed_trainer_pretrained.py`）
2. コードを編集して保存（Cmd+S / Ctrl+S）
3. Ruffによる自動フォーマットが実行されることを確認

### 5. ポート転送動作確認

TensorBoardとOptuna Dashboardのポート転送が機能していることを確認:

```bash
# TensorBoard起動（例）
tensorboard --logdir=exp/

# ブラウザでlocalhost:6006にアクセス可能であることを確認
```

## テスト結果の記録

自動テストが全てパスした場合、以下のサマリーが表示されます:

```
==========================================
テスト結果サマリー
==========================================
✓ ALL TESTS PASSED
Task 4.2: Devcontainer起動テスト成功
```

## トラブルシューティング

### Git submodule初期化失敗

**症状**: `✗ sebbs submodule NOT initialized or empty`

**原因**: Git submoduleが初期化されていない

**対処法**:
```bash
git submodule update --init --recursive
```

### 名前付きボリューム書き込み不可

**症状**: `✗ {directory} directory is NOT writable`

**原因**: ボリュームマウント失敗または権限問題

**対処法**:
```bash
# ボリューム状態確認
docker volume ls
docker volume inspect dcase-data

# ボリューム再作成（データ消失に注意）
docker volume rm dcase-data
# devcontainer再起動でボリューム自動作成
```

### Claude Code CLI未インストール

**症状**: `✗ Claude Code CLI is NOT installed`

**原因**: Claude Code featureのインストール失敗

**対処法**:
1. devcontainer.jsonのfeaturesセクション確認
2. devcontainer再ビルド（VS Code Command Palette → "Dev Containers: Rebuild Container"）

### システムライブラリ未インストール

**症状**: `✗ {library} is NOT installed`

**原因**: Dockerfile内のapt-get install失敗

**対処法**:
1. Dockerfile確認（.devcontainer/Dockerfile）
2. devcontainer再ビルド（`--no-cache`オプション推奨）

## 関連ドキュメント

- [DEVCONTAINER_GUIDE.md](../DESED_task/dcase2024_task4_baseline/DEVCONTAINER_GUIDE.md) - 詳細なトラブルシューティングガイド
- [tasks.md](../.kiro/specs/devcontainer-safe-impl/tasks.md) - タスク一覧
- [design.md](../.kiro/specs/devcontainer-safe-impl/design.md) - 技術設計書

## 要件トレーサビリティ

このテストは以下の要件をカバーしています:
- 要件1.6: Claude Code feature統合
- 要件4.1-4.2: Git submodule初期化
- 要件5.1-5.3: VS Code拡張インストール
- 要件6.1-6.5: 名前付きボリュームマウント

詳細は[requirements.md](../.kiro/specs/devcontainer-safe-impl/requirements.md)を参照してください。
