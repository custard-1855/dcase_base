# 実装タスク

## タスク一覧

### Phase 1: インフラストラクチャ設定ファイルの作成

- [x] 1. Devcontainer設定ファイルの作成
- [x] 1.1 (P) `.devcontainer/devcontainer.json`を作成する
  - VS Code Dev Containersの統合設定ファイルを作成
  - Claude Code feature (`ghcr.io/anthropics/devcontainer-features/claude-code:1`)を統合
  - VS Code拡張機能の自動インストール設定（Python、Ruff、MyPy、Pylance）を追加
  - ポート転送設定（6006: TensorBoard、8080: Optuna Dashboard）を追加
  - 環境変数`CLAUDE_CONFIG_DIR`を`/home/vscode/.claude`に設定
  - リモートユーザーを`vscode`に設定
  - 5つの名前付きボリュームマウント設定（data、embeddings、exp、wandb、claude-code-config）を追加
  - `postCreateCommand`スクリプトを定義（uv sync、gh auth setup-git、git submodule update、pre-commit install）
  - 保存時自動フォーマット、インポート整理、Python interpreter path設定を含める
  - _Requirements: 1.1, 1.4, 1.5, 1.6, 1.7, 1.8, 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.3, 9.2, 9.3_

- [x] 1.2 (P) `.devcontainer/Dockerfile`を作成する
  - Multi-stage Dockerfileを実装（Stage 1: uv、Stage 2: Base + SystemLibs、Stage 3: Dependencies）
  - Stage 1: `ghcr.io/astral-sh/uv:0.9.16`からuvバイナリをコピー
  - Stage 2: `python:3.12-slim-bookworm`をベースイメージとして使用
  - Stage 2: SoX、FFmpeg、libsndfile1-dev、git、GitHub CLIをAPT経由でインストール
  - Stage 2: `uv python install 3.12`でPython 3.12をインストール
  - Stage 2: `useradd -m -s /bin/bash vscode`で非rootユーザーを作成
  - Stage 2: `LABEL build_timestamp`でビルド時のタイムスタンプを記録
  - Stage 3: `pyproject.toml`と`uv.lock`をコピーし、`uv sync --frozen --all-groups`で依存関係をインストール
  - Stage 3: `USER vscode`で非rootユーザーに切り替え
  - uvキャッシュマウント（`--mount=type=cache,target=/root/.cache/uv`）を使用してビルド時間を最適化
  - _Requirements: 1.2, 1.3, 1.7, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.5, 7.1, 7.2, 7.4, 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 1.3 (P) `.dockerignore`ファイルを作成する
  - `.env`、`credentials.json`、`.ssh/`、`.aws/`等の機密ファイルをDocker build contextから除外
  - `data/`、`embeddings/`、`exp/`、`wandb/`等の大容量ディレクトリを除外（ボリュームマウントで管理）
  - `.git/`、`__pycache__/`、`*.pyc`、`.DS_Store`等の不要ファイルを除外
  - ビルド時間短縮とセキュリティ向上を実現
  - _Requirements: 7.4_

### Phase 2: 起動時初期化スクリプトの実装

- [x] 2. postCreateCommandスクリプトの実装
- [x] 2.1 postCreateCommand統合スクリプトを作成する
  - `uv sync`を実行し、ワークスペースマウント後の依存関係インストールを完了
  - `gh auth setup-git`を実行し、GitHub CLI credential helperを設定（事前にgh auth login実行前提）
  - `git submodule update --init --recursive`を実行し、sebbsサブモジュールを初期化
  - 注: PSDS_EvalはPyPIパッケージ（psds-eval）として管理されており、submoduleではない
  - `pre-commit install`を実行し、Git hooksを設定
  - システムライブラリ検証（`which sox ffmpeg git gh`）を実行
  - ディレクトリ書き込み権限検証（data、embeddings、exp、wandbへのtouch/rmテスト）を実行
  - リソース不足チェック（メモリ < 2048 MB、ディスク < 10 GB）を実行し、警告を表示
  - 各コマンド失敗時もコンテナ起動継続（エラーメッセージ表示）
  - `devcontainer.json`の`postCreateCommand`フィールドでスクリプトを参照
  - _Requirements: 2.3, 2.5, 3.5, 4.1, 4.2, 4.3, 5.6, 6.6, 9.4_

### Phase 3: ドキュメント作成

- [x] 3. ドキュメント作成
- [x] 3.1 (P) README.mdにDevcontainerセクションを追加する
  - Devcontainer概要（目的、利点）を記載
  - 事前準備（Docker Desktop 20.10+インストール、gh auth login実行）を記載
  - セットアップ手順（VS Codeで"Reopen in Container"実行）を記載
  - 基本的な使用方法（訓練、推論、可視化コマンド例）を記載
  - トラブルシューティングガイドへのリンク（`DEVCONTAINER_GUIDE.md`）を追加
  - _Requirements: 10.1_

- [x] 3.2 (P) DEVCONTAINER_GUIDE.mdを作成する
  - 詳細セットアップ手順（事前準備の詳細、gh auth loginの重要性、wandb APIキー取得方法（オプショナル））を記載
  - トラブルシューティングガイド（システムライブラリエラー、submodule初期化失敗、ボリューム権限問題、gh auth login未実行エラー）を記載
  - FAQ（GPU対応、wandb使用方法、ディスク容量管理、docker volume prune手順）を記載
  - `devcontainer.json`と`Dockerfile`のコメント付き解説を提供
  - リソース不足時の対処法（メモリ制限増加、バッチサイズ削減、ボリューム削除）を記載
  - _Requirements: 10.2, 10.3, 10.4, 10.5_

### Phase 4: 統合テストと検証

- [x] 4. Devcontainer統合テストの実行
- [x] 4.1 ビルド検証テストを実行する
  - `docker pull python:3.12-slim-bookworm`でベースイメージ取得成功を確認
  - Dockerfile内でシステムライブラリインストール成功を検証（`which sox ffmpeg git gh && dpkg -l | grep libsndfile`）
  - Dockerfile内でuv正常動作を検証（`uv --version && uv python list`）
  - Dockerfile内でPython依存関係完全インストールを検証（`uv sync --frozen && uv pip list | grep pytorch-lightning`）
  - 各検証失敗時の復旧手順をドキュメントに反映
  - _Requirements: 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 8.1, 8.2, 8.3_

- [x] 4.2 Devcontainer起動テストを実行する
  - VS Codeで"Reopen in Container"を実行し、コンテナ起動成功を確認
  - postCreateCommand完全実行を確認（"postCreateCommand completed"表示）
  - Git submodule初期化成功を確認（`ls -la sebbs`で内容存在確認、実際のサブモジュールはsebbsのみ）
  - VS Code拡張インストール完了を確認（Python、Ruff、MyPy、Pylance）
  - 名前付きボリューム書き込み可能性を確認（`touch /workspace/data/test.txt`等）
  - Claude Code CLI動作確認（`claude --version`）
  - 自動テストスクリプト作成（`tests/test_devcontainer_startup.sh`）
  - テストドキュメント作成（`tests/README_TEST_DEVCONTAINER.md`）
  - _Requirements: 1.6, 4.1, 4.2, 5.1, 5.2, 5.3, 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 4.3 機能テストを実行する
  - PyTorch Lightning訓練スクリプト実行可能性を確認（小規模データセットでCPU訓練開始確認）
  - Pythonファイル保存時自動フォーマット実行を確認（Ruff動作確認）
  - MyPy型チェック実行を確認（VS Code Problems paneで型エラー表示確認）
  - pre-commitフック実行を確認（`git commit -m "test"`でフック動作確認）
  - ポート転送動作確認（TensorBoard、Optuna Dashboard）
  - _Requirements: 1.5, 5.4, 5.5, 5.6_

### Phase 5: セキュリティ検証

- [x] 5. セキュリティ設定の検証
- [x] 5.1 Claude Codeファイルアクセス制限を検証する
  - Claude Codeが/workspace配下のみアクセス可能であることを確認
  - Claude Codeが/home/vscode/.sshにアクセス不可であることを確認
  - Claude Codeがホスト/homeにアクセス不可であることを確認
  - セキュリティ分離がClaude Code feature側で実装済みであることを確認（追加設定不要）
  - テストスクリプト作成: `tests/test_security_claude_code_access.py`
  - 検証レポート作成: `.devcontainer/SECURITY_VERIFICATION.md`
  - _Requirements: 7.6_

- [x] 5.2 機密情報管理を検証する
  - `.dockerignore`に機密ファイル（.env、credentials.json、.ssh/、.aws/）が記載されていることを確認
  - Dockerfile内でENV経由の機密情報埋め込みが行われていないことを確認
  - devcontainer.json `remoteEnv`でwandb APIキーがコメントアウトされていることを確認（オプショナル設定）
  - 非rootユーザー（vscode）でプロセス実行されていることを確認
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

### Phase 6: リソース管理とパフォーマンス検証

- [x] 6. リソース管理の検証
- [x] 6.1 メモリ・ディスク容量管理を検証する
  - postCreateCommand時のリソース不足チェック動作を確認（メモリ < 2048 MB警告）
  - ディスク容量不足警告動作を確認（df -h出力チェック）
  - Docker Desktop設定でメモリ制限増加方法をドキュメントに記載
  - docker volume prune手順をドキュメントに記載
  - _Requirements: 9.2, 9.3, 9.4, 9.5_

- [x] 6.2 ビルド時間最適化を検証する
  - Multi-stage Dockerfileキャッシュヒット効果を確認（初回約5-7分、2回目以降約1-2分）
  - uvキャッシュマウント効果を確認（Python依存関係インストール時間短縮）
  - APTキャッシュ有効化効果を確認（システムライブラリインストール時間短縮）
  - ビルド時間計測結果をドキュメントに記載
  - _Requirements: 9.5_

## タスク実行時の注意事項

### 並列実行可能タスク（(P)マーク）
- Phase 1（1.1、1.2、1.3）: 独立した設定ファイル作成のため並列実行可能
- Phase 3（3.1、3.2）: 独立したドキュメント作成のため並列実行可能

### 順次実行必須タスク
- Phase 2（2.1）: Phase 1完了後に実行（devcontainer.json依存）
- Phase 4（4.1-4.3）: Phase 1-3完了後に実行（統合テスト）
- Phase 5（5.1-5.2）: Phase 4完了後に実行（セキュリティ検証）
- Phase 6（6.1-6.2）: Phase 4完了後に実行（リソース検証）

### 重要な前提条件
- **gh auth login事前実行**: Git submodule初期化成功のため、ホスト環境で事前に`gh auth login`を実行する必要がある
- **Docker Desktop**: 20.10+がホストにインストール済みである必要がある
- **VS Code**: Dev Containers拡張がインストール済みである必要がある

### 品質チェックリスト
- [x] すべての要件（1.1-10.5）がタスクにマッピングされている
- [x] タスク依存関係が明確に定義されている
- [x] テストタスク（Phase 4-6）が含まれている
- [x] ドキュメント作成タスク（Phase 3）が含まれている
- [x] セキュリティ検証タスク（Phase 5）が含まれている

## タスク完了基準

### Phase 1完了基準
- `.devcontainer/devcontainer.json`、`.devcontainer/Dockerfile`、`.dockerignore`が作成済み
- 各ファイルが要件をすべて満たしている

### Phase 2完了基準
- postCreateCommandスクリプトが実装済み
- 各コマンド（uv sync、gh auth setup-git、git submodule、pre-commit install、検証スクリプト）が正常動作する

### Phase 3完了基準
- README.mdにDevcontainerセクション追加済み
- DEVCONTAINER_GUIDE.md作成済み（詳細手順、トラブルシューティング、FAQ含む）

### Phase 4完了基準
- ビルド検証テスト成功
- Devcontainer起動テスト成功
- 機能テスト成功

### Phase 5完了基準
- Claude Codeファイルアクセス制限検証完了
- 機密情報管理検証完了

### Phase 6完了基準
- メモリ・ディスク容量管理検証完了
- ビルド時間最適化検証完了
