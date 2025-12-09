# Requirements Document

## Project Description (Input)
このプロジェクトをDevcontainer内で安全に実装したい

## Introduction
このプロジェクトは、DCASE 2024 Task 4 Sound Event Detection (SED) システムの開発環境をDevcontainer化し、開発者が安全かつ一貫性のある環境で作業できるようにすることを目的としています。現在のプロジェクトはPython 3.11+、PyTorch Lightning、および多数の音声処理ライブラリに依存しており、これらの依存関係を適切に管理し、再現可能な開発環境を提供する必要があります。

## Requirements

### Requirement 1: Devcontainer設定ファイルの作成
**Objective:** As a 開発者, I want Devcontainer設定ファイルが自動生成される, so that 環境構築の手間を削減できる

#### Acceptance Criteria
1. The Devcontainer System shall `.devcontainer/devcontainer.json`を作成する
2. The Devcontainer System shall `.devcontainer/Dockerfile`を作成する
3. When 設定ファイルが作成される, the Devcontainer System shall プロジェクトのPython 3.11+要件を満たすベースイメージを指定する
4. The Devcontainer System shall VS Code拡張機能(Python, Ruff, MyPy)を設定に含める
5. The Devcontainer System shall ポート転送設定(TensorBoard, Optuna Dashboard用)を含める
6. The Devcontainer System shall Claude Code feature (`ghcr.io/anthropics/devcontainer-features/claude-code:1.0`)を含める
7. The Devcontainer System shall コンテナユーザーを`vscode`に設定する
8. The Devcontainer System shall 環境変数`CLAUDE_CONFIG_DIR`を`/home/vscode/.claude`に設定する

### Requirement 2: Python環境とパッケージ管理
**Objective:** As a 開発者, I want Python環境とパッケージが自動的に設定される, so that 依存関係の問題を回避できる

#### Acceptance Criteria
1. When Devcontainerが起動する, the Devcontainer System shall Python 3.12をインストールする
2. The Devcontainer System shall uvパッケージマネージャーをインストールする
3. When パッケージマネージャーがインストールされる, the Devcontainer System shall `pyproject.toml`の依存関係を自動インストールする
4. The Devcontainer System shall 開発用依存関係(mypy, ruff, pandas-stubs, types-setuptools)をインストールする
5. While コンテナが起動中, the Devcontainer System shall 仮想環境を自動的にアクティベートする

### Requirement 3: システム依存関係のインストール
**Objective:** As a 開発者, I want 音声処理に必要なシステムライブラリが自動インストールされる, so that 音声処理機能が正常に動作する

#### Acceptance Criteria
1. The Devcontainer System shall SoX(Sound eXchange)をインストールする
2. The Devcontainer System shall FFmpegをインストールする
3. The Devcontainer System shall libsndfileをインストールする
4. The Devcontainer System shall 必要なCUDA/GPU関連ライブラリをインストールする
5. When システム依存関係がインストールされる, the Devcontainer System shall インストール成功を検証する

### Requirement 4: Gitサブモジュールの自動初期化
**Objective:** As a 開発者, I want Gitサブモジュールが自動的に初期化される, so that 手動での初期化作業が不要になる

#### Acceptance Criteria
1. When Devcontainerが起動する, the Devcontainer System shall `git submodule update --init --recursive`を実行する
2. The Devcontainer System shall sebbsサブモジュールを初期化する
3. If サブモジュール初期化が失敗した, then the Devcontainer System shall エラーメッセージを表示する

### Requirement 5: 開発ツールの統合
**Objective:** As a 開発者, I want 開発ツールがVS Code内で統合される, so that 効率的に開発できる

#### Acceptance Criteria
1. The Devcontainer System shall Ruffリンターを自動設定する
2. The Devcontainer System shall MyPy型チェッカーを自動設定する
3. The Devcontainer System shall Pytestテストランナーを設定する
4. When ファイルが保存される, the Devcontainer System shall 自動フォーマットを実行する
5. The Devcontainer System shall インポートの自動整理を有効化する
6. When Devcontainerが起動完了する, the Devcontainer System shall `pre-commit install`を自動実行する

### Requirement 6: データディレクトリのマウント設定
**Objective:** As a 開発者, I want データディレクトリが適切にマウントされる, so that データの永続性が保証される

#### Acceptance Criteria
1. The Devcontainer System shall `data/`ディレクトリをボリュームマウントする
2. The Devcontainer System shall `embeddings/`ディレクトリをボリュームマウントする
3. The Devcontainer System shall `exp/`ディレクトリをボリュームマウントする
4. The Devcontainer System shall `wandb/`ディレクトリをボリュームマウントする
5. The Devcontainer System shall Claude設定用の名前付きボリューム(`claude-code-config-${devcontainerId}`)を`/home/vscode/.claude`にマウントする
6. When マウントが完了する, the Devcontainer System shall ディレクトリの書き込み権限を検証する

### Requirement 7: セキュリティ設定
**Objective:** As a 開発者, I want セキュアな開発環境が提供される, so that セキュリティリスクを最小化できる

#### Acceptance Criteria
1. The Devcontainer System shall 非rootユーザー(`vscode`)でコンテナを実行する
2. The Devcontainer System shall 最小限の権限でプロセスを実行する
3. The Devcontainer System shall 機密情報を環境変数として扱う
4. If 機密ファイル(.env, credentials.json)が存在する, then the Devcontainer System shall それらをコンテナ外に配置する
5. The Devcontainer System shall ネットワークアクセスを必要最小限に制限する
6. The Devcontainer System shall Claude Codeのファイルアクセスをプロジェクトディレクトリ配下に制限する

### Requirement 8: 実験環境の再現性
**Objective:** As a 研究者, I want 実験環境が完全に再現可能である, so that 実験結果の信頼性が保証される

#### Acceptance Criteria
1. The Devcontainer System shall Pythonバージョンを固定する(3.12)
2. The Devcontainer System shall パッケージバージョンを`pyproject.toml`に従って固定する
3. The Devcontainer System shall PyTorch Lightning 1.9.xを維持する
4. When 環境が構築される, the Devcontainer System shall 依存関係のハッシュを検証する
5. The Devcontainer System shall ビルド時のタイムスタンプを記録する

### Requirement 9: パフォーマンスとリソース管理
**Objective:** As a 開発者, I want コンテナリソースが適切に管理される, so that 開発作業がスムーズに行える

#### Acceptance Criteria
1. The Devcontainer System shall GPUアクセスを有効化する
2. The Devcontainer System shall 適切なメモリ制限を設定する
3. The Devcontainer System shall CPUコア数を設定可能にする
4. When リソース不足が発生する, the Devcontainer System shall 警告を表示する
5. The Devcontainer System shall ディスクI/Oを最適化する(キャッシュ設定)

### Requirement 10: ドキュメントとガイド
**Objective:** As a 新規開発者, I want Devcontainerの使用方法が明確に文書化される, so that スムーズに開発を開始できる

#### Acceptance Criteria
1. The Devcontainer System shall README.mdにDevcontainerセットアップ手順を記載する
2. The Devcontainer System shall トラブルシューティングガイドを提供する
3. The Devcontainer System shall よくある質問(FAQ)セクションを含める
4. When 設定ファイルが変更される, the Devcontainer System shall 変更内容をドキュメントに反映する
5. The Devcontainer System shall コメント付きの設定例を提供する
