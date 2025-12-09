# 研究 & 設計決定ログ

## サマリー
- **Feature**: `devcontainer-safe-impl`
- **Discovery Scope**: Complex Integration（新規Devcontainer作成、マルチレイヤー依存関係、CPU専用環境、セキュリティ考慮）
- **主要な発見事項**:
  1. 既存の.devcontainerは存在せず、完全に新規作成が必要
  2. Python 3.12固定、PyTorch Lightning 1.9.x固定、uvパッケージマネージャー使用が前提
  3. Claude Code公式featureを使用することで、セキュリティ分離とNode.js依存関係が自動管理される
  4. **CPU専用環境採用**: Mac環境での迅速なビルドを優先、python:3.12-slim-bookwormベースイメージ使用
  5. **GitHub CLI (gh) 認証**: gh auth loginによりSSH鍵不要でsubmodule初期化可能
  6. **wandb APIキーは不要**: ローカル開発自動化のみ、後から追加可能

## 研究ログ

### プロジェクト現状分析
- **コンテキスト**: 既存のDCASE 2024 Task 4 SEDプロジェクトのDevcontainer化要件
- **調査対象**: pyproject.toml, .python-version, ディレクトリ構造
- **発見事項**:
  - Python 3.12が.python-versionで指定済み
  - pyproject.tomlではrequires-python = ">=3.11"（3.12で互換性あり）
  - PyTorch Lightning 1.9.*に固定（再現性重視、2.x系は使用しない）
  - 開発依存関係: mypy, ruff, pandas-stubs, types-setuptools
  - 音声処理ライブラリ: sox, soxbindings, resampy, torchaudio (>=2.8.0)
  - 実験管理: wandb (>=0.22.3), tensorboard (>=2.20.0), optuna (>=4.6.0)
  - .devcontainerディレクトリは存在しない（新規作成）
- **影響**:
  - Dockerfileでは音声処理用システムライブラリ（SoX, FFmpeg, libsndfile）のインストールが必須
  - uvを使ったpyproject.toml準拠のインストールフローが必要
  - PyTorch Lightning 1.9.xとの互換性を保つPyTorchバージョン選定が重要

### Devcontainer ベストプラクティス（Python ML/PyTorch）
- **コンテキスト**: 2025年現在のML向けDevcontainer構築標準の調査
- **ソース**:
  - PyTorch公式.devcontainer/README.md
  - Red Hat Emerging Technologies ガイド
  - Medium記事（Immutable development environments for PyTorch）
- **発見事項**:
  - **Base Image推奨**: NVIDIA CUDA base images (例: `nvidia/cuda:12.1.0-base-ubuntu22.04`)またはそれ以降
  - **GPU設定**: ホストにNVIDIA DriverとNVIDIA Container Toolkitのみ必要（CUDA Toolkit不要）
  - **runArgs**: `["--gpus", "all"]`でGPU全体を公開
  - **再現性**: 依存関係のバージョン固定（例: torch==2.5.0）、タイムスタンプ記録
  - **Python 3.12サポート**: 2025年時点でAI開発の標準バージョン（JITパフォーマンス向上）
  - **Extensions自動インストール**: devcontainer.json内customizationsで指定（Python, Jupyter, Pylance等）
- **影響**:
  - Dockerfileベースイメージは`nvidia/cuda`系を採用
  - devcontainer.jsonにrunArgsとGPU関連環境変数を設定
  - 要件1.6で指定されたPython, Ruff, MyPy拡張機能をcustomizationsに記載

### Claude Code統合
- **コンテキスト**: Anthropic公式のClaude Code devcontainer feature統合方法
- **ソース**:
  - https://docs.claude.com/en/docs/claude-code/devcontainer
  - https://github.com/anthropics/devcontainer-features
  - DEV Community記事: Running Claude Code inside your dev containers
- **発見事項**:
  - **公式Feature**: `ghcr.io/anthropics/devcontainer-features/claude-code:1`
  - **Node.js依存**: Claude CodeはNode.js/npm必須（featureが自動インストール試行、Debian/Ubuntu/Alpine/Fedora/RHEL/CentOS対応）
  - **セキュリティ分離**: Claude Codeは/workspace（プロジェクト）とマウント済みディレクトリのみアクセス可能、ホストのSSH鍵やブラウザデータは非アクセス
  - **環境変数**: `CLAUDE_CONFIG_DIR=/home/vscode/.claude`設定推奨
  - **名前付きボリューム**: `claude-code-config-${devcontainerId}`を`/home/vscode/.claude`にマウント（設定永続化）
  - **remoteUser**: `vscode`ユーザーで接続（非root）
- **影響**:
  - 要件1.6, 1.8のClaude Code feature設定とCLAUDE_CONFIG_DIR環境変数設定が明確化
  - 要件6.5の名前付きボリュームマウント仕様が確定
  - 要件7.6のファイルアクセス制限はClaude Code側で実装済み（追加設定不要）

### uvパッケージマネージャー × Docker/Devcontainer
- **コンテキスト**: 2025年時点のuvを使ったDocker/Devcontainer構築標準
- **ソース**:
  - https://docs.astral.sh/uv/guides/integration/docker/（公式ドキュメント）
  - Depot: Optimal Dockerfile for Python with uv
  - GitHub: a5chin/python-uv, mjun0812/python-project-template
- **発見事項**:
  - **uvバージョン固定**: `COPY --from=ghcr.io/astral-sh/uv:0.9.16 /usr/local/bin/uv /usr/local/bin`
  - **マルチステージビルド**: 依存関係レイヤー分離（`uv sync --no-install-project`で推移的依存のみ先行インストール）
  - **キャッシュマウント**: `RUN --mount=type=cache,target=/root/.cache/uv`でビルド高速化
  - **Python version install**: `uv python install 3.12`でコンテナ内Python環境構築
  - **VSCode bind-mount問題**: VSCodeが/workspaces/<project>にバインドマウントするため、postCreateCommand等で`uv sync`再実行が必要
  - **Modern template**: uv + ruff + devcontainer + Claude Code統合テンプレートが存在（参考実装）
- **影響**:
  - Dockerfileはマルチステージ構成（uv layer + 依存関係 layer + プロジェクトlayer）
  - 要件2.2, 2.3のuvインストールとpyproject.toml自動インストールの実装方針が明確化
  - postCreateCommandで`uv sync`実行を追加（要件2.5の仮想環境アクティベート対応）

### GPU/CUDA/非rootユーザーセキュリティ
- **コンテキスト**: GPU有効化と非rootユーザー実行の両立
- **ソース**:
  - NVIDIA Containers User Guide
  - Stack Overflow: Using GPU from a docker container
  - Chainguard pytorch image overview
- **発見事項**:
  - **ホスト要件**: NVIDIA DriverとNVIDIA Container Toolkitのみ（CUDA Toolkit不要）
  - **runArgs**: `["--gpus", "all"]`または`["--gpus", "device=0,1"]`でGPU指定
  - **環境変数**: `NVIDIA_VISIBLE_DEVICES=all`, `CUDA_VISIBLE_DEVICES=all`設定
  - **非rootユーザー**: devcontainer.jsonの`remoteUser: "vscode"`で非rootユーザー接続
  - **User ID mapping**: `-u $(id -u):$(id -g)`はwarningが出る可能性あり、remoteUserオプション使用推奨
  - **セキュリティ強化**: Chainguard imagesは/home/nonrootで非rootユーザー実行、sudo削除オプション
  - **権限最小化**: GPUアクセスはNVIDIA Container Toolkit経由で制御、直接デバイスアクセス不要
- **影響**:
  - 要件3.4のCUDA/GPU関連ライブラリはベースイメージ（nvidia/cuda）に含まれる
  - 要件7.1, 7.2の非rootユーザー実行はremoteUser設定で実現
  - 要件9.1のGPUアクセス有効化はrunArgsと環境変数で実現

### Devcontainer Features
- **コンテキスト**: 公式devcontainer featuresの調査と選定
- **ソース**: https://containers.dev/features
- **発見事項**:
  - **Python feature**: `ghcr.io/devcontainers/features/python:1`でバージョン指定可能
  - **Docker-outside-of-docker**: `ghcr.io/devcontainers/features/docker-outside-of-docker:1`でホストDocker利用
  - **Git with LFS**: `ghcr.io/devcontainers/features/git-lfs:1`
  - **Node.js**: Claude Code依存、feature自動インストールまたは明示的追加
  - **設定形式**: featuresセクションに`"ghcr.io/.../feature:version": { options }`形式で記載
- **影響**:
  - Python featureは使用せず、Dockerfile内でuv経由のPython 3.12インストールを採用（より細かい制御）
  - Git submodule要件（要件4）のため、Git LFS featureは不要（標準gitで対応可能）
  - Claude Code feature以外は最小限に留める（シンプル性重視）

## アーキテクチャパターン評価

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| Simple Dockerfile | 単一Dockerfileで全依存関係をインストール | シンプル、理解しやすい | ビルド時間長い、キャッシュ効率悪い | 小規模プロジェクト向き |
| Multi-stage Dockerfile | uvコピー → 依存関係 → プロジェクトの段階的ビルド | ビルド高速化、レイヤーキャッシュ効率良 | 設定が複雑、デバッグ難しい | **推奨**: ML環境で標準的 |
| Feature-based | Python feature等を使用 | 設定簡潔、メンテナンス容易 | uvとの統合不明確、カスタマイズ制約 | 今回は採用しない |
| Docker Compose | 複数サービス分離（DB, App等） | マイクロサービス対応 | 単一プロジェクトには過剰、複雑性増 | 今回は不要 |

**選定**: **Multi-stage Dockerfile** + **Claude Code feature** + **devcontainer.json統合設定**

## 設計決定

### Decision: ベースイメージ選定（CPU専用環境）
- **コンテキスト**: Mac環境での迅速な開発自動化、GPU拡張性の将来的な保持
- **検討した代替案**:
  1. `nvidia/cuda:12.1.0-base-ubuntu22.04` — GPU環境、重い（約1.5GB）、Mac環境で意味なし
  2. `nvidia/cuda:12.1.0-devel-ubuntu22.04` — 開発ツール込み（さらに重い）
  3. `pytorch/pytorch:2.x` — PyTorch公式イメージ（Python 3.11、Lightning 2.x互換性問題）
  4. **`python:3.12-slim-bookworm`** — **採用**: 軽量（約50MB）、公式Docker image、CPU専用
- **選定アプローチ**: **python:3.12-slim-bookworm**
- **理論的根拠**:
  - 軽量（約50MB vs nvidia/cuda:12.1.0-baseの約1.5GB）、Mac環境での高速ビルド
  - 公式Docker image（Debian 12 Bookworm）、長期サポート
  - CPU専用環境に最適、GPU不要なため依存関係最小化
  - PyTorch Lightning 1.9.x + torchaudio >=2.8.0（CPU版）完全互換
  - 将来的なGPU拡張性は設計レベルで保持（ベースイメージ切り替えのみ）
- **トレードオフ**:
  - **利点**: イメージサイズ小、ビルド高速、Mac環境で最適、完全制御
  - **欠点**: GPU訓練不可（CPU推論のみ）、システムライブラリ手動インストール必要（SoX, FFmpeg, libsndfile）
- **フォローアップ**:
  - ビルド時にapt-get updateとapt-get install -yでシステム依存関係インストール（要件3）
  - GitHub CLI (gh)インストール追加（公式インストール方法使用）
  - 将来的なGPU拡張時は、ARGまたはdockerfile切り替えで対応

### Decision: uvによるパッケージ管理戦略
- **コンテキスト**: pyproject.toml依存関係の再現可能なインストール
- **検討した代替案**:
  1. pip + requirements.txt — 従来方式、バージョン固定難しい
  2. poetry — ロック管理強力だがuv採用済みプロジェクトと不一致
  3. uv + pyproject.toml — 高速、ロックファイル生成、steering準拠
- **選定アプローチ**: **uv + pyproject.toml** with lock file
- **理論的根拠**:
  - steeringドキュメントで明示的にuv採用（tech.md）
  - uv syncでpyproject.toml依存関係の完全再現（要件8.2, 8.4）
  - マルチステージビルドでキャッシュ最適化（公式ドキュメント推奨）
- **トレードオフ**:
  - **利点**: ビルド高速、依存関係ハッシュ検証、完全再現性
  - **欠点**: postCreateCommandでuv sync再実行必要（VSCodeマウント特性）
- **フォローアップ**:
  - Dockerfile内でuv.lock生成とuv sync --frozen実行
  - postCreateCommandでワークスペースマウント後のuv sync実行（要件2.5）

### Decision: Claude Code統合方法
- **コンテキスト**: Claude Code安全統合と設定永続化
- **検討した代替案**:
  1. Claude Code手動インストール — 煩雑、再現性低い
  2. Dockerfile内でnpmインストール — バージョン管理難しい
  3. 公式Claude Code feature — **採用**、自動管理、セキュリティ考慮済み
- **選定アプローチ**: **ghcr.io/anthropics/devcontainer-features/claude-code:1**
- **理論的根拠**:
  - Anthropic公式サポート、自動アップデート
  - Node.js依存関係自動解決
  - セキュリティ分離がfeature側で実装済み（/workspaceアクセス制限）
  - 名前付きボリュームで設定永続化（${devcontainerId}で複数プロジェクト分離）
- **トレードオフ**:
  - **利点**: メンテナンス不要、セキュリティベストプラクティス適用済み、設定シンプル
  - **欠点**: feature更新に依存、内部動作のカスタマイズ不可
- **フォローアップ**:
  - devcontainer.jsonにfeature追加、CLAUDE_CONFIG_DIR環境変数設定（要件1.6, 1.8）
  - mounts設定で`claude-code-config-${devcontainerId}`ボリュームマウント（要件6.5）

### Decision: Git submodule自動初期化タイミング
- **コンテキスト**: sebbsサブモジュールの初期化タイミング（注: PSDS_EvalはPyPIパッケージ psds-eval として管理、submoduleではない）
- **検討した代替案**:
  1. Dockerfile RUN内 — ビルド時実行、.gitアクセス不可の可能性
  2. postCreateCommand — **採用**、ワークスペースマウント後実行
  3. postStartCommand — コンテナ起動毎実行、冗長
- **選定アプローチ**: **postCreateCommand**で`git submodule update --init --recursive`実行
- **理論的根拠**:
  - devcontainerはワークスペースをマウント後にpostCreateCommand実行（.git利用可能）
  - 初回作成時のみ実行、2回目以降はスキップされる
  - 要件4.1の期待動作と一致
- **トレードオフ**:
  - **利点**: .gitアクセス確実、初回のみ実行、エラーハンドリング可能
  - **欠点**: ビルド時ではないため初回起動時間が若干増加
- **フォローアップ**:
  - postCreateCommandで`uv sync && git submodule update --init --recursive && pre-commit install`実行
  - エラー時のメッセージ表示（要件4.3）はコマンド失敗時に自動表示

### Decision: データディレクトリマウント戦略
- **コンテキスト**: data/, embeddings/, exp/, wandb/の永続化とパフォーマンス
- **検討した代替案**:
  1. バインドマウント — ホストパス依存、ポータビリティ低い
  2. 名前付きボリューム — **採用**、Docker管理、ポータブル
  3. コンテナ内部ストレージ — 永続性なし、データ消失リスク
- **選定アプローチ**: **名前付きボリューム** for data/embeddings/exp/wandb
- **理論的根拠**:
  - Dockerがボリューム管理、ホスト環境非依存
  - I/O性能最適化（要件9.5）
  - 書き込み権限自動設定（vscodeuserアクセス可能）
  - Claude Config用ボリュームと一貫性
- **トレードオフ**:
  - **利点**: ポータブル、パフォーマンス良、権限問題なし
  - **欠点**: ホスト側から直接アクセス難しい（docker volume inspectで確認可能）
- **フォローアップ**:
  - devcontainer.jsonのmountsセクションで4ディレクトリを名前付きボリュームとして定義（要件6.1-6.4）
  - postCreateCommandでディレクトリ存在確認（mkdir -p）とパーミッション検証（要件6.6）

### Decision: Git認証方法（GitHub CLI credential helper）
- **コンテキスト**: Git submodule初期化のためのGitHub認証、SSH鍵不要な方法
- **検討した代替案**:
  1. SSH鍵設定 — 煩雑、ホスト環境依存、セキュリティリスク
  2. Personal Access Token — 環境変数管理必要、漏洩リスク
  3. **GitHub CLI (gh) credential helper** — **採用**: 簡便、セキュア、公式サポート
- **選定アプローチ**: **gh auth login（ホスト）+ gh auth setup-git（devcontainer）**
- **理論的根拠**:
  - ホスト環境でgh auth login実行により、認証情報がホストに保存される
  - devcontainer内でgh auth setup-gitにより、Git credential helperがgh使用に設定される
  - SSH鍵不要、Token管理不要、ホストの認証情報を安全に継承
  - 公式GitHub CLI推奨パターン
- **トレードオフ**:
  - **利点**: SSH鍵不要、簡便、セキュア、公式サポート
  - **欠点**: gh auth login事前実行必要（ドキュメントで明記）
- **フォローアップ**:
  - DockerfileでGitHub CLI (gh)インストール（公式インストール方法使用）
  - postCreateCommandでgh auth setup-git実行
  - DEVCONTAINER_GUIDE.mdに事前準備セクション追加、gh auth loginの重要性を明記

### Decision: セキュリティ設定アプローチ
- **コンテキスト**: 非rootユーザー実行、ネットワーク制限、機密情報保護
- **検討した代替案**:
  1. rootユーザー実行 + sudo削除 — 初期root必要、リスク高い
  2. remoteUser: vscode — **採用**、devcontainer標準、権限最小
  3. カスタムユーザー作成 — 複雑、メンテナンス負荷
- **選定アプローチ**: **remoteUser: "vscode"** + **Dockerfile内vscodeuserの作成**
- **理論的根拠**:
  - devcontainer標準パターン（要件7.1）
  - CPU専用環境のため、GPUアクセス権限不要
  - Claude Code featureがvscodeユーザー前提
  - 要件7.6のファイルアクセス制限はClaude Code側実装済み
- **トレードオフ**:
  - **利点**: セキュリティベストプラクティス、Claude Code統合スムーズ、権限問題最小
  - **欠点**: 一部システムレベル操作にsudo必要（ただしdevcontainer環境では稀）
- **フォローアップ**:
  - Dockerfileでvscodeuserを作成（RUN useradd -m -s /bin/bash vscode）
  - 環境変数を.env経由で管理（要件7.3）、.envファイルはコンテナ外配置推奨（要件7.4）
  - ネットワーク制限（要件7.5）は明示的設定不要（デフォルトでホスト以外アクセス不可）

### Decision: wandb統合（オプショナル機能）
- **コンテキスト**: ローカル開発自動化のみが目的、wandb APIキー不要
- **検討した代替案**:
  1. wandb APIキー必須 — ローカル開発には過剰、初回セットアップ障壁
  2. **wandb APIキーオプショナル** — **採用**: 初期実装不要、後から追加可能
  3. wandb完全削除 — 将来的な拡張性喪失
- **選定アプローチ**: **wandb APIキーはオプショナル**、devcontainer.jsonにコメントアウトで記載
- **理論的根拠**:
  - ローカル開発自動化のみが目的（要件明確化）
  - TensorBoardとOptunaで代替可能
  - wandb未認証時はローカルログのみ保存（graceful degradation）
  - 将来的な追加は環境変数設定のみで対応可能
- **トレードオフ**:
  - **利点**: 初回セットアップ簡便、wandb未使用時でも動作
  - **欠点**: wandb機能使用時は手動設定必要
- **フォローアップ**:
  - devcontainer.jsonのremoteEnvにWANDB_API_KEYをコメントアウトで記載
  - DEVCONTAINER_GUIDE.mdにwandb追加手順記載（オプショナル機能）
  - TensorBoardとOptunaのポート転送設定（forwardPorts: [6006, 8080]）

## リスクと軽減策
- **リスク1**: PyTorch Lightning 1.9.xとPyTorch 2.x（CPU版）の互換性問題
  - **軽減策**: torchaudio>=2.8.0（CPU版）指定、uv lockで依存関係固定、初回ビルド時に互換性検証テスト実行
- **リスク2**: GitHub CLI (gh)インストール失敗
  - **軽減策**: 公式インストール方法使用（GPGキー検証）、エラーハンドリング追加、ドキュメントに手動インストール手順記載
- **リスク3**: システムライブラリ（SoX, FFmpeg, libsndfile）インストール失敗
  - **軽減策**: Dockerfile内でapt-get update後に個別インストール、エラーログ出力、要件3.5で成功検証
- **リスク4**: uvとVSCodeマウントタイミング問題
  - **軽減策**: postCreateCommandでuv sync再実行、公式ドキュメント推奨パターン採用
- **リスク5**: gh auth login未実行によるsubmodule初期化失敗
  - **軽減策**: エラーメッセージで事前実行を明示的に案内、DEVCONTAINER_GUIDE.mdに事前準備セクション追加
- **リスク6**: CPU専用環境での訓練速度低下
  - **軽減策**: ドキュメントに推奨用途明記（コード開発、小規模実験、デバッグ、単体テスト）、将来的なGPU拡張手順提供
- **リスク5**: 名前付きボリュームのディスク容量不足
  - **軽減策**: Docker volumeのpruneコマンド案内、要件9.4でリソース不足時警告表示
- **リスク6**: pre-commit hookの破損または互換性問題
  - **軽減策**: postCreateCommandでpre-commit install実行前にgit status確認、失敗時はワーニング表示のみ（ブロックしない）

## 参考文献
- [Devcontainer - Claude Docs](https://docs.claude.com/en/docs/claude-code/devcontainer)
- [Anthropic devcontainer-features GitHub](https://github.com/anthropics/devcontainer-features)
- [Using uv in Docker - Official Guide](https://docs.astral.sh/uv/guides/integration/docker/)
- [NVIDIA Containers User Guide](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html)
- [Dev Container Features Registry](https://containers.dev/features)
- [Production-ready Python Docker with uv - hynek.me](https://hynek.me/articles/docker-uv/)
- [PyTorch DevContainer README](https://github.com/pytorch/pytorch/blob/main/.devcontainer/README.md)
- [Red Hat: PyTorch, containers, and NVIDIA guide](https://next.redhat.com/2025/08/26/a-developers-guide-to-pytorch-containers-and-nvidia-solving-the-puzzle/)
