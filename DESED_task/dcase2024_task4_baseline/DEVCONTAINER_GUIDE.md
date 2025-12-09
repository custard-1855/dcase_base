# Devcontainer セットアップガイド

DCASE 2024 Task 4 Sound Event Detection (SED) システムのDevcontainer開発環境の詳細セットアップガイドです。

## 目次

- [事前準備](#事前準備)
- [セットアップ手順](#セットアップ手順)
- [セキュリティとデータ保護](#セキュリティとデータ保護)
- [データ復旧手順](#データ復旧手順)
- [トラブルシューティング](#トラブルシューティング)
- [FAQ](#faq)
- [設定ファイル解説](#設定ファイル解説)

## 事前準備

Devcontainerを使用する前に、以下のツールとセットアップが必要です。

### Docker Desktop のインストール

**必須バージョン**: Docker Engine 20.10以上

#### Mac/Windows の場合

1. [Docker Desktop](https://www.docker.com/products/docker-desktop/)をダウンロード
2. インストーラーを実行し、指示に従ってインストール
3. Docker Desktopを起動し、動作確認

```bash
# バージョン確認
docker --version
# 出力例: Docker version 24.0.0, build abc123
```

#### リソース設定

Docker Desktopの設定で、以下のリソースを推奨します:

- **メモリ**: 8GB以上（最低4GB）
- **CPU**: 4コア以上
- **ディスク容量**: 50GB以上の空き容量

**設定方法** (Docker Desktop):
1. Docker Desktop → Settings (歯車アイコン) → Resources
2. Memory、CPUs、Disk image sizeを調整
3. "Apply & Restart"をクリック

### GitHub CLI (gh) 認証

**重要**: Git submodule（sebbs）の初期化にはGitHub認証が必要です。

**注意**: PSDS_EvalはPyPIパッケージ（psds-eval）として管理されており、submoduleではありません。実際のsubmoduleは**sebbsのみ**です。

#### GitHub CLI インストール

**Mac (Homebrew)**:
```bash
brew install gh
```

**Windows (WinGet)**:
```bash
winget install --id GitHub.cli
```

**Linux (Debian/Ubuntu)**:
```bash
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh
```

#### GitHub 認証の実行

**事前にホスト環境で実行する必要があります**:

```bash
# GitHub認証（ブラウザで認証フロー開始）
gh auth login

# 対話形式で以下を選択:
# - What account do you want to log into? → GitHub.com
# - What is your preferred protocol for Git operations? → HTTPS
# - Authenticate Git with your GitHub credentials? → Yes
# - How would you like to authenticate GitHub CLI? → Login with a web browser

# 認証状態の確認
gh auth status
# 出力例:
# github.com
#   ✓ Logged in to github.com as your-username (keyring)
#   ✓ Git operations for github.com configured to use https protocol.
```

**なぜ必要か**:
- Git submodule（sebbs）は非公開リポジトリまたは認証が必要な場合があります
- `gh auth setup-git`コマンドにより、GitがGitHub CLI経由で認証情報を使用します
- SSH鍵の設定が不要になります

### VS Code Dev Containers 拡張

VS Codeで[Dev Containers拡張](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)をインストール:

1. VS Codeを起動
2. 拡張機能タブ（Cmd/Ctrl+Shift+X）を開く
3. "Dev Containers"を検索してインストール

### wandb APIキー（オプショナル）

**初期セットアップでは不要**です。TensorBoardとOptunaのみ使用します。

wandbを使用する場合は、後から以下の手順で追加できます:

1. [wandb.ai](https://wandb.ai/)でアカウント作成
2. [APIキー取得ページ](https://wandb.ai/authorize)でAPIキーをコピー
3. ホスト環境で環境変数を設定:
   ```bash
   # ~/.bashrc または ~/.zshrc に追加
   export WANDB_API_KEY=your_api_key_here
   ```
4. `.devcontainer/devcontainer.json`の`remoteEnv`セクションで以下をアンコメント:
   ```json
   "remoteEnv": {
     "CLAUDE_CONFIG_DIR": "/home/vscode/.claude",
     "WANDB_API_KEY": "${localEnv:WANDB_API_KEY}"  // この行をアンコメント
   }
   ```
5. Devcontainerを再ビルド（"Dev Containers: Rebuild Container"）

## セットアップ手順

### 1. プロジェクトをクローン

```bash
git clone https://github.com/your-repo/dcase_base.git
cd dcase_base
```

### 2. VS Codeでプロジェクトを開く

```bash
code .
```

### 3. Devcontainerで開く

1. VS Codeの左下の緑色アイコン（リモート接続）をクリック
2. "Reopen in Container"を選択
3. 初回ビルドが開始（5-7分程度）

**ビルドプロセス**:
- ベースイメージ取得（python:3.12-slim-bookworm）
- システムライブラリインストール（SoX、FFmpeg、libsndfile、git、GitHub CLI）
- Python依存関係インストール（uv sync）
- Claude Code CLIインストール

### 4. 起動確認

VS Code Terminalで以下のコマンドを実行し、環境を確認:

```bash
# Python バージョン確認
python --version
# 出力: Python 3.12.x

# システムライブラリ確認
which sox ffmpeg git gh
# すべてパスが表示されること

# Git submodule 確認（sebbsのみが実際のsubmodule）
ls -la DESED_task/dcase2024_task4_baseline/sebbs
# ディレクトリに内容が存在すること

# psds-eval パッケージ確認（PyPIからインストール済み）
uv pip show psds-eval

# Claude Code 確認
claude --version
```

### 5. 開発開始

訓練スクリプトを実行:

```bash
cd DESED_task/dcase2024_task4_baseline
python train_pretrained.py
```

## セキュリティとデータ保護

### Claude Codeの不慮のコマンド実行でホストシステムは安全か?

**結論: はい、ホストシステムのデータは保護されています。**

Devcontainer環境は、Claude Codeの不慮のコマンド実行（例: `rm -rf /`）でもホストシステムのデータを保護する多層防御策を実装しています。

#### 1. コンテナファイルシステムの分離

**仕組み**:
- コンテナ内のファイルシステムはホストから完全に分離されています
- `/workspace`はDocker bind mountとしてプロジェクトディレクトリのみマウント
- `/usr`、`/etc`等のシステムディレクトリはコンテナ内の独立したファイルシステム

**効果**:
- コンテナ内での破壊的コマンドはコンテナ内のみ影響
- ホストの`/Users`、`/Applications`、`/System`等には一切影響なし

**検証方法**:
```bash
# Devcontainer内で実行
ls -la /workspace  # プロジェクトファイルのみ
ls -la /home       # コンテナ内の独立した/home（ホストとは別）
```

#### 2. 非rootユーザー実行による権限制限

**仕組み**:
- すべてのプロセスは非rootユーザー（`vscode`）で実行
- システムディレクトリ（`/usr/bin`、`/etc`等）への書き込み権限なし

**効果**:
- 破壊的コマンド実行時もシステムファイル削除不可
- Claude Codeの誤操作でもコンテナシステムは保護される

**検証方法**:
```bash
# Devcontainer内で実行
whoami  # 出力: vscode

# システムディレクトリへの書き込みテスト（失敗することを確認）
touch /usr/bin/test_file
# エラー: touch: cannot touch '/usr/bin/test_file': Permission denied
```

#### 3. Claude Codeファイルアクセス制限

**仕組み**:
- Claude Code feature側でファイルアクセス範囲を制限
- アクセス可能: `/workspace`配下、`/home/vscode/.claude`のみ
- アクセス不可: `/home/vscode/.ssh`、`/etc/passwd`、ホストファイルシステム

**効果**:
- Claude Codeはプロジェクトファイルのみ操作可能
- 機密ファイル（SSH鍵、システム設定等）は保護される

**実装**:
Anthropic公式Claude Code feature (`ghcr.io/anthropics/devcontainer-features/claude-code:1`) により自動適用。追加設定不要。

#### 4. 名前付きボリュームによるデータ永続化

**仕組み**:
- 重要なデータ（`data/`、`embeddings/`、`exp/`、`wandb/`）は名前付きボリュームとして管理
- ボリュームはDockerホストで管理され、コンテナ削除後も保持

**効果**:
- コンテナ内でデータが削除されても、ボリューム自体は保護される
- コンテナ再作成時に同じボリュームをマウントすることでデータ復元可能

**検証方法**:
```bash
# ホスト環境で実行
docker volume ls
# 出力例:
# DRIVER    VOLUME NAME
# local     dcase-data
# local     dcase-embeddings
# local     dcase-exp
# local     dcase-wandb
```

### セキュリティ検証テストの実行

プロジェクトには包括的なセキュリティ検証テストが含まれています。

**テスト実行方法**:
```bash
# ホストシステム保護検証テスト
uv run python tests/test_security_host_protection.py

# Claude Codeファイルアクセス制限検証テスト
uv run python tests/test_security_claude_code_access.py

# 機密情報管理検証テスト
uv run python tests/test_security_secrets_management.py
```

**期待される出力**:
```
【結論】
✓ Claude Codeの不慮のコマンド実行でもホストシステムのデータは保護されます:
  1. コンテナ内のファイルシステムはホストから分離されています
  2. 非rootユーザー（vscode）でシステムディレクトリへの書き込みが制限されています
  3. Claude Code feature側で/workspace配下のみアクセス可能に制限されています
  4. 名前付きボリューム（data、embeddings、exp、wandb）はコンテナ削除後も保持されます
```

## データ復旧手順

### コンテナ内のデータが吹き飛んだ場合の復旧方法

コンテナ内で誤ってデータを削除した場合でも、名前付きボリュームによる永続化により復旧可能です。

#### シナリオ1: ワークスペース内のファイル削除（Git管理対象）

**症状**:
```bash
# 誤って実行してしまった例
rm -rf DESED_task/dcase2024_task4_baseline/local/*.py
```

**復旧方法**:
```bash
# Devcontainer内で実行
cd DESED_task/dcase2024_task4_baseline

# Gitでファイルを復元
git checkout -- local/*.py

# または、最新コミットからすべて復元
git reset --hard HEAD
```

**注意**: Git管理されていないファイル（`data/`、`exp/`等）はこの方法では復元できません。

#### シナリオ2: 名前付きボリューム内のデータ削除

**症状**:
```bash
# 誤って実行してしまった例
rm -rf /workspace/data/*
rm -rf /workspace/exp/*
```

**重要**: 名前付きボリューム内のデータは**Dockerボリュームに永続化**されています。コンテナを削除しても、ボリューム自体は保持されます。

**復旧方法（ボリュームが残っている場合）**:

1. **コンテナを再起動**
   ```bash
   # VS Code Command Palette (Cmd/Ctrl+Shift+P)
   # "Dev Containers: Rebuild Container"を選択
   ```

2. **ボリュームが正しくマウントされているか確認**
   ```bash
   # Devcontainer内で実行
   ls -la /workspace/data
   ls -la /workspace/exp
   ```

3. **データが復元されていることを確認**
   - ボリュームが削除されていない限り、データは保持されています
   - コンテナ削除 ≠ ボリューム削除

#### シナリオ3: 名前付きボリューム自体を削除してしまった

**症状**:
```bash
# ホスト環境で誤って実行してしまった例
docker volume rm dcase-data
docker volume rm dcase-exp
```

**復旧方法**:

**重要**: ボリューム削除後は**データ復元不可**です。バックアップがない場合は、データセットを再ダウンロードするか、実験を再実行する必要があります。

1. **データセット再ダウンロード（`data/`の場合）**
   ```bash
   # Devcontainer内で実行
   cd DESED_task/dcase2024_task4_baseline
   python generate_dcase_task4_2024.py
   ```

2. **BEATs埋め込み再生成（`embeddings/`の場合）**
   ```bash
   cd DESED_task/dcase2024_task4_baseline
   python extract_embeddings.py
   ```

3. **実験結果の再実行（`exp/`の場合）**
   ```bash
   cd DESED_task/dcase2024_task4_baseline
   python train_pretrained.py
   ```

#### シナリオ4: コンテナ全体を削除してしまった

**症状**:
```bash
# ホスト環境で誤って実行してしまった例
docker container rm -f <container_id>
```

**復旧方法**:

**朗報**: コンテナ削除でも名前付きボリュームは保持されます！

1. **VS Codeで"Reopen in Container"を実行**
   ```bash
   # VS Code: 左下の緑色アイコン → "Reopen in Container"
   ```

2. **コンテナが再作成され、既存のボリュームがマウントされる**
   ```bash
   # Devcontainer内で実行
   ls -la /workspace/data
   ls -la /workspace/embeddings
   ls -la /workspace/exp
   # データが復元されていることを確認
   ```

3. **postCreateCommandが自動実行される**
   - uv sync（依存関係インストール）
   - git submodule（sebbs初期化）
   - pre-commit（フック設定）

#### シナリオ5: ホスト環境でプロジェクトディレクトリを削除してしまった

**症状**:
```bash
# ホスト環境で誤って実行してしまった例
rm -rf ~/work/dcase_base
```

**復旧方法**:

1. **Gitリポジトリから再クローン**
   ```bash
   # ホスト環境で実行
   git clone https://github.com/your-repo/dcase_base.git
   cd dcase_base
   ```

2. **VS Codeで"Reopen in Container"を実行**
   ```bash
   code .
   # VS Code: "Reopen in Container"
   ```

3. **名前付きボリューム内のデータは保持されている**
   ```bash
   # Devcontainer内で実行
   ls -la /workspace/data
   ls -la /workspace/exp
   # ボリューム削除していない限り、データは復元されている
   ```

### データバックアップのベストプラクティス

#### 定期的なボリュームバックアップ

**推奨頻度**: 重要な実験前後、週1回

1. **ボリュームを別ボリュームにバックアップ**
   ```bash
   # ホスト環境で実行
   docker run --rm \
     -v dcase-exp:/source \
     -v dcase-exp-backup:/backup \
     alpine sh -c "cd /source && tar cf - . | (cd /backup && tar xf -)"
   ```

2. **ボリュームをホストディレクトリにエクスポート**
   ```bash
   # ホスト環境で実行
   docker run --rm \
     -v dcase-exp:/source \
     -v ~/backups/dcase-exp:/backup \
     alpine sh -c "cd /source && tar cf - . | (cd /backup && tar xf -)"
   ```

3. **特定ファイルのみバックアップ**
   ```bash
   # Devcontainer内で実行
   # 重要なチェックポイントをコピー
   cp -r exp/best_model ~/backups/
   ```

#### 実験結果のクラウド同期（wandb推奨）

**wandb使用時の自動バックアップ**:
```bash
# Devcontainer内で実行
# wandb設定（初回のみ）
wandb login

# 訓練時に自動アップロード
python train_pretrained.py
# 実験結果が自動的にwandb.aiにバックアップされる
```

**wandb未使用時のローカルバックアップ**:
```bash
# 重要な実験結果をtarで圧縮
cd DESED_task/dcase2024_task4_baseline
tar czf ~/backups/exp_$(date +%Y%m%d).tar.gz exp/best_model/
```

### ボリューム管理のベストプラクティス

#### ボリューム使用状況確認

```bash
# ホスト環境で実行
docker system df -v

# ボリューム一覧と使用量
docker volume ls
docker volume inspect dcase-data
```

#### ボリューム削除時の注意

**安全な削除手順**:
1. **削除前に必ずバックアップ**
2. **不要なボリュームのみ削除**
   ```bash
   # wandb未使用の場合のみ削除
   docker volume rm dcase-wandb
   ```
3. **実験結果は選択的に削除**
   ```bash
   # Devcontainer内で古い実験のみ削除
   cd DESED_task/dcase2024_task4_baseline/exp
   rm -rf old_experiment_20240101/
   ```

**危険な操作（実行前に必ず確認）**:
```bash
# 全ボリューム削除（データ消失リスク）
docker volume prune --all  # 実行しないでください

# 特定ボリューム削除（不可逆）
docker volume rm dcase-data  # バックアップ済みの場合のみ
```

## トラブルシューティング

### ビルド検証テスト失敗（Build Validation Test Failures）

プロジェクトに含まれるビルド検証テスト (`tests/test_devcontainer_build.sh`) を実行して、環境構築の成功を確認できます。

**テスト実行方法**:
```bash
bash tests/test_devcontainer_build.sh
```

#### Test 0: ベースイメージ取得失敗

**症状**:
```
✗ Base image pull failed
  Recovery: Check Docker Hub connection or proxy settings
```

**原因**:
- Docker Hub接続不可
- プロキシ設定未構成

**対処法**:
1. Docker Hub接続確認
   ```bash
   docker pull python:3.12-slim-bookworm
   ```

2. プロキシ環境の場合、Docker Desktopでプロキシ設定
   - Docker Desktop → Settings → Resources → Proxies
   - Manual proxy configurationを有効化

#### Test 2: Dockerfileビルド失敗

**症状**:
```
✗ Dockerfile build failed
  Recovery: Check build logs above for specific errors
```

**一般的な原因と対処法**:

1. **APTリポジトリ接続失敗**
   ```bash
   # キャッシュクリアして再ビルド
   docker build --no-cache -f .devcontainer/Dockerfile -t test .
   ```

2. **GitHub Container Registry接続失敗**
   - ghcr.io/astral-sh/uv:0.9.16取得失敗
   - ネットワーク確認、再実行で解決することが多い

3. **ディスク容量不足**
   ```bash
   # 不要なDockerリソースを削除
   docker system prune -a --volumes
   ```

#### Test 3: Python 3.12検証失敗

**症状**:
```
✗ Python 3.12 not detected. Found: Python 3.x.x
  Recovery: Verify 'uv python install 3.12' in Dockerfile
```

**対処法**:
1. `.devcontainer/Dockerfile`のStage 2で`uv python install 3.12`が実行されているか確認
2. uvバイナリが正しくコピーされているか確認 (Test 5参照)

#### Test 5: uvパッケージマネージャー検証失敗

**症状**:
```
✗ uv is NOT installed
  Recovery: Verify COPY --from=uv in Dockerfile Stage 2
```

**対処法**:
1. `.devcontainer/Dockerfile`のStage 1でuvイメージが正しく指定されているか確認:
   ```dockerfile
   FROM ghcr.io/astral-sh/uv:0.9.16 AS uv
   ```

2. Stage 2でCOPYコマンドが正しいか確認:
   ```dockerfile
   COPY --from=uv /uv /usr/local/bin/uv
   ```

#### Test 6: Python依存関係検証失敗

**症状**:
```
✗ Python dependencies not installed (found 0 packages)
  Recovery: Verify pyproject.toml specifies 'pytorch-lightning==1.9.*'
```

**原因**:
- `uv sync`失敗
- `pyproject.toml`不整合
- `uv.lock`破損

**対処法**:
1. `pyproject.toml`でPyTorch Lightning 1.9.xが指定されているか確認:
   ```toml
   dependencies = [
       "pytorch-lightning==1.9.*",
       # ...
   ]
   ```

2. `uv.lock`再生成:
   ```bash
   uv lock --upgrade
   ```

3. Dockerfile Stage 3で`uv sync`が実行されているか確認

### システムライブラリインストール失敗

**症状**:
```
Failed to install audio libraries. Check network connection and retry.
```

**原因**:
- APT repository接続不可
- ネットワークタイムアウト
- パッケージ名変更

**対処法**:

1. ネットワーク接続確認
   ```bash
   ping google.com
   ```

2. Dockerキャッシュクリア後に再ビルド
   ```bash
   # VS Code Command Palette (Cmd/Ctrl+Shift+P)
   # "Dev Containers: Rebuild Container Without Cache"を選択
   ```

3. プロキシ環境の場合は`.devcontainer/Dockerfile`に以下を追加:
   ```dockerfile
   # Stage 2のapt-get前に追加
   ENV http_proxy=http://your-proxy:port
   ENV https_proxy=http://your-proxy:port
   ```

### Git submodule 初期化失敗

**症状**:
```
Error: Git submodule initialization failed. sebbs may be unavailable.
```

**注意**: PSDS_EvalはPyPIパッケージ（psds-eval）として管理されており、submoduleではありません。実際のsubmoduleは**sebbsのみ**です。

**原因**:
- `gh auth login`未実行
- GitHub認証エラー
- ネットワークエラー

**対処法**:

1. **最重要**: ホスト環境で`gh auth login`を実行
   ```bash
   # ホスト環境（Devcontainer外）で実行
   gh auth login
   gh auth status  # 認証確認
   ```

2. Devcontainerを再起動
   ```bash
   # VS Code Command Palette
   # "Dev Containers: Rebuild Container"
   ```

3. 手動でsubmodule初期化
   ```bash
   # Devcontainer内で実行
   cd DESED_task/dcase2024_task4_baseline
   gh auth setup-git  # credential helper設定
   git submodule update --init --recursive
   ```

4. `.gitmodules`ファイルを確認（破損していないか）
   ```bash
   cat .gitmodules
   ```

### gh auth login 未実行エラー

**症状**:
```
Warning: gh auth setup-git failed. Run 'gh auth login' on host machine first, then restart container.
```

**原因**:
ホスト環境でGitHub CLI認証が未実行

**対処法**:

1. Devcontainerを一旦停止
   ```bash
   # VS Code Command Palette
   # "Dev Containers: Close Remote Connection"
   ```

2. **ホスト環境**で認証実行
   ```bash
   gh auth login
   ```

3. Devcontainerで再度開く
   ```bash
   # VS Code: "Reopen in Container"
   ```

### ボリューム権限問題

**症状**:
```
Warning: data not writable. Data persistence may fail. Check volume mounts.
```

**原因**:
- Dockerボリュームマウント失敗
- ユーザーパーミッション問題

**対処法**:

1. ボリューム一覧確認
   ```bash
   docker volume ls
   # dcase-data, dcase-embeddings, dcase-exp, dcase-wandb が存在すること
   ```

2. ボリューム詳細確認
   ```bash
   docker volume inspect dcase-data
   ```

3. ボリューム再作成
   ```bash
   # Devcontainerを停止
   # VS Code: "Close Remote Connection"

   # ホスト環境でボリューム削除
   docker volume rm dcase-data dcase-embeddings dcase-exp dcase-wandb

   # Devcontainerを再ビルド
   # VS Code: "Rebuild Container"
   ```

4. 権限確認
   ```bash
   # Devcontainer内で実行
   ls -ld data embeddings exp wandb
   # vscodeuserの所有権であること
   ```

### メモリ不足警告

**症状**:
```
Warning: Low memory detected (1024 MB available). Training may fail. Increase Docker memory limit.
```

**原因**:
Docker Desktopのメモリ制限が不足

**対処法**:

1. **Docker Desktop設定でメモリ増加**
   - Docker Desktop → Settings → Resources → Memory
   - 8GB以上に設定（最低4GB）
   - "Apply & Restart"

2. **バッチサイズ削減**
   ```yaml
   # DESED_task/dcase2024_task4_baseline/confs/default.yaml
   training:
     batch_size: 16  # デフォルト32から16に削減
   ```

3. **CPUのみで軽量実験**
   ```bash
   # 小規模データセットで動作確認
   python train_pretrained.py training.max_epochs=1 data.limit_train_batches=10
   ```

### ディスク容量不足警告

**症状**:
```
Warning: Low disk space (5 GB available). Data storage may fail. Clean up volumes or increase disk.
```

**原因**:
- Dockerボリューム肥大化
- 実験結果蓄積
- Dockerイメージキャッシュ

**対処法**:

1. **不要なボリューム削除**
   ```bash
   # 未使用ボリューム一括削除（注意: データ消失）
   docker volume prune

   # 特定ボリューム削除
   docker volume rm dcase-wandb  # wandb使用していない場合
   ```

2. **不要な実験結果削除**
   ```bash
   # Devcontainer内で実行
   cd DESED_task/dcase2024_task4_baseline/exp
   rm -rf old_experiment_*  # 古い実験削除
   ```

3. **Dockerシステム全体クリーンアップ**
   ```bash
   # ホスト環境で実行（注意: すべての未使用リソース削除）
   docker system prune -a --volumes
   ```

4. **Docker Desktop設定でディスク容量増加**
   - Docker Desktop → Settings → Resources → Disk image size
   - 100GB以上に設定

### uvインストール失敗

**症状**:
```
Failed to fetch uv binary from ghcr.io. Check network or use local uv installation.
```

**原因**:
GitHub Container Registry接続不可

**対処法**:

1. ネットワーク確認

2. `.devcontainer/Dockerfile`を修正してローカルインストール使用:
   ```dockerfile
   # Stage 1をコメントアウト
   # FROM ghcr.io/astral-sh/uv:0.9.16 AS uv

   # Stage 2のCOPYをpip installに変更
   FROM python:3.12-slim-bookworm AS base
   # COPY --from=uv /usr/local/bin/uv /usr/local/bin/uv
   RUN pip install uv  # この行を追加
   ```

### Python依存関係インストール失敗

**症状**:
```
Failed to install Python dependencies. Check pyproject.toml and uv.lock integrity.
```

**原因**:
- `pyproject.toml`不整合
- `uv.lock`破損
- PyPI接続不可

**対処法**:

1. **uv.lock再生成**
   ```bash
   # Devcontainer内で実行
   uv lock --upgrade
   ```

2. **pyproject.toml検証**
   ```bash
   # 依存関係確認
   cat pyproject.toml
   ```

3. **再ビルド**
   ```bash
   # VS Code: "Rebuild Container Without Cache"
   ```

### Claude Code動作確認失敗

**症状**:
```
claude: command not found
```

**原因**:
Claude Code feature自動インストール失敗

**対処法**:

1. Node.jsインストール確認
   ```bash
   node --version
   npm --version
   ```

2. 手動でClaude Code CLIインストール
   ```bash
   # Node.jsが導入済みの場合
   npm install -g @anthropic-ai/claude-code
   ```

3. `.devcontainer/devcontainer.json`のfeature設定確認:
   ```json
   "features": {
     "ghcr.io/anthropics/devcontainer-features/claude-code:1": {}
   }
   ```

## FAQ

### GPU 対応は可能ですか？

**現在の実装はCPU専用**です。将来的なGPU拡張は可能ですが、初期実装では対象外としています。

**GPU対応手順（将来的な拡張）**:

1. `.devcontainer/Dockerfile`のベースイメージ変更:
   ```dockerfile
   # FROM python:3.12-slim-bookworm AS base
   FROM nvidia/cuda:12.1.0-base-ubuntu22.04 AS base
   ```

2. `.devcontainer/devcontainer.json`にrunArgs追加:
   ```json
   "runArgs": [
     "--gpus", "all"
   ]
   ```

3. 環境変数追加:
   ```json
   "remoteEnv": {
     "NVIDIA_VISIBLE_DEVICES": "all",
     "CUDA_VISIBLE_DEVICES": "all"
   }
   ```

**注意**: Mac環境ではGPUサポート不可（NVIDIAドライバ不要のためCPU版を採用）

### wandb を使用するには？

wandbは**オプショナル**機能です。初期セットアップでは不要ですが、後から追加可能です。

**手順**:

1. wandb APIキー取得
   - [wandb.ai](https://wandb.ai/)でアカウント作成
   - [APIキーページ](https://wandb.ai/authorize)でキーをコピー

2. ホスト環境で環境変数設定
   ```bash
   # ~/.bashrc または ~/.zshrc に追加
   export WANDB_API_KEY=your_api_key_here
   source ~/.bashrc  # 反映
   ```

3. `.devcontainer/devcontainer.json`を編集
   ```json
   "remoteEnv": {
     "CLAUDE_CONFIG_DIR": "/home/vscode/.claude",
     "WANDB_API_KEY": "${localEnv:WANDB_API_KEY}"  // アンコメント
   }
   ```

4. Devcontainer再ビルド
   ```bash
   # VS Code: "Rebuild Container"
   ```

5. wandb認証確認
   ```bash
   # Devcontainer内で実行
   python -c "import wandb; wandb.login()"
   ```

**wandb未使用時の動作**:
- ローカルログのみ保存（`wandb/`ディレクトリ）
- TensorBoardで可視化可能
   ```bash
   tensorboard --logdir=exp/
   # ブラウザで http://localhost:6006 を開く
   ```

### ディスク容量を管理するには？

Devcontainerの長期使用でディスク容量が不足する場合があります。

**定期メンテナンス**:

1. **Dockerボリューム使用状況確認**
   ```bash
   docker system df -v
   ```

2. **未使用ボリューム削除**
   ```bash
   # 安全な削除（未使用のみ）
   docker volume prune

   # 確認プロンプトをスキップ
   docker volume prune -f
   ```

3. **特定ボリューム削除**
   ```bash
   # wandb未使用の場合
   docker volume rm dcase-wandb

   # 実験結果リセット（注意: データ消失）
   docker volume rm dcase-exp
   ```

4. **Dockerイメージキャッシュ削除**
   ```bash
   # 未使用イメージ削除
   docker image prune -a
   ```

5. **システム全体クリーンアップ**
   ```bash
   # 注意: すべての未使用リソース削除（コンテナ、ボリューム、イメージ）
   docker system prune -a --volumes

   # 実行前に確認
   docker system df
   ```

**推奨スケジュール**:
- 週1回: `docker volume prune`
- 月1回: `docker image prune -a`
- 必要時: 古い実験結果を手動削除（`exp/`ディレクトリ内）

### TensorBoard/Optuna Dashboardへのアクセス方法は？

Devcontainerはポート転送を自動設定します。

**TensorBoard**:
```bash
# Devcontainer内で実行
cd DESED_task/dcase2024_task4_baseline
tensorboard --logdir=exp/

# ブラウザで http://localhost:6006 を開く
```

**Optuna Dashboard**:
```bash
# Devcontainer内で実行
optuna-dashboard sqlite:///exp/optuna.db

# ブラウザで http://localhost:8080 を開く
```

**ポート転送設定** (`.devcontainer/devcontainer.json`):
```json
"forwardPorts": [6006, 8080]
```

### pre-commitフックが動作しない

**症状**:
```
Warning: pre-commit install failed. Git hooks not activated.
```

**対処法**:

1. 手動でインストール
   ```bash
   cd DESED_task/dcase2024_task4_baseline
   pre-commit install
   ```

2. `.git/hooks/`権限確認
   ```bash
   ls -la .git/hooks/
   # pre-commitファイルが実行可能であること
   ```

3. フック動作確認
   ```bash
   git commit -m "test"
   # pre-commitフックが実行されること
   ```

### コンテナ再起動時にsubmodule内容が消える

**原因**:
Git submoduleはワークスペース（bind mount）に保存されるため、通常は永続化されます。消える場合はpostCreateCommand失敗が原因です。

**対処法**:

1. postCreateCommand出力確認
   ```bash
   # VS Code Terminal で確認
   # "postCreateCommand completed"が表示されるか
   ```

2. 手動で再初期化
   ```bash
   git submodule update --init --recursive
   ```

3. gh auth login確認
   ```bash
   gh auth status
   ```

## 設定ファイル解説

### devcontainer.json

Devcontainer統合設定ファイル。VS Code、Docker、Claude Codeの橋渡し役を担います。

**主要セクション**:

```json
{
  "name": "DCASE 2024 Task 4 Dev Container (CPU)",

  // Dockerfileビルド指示
  "build": {
    "dockerfile": "Dockerfile"
  },

  // Claude Code feature統合（Anthropic公式）
  "features": {
    "ghcr.io/anthropics/devcontainer-features/claude-code:1": {}
  },

  // 名前付きボリュームマウント（データ永続化）
  "mounts": [
    "source=dcase-data,target=/workspace/data,type=volume",
    "source=dcase-embeddings,target=/workspace/embeddings,type=volume",
    "source=dcase-exp,target=/workspace/exp,type=volume",
    "source=dcase-wandb,target=/workspace/wandb,type=volume",
    "source=claude-code-config-${devcontainerId},target=/home/vscode/.claude,type=volume"
  ],

  // 非rootユーザー（セキュリティ）
  "remoteUser": "vscode",

  // 環境変数設定
  "remoteEnv": {
    "CLAUDE_CONFIG_DIR": "/home/vscode/.claude"
    // "WANDB_API_KEY": "${localEnv:WANDB_API_KEY}"  // オプショナル
  },

  // VS Code拡張機能自動インストール
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",           // Python基本機能
        "ms-python.vscode-pylance",   // 型チェック
        "charliermarsh.ruff",         // Linter/Formatter
        "ms-python.mypy-type-checker" // 型チェッカー
      ],
      "settings": {
        "editor.formatOnSave": true,  // 保存時自動フォーマット
        "editor.codeActionsOnSave": {
          "source.organizeImports": true  // インポート自動整理
        },
        "python.defaultInterpreterPath": "/workspace/.venv/bin/python",
        "[python]": {
          "editor.defaultFormatter": "charliermarsh.ruff"
        }
      }
    }
  },

  // ポート転送（TensorBoard、Optuna Dashboard）
  "forwardPorts": [6006, 8080],

  // 起動後初期化コマンド
  "postCreateCommand": "uv sync && gh auth setup-git && git submodule update --init --recursive && pre-commit install"
}
```

**重要設定の解説**:

- **features**: Claude Code公式featureを使用。Node.js依存関係が自動解決されます。
- **mounts**: 名前付きボリュームによりデータ永続化。ホスト非依存でポータブルです。
- **remoteUser**: 非rootユーザー（vscode）で実行し、セキュリティリスクを最小化。
- **postCreateCommand**: コンテナ起動後に自動実行。uv sync（依存関係）、gh auth setup-git（Git認証）、git submodule（サブモジュール初期化）、pre-commit（フック）を順次実行。

### Dockerfile

Multi-stage Dockerfileによる軽量ビルド。

**Stage 1: uvバイナリ取得**
```dockerfile
FROM ghcr.io/astral-sh/uv:0.9.16 AS uv
```
- 公式uvバイナリをコピー
- バージョン固定により再現性保証

**Stage 2: ベース環境構築**
```dockerfile
FROM python:3.12-slim-bookworm AS base

# uvバイナリコピー
COPY --from=uv /usr/local/bin/uv /usr/local/bin/uv

# システムライブラリインストール
RUN apt-get update && apt-get install -y \
    sox \          # 音声処理（torchaudio/resampyバックエンド）
    ffmpeg \       # 音声処理
    libsndfile1-dev \  # 音声ファイルI/O
    git \          # Git submodule管理
    curl \         # GitHub CLI取得
    && rm -rf /var/lib/apt/lists/*  # APTキャッシュ削除（イメージサイズ削減）

# GitHub CLI公式インストール方法
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12インストール
RUN uv python install 3.12

# 非rootユーザー作成（セキュリティ）
RUN useradd -m -s /bin/bash vscode

# ビルドタイムスタンプ記録（再現性）
LABEL build_timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
```

**Stage 3: Python依存関係インストール**
```dockerfile
FROM base AS dependencies

WORKDIR /workspace

# pyproject.tomlとuv.lockコピー
COPY pyproject.toml uv.lock ./

# uv sync --frozen: ロックファイル厳格検証、ハッシュ確認
# --all-groups: 開発依存関係（mypy、ruff）含む
# --mount=type=cache: uvキャッシュ再利用（ビルド高速化）
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-groups

# 非rootユーザーに切り替え
USER vscode
```

**重要な設計判断**:

- **python:3.12-slim-bookworm**: 軽量（約50MB）、Mac環境での高速ビルド、CPU専用に最適
- **Multi-stage build**: 各ステージで責任分離、キャッシュヒット率向上
- **uv sync --frozen**: uv.lockハッシュ検証により完全再現性保証
- **APTキャッシュ削除**: `rm -rf /var/lib/apt/lists/*`でイメージサイズ削減
- **USER vscode**: 非rootユーザーでプロセス実行（セキュリティベストプラクティス）

## 参考リンク

- [VS Code Dev Containers公式ドキュメント](https://code.visualstudio.com/docs/devcontainers/containers)
- [Docker公式ドキュメント](https://docs.docker.com/)
- [GitHub CLI公式ドキュメント](https://cli.github.com/manual/)
- [uv公式ドキュメント](https://docs.astral.sh/uv/)
- [Claude Code公式ドキュメント](https://docs.anthropic.com/en/docs/claude-code)

## 問題報告

問題が解決しない場合は、以下の情報を含めて報告してください:

1. エラーメッセージ全文
2. 実行環境（OS、Docker version）
3. `docker system df -v`出力
4. `gh auth status`出力
5. `.devcontainer/devcontainer.json`と`Dockerfile`の変更内容（カスタマイズしている場合）
