# Research & Design Decisions: 実験ディレクトリ構造リファクタリング

## Summary
- **Feature**: `refactor-experiment-structure`
- **Discovery Scope**: Extension（既存システムの拡張）
- **Key Findings**:
  - wandbは `dir` パラメータでカスタムディレクトリを指定可能だが、デフォルトは `wandb/run-{timestamp}-{id}/` 形式
  - PyTorch Lightning `ModelCheckpoint` は `dirpath` で明示的なチェックポイント保存先を指定可能
  - 現在の実装は `wandb.init()` の `name` パラメータに疑似的な階層構造（スラッシュ区切り）を渡しているが、ファイルシステム上は平坦
  - wandb UIは階層的フォルダ構造を公式にサポートしていない（プロジェクト、グループ、タグによる組織化が推奨）
  - **実行モード管理の問題**: 現在、`use_wandb=True` の場合、training/test-only/feature-extraction すべてで新しいwandB runが作成される（wandB初期化が実行モードを考慮していない）

## Research Log

### wandbのカスタムディレクトリ設定

**Context**: Requirements 3（wandb統合とパス解決）を実現するため、wandbの `dir` パラメータによるカスタムディレクトリ設定方法を調査。

**Sources Consulted**:
- [WandB Documentation: Configure experiments](https://docs.wandb.ai/models/track/config)
- [WandB GitHub Issue #3077: Organize simulations in sub-folders](https://github.com/wandb/wandb/issues/3077)
- [PyTorch Lightning WandB Logger](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html)

**Findings**:
- `wandb.init(dir="/custom/path")` でベースディレクトリを指定可能
- ただし、wandbは内部的に `{dir}/run-{timestamp}-{id}/` 形式のサブディレクトリを常に作成する
- `name` パラメータはwandB UIでの表示名として機能（スラッシュを含めても階層フォルダにはならない）
- `group` パラメータと `tags` で実験をUI上で組織化することが推奨されている

**Implications**:
- Requirement 1の階層的ディレクトリ構造を実現するには、`wandb.init(dir="{experiments}/{category}/{method}/{variant}/")` として親ディレクトリを制御
- wandbのrun IDベースのサブディレクトリは保持し、symlink/metadataでマッピングを提供（Requirement 3.3）
- wandb UIでの表示と、ローカルファイルシステムの階層構造を分離して設計

### PyTorch Lightning ModelCheckpointのdirpath制御

**Context**: Requirement 2（成果物の統合管理）のため、チェックポイント保存先のカスタマイズ方法を調査。

**Sources Consulted**:
- [PyTorch Lightning ModelCheckpoint API](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html)
- [PyTorch Lightning Checkpointing Guide](https://lightning.ai/docs/pytorch/stable/common/checkpointing.html)

**Findings**:
- `ModelCheckpoint(dirpath="/custom/checkpoint/path/")` で明示的に保存先を指定可能
- Loggerが存在する場合、デフォルトでは `{logger.log_dir}/checkpoints/` に保存される
- `filename` パラメータでメトリクスを含む動的ファイル名を設定可能（例: `{epoch:02d}-{val_loss:.2f}`）
- `save_top_k` でベストモデルの保存数を制御

**Implications**:
- 実験ディレクトリ内の `checkpoints/` サブディレクトリにModelCheckpointのdirpathを設定
- 既存の `train_pretrained.py` では `desed_training._wandb_checkpoint_dir` を使用してdirpathを決定しているため、この仕組みを統一的に拡張
- ファイル名にepochやmetricを含めることで、チェックポイントの識別性を向上

### 既存実装の統合パターン分析

**Context**: 既存の `sed_trainer_pretrained.py` と `train_pretrained.py` のwandb統合の実装パターンを分析。

**Findings**:
- `sed_trainer_pretrained.py` の `_init_wandb_project()` メソッドで `wandb.init()` を呼び出し
- `name` パラメータに `self.hparams["net"]["wandb_dir"]` を渡している（コマンドライン引数 `--wandb_dir` から設定）
- `self._wandb_checkpoint_dir = os.path.join(wandb.run.dir, "checkpoints")` でチェックポイントディレクトリを生成
- `train_pretrained.py` ではこの `_wandb_checkpoint_dir` の存在をチェックし、ModelCheckpointのdirpathに使用

**Implications**:
- 新しい階層的ディレクトリ構造を導入するには、`_init_wandb_project()` メソッドを拡張
- YAML設定ファイルに `experiment_naming` セクションを追加し、category/method/variantを定義
- `dir` パラメータで親ディレクトリを制御し、`_wandb_checkpoint_dir` の計算ロジックを更新
- 後方互換性のため、`wandb_dir` が指定されている場合は従来の動作を維持

### YAML設定による実験命名

**Context**: Requirement 4（設定駆動の実験命名）のため、現在の設定ファイル構造と命名パターンを分析。

**Findings**:
- 現在の `pretrained.yaml` には `net.use_wandb` と `net.wandb_dir` が存在
- `wandb_dir` はコマンドライン引数で上書き可能（例: `--wandb_dir "150/cmt_apply-unlabeled/CMT_use_neg_sample"`）
- スラッシュ区切りの文字列で疑似的な階層を表現しているが、これはwandB UIのrun名としてのみ機能

**Implications**:
- Requirement 4を満たすため、新しい設定項目を導入:
  ```yaml
  experiment:
    category: "baseline"
    method: "cmt"
    variant: "use_neg_sample"
    base_dir: "experiments"  # Optional: デフォルトは "experiments"
  ```
- テンプレート機能（`{method}_{variant}_{timestamp}`）はパス生成ユーティリティで実装
- 環境変数置換（`$SCRATCH_DIR/experiments`）は `os.path.expandvars()` で対応

### Hydra統合の検討

**Context**: 階層的な設定管理の拡張性を評価するため、Hydraフレームワークの採用可否を調査。

**Sources Consulted**:
- [WandB + Hydra Integration](https://docs.wandb.ai/guides/integrations/hydra/)

**Findings**:
- Hydraは階層的な設定の動的生成とcompositionをサポート
- wandbとの統合が公式にサポートされている
- ただし、既存のYAML + argparse構成から移行するにはコスト大

**Implications**:
- 本リファクタリングではHydra導入を見送り、既存のYAML + argparse構造を維持
- 将来的な拡張オプションとして `research.md` に記録（Non-Goal扱い）

### 実行モード管理と不要なwandBログ生成の問題

**Context**: Requirement 5（実行モード別のwandBログ管理）を追加するため、既存コードの実行モード判定とwandB初期化の関係を調査。

**Sources Consulted**:
- `local/sed_trainer_pretrained.py` — SEDTask4.__init__() と _init_wandb_project()
- `train_pretrained.py` — test_from_checkpoint と eval_from_checkpoint 引数処理
- `visualize/get_features/extract_inference_features.py` — 特徴量抽出時のSEDTask4インスタンス化

**Findings**:
- **現在の実行モード判定**:
  - Training: `test_state_dict is None` (train_pretrained.py:397, 538, 610)
  - Test-only: `--test_from_checkpoint` 引数が指定されている (train_pretrained.py:742-747)
  - Evaluation: `--eval_from_checkpoint` 引数が指定され、`evaluation=True` (train_pretrained.py:744-746)
  - Feature extraction: `extract_inference_features.py` で明示的に `evaluation=True` を渡す
- **wandB初期化の問題**:
  - `SEDTask4.__init__()` は `use_wandb` フラグのみをチェック (sed_trainer_pretrained.py:182-183)
  - `evaluation` パラメータと `fast_dev_run` パラメータは存在するが、`_init_wandb_project()` 内で**考慮されていない**
  - 結果: `use_wandb=True` であれば、test-only時や特徴量抽出時も不要なwandB runが作成される
- **`evaluation` パラメータの現在の用途**:
  - `self.evaluation = evaluation` として保存 (sed_trainer_pretrained.py:224)
  - Training時のloss計算を制御 (`if not self.evaluation:` でスキップ) (sed_trainer_pretrained.py:1479)
  - しかし、wandB初期化の制御には使われていない

**Implications**:
- Requirement 5を実現するには、`_init_wandb_project()` メソッドで実行モードをチェックし、wandB初期化を条件分岐させる必要がある
- 実行モードの明示的な設定（YAML `mode:` または `--mode` 引数）を導入し、暗黙的な判定ロジックを減らす
- `evaluation=True` または `fast_dev_run=True` の場合、デフォルトでwandBを無効化する戦略を採用
- manifest.jsonに実行モード（`"mode": "train" | "test" | "inference" | "feature_extraction"`）を記録

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| **Option 1: Wrapper Pattern** | 既存の `wandb.init()` を薄いラッパー関数で包み、カスタムディレクトリロジックを注入 | 最小限の変更、後方互換性維持、既存コード構造を保持 | ラッパー層の保守コスト | ✅ 選択 |
| **Option 2: LightningModule Refactor** | `SEDTask4.__init__()` で実験ディレクトリ管理を完全に統合 | 一元管理、型安全性向上 | 既存コードの大幅変更、リスク高 | Non-Goal（将来の改善候補） |
| **Option 3: Hydra Migration** | Hydraフレームワークに完全移行し、設定管理を置き換え | 強力な階層的設定、プロフェッショナルツール | 学習コスト大、既存YAMLとの非互換性 | 却下（本リファクタリングのスコープ外） |

## Design Decisions

### Decision 1: 階層的ディレクトリ構造とwandb runディレクトリの共存

**Context**: wandbは内部的に `run-{timestamp}-{id}/` 形式のディレクトリを常に作成するため、完全なカスタム命名は不可能。

**Alternatives Considered**:
1. **wandbのデフォルトディレクトリを完全に無視** → 設定ファイルと実際の成果物の場所が分離され、混乱を招く
2. **wandb.init(dir=)で親ディレクトリのみ制御し、run IDディレクトリを許容** → 階層構造とwandbの仕組みを両立
3. **wandbを使用せず、カスタムロガーのみ使用** → 既存のwandb統合を破壊

**Selected Approach**: Option 2（wandb.init(dir=)で親ディレクトリ制御）

**Rationale**:
- wandbのrun IDベースのディレクトリは一意性を保証するため、timestamp付きvariantディレクトリとして機能
- `experiments/{category}/{method}/{variant}/run-{timestamp}-{id}/` という構造で、階層性と一意性を両立
- Requirement 1.5の「unique identifiers (timestamp or counter)」に自然に対応

**Trade-offs**:
- 利点: wandbの内部構造を尊重し、デバッグとトラブルシューティングが容易
- 妥協点: 完全なカスタム命名（例: `{variant}/` のみ）は不可能だが、親ディレクトリで階層性を確保

**Follow-up**:
- symlinkや manifest ファイルで run ID → 実験名のマッピングを提供（Requirement 3.3）

### Decision 2: 成果物の保存階層

**Context**: Requirement 2で定義された成果物（checkpoints, metrics, inference, visualizations, config）の保存構造。

**Alternatives Considered**:
1. **wandb runディレクトリ直下にフラットに配置** → 既存の `wandb.run.dir/checkpoints/` パターンを踏襲するが、他の成果物の管理が不明瞭
2. **wandb runディレクトリ内にサブディレクトリを作成** → すべての成果物を体系的に管理
3. **wandbディレクトリと並列に独立した成果物ディレクトリを作成** → wandbと分離するが、実験の統一性が損なわれる

**Selected Approach**: Option 2（wandb runディレクトリ内にサブディレクトリ）

**Rationale**:
- 単一の実験ディレクトリ配下ですべての成果物を管理
- 既存の `checkpoints/` パターンを `inference/`, `visualizations/`, `config/` に拡張
- wandbの自動同期機能（wandb cloud sync）の恩恵を受けられる

**Trade-offs**:
- 利点: 一貫性、発見可能性、wandbとの統合
- 欠点: wandbディレクトリサイズが増大する可能性（ただし、`.wandbignore` で制御可能）

**Follow-up**:
- 可視化データ（UMAP plot等）の大容量ファイルは `.wandbignore` で同期対象外とするオプションを提供

### Decision 3: 設定駆動命名とテンプレート機能のスコープ

**Context**: Requirement 4で定義された設定駆動の命名とテンプレート機能。

**Alternatives Considered**:
1. **Jinja2テンプレートエンジンの導入** → 強力だが、依存関係増加
2. **Python str.format()による単純な文字列置換** → 軽量、標準ライブラリのみ
3. **固定スキーマのみ（テンプレート機能なし）** → シンプルだが柔軟性不足

**Selected Approach**: Option 2（str.format()による文字列置換）

**Rationale**:
- Requirement 4.2の「テンプレート（例: `{method}_{variant}_{timestamp}`）」は基本的なプレースホルダー置換で十分
- 標準ライブラリの `str.format()` と `datetime` で実現可能
- 過度に複雑なテンプレートエンジンは保守コストに見合わない

**Trade-offs**:
- 利点: 軽量、学習コスト低、依存関係なし
- 欠点: 複雑な条件分岐やループは不可（ただし、現在の要件では不要）

**Follow-up**:
- `{timestamp}` は ISO8601 形式（`YYYYMMDD_HHMMSS`）で統一
- `{uuid4}` などの追加プレースホルダーは将来の拡張として検討

### Decision 4: パス解決システムの実装方式

**Context**: Requirement 3で定義されたwandB統合とパス解決のヘルパー関数。

**Alternatives Considered**:
1. **LightningModuleに直接統合** → 強い結合、テスト困難
2. **独立したユーティリティモジュール** → 疎結合、テスト容易、再利用可能
3. **既存の `SEDTask4` クラスのメソッド拡張** → 既存パターンを踏襲

**Selected Approach**: Option 2（独立したユーティリティモジュール） + Option 3（`SEDTask4` メソッド拡張）のハイブリッド

**Rationale**:
- `local/experiment_dir.py` として新しいモジュールを作成し、パス解決ロジックを集約
- `SEDTask4._init_wandb_project()` からこのモジュールを呼び出し、実験ディレクトリを設定
- `get_checkpoint_dir()`, `get_inference_dir()` などのヘルパー関数は `experiment_dir.py` に実装

**Trade-offs**:
- 利点: 単一責任原則、テスト容易性、再利用性
- 欠点: 新しいモジュールの導入（ただし、小規模で焦点が絞られている）

**Follow-up**:
- `experiment_dir.py` にはロギング機能を含め、ディレクトリ作成時にログ出力

### Decision 5: 実行モード別のwandB初期化制御

**Context**: Requirement 5（実行モード別のwandBログ管理）を実現するため、wandB初期化のタイミングと条件を決定。

**Alternatives Considered**:
1. **明示的なモード指定を必須化** → ユーザーが常に `--mode` 引数を指定、既存スクリプトとの互換性なし
2. **既存パラメータからの自動推論** → `evaluation`, `test_state_dict`, `fast_dev_run` から自動判定、後方互換性維持
3. **ハイブリッドアプローチ** → 明示的指定があればそれを優先、なければ自動推論

**Selected Approach**: Option 3（ハイブリッドアプローチ）

**Rationale**:
- 新しいコードでは明示的な `mode` 指定を推奨（YAML `experiment.mode` または `--mode` 引数）
- 既存のスクリプトでは自動推論で後方互換性を確保:
  - `evaluation=True` → `inference` モード
  - `fast_dev_run=True` → `train` モード（ただしwandB無効化）
  - `test_state_dict is not None` → `test` モード
  - それ以外 → `train` モード
- `_init_wandb_project()` メソッドで実行モードをチェックし、`inference` または `feature_extraction` モードではwandB初期化をスキップ

**Trade-offs**:
- 利点: 既存コードとの後方互換性、明示性と自動推論のバランス
- 欠点: 自動推論ロジックの保守コスト（ただし、単純な条件分岐のみ）

**Follow-up**:
- `test` モードでのwandB初期化は設定可能に（`log_test_to_wandb: false` でデフォルト無効化）
- manifest.jsonに `mode` フィールドを必須化し、後からログの目的を識別可能にする

## Risks & Mitigations

### Risk 1: 既存のwandB run履歴との互換性

**Mitigation**:
- コマンドライン引数 `--wandb_dir` が指定されている場合は従来の動作を維持（後方互換モード）
- 新しい設定項目 `experiment.category/method/variant` が存在する場合のみ、階層的ディレクトリ構造を適用

### Risk 2: 大容量可視化ファイルのwandB同期オーバーヘッド

**Mitigation**:
- `.wandbignore` ファイルを自動生成し、`visualizations/*.png` や `inference/*.wav` を同期対象外とするオプションを提供
- 設定ファイルに `wandb_sync_artifacts: false` フラグを追加

### Risk 3: パス長制限（Windows等）

**Mitigation**:
- Requirement 1.4で定義された検証機能を実装
- パス長が制限を超える場合は警告を出し、variant名の短縮を提案
- 最大パス長を260文字（Windows制限）以下に保つバリデーション

### Risk 4: 並行実験での競合

**Mitigation**:
- Requirement 1.5の「unique identifiers」としてwandbのrun IDを活用
- timestampだけでなくrun IDによる一意性を保証

## References

### Official Documentation
- [WandB API Reference: wandb.init](https://docs.wandb.ai/ref/python/init) — wandb.init()の公式パラメータリファレンス
- [PyTorch Lightning ModelCheckpoint](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html) — ModelCheckpointコールバックの詳細
- [WandB Configuration Management](https://docs.wandb.ai/guides/track/config) — 実験設定の管理方法

### Community Discussions
- [GitHub Issue #3077: Organize simulations in sub-folders](https://github.com/wandb/wandb/issues/3077) — wandBでの階層的フォルダ構造に関するコミュニティディスカッション
- [WandB Best Practices Guide](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTY1MjQz) — PyTorch Lightning + wandbのベストプラクティス

### Internal Project References
- `DESED_task/dcase2024_task4_baseline/local/sed_trainer_pretrained.py` — 既存のwandB統合実装
- `DESED_task/dcase2024_task4_baseline/train_pretrained.py` — トレーニングスクリプトとModelCheckpoint設定
- `.kiro/steering/structure.md` — プロジェクト構造の原則とパターン
