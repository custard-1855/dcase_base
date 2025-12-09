# ギャップ分析: 実験ディレクトリ構造リファクタリング

## 1. 現状調査サマリー

### 既存のディレクトリ構造

現在のプロジェクトでは、以下のような実験成果物の保存構造が採用されている:

```
DESED_task/dcase2024_task4_baseline/
├── exp/
│   └── 2024_baseline/               # TensorBoard logger による命名
│       └── version_X/               # 自動バージョニング
│           ├── checkpoints/
│           └── hparams.yaml
├── wandb/
│   ├── run-{timestamp}-{id}/        # wandbデフォルト形式
│   │   ├── checkpoints/             # 新規追加された統合ディレクトリ
│   │   ├── files/
│   │   └── logs/
│   └── latest-run -> symlink
├── visualize/get_features/
│   ├── inference_configs/
│   └── inference_outputs/           # 推論結果（手動管理）
└── output/                          # カスタム出力（用途不明瞭）
```

### 既存のwandb統合状況

#### train_pretrained.py (lines 504-508, 539-545)
```python
logger = TensorBoardLogger(
    os.path.dirname(config["log_dir"]),
    config["log_dir"].split("/")[-1],
)

# wandbが有効な場合はwandbのcheckpointディレクトリを使用
if hasattr(desed_training, '_wandb_checkpoint_dir') and desed_training._wandb_checkpoint_dir:
    checkpoint_dir = desed_training._wandb_checkpoint_dir
else:
    checkpoint_dir = logger.log_dir
```

#### sed_trainer_pretrained.py (lines 182-186, 417-441)
```python
if self.hparams["net"]["use_wandb"]:
    self._init_wandb_project()
else:
    self._wandb_checkpoint_dir: str | None = None

def _init_wandb_project(self) -> None:
    wandb.init(project=PROJECT_NAME, name=self.hparams["net"]["wandb_dir"])

    if wandb.run is not None:
        self._wandb_checkpoint_dir = os.path.join(wandb.run.dir, "checkpoints")
        os.makedirs(self._wandb_checkpoint_dir, exist_ok=True)
```

**重要な発見**:
- wandbの `name` パラメータにコマンドライン引数 `--wandb_dir` が直接渡されている
- これが実質的な「実験名」として機能しているが、階層的構造は未サポート
- wandbは内部的に `run-{timestamp}-{id}` 形式のディレクトリを作成
- チェックポイントは `wandb.run.dir/checkpoints/` に保存されるよう修正済み

### 実験命名の現状

#### コマンドライン引数による制御 (run_exp_cmt.sh, train_pretrained.py)
```bash
# run_exp_cmt.sh (line 4, 40)
BASE_WANDB_DIR="150/cmt_apply-unlabeled/"
uv run train_pretrained.py \
    --wandb_dir ${BASE_WANDB_DIR}/CMT_use_neg_sample \
    --cmt --use_neg_sample

# train_pretrained.py (lines 702-738)
parser.add_argument("--wandb_dir")
if args.wandb_dir is not None:
    configs["net"]["wandb_dir"] = args.wandb_dir
```

**実験命名パターンの実態**:
- スラッシュ区切りで疑似的な階層構造を表現: `"150/cmt_apply-unlabeled/CMT_use_neg_sample"`
- しかし、ファイルシステム上は `wandb/run-{timestamp}-{id}/` という平坦な構造
- 階層情報はwandb UIのrun名としてのみ保持される

### 設定ファイルの現状 (pretrained.yaml)

```yaml
net:
  use_wandb: False                   # デフォルトではwandb無効
  wandb_dir: "None"                  # 文字列 "None"（実質未設定）

# 実験パラメータはYAMLに直接記載
cmt:
  enabled: False
  phi_frame: 0.5
sebbs:
  enabled: false
```

**設定駆動の実装状況**:
- CMT/SEBBsなどの実験パラメータはYAMLで管理可能
- しかし、実験名の階層構造（category/method/variant）はYAMLに記載されていない
- コマンドライン引数が実質的な実験名制御手段となっている

---

## 2. 要件と既存資産のマッピング

### Requirement 1: 階層的実験ディレクトリ構造

| 受入基準 | 既存資産 | ギャップ |
|---------|---------|---------|
| AC1: `experiments/{category}/{method}/{variant}/` 構造を作成 | ❌ なし | **Missing** - ディレクトリ作成ロジックが存在しない |
| AC2: 親ディレクトリの自動作成 | ⚠️ `os.makedirs(self._wandb_checkpoint_dir, exist_ok=True)` (L424) | **Extend** - wandb内部用、汎用化が必要 |
| AC3: 3階層サポート (category/method/variant) | ❌ なし | **Missing** - 階層構造の概念がない |
| AC4: ファイルシステム互換性検証 | ❌ なし | **Missing** - パス検証機能が未実装 |
| AC5: 同一階層での一意性確保（タイムスタンプ/カウンタ付加） | ✅ wandb自動生成: `run-{timestamp}-{id}` | **Constraint** - wandbの命名規則と統合が必要 |

**分析**:
- 既存のwandb統合は平坦な構造を前提としている
- 階層的構造の実装は新規機能として追加が必要
- wandbの自動ID生成機能は維持すべき（一意性保証のため）

### Requirement 2: 実験成果物の統合管理

| 受入基準 | 既存資産 | ギャップ |
|---------|---------|---------|
| AC1: `{experiment_dir}/checkpoints/` にチェックポイント保存 | ✅ `self._wandb_checkpoint_dir` (L423-424) | **Extend** - wandb専用、汎用化が必要 |
| AC2: `{experiment_dir}/metrics/` にメトリクス保存 | ⚠️ wandb.log() による自動保存 | **Integrate** - ファイルとして保存する仕組みが必要 |
| AC3: `{experiment_dir}/inference/` に推論結果保存 | ⚠️ `visualize/get_features/inference_outputs/` (分離) | **Relocate** - 実験ディレクトリ内に移動 |
| AC4: `{experiment_dir}/visualizations/` に可視化保存 | ⚠️ `visualize/` 配下に分散 | **Relocate** - 実験ディレクトリ内に移動 |
| AC5: `{experiment_dir}/config/` に設定スナップショット保存 | ⚠️ wandb自動保存 + `hparams.yaml` | **Extend** - 明示的な設定保存機構 |
| AC6: マニフェストファイル生成 | ❌ なし | **Missing** - 新規実装が必要 |

**分析**:
- チェックポイント管理は部分的に実装済み（wandb連携部分のみ）
- メトリクスはwandb経由でクラウド保存されるが、ローカルファイルとしての保存は未実装
- 推論結果と可視化は実験とは独立した場所に保存されている（構造的な分離）
- マニフェストファイル生成は新規機能

### Requirement 3: wandb統合とパス解決

| 受入基準 | 既存資産 | ギャップ |
|---------|---------|---------|
| AC1: wandbデフォルトディレクトリを上書き | ⚠️ `wandb.init(name=...)` のみ | **Missing** - `dir` パラメータ未使用 |
| AC2: 初期化時にカスタム `dir` パラメータを注入 | ❌ なし | **Missing** - 実装が必要 |
| AC3: wandb run IDと実験パスのマッピング維持 | ⚠️ `wandb.run.dir` 使用 | **Constraint** - 逆方向マッピング（ID→パス）が未実装 |
| AC4: パス解決ヘルパー関数提供 | ❌ なし | **Missing** - `get_checkpoint_dir()` 等の実装が必要 |
| AC5: run IDからの実験パス解決（100ms以内） | ❌ なし | **Missing** - マッピングテーブルまたはメタデータファイルが必要 |

**分析**:
- 現在の実装は wandb の自動ディレクトリ作成に依存
- カスタムディレクトリへの配置機能は未実装
- パス解決機能が存在しないため、実験成果物の参照が困難

### Requirement 4: 設定駆動の実験命名

| 受入基準 | 既存資産 | ギャップ |
|---------|---------|---------|
| AC1: YAML設定で実験命名パラメータをサポート | ⚠️ 部分的: `cmt`, `sebbs` パラメータのみ | **Extend** - `category`, `method`, `variant` フィールドが必要 |
| AC2: テンプレート機能（`{method}_{variant}_{timestamp}`） | ❌ なし | **Missing** - テンプレート展開機能が必要 |
| AC3: 設定ロード時の命名パラメータ検証 | ❌ なし | **Missing** - バリデーション機能が必要 |
| AC4: デフォルト値のフォールバック | ❌ なし | **Missing** - デフォルト値設定が必要 |
| AC5: 環境変数置換（`$SCRATCH_DIR/experiments`） | ❌ なし | **Missing** - 環境変数展開機能が必要 |

**分析**:
- 現在の設定ファイルは実験パラメータの管理には対応
- 実験命名の階層構造はコマンドライン引数に依存（YAMLには未記載）
- テンプレート機能や環境変数置換は未実装

---

## 3. 実装アプローチの評価

### Option A: 既存コンポーネントの拡張

**拡張対象ファイル**:
1. `sed_trainer_pretrained.py` (L417-441: `_init_wandb_project`)
2. `train_pretrained.py` (L504-508: logger設定, L539-545: checkpoint_dir設定)

**拡張内容**:
```python
# sed_trainer_pretrained.py
def _init_wandb_project(self) -> None:
    # 新機能: 階層的実験パスの構築
    exp_path = self._build_experiment_path()  # category/method/variant

    # wandbのdirパラメータに実験パスを指定
    wandb.init(
        project=PROJECT_NAME,
        name=self.hparams["experiment"]["name"],
        dir=exp_path,  # ← 新規追加
    )

    # サブディレクトリ作成
    self._create_experiment_subdirs(exp_path)

    # マッピングファイル生成
    self._save_experiment_metadata(wandb.run.id, exp_path)
```

**互換性評価**:
- ✅ 既存のwandb連携コードを活用可能
- ✅ TensorBoardLoggerとの共存が可能
- ⚠️ wandb無効時のフォールバック処理が必要
- ❌ `visualize/` 配下の推論スクリプトは別途修正が必要

**複雑性とメンテナンス性**:
- **認知負荷**: 中程度 - `_init_wandb_project` の責務が増加（パス構築、マッピング管理）
- **単一責任原則**: やや違反 - wandb初期化とディレクトリ管理が混在
- **ファイルサイズ**: `sed_trainer_pretrained.py` が肥大化（現在 ~1400行 → 推定 ~1600行）

**トレードオフ**:
- ✅ 新規ファイル数が少ない（開発速度向上）
- ✅ 既存のwandb統合パターンを踏襲（学習コスト低）
- ❌ `SEDTask4` クラスの責務が過剰になる
- ❌ 実験管理ロジックがLightningModuleに埋め込まれる（再利用性低下）

---

### Option B: 新規コンポーネントの作成

**新規作成ファイル**:
```
local/
├── experiment_manager.py         # 実験ディレクトリ管理
│   ├── ExperimentPathBuilder     # パス構築ロジック
│   ├── ExperimentArtifactManager # 成果物保存・マニフェスト生成
│   └── ExperimentMetadata        # run ID <-> パス マッピング
└── config_utils.py               # 設定テンプレート展開
    └── ExperimentConfigResolver   # YAML設定 + テンプレート + 環境変数
```

**統合ポイント**:
```python
# sed_trainer_pretrained.py
from local.experiment_manager import ExperimentPathBuilder, ExperimentArtifactManager

def _init_wandb_project(self) -> None:
    # 実験パス構築を専用クラスに委譲
    path_builder = ExperimentPathBuilder(self.hparams["experiment"])
    exp_path = path_builder.build_path()

    wandb.init(project=PROJECT_NAME, name=path_builder.get_name(), dir=exp_path)

    # 成果物管理を専用クラスに委譲
    self.artifact_manager = ExperimentArtifactManager(exp_path)
    self.artifact_manager.setup_directories()
    self.artifact_manager.save_metadata(wandb.run.id)
```

**責務境界**:
- **ExperimentPathBuilder**: 階層パス構築、一意性確保、パス検証
- **ExperimentArtifactManager**: サブディレクトリ作成、成果物保存、マニフェスト生成
- **ExperimentMetadata**: run ID <-> パス マッピングの永続化・検索
- **SEDTask4**: 学習ロジック、メトリクス計算（実験管理から分離）

**トレードオフ**:
- ✅ 関心の分離が明確（テスト容易性向上）
- ✅ 再利用性が高い（他プロジェクトへの展開可能）
- ✅ `SEDTask4` の責務が軽減（単一責任原則）
- ❌ ファイル数が増加（ナビゲーション複雑化）
- ❌ インターフェース設計に時間が必要

---

### Option C: ハイブリッドアプローチ

**段階的実装戦略**:

**Phase 1: 最小限の拡張（MVP）**
- `sed_trainer_pretrained.py` に `_build_experiment_path()` メソッドを追加
- wandb の `dir` パラメータをカスタムパスに設定
- 基本的なサブディレクトリ作成（checkpoints/metrics/config のみ）
- **対象要件**: Requirement 1 (AC1-3), Requirement 2 (AC1, AC5)

**Phase 2: マッピング機能の追加**
- `local/experiment_metadata.py` を新規作成
- run ID <-> パス マッピングの永続化
- パス解決ヘルパー関数の実装
- **対象要件**: Requirement 3 (AC3-5)

**Phase 3: 成果物管理の統合**
- `visualize/` スクリプトを修正し、実験ディレクトリ内に保存
- マニフェストファイル生成機能
- **対象要件**: Requirement 2 (AC3-4, AC6)

**Phase 4: 設定駆動の実装**
- `confs/pretrained.yaml` に `experiment` セクション追加
- テンプレート展開と環境変数置換
- **対象要件**: Requirement 4 (全AC)

**リスク軽減**:
- ✅ 段階的なロールアウト（既存機能への影響を最小化）
- ✅ 各フェーズで動作確認可能（増分テスト）
- ✅ 必要に応じてリファクタリング（Phase 2以降で設計改善）
- ⚠️ Phase 1が肥大化した場合、Phase 2でリファクタリングコストが発生

**トレードオフ**:
- ✅ 初期開発速度とメンテナンス性のバランス
- ✅ リスク分散（段階的な検証）
- ❌ 設計の一貫性が損なわれる可能性（Phase間で設計思想が変わるリスク）
- ❌ 複数フェーズにわたる調整コスト

---

## 4. 複雑性とリスクの評価

### 実装規模: M (Medium, 3-7日)

**根拠**:
- 既存の wandb 統合パターンは理解済み（調査完了）
- パス構築ロジックは標準的な文字列操作（複雑性低）
- 既存の `ModelCheckpoint` コールバックとの統合が必要（中程度の複雑性）
- マッピングファイルの永続化は軽量（JSON/YAML形式で十分）
- 推論スクリプトの修正は影響範囲が限定的

**タスク分解**:
1. 階層的パス構築ロジック (1日)
2. wandb統合とサブディレクトリ作成 (1日)
3. マッピング機能とヘルパー関数 (1日)
4. 設定ファイル拡張とテンプレート展開 (1日)
5. 推論スクリプト修正 (1日)
6. テストとドキュメント (2日)

### リスク: Medium

**技術的リスク**:
- **wandb の `dir` パラメータの挙動**: wandb は指定ディレクトリ配下に `wandb/run-*` を作成する可能性
  - **軽減策**: wandb ドキュメント確認 + 事前検証実験
- **PyTorch Lightning の ModelCheckpoint との統合**: チェックポイントパスの設定タイミングが不適切な場合、保存先が二重管理される
  - **軽減策**: `train_pretrained.py` L539-545 の既存ロジックを踏襲
- **既存実験との後方互換性**: wandb無効時のフォールバックが不適切な場合、既存の実験スクリプトが動作しなくなる
  - **軽減策**: `use_wandb=False` 時は従来のTensorBoardLogger動作を保証

**運用リスク**:
- **ディレクトリ命名の一貫性**: 手動でのコマンドライン引数指定に依存すると、命名規則が統一されない
  - **軽減策**: YAML設定に `experiment.category/method/variant` を必須化
- **ディスク容量**: 実験成果物が統合管理されることで、1実験あたりのディスク使用量が増加
  - **軽減策**: マニフェストファイルでサイズ監視、古い実験の自動アーカイブ機能（将来拡張）

**スケジュールリスク**:
- **推論スクリプトの影響範囲**: `visualize/` 配下の複数スクリプトが `inference_outputs/` を参照している場合、修正範囲が拡大
  - **軽減策**: Grep で全参照箇所を特定済み（2ファイルのみ: `check_feature_properties.py`, `extract_inference_features.py`）

---

## 5. 既知の制約と依存関係

### アーキテクチャ制約

1. **PyTorch Lightning フレームワーク**:
   - `SEDTask4` は `pl.LightningModule` を継承
   - `__init__` でのwandb初期化タイミングが固定（L182-186）
   - **影響**: 実験パス構築ロジックも初期化時に実行する必要がある

2. **wandb ライフサイクル**:
   - `wandb.init()` は1プロセスにつき1回のみ呼び出し可能
   - `wandb.run.dir` は初期化後に確定
   - **影響**: パス構築は `wandb.init()` より前に完了する必要がある

3. **TensorBoardLogger との共存**:
   - `train_pretrained.py` L504-508 で TensorBoardLogger を明示的に使用
   - wandb無効時はこれがメインのロガー
   - **影響**: wandb無効時のフォールバックとして TensorBoardLogger の `log_dir` を使用

### 既存の統合ポイント

1. **ModelCheckpoint コールバック** (`train_pretrained.py` L554-561):
   ```python
   ModelCheckpoint(
       checkpoint_dir,  # ← wandb有効時は _wandb_checkpoint_dir
       monitor="val/obj_metric",
       save_top_k=1,
   )
   ```
   - **依存**: `checkpoint_dir` の値に基づいてファイルが保存される
   - **統合要件**: 新しい実験ディレクトリ構造に合わせて `checkpoint_dir` を設定

2. **コマンドライン引数解析** (`train_pretrained.py` L702-738):
   ```python
   parser.add_argument("--wandb_dir")
   configs["net"]["wandb_dir"] = args.wandb_dir
   ```
   - **依存**: 既存の `--wandb_dir` 引数が実験名として機能
   - **統合要件**: 新しい階層的命名規則との互換性を保つ

3. **推論特徴量抽出** (`visualize/get_features/extract_inference_features.py`):
   ```python
   parser.add_argument("--output_dir", default="inference_outputs/baseline")
   ```
   - **依存**: 固定パスに推論結果を保存
   - **統合要件**: 実験ディレクトリ内に動的にパスを生成する仕組みが必要

### パフォーマンス制約

- **AC5 (Requirement 3)**: run IDからの実験パス解決を100ms以内で実行
  - **実装オプション**:
    - JSON/YAMLマッピングファイル（単純、十分高速）
    - SQLiteデータベース（オーバーキル）
  - **推奨**: JSON形式のマッピングファイル（`experiments_metadata.json`）

---

## 6. 要調査項目（Research Needed）

以下の項目は設計フェーズで詳細調査が必要:

### 1. wandb の `dir` パラメータ挙動
- **質問**: `wandb.init(dir="experiments/cat1/method1/var1")` を指定した場合、wandbは:
  - a) `experiments/cat1/method1/var1/` 直下に成果物を保存するか
  - b) `experiments/cat1/method1/var1/wandb/run-*` という階層を作成するか
- **調査方法**: 小規模な検証スクリプトで実験
- **設計への影響**: ディレクトリ構造の最終形に影響

### 2. TensorBoardLogger と wandb の同時使用
- **質問**: 両方のロガーが有効な場合、ログファイルの保存場所はどうなるか
- **調査方法**: 既存コードで `use_wandb=True` かつ TensorBoardLogger 有効時の動作確認
- **設計への影響**: ログファイルの統合管理方法

### 3. 既存実験データの移行
- **質問**: 既に `wandb/run-*/` 形式で保存された実験データを新構造に移行するか
- **調査方法**: ステークホルダーへのヒアリング
- **設計への影響**: マイグレーションスクリプトの必要性

### 4. マニフェストファイルのスキーマ
- **質問**: マニフェストに含めるメタデータの詳細（ファイルハッシュ、Git commit ID など）
- **調査方法**: 類似プロジェクトのベストプラクティス調査（MLflow, DVC など）
- **設計への影響**: マニフェスト生成ロジックの複雑性

---

## 7. 推奨実装戦略と次ステップ

### 推奨アプローチ: **Option C (ハイブリッド)** を採用

**理由**:
1. **リスク分散**: 段階的実装により、各フェーズで動作確認が可能
2. **柔軟性**: Phase 1で最小限の機能を提供し、フィードバックに基づいてPhase 2以降を調整
3. **メンテナンス性**: Phase 2以降で新規コンポーネントを導入することで、最終的に Option B の設計品質を達成
4. **開発速度**: Phase 1は既存コードの拡張で迅速に実装可能

### Phase 1 実装スコープ (MVP)

**含まれる機能**:
- ✅ 階層的実験パス構築（`experiments/{category}/{method}/{variant}/`）
- ✅ wandb の `dir` パラメータ設定
- ✅ 基本サブディレクトリ作成（checkpoints/metrics/config）
- ✅ 設定スナップショット保存
- ✅ wandb無効時のフォールバック（TensorBoardLogger経由）

**除外される機能** (Phase 2以降):
- ❌ run ID <-> パス マッピング（手動での実験検索で代替）
- ❌ 推論結果の自動統合（既存の `inference_outputs/` を継続使用）
- ❌ マニフェストファイル生成
- ❌ YAML設定のテンプレート展開

**Phase 1 成功基準**:
1. `--wandb_dir "category/method/variant"` を指定した場合、`experiments/category/method/variant/run-*/` が作成される
2. チェックポイントが `experiments/category/method/variant/run-*/checkpoints/` に保存される
3. 設定ファイルが `experiments/category/method/variant/run-*/config/` に保存される
4. `use_wandb=False` の場合、従来通り `exp/2024_baseline/` に保存される

### 設計フェーズへの引継ぎ事項

**要調査項目の優先順位**:
1. **高**: wandb `dir` パラメータの挙動（Phase 1実装に直接影響）
2. **中**: 既存実験データの移行方針（ステークホルダー判断が必要）
3. **低**: マニフェストスキーマ（Phase 3で詳細化）

**設計判断が必要な項目**:
- YAML設定の `experiment` セクションのスキーマ設計（category/method/variant の必須/任意）
- コマンドライン引数とYAML設定の優先順位（引数 > YAML か、エラーにするか）
- 一意性確保のためのタイムスタンプフォーマット（ISO 8601 vs Unix timestamp）

**既存コードへの影響範囲**:
- **低影響**: `sed_trainer_pretrained.py` (1メソッド追加、既存メソッド修正)
- **低影響**: `train_pretrained.py` (checkpoint_dir設定箇所のみ)
- **影響なし**: `desed_task/` (コアライブラリは変更不要)
- **Phase 2以降**: `visualize/` (推論スクリプト群)

---

## 8. まとめ

### ギャップ分析の結論

現在のコードベースは、wandb統合の基礎は実装されているものの、**階層的な実験管理機能は全体的に欠落している**。以下の点が主要なギャップ:

1. **ディレクトリ構造**: 平坦な `wandb/run-*` 形式 → 階層的な `experiments/{category}/{method}/{variant}/` への変換が必要
2. **パス解決**: run ID から実験パスを引く仕組みが存在しない
3. **成果物統合**: 推論結果・可視化が実験とは別の場所に保存されている
4. **設定駆動**: 実験名の階層構造がYAMLに記載されておらず、コマンドライン引数に依存

### 実装の実現可能性

**実現可能**: すべての要件は既存技術スタックで実装可能
- wandb の柔軟な設定オプション（`dir`, `name`）を活用
- PyTorch Lightning の拡張ポイント（`__init__`, callbacks）を利用
- 既存のYAML設定機構を拡張

**推定工数**: **Medium (5-7日)**
- Phase 1 MVP: 3日
- テスト・調整: 2日
- ドキュメント: 1日

**推奨される次のステップ**:
1. `/kiro:spec-design refactor-experiment-structure` を実行し、技術設計書を作成
2. Phase 1 実装範囲の詳細設計（クラス設計、インターフェース定義）
3. wandb `dir` パラメータの挙動検証（小規模実験）
4. Phase 1 実装開始

---

**生成日時**: 2025-12-09
**対象Specification**: refactor-experiment-structure
**フェーズ**: Gap Analysis Complete → Design Phase Ready
