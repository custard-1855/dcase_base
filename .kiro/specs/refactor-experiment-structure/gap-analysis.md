# Gap Analysis: refactor-experiment-structure

## Analysis Summary

**スコープ**: wandbのデフォルトディレクトリ構造から、カスタム階層的実験ディレクトリ構造への移行

**主要な課題**:
- wandb統合コードの変更（現在wandb.initで自動ディレクトリ生成）
- チェックポイント、推論結果、可視化ツールの複数箇所への散在したパス参照
- 設定ファイル駆動の実験命名システムの不在
- 既存の3つの異なる出力ディレクトリ（`wandb/`, `exp/`, `visualize/visualization_outputs/`）の統合

**推奨アプローチ**: Hybrid (Option C) - 既存コンポーネントの拡張と新規パス管理モジュールの作成

---

## 1. Current State Investigation

### 1.1 Key Files and Directory Layout

**実験管理関連の主要ファイル**:
- `local/sed_trainer_pretrained.py`: PyTorch Lightning module（wandb初期化、チェックポイント管理）
- `train_pretrained.py`: トレーニングエントリーポイント（ModelCheckpoint設定、ロガー初期化）
- `confs/pretrained.yaml`: 設定ファイル（wandb関連設定を含む）
- `visualize/get_features/extract_inference_features.py`: 推論結果の保存
- `visualize/visualize_*.py`: 可視化スクリプト（UMAP、信頼性ダイアグラム、Grad-CAM）

**現在の出力ディレクトリ構造**:
```
DESED_task/dcase2024_task4_baseline/
├── exp/                          # TensorBoardLogger default output
│   └── {experiment_name}/
├── wandb/                        # wandb default structure
│   └── run-{timestamp}-{id}/
│       ├── files/
│       └── checkpoints/          # カスタム追加（_init_wandb_project内）
└── visualize/
    ├── inference_outputs/        # 推論結果（暗黙的）
    │   └── {model_name}/
    └── visualization_outputs/    # 可視化結果（ハードコード）
        ├── umap/
        ├── reliability/
        └── gradcam/
```

### 1.2 Dominant Architecture Patterns

**パス生成パターン**:
1. **wandb自動ディレクトリ生成** (`sed_trainer_pretrained.py:412-418`):
   ```python
   wandb.init(project=PROJECT_NAME, name=self.hparams["net"]["wandb_dir"])
   self._wandb_checkpoint_dir = os.path.join(wandb.run.dir, "checkpoints")
   os.makedirs(self._wandb_checkpoint_dir, exist_ok=True)
   ```

2. **TensorBoardLogger使用** (`train_pretrained.py:504-508`):
   ```python
   logger = TensorBoardLogger(
       os.path.dirname(config["log_dir"]),
       config["log_dir"].split("/")[-1],
   )
   ```

3. **ModelCheckpoint動的ディレクトリ設定** (`train_pretrained.py:540-544`):
   ```python
   if hasattr(desed_training, '_wandb_checkpoint_dir') and desed_training._wandb_checkpoint_dir:
       checkpoint_dir = desed_training._wandb_checkpoint_dir
   else:
       checkpoint_dir = logger.log_dir
   ```

4. **可視化ツールのハードコードパス** (`visualize_umap.py:9-10`):
   ```python
   --input_dirs inference_outputs/baseline inference_outputs/cmt_normal
   --output_dir visualization_outputs/umap
   ```

**依存関係パターン**:
- PyTorch Lightning → wandb/TensorBoard → ファイルシステム
- 可視化ツール → 推論結果の暗黙的パス（`inference_outputs/`）
- チェックポイントローダー → `inference_metadata.json`内のチェックポイントパス

### 1.3 Integration Surfaces

**現在の統合ポイント**:
- `SEDTask4._init_wandb_project()`: wandb初期化とチェックポイントディレクトリ作成
- `train_pretrained.py:single_run()`: ロガーとコールバック設定
- `ModelCheckpoint`: チェックポイント保存パス
- 可視化スクリプト: `--input_dirs`引数でデータロード

**データフロー**:
```
設定ファイル (YAML)
    ↓
train_pretrained.py (ロガー初期化、checkpoint_dir決定)
    ↓
SEDTask4 (wandb.init、_wandb_checkpoint_dir設定)
    ↓
ModelCheckpoint (best.ckpt保存)
    ↓
推論スクリプト (inference_metadata.json生成)
    ↓
可視化スクリプト (結果ロード、可視化出力)
```

---

## 2. Requirements Feasibility Analysis

### 2.1 Requirements Mapping

#### Requirement 1: 階層的実験ディレクトリ構造

**技術的ニーズ**:
- パス生成ロジック: `experiments/{category}/{method}/{variant}/`
- ディレクトリ検証: ファイルシステム互換性チェック
- 衝突解決: タイムスタンプ/カウンター追加ロジック

**Gaps**:
- ❌ **Missing**: カスタムディレクトリレイアウト生成ロジック（現在wandb/exp固定）
- ❌ **Missing**: ファイル名検証ユーティリティ（OS依存の無効文字チェック）
- ⚠️ **Constraint**: PyTorch Lightning 1.9.xの`Trainer.logger`との互換性保持

**実現可能性**: ✅ **High** - 既存のパス操作パターン（`os.makedirs`使用42箇所）を拡張可能

---

#### Requirement 2: 実験成果物の統合管理

**技術的ニーズ**:
- サブディレクトリ構造: `checkpoints/`, `metrics/`, `inference/`, `visualizations/`, `config/`
- マニフェスト生成: 実験終了時のアーティファクトメタデータ

**Gaps**:
- ❌ **Missing**: 統合ディレクトリ構造（現在3箇所に散在）
- ❌ **Missing**: アーティファクトマニフェスト生成ロジック
- ⚠️ **Constraint**: TensorBoard/wandb既存ログとの互換性

**実現可能性**: ✅ **Medium-High** - 既存の保存ロジックをラップして統合可能だが、複数箇所の修正が必要

---

#### Requirement 3: wandb統合とパス解決

**技術的ニーズ**:
- wandb `dir`パラメータオーバーライド
- シンボリックリンク/メタデータマッピング（run ID ↔ 実験パス）
- ヘルパー関数: `get_checkpoint_dir()`, `get_inference_dir()`

**Gaps**:
- ❌ **Missing**: パス解決システム（現在は属性直接参照）
- ⚠️ **Unknown**: wandbの`dir`パラメータ使用時の既存機能への影響（トラッキング、自動アップロード）
- ✅ **Existing**: `_wandb_checkpoint_dir`属性（部分的解決）

**実現可能性**: ⚠️ **Medium** - wandb APIとの統合テストが必要（Research Needed）

**Research Needed**:
- wandb.init(dir=...)使用時のファイル同期動作
- シンボリックリンク vs メタデータファイルの性能トレードオフ

---

#### Requirement 4: 設定駆動の実験命名

**技術的ニーズ**:
- YAML設定拡張: `experiment.category`, `experiment.method`, `experiment.variant`
- テンプレート展開: `{method}_{variant}_{timestamp}`
- 環境変数展開: `$SCRATCH_DIR/experiments`

**Gaps**:
- ❌ **Missing**: 実験命名設定スキーマ（現在`wandb_dir`のみ）
- ❌ **Missing**: テンプレート展開ロジック
- ✅ **Existing**: YAML設定読み込み基盤（`train_pretrained.py:707-708`）

**実現可能性**: ✅ **High** - 既存の設定システムを拡張するのみ

---

### 2.2 Complexity Signals

**タスクタイプ**: Workflow (複数コンポーネント間の連携変更)

**複雑度指標**:
- 変更が必要なファイル数: 5-8ファイル（trainer, train script, visualize scripts, config）
- 統合ポイント数: 4箇所（wandb, TensorBoard, ModelCheckpoint, visualize tools）
- 外部依存: wandb, PyTorch Lightning, pathlib/os

---

## 3. Implementation Approach Options

### Option A: Extend Existing Components

**拡張対象ファイル**:
- `local/sed_trainer_pretrained.py`: `_init_wandb_project()`メソッド拡張
- `train_pretrained.py`: ロガー初期化部分の修正
- `confs/pretrained.yaml`: 実験命名パラメータ追加

**互換性評価**:
- ✅ 既存の`_wandb_checkpoint_dir`属性パターンを踏襲
- ✅ YAML設定拡張は後方互換（デフォルト値提供）
- ⚠️ wandb/TensorBoardロガーとの統合動作要確認

**複雑度と保守性**:
- ✅ 最小限の新規ファイル（パス生成ユーティリティのみ追加）
- ⚠️ `sed_trainer_pretrained.py`の責任範囲拡大（パス管理ロジック追加）
- ⚠️ 可視化スクリプトは各々個別修正が必要

**Trade-offs**:
- ✅ 学習曲線が緩やか（既存パターン踏襲）
- ✅ 既存インフラ（wandb/TensorBoard）活用
- ❌ トレーナーモジュールの肥大化リスク
- ❌ パス管理ロジックの再利用性低下

---

### Option B: Create New Components

**新規作成コンポーネント**:
- `local/experiment_manager.py`: 実験ディレクトリ構造管理クラス
  ```python
  class ExperimentManager:
      def __init__(self, config: dict):
          self.category = config["experiment"]["category"]
          self.method = config["experiment"]["method"]
          self.variant = config["experiment"]["variant"]
          self.base_dir = Path(config["experiment"]["base_dir"])

      def get_experiment_dir(self) -> Path: ...
      def get_checkpoint_dir(self) -> Path: ...
      def get_inference_dir(self) -> Path: ...
      def get_visualization_dir(self) -> Path: ...
      def create_manifest(self) -> None: ...
  ```
- `local/experiment_config.py`: 設定検証とテンプレート展開
- `local/experiment_paths.py`: パス解決ヘルパー

**統合ポイント**:
- `SEDTask4.__init__()`: ExperimentManager初期化
- `train_pretrained.py:single_run()`: ExperimentManagerからパス取得
- 可視化スクリプト: `experiment_paths`ユーティリティインポート

**責任境界**:
- `ExperimentManager`: ディレクトリ構造生成、アーティファクト管理
- `SEDTask4`: モデルトレーニングロジック（パス管理から分離）
- `experiment_paths`: 読み取り専用パス解決（可視化ツール向け）

**Trade-offs**:
- ✅ 明確な責任分離（Single Responsibility Principle）
- ✅ 可視化ツールでの再利用性高い
- ✅ テスト容易性向上
- ❌ 新規ファイル追加（学習コスト）
- ❌ 既存コードとの統合箇所増加

---

### Option C: Hybrid Approach ⭐ **Recommended**

**戦略**:
1. **Phase 1**: コア機能実装（新規コンポーネント作成）
   - `local/experiment_manager.py`: ExperimentManager作成
   - `confs/pretrained.yaml`: 実験命名設定追加（デフォルト値で後方互換）
   - `SEDTask4`: ExperimentManagerインスタンス化とwandб統合

2. **Phase 2**: 既存コンポーネント統合
   - `train_pretrained.py`: ModelCheckpointパスをExperimentManager経由に変更
   - `sed_trainer_pretrained.py`: `exp_dir`プロパティをExperimentManager.get_experiment_dir()にリダイレクト

3. **Phase 3**: 可視化ツール移行（段階的）
   - `visualize/get_features/extract_inference_features.py`: 出力パスをExperimentManager経由に
   - 他の可視化スクリプトは後方互換性保持（オプショナル引数で新パス対応）

**段階的実装理由**:
- Phase 1で基盤完成（トレーニングワークフロー中断なし）
- Phase 2でメインワークフロー移行
- Phase 3は既存スクリプトを壊さない（段階的移行可能）

**リスク軽減**:
- 設定にフィーチャーフラグ追加: `experiment.use_custom_structure: false` (デフォルト)
- 既存の`wandb/`ディレクトリも並行利用可能（移行期間中）
- ExperimentManager内でフォールバックロジック実装

**Trade-offs**:
- ✅ 段階的ロールアウト（リスク分散）
- ✅ 既存ワークフロー中断なし
- ✅ 新規システムの段階的検証
- ⚠️ 移行期間中の複雑性（2つのパスシステム併存）
- ⚠️ フィーチャーフラグ管理のオーバーヘッド

---

## 4. Implementation Complexity & Risk

### Effort Estimation

**Size**: **M (Medium, 3-7 days)**

**内訳**:
- ExperimentManager実装: 1-2日（ディレクトリ生成、マニフェスト、パス解決）
- SEDTask4/train_pretrained統合: 1-2日（wandb統合、ModelCheckpoint修正）
- 設定システム拡張: 0.5-1日（YAML検証、テンプレート展開）
- 可視化ツール修正: 1-2日（3-4スクリプトのパス参照変更）
- テスト・検証: 1日（統合テスト、既存実験との互換性確認）

**根拠**: 既存パターン踏襲（`os.makedirs`パターン42箇所）により複雑なファイルシステム操作なし。wandb統合は既存コード（`_init_wandb_project`）をベースに拡張。

---

### Risk Assessment

**Risk Level**: **Medium**

**High-risk Areas**:
1. **wandb統合の予期しない動作** (Medium)
   - `wandb.init(dir=...)`使用時のファイル同期への影響不明
   - **Mitigation**: 早期プロトタイプでwandб動作検証、公式ドキュメント確認

2. **可視化ツールのパス依存** (Medium)
   - `inference_outputs/`ディレクトリへのハードコード参照が複数箇所
   - **Mitigation**: Phase 3で段階的移行、後方互換性保持

3. **ModelCheckpointとwandbの相互作用** (Low-Medium)
   - カスタムチェックポイントディレクトリとwandб自動アップロード
   - **Mitigation**: `wandb.save()`明示的呼び出しで対応

**Low-risk Areas**:
- YAML設定拡張（既存システム安定）
- ディレクトリ生成ロジック（標準ライブラリのみ使用）
- パス解決ヘルパー（読み取り専用、副作用なし）

---

## 5. Recommendations for Design Phase

### Preferred Approach

**Option C (Hybrid)** を推奨:
- Phase 1-2でコアワークフローを移行（Medium effort, Medium risk）
- Phase 3の可視化ツール移行は後続タスクとして分離可能
- フィーチャーフラグにより既存システムとの並行稼働

### Key Decisions to Make in Design Phase

1. **wandb統合戦略**:
   - `wandb.init(dir=custom_path)` vs シンボリックリンク vs メタデータファイル
   - wandбの既存機能（自動アップロード、run管理）への影響範囲

2. **ディレクトリ構造詳細**:
   - マニフェストファイル形式（JSON/YAML）
   - メタデータ内容（タイムスタンプ、git commit hash、ハイパーパラメータ）

3. **後方互換性ポリシー**:
   - 既存の`wandb/`, `exp/`ディレクトリサポート期間
   - 移行ツール提供の要否

4. **エラーハンドリング**:
   - ディレクトリ作成失敗時のフォールバック
   - 無効な実験名のバリデーション

### Research Items to Carry Forward

1. **wandb API Deep Dive** (Priority: High):
   - `dir`パラメータのドキュメント精読
   - カスタムディレクトリ使用時のrun.dir動作
   - 自動ファイル同期の挙動変化

2. **Performance Impact** (Priority: Medium):
   - シンボリックリンク vs メタデータファイルの読み取り性能
   - 大量実験時のディレクトリ構造スキャン性能

3. **Visualization Tool Integration** (Priority: Low):
   - 既存スクリプトのパス参照パターン詳細調査
   - 共通パス解決ライブラリの設計

---

## 6. Requirement-to-Asset Map

| Requirement | Existing Assets | Gap Status | Notes |
|-------------|----------------|-----------|-------|
| **Req 1.1**: `experiments/{category}/{method}/{variant}/` | `os.makedirs` (42 usages) | **Missing** | パターン存在、カスタムレイアウトロジック不在 |
| **Req 1.2**: 親ディレクトリ自動作成 | `os.makedirs(..., exist_ok=True)` | ✅ **Exists** | `sed_trainer_pretrained.py:418` |
| **Req 1.3**: 3階層サポート | - | **Missing** | 新規実装必要 |
| **Req 1.4**: ファイル名検証 | - | **Missing** | pathlib.Path使用の基盤あり |
| **Req 1.5**: 衝突解決（タイムスタンプ） | wandb run ID生成 | **Constraint** | wandbパターン参考可能 |
| **Req 2.1**: checkpoints/ | `_wandb_checkpoint_dir` | ✅ **Partial** | wandb内のみ実装済 |
| **Req 2.2**: metrics/ | TensorBoardLogger | **Constraint** | 既存ログとの統合必要 |
| **Req 2.3**: inference/ | `visualize/get_features/` | **Constraint** | ハードコードパス変更必要 |
| **Req 2.4**: visualizations/ | `visualization_outputs/` | **Constraint** | 3-4スクリプト修正 |
| **Req 2.5**: config/ | `wandb.save(config_file)` | ✅ **Partial** | 拡張可能 |
| **Req 2.6**: マニフェスト生成 | - | **Missing** | 新規実装 |
| **Req 3.1**: wandb dir override | `wandb.init()` | **Unknown** | API調査必要 |
| **Req 3.2**: wandb初期化時インジェクション | `_init_wandb_project()` | ✅ **Exists** | 拡張ポイント明確 |
| **Req 3.3**: run ID ↔ パス マッピング | - | **Missing** | シンボリックリンク or JSON |
| **Req 3.4**: ヘルパー関数 | `_wandb_checkpoint_dir` | **Missing** | 統一インターフェース化 |
| **Req 3.5**: パス解決性能（<100ms） | - | **Unknown** | ベンチマーク必要（低優先度） |
| **Req 4.1**: YAML命名パラメータ | `wandb_dir` | **Missing** | スキーマ拡張 |
| **Req 4.2**: テンプレート展開 | - | **Missing** | str.format()で実装可能 |
| **Req 4.3**: パラメータ検証 | `yaml.safe_load()` | ✅ **Exists** | バリデーション追加のみ |
| **Req 4.4**: デフォルト値フォールバック | - | **Missing** | 設計必要 |
| **Req 4.5**: 環境変数展開 | `os.environ` | ✅ **Exists** | `os.path.expandvars()`利用 |

---

## 7. Gap Analysis Approach

本分析は `.kiro/settings/rules/gap-analysis.md` フレームワークに従って実施:

- ✅ **Current State Investigation**: 5主要ファイル、42箇所のパス操作パターン特定
- ✅ **Requirements Feasibility Analysis**: 4要件を19個の受入基準にマッピング
- ✅ **Multiple Implementation Options**: 3アプローチ評価（Extend/New/Hybrid）
- ✅ **Complexity & Risk Assessment**: M effort, Medium risk（根拠明示）
- ✅ **Explicit Gaps**: 8 Missing, 6 Constraint, 3 Unknown（Research Needed明示）

---

## Next Steps

1. **Gap Analysis Review**: 本ドキュメント確認とフィードバック
2. **Proceed to Design**: `/kiro:spec-design refactor-experiment-structure` 実行
3. **wandb Research**: 設計フェーズ前にwandб統合動作の簡易検証推奨

**注**: 本分析は情報提供を目的とし、最終実装決定は設計フェーズで行う。複数の実現可能な選択肢を提示している。
