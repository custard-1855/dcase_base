# 実装タスク

## タスク概要

このドキュメントは、実験ディレクトリ構造リファクタリング機能の実装タスクを定義します。requirements.mdとdesign.mdで承認された内容に基づき、実装作業を段階的に進めるためのタスクリストを提供します。

---

## タスクリスト

- [ ] 1. 実験ディレクトリ管理モジュールの実装
- [x] 1.1 (P) ExecutionMode enumとExperimentConfigデータクラスの実装
  - `ExecutionMode` enum（TRAIN, TEST, INFERENCE, FEATURE_EXTRACTION）を定義
  - `ExperimentConfig` dataclass（mode, category, method, variant, base_dir, template, log_test_to_wandb）を実装
  - frozen=Trueで不変性を強制
  - デフォルト値を設定（mode=TRAIN, category="default", method="baseline", variant="v1"）
  - _Requirements: 1.3, 4.1, 4.3, 4.4, 5.1, 5.7_

- [x] 1.2 (P) 実行モード検出機能の実装
  - `detect_execution_mode()` 関数を実装
  - 優先順位:明示的指定（hparams["experiment"]["mode"]） → evaluation → test_state_dict → fast_dev_run → デフォルト（TRAIN）
  - 判定結果をログ出力して透明性を確保
  - _Requirements: 5.1, 5.8_

- [x] 1.3 (P) 階層的ディレクトリパス生成機能の実装
  - `build_experiment_path()` 関数を実装
  - モード別レイアウト：`experiments/{mode}/{category}/{method}/{variant}/`
  - 親ディレクトリの自動作成（`os.makedirs(exist_ok=True)`）
  - 環境変数置換のサポート（`os.path.expandvars()`）
  - テンプレート機能（`apply_template()`）の実装
  - _Requirements: 1.1, 1.2, 1.3, 4.2, 4.5_

- [x] 1.4 (P) パス検証機能の実装
  - `validate_path()` 関数を実装
  - 無効文字チェック（正規表現で `<>:"|?*` を検出）
  - パス長検証（Windows互換のため260文字制限）
  - 検証エラー時に詳細なエラーメッセージと修正提案を表示
  - _Requirements: 1.4_

- [x] 1.5 (P) wandB初期化制御機能の実装
  - `should_initialize_wandb()` 関数を実装
  - ロジック：TRAIN → True、TEST → config.log_test_to_wandb、INFERENCE/FEATURE_EXTRACTION → False
  - モード別の無効化メッセージをログ出力
  - _Requirements: 5.2, 5.3, 5.4, 5.6_

- [x] 1.6 (P) 成果物ディレクトリ管理機能の実装
  - `create_artifact_dirs()` 関数を実装
  - サブディレクトリ作成：checkpoints/, metrics/, inference/, visualizations/, config/
  - 作成されたパス辞書を返却
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 1.7 (P) manifest.json生成機能の実装
  - `generate_manifest()` 関数を実装
  - フィールド：run_id, experiment_path, mode, category, method, variant, created_at, parent_run_id, config
  - JSON形式（UTF-8、indent=2）で保存
  - inference/feature_extractionモードではrun_id=None
  - _Requirements: 2.6, 3.3, 5.5, 5.9_

- [x] 1.8 パス解決ヘルパー関数の実装
  - `get_experiment_dir()` 関数を実装（manifest.json検索 → ディレクトリスキャンのフォールバック）
  - `get_checkpoint_dir()`, `get_inference_dir()`, `get_visualization_dir()` 関数を実装
  - mode引数による検索スコープの絞り込み
  - 100ms以内のパフォーマンス要件を満たす実装
  - manifest破損時のフォールバック処理
  - _Requirements: 3.3, 3.4, 3.5_

- [x] 1.9 (P) ExperimentDirManagerモジュールのテスト作成
  - `detect_execution_mode()` の単体テスト（明示的指定、自動推論、デフォルト動作）
  - `build_experiment_path()` のテスト（モード別レイアウト、無効文字、パス長超過）
  - `should_initialize_wandb()` のテスト（各モードの戻り値検証）
  - `validate_path()` のテスト（Windows無効文字、260文字制限）
  - ExperimentConfigのテスト（デフォルト値、frozen不変性、無効mode値）
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 5.1, 5.2, 5.3, 5.4_

- [ ] 2. SEDTask4のwandB初期化拡張
- [x] 2.1 _init_wandb_project()メソッドのモード対応リファクタリング
  - 実行モード検出の統合（`ExperimentDirManager.detect_execution_mode()`を呼び出し）
  - レガシーモード（`--wandb_dir`）、新モード（ExperimentConfig）、デフォルトの3パターンを実装
  - 優先順位：レガシーモード > 新モード > デフォルト
  - inference/feature_extractionモード時、wandB初期化をスキップ
  - wandB有効時、カスタムディレクトリパスを注入（`wandb.init(dir=...)`）
  - self.execution_mode属性を設定
  - self._wandb_checkpoint_dir属性を設定（wandB無効時はNone）
  - _Requirements: 3.1, 3.2, 5.2, 5.3, 5.4, 5.6_

- [x] 2.2 非wandBモード時のディレクトリ作成処理
  - inference/feature_extractionモード時、タイムスタンプベースのディレクトリを作成
  - 成果物サブディレクトリを作成（`ExperimentDirManager.create_artifact_dirs()`）
  - self._inference_dir属性を設定
  - manifest.json生成（run_id=None）
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 5.4_

- [x] 2.3 wandB有効モード時のartifactディレクトリとmanifest生成
  - wandb.run.dirからexperiment_dirを取得
  - 成果物サブディレクトリを作成
  - manifest.json生成（mode含む）
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.1, 3.2, 5.5_

- [x] 2.4 SEDTask4統合テストの作成
  - 新モード（YAML experiment セクション、mode=train）でディレクトリ構造が正しく作成されることを確認
  - 新モード（mode=inference）でwandB初期化がスキップされ、inferenceディレクトリが作成されることを確認
  - レガシーモード（`--wandb_dir`）で従来の動作が維持されることを確認
  - wandB無効時にデフォルトフォールバックが動作することを確認
  - TESTモード（log_test_to_wandb有効/無効）の動作を確認
  - _Requirements: 3.1, 3.2, 5.2, 5.3, 5.4_

- [ ] 3. train_pretrained.pyのCLI・設定拡張
- [x] 3.1 YAML設定のexperimentセクション解析機能
  - `prepare_run()` 関数内でYAML読み込み後、experimentセクションの存在確認
  - experimentセクションが存在しない場合、空の辞書を設定
  - ExperimentConfigインスタンスを生成してバリデーション
  - 生成されたexperiment構造をログ出力
  - _Requirements: 4.1, 4.3, 4.4_

- [x] 3.2 --mode CLI引数の追加
  - argparseに`--mode`引数を追加（choices: train, test, inference, feature_extraction）
  - CLI引数が指定された場合、YAML設定を上書き
  - 明示的なモード指定をログ出力
  - _Requirements: 5.7_

- [x] 3.3 レガシーモード（--wandb_dir）との共存処理
  - `--wandb_dir` 引数が指定されている場合、configs["wandb"]["wandb_dir"]に設定
  - レガシーモード使用時の警告ログを出力
  - 優先順位の明確化（レガシーモードが最優先）
  - _Requirements: 3.1_

- [x] 3.4 test_state_dict設定のモード推論サポート
  - `--test_from_checkpoint` または `--eval_from_checkpoint` 引数からチェックポイントをロード
  - test_model_state_dictをSEDTask4に渡す（_test_state_dictパラメータ）
  - モード推論に使用されるログ出力
  - _Requirements: 5.8_

- [x] 3.5 ModelCheckpointのdirpath設定統合
  - SEDTask4._wandb_checkpoint_dirからcheckpoint_dirを決定
  - wandB無効時はlogger.log_dirからフォールバック
  - ModelCheckpointのdirpath引数に設定
  - _Requirements: 2.1_

- [x] 3.6 train_pretrained.py統合テストの作成
  - YAML experiment セクション解析とExperimentConfig生成の確認
  - CLI `--mode` 引数による上書きの確認
  - レガシーモードとの競合処理の確認
  - ModelCheckpoint dirpath決定ロジックの確認
  - _Requirements: 4.1, 4.3, 5.7, 5.8_

- [ ] 4. E2Eテストとドキュメント整備
- [x] 4.1 trainモードE2Eテストの作成
  - `uv run train_pretrained.py --confs pretrained.yaml` で実験開始
  - `experiments/train/{category}/{method}/{variant}/run-*/` が作成されることを確認
  - checkpoints/, metrics/, config/ サブディレクトリの存在確認
  - manifest.jsonの内容検証（mode="train"、run_id存在）
  - _Requirements: 1.1, 1.2, 2.1, 2.2, 2.5, 2.6, 5.2, 5.5_

- [x] 4.2 inferenceモードE2Eテストの作成
  - `--mode inference` 引数付きで推論スクリプト実行
  - wandB runが生成されないことを確認
  - `experiments/inference/{category}/{method}/{variant}/run-*/` が作成されることを確認
  - manifest.jsonの内容検証（mode="inference"、run_id=null）
  - _Requirements: 1.1, 2.3, 2.6, 5.4, 5.5, 5.6_

- [x] 4.3 (P) パス解決パフォーマンステストの作成
  - `get_experiment_dir()` が100ms以内に完了することを確認
  - 1000実験のmanifestファイルスキャンが1秒以内に完了することを確認
  - mode引数による検索スコープ絞り込みの効果を測定
  - _Requirements: 3.5_

- [x] 4.4 (P) 並行実験分離テストの作成
  - 同じcategory/method/variantで複数training実験を同時実行
  - run IDで分離されることを確認
  - 複数inference実行が異なるタイムスタンプディレクトリに分離されることを確認
  - _Requirements: 1.5_

- [x] 4.5 (P) manifest破損時のフォールバックテストの作成
  - manifest.jsonを破損させた状態でget_experiment_dir()を呼び出し
  - フォールバック処理（ディレクトリスキャン）が動作することを確認
  - 警告ログが出力されることを確認
  - _Requirements: 3.3, 3.5_

- [x] 4.6 実装ドキュメントの更新
  - README.mdまたは該当ドキュメントに新しい実験ディレクトリ構造を記載
  - experimentセクションのYAML設定例を追加
  - `--mode` CLI引数の使用方法を説明
  - レガシーモードとの違いを明記
  - モード別のwandB動作（有効/無効）を説明
  - _Requirements: 全般_

---

## タスク統計

- **合計**: 6メジャータスク、26サブタスク
- **全要件カバレッジ**: 9要件（Requirements 1-5）のすべての受入基準をカバー
- **並列実行可能タスク**: 15サブタスク（(P)マーク付き）
- **推定作業時間**: 各サブタスク1-3時間、合計約30-70時間

---

## 要件トレーサビリティマトリクス

| 要件ID | 要件概要 | 対応タスク |
|--------|---------|----------|
| 1.1 | 階層的ディレクトリ構造 | 1.3, 4.1, 4.2 |
| 1.2 | 親ディレクトリの自動作成 | 1.3, 4.1 |
| 1.3 | 3階層サポート + モード階層 | 1.1, 1.3, 1.9 |
| 1.4 | パス検証 | 1.4, 1.9 |
| 1.5 | 一意識別子付与 | 4.4 |
| 2.1 | checkpoints管理 | 1.6, 2.2, 2.3, 3.5, 4.1 |
| 2.2 | metrics管理 | 1.6, 2.2, 2.3, 4.1 |
| 2.3 | inference管理 | 1.6, 2.2, 2.3, 4.2 |
| 2.4 | visualizations管理 | 1.6, 2.2, 2.3 |
| 2.5 | config管理 | 1.6, 2.2, 2.3, 4.1 |
| 2.6 | manifest生成 | 1.7, 2.2, 2.3, 4.1, 4.2 |
| 3.1 | wandbデフォルトディレクトリのオーバーライド | 2.1, 2.3, 3.3 |
| 3.2 | wandb初期化時のパス注入 | 2.1, 2.3 |
| 3.3 | symlink/metadataマッピング | 1.7, 1.8, 4.5 |
| 3.4 | ヘルパー関数提供 | 1.8 |
| 3.5 | パス解決パフォーマンス（100ms以内） | 1.8, 4.3, 4.5 |
| 4.1 | YAML設定サポート | 1.1, 3.1, 3.6 |
| 4.2 | テンプレート機能 | 1.3 |
| 4.3 | 必須パラメータ検証 | 1.1, 3.1, 3.6 |
| 4.4 | デフォルト値フォールバック | 1.1, 3.1 |
| 4.5 | 環境変数置換 | 1.3 |
| 5.1 | 4つの実行モード認識 | 1.1, 1.2, 1.9 |
| 5.2 | trainモードでwandB run作成 | 1.5, 1.9, 2.1, 2.4, 4.1 |
| 5.3 | testモードで最小ログまたは再利用 | 1.5, 1.9, 2.1, 2.4 |
| 5.4 | inference/feature_extraction時wandB無効化 | 1.5, 1.9, 2.1, 2.2, 2.4, 4.2 |
| 5.5 | manifestにmode記録 | 1.7, 2.3, 4.1, 4.2 |
| 5.6 | モード別無効化メッセージ | 1.5, 2.1, 4.2 |
| 5.7 | モード明示指定（YAML/CLI） | 1.1, 3.2, 3.6 |
| 5.8 | モード自動推論 | 1.2, 3.4, 3.6 |
| 5.9 | parent_run_id参照リンク | 1.7 |

---

## 次のステップ

タスクが承認されたら、以下のコマンドで実装を開始してください：

```bash
# 特定のタスク実行（推奨）
/kiro:spec-impl refactor-experiment-structure 1.1

# 複数タスク実行（注意：コンテキストの肥大化に注意）
/kiro:spec-impl refactor-experiment-structure 1.1,1.2

# 全タスク実行（非推奨：コンテキストが大きくなりすぎる可能性）
/kiro:spec-impl refactor-experiment-structure
```

**重要**: タスク間でコンテキストをクリアして、クリーンな状態を保つことを推奨します。
