# Requirements Document

## Project Description (Input)
SEBBs

## Introduction

本仕様は、SEBBs（Sound Event Bounding Boxes）パッケージのリファクタリングを定義する。SEBBsはDCASE 2024 Task 4における音響イベント検出の後処理を担う重要なコンポーネントであり、変化点検出とセグメント統合を通じて高精度なイベント境界予測を実現する。

現状、SEBBsパッケージはgit submoduleとして管理されており、直接編集が制約される。本リファクタリングでは、submoduleを変更せずに、型安全性、保守性、ドキュメント品質を向上させるラッパーレイヤーを構築する。

## Requirements

### Requirement 1: 型定義モジュールの実装

**Objective:** As a developer, I want 明確な型定義を持つモジュール, so that 型安全性が向上し、IDEの補完機能が有効に機能する

#### Acceptance Criteria

1. The SEBBs Wrapper shall SEBBデータ型（SEBB, Detection, SEBBList, DetectionList）を定義する
2. The SEBBs Wrapper shall スコアデータ型（Scores, GroundTruth, AudioDurations）を定義する
3. The SEBBs Wrapper shall パラメータ型（ClasswiseParam, OptionalClasswiseParam）を定義する
4. The SEBBs Wrapper shall 設定型（PredictorConfig, TuningConfig, EvaluationConfig）をTypedDictで定義する
5. The SEBBs Wrapper shall PredictorProtocolインターフェースを定義する
6. The SEBBs Wrapper shall すべての型定義にdocstringを付与する

### Requirement 2: Predictorラッパーの実装

**Objective:** As a developer, I want CSEBBsPredictorの型安全なラッパー, so that submoduleの機能を保持しつつ、より明確なインターフェースで利用できる

#### Acceptance Criteria

1. When SEBBsPredictorが初期化される, then the SEBBs Wrapper shall step_filter_length、merge_threshold_abs、merge_threshold_rel、detection_threshold、sound_classesパラメータを受け入れる
2. The SEBBs Wrapper shall predict()メソッドでSEBBsを予測する機能を提供する
3. The SEBBs Wrapper shall detect()メソッドで検出閾値を適用した予測を提供する
4. The SEBBs Wrapper shall detection_thresholding()メソッドで既存SEBBsへの閾値適用を提供する
5. The SEBBs Wrapper shall from_config()クラスメソッドで設定辞書からのインスタンス生成を提供する
6. The SEBBs Wrapper shall copy()メソッドでインスタンスの複製を提供する
7. The SEBBs Wrapper shall すべてのメソッドに型アノテーションと詳細なdocstringを付与する
8. When SEBBsPredictorが使用される, then the SEBBs Wrapper shall 内部でCSEBBsPredictorにデリゲートする
9. The SEBBs Wrapper shall プロパティアクセスで各パラメータの取得を可能にする

### Requirement 3: Tunerラッパーの実装

**Objective:** As a developer, I want ハイパーパラメータチューニングの型安全なインターフェース, so that PSDSやcollar-based F1の最適化が容易になる

#### Acceptance Criteria

1. The SEBBs Wrapper shall tune()静的メソッドで汎用的なグリッドサーチを提供する
2. The SEBBs Wrapper shall tune_for_psds()メソッドでPSDS最適化のための簡潔なインターフェースを提供する
3. The SEBBs Wrapper shall tune_for_collar_based_f1()メソッドでcollar-based F1最適化のための簡潔なインターフェースを提供する
4. The SEBBs Wrapper shall cross_validation()メソッドでleave-one-out交差検証を提供する
5. When チューニングメソッドが呼び出される, then the SEBBs Wrapper shall 内部でcsebbs.tune()を呼び出し、結果をラップする
6. The SEBBs Wrapper shall すべてのチューニングメソッドに型アノテーションと使用例を含むdocstringを付与する
7. The SEBBs Wrapper shall カスタム選択関数のサポートを維持する

### Requirement 4: Evaluatorラッパーの実装

**Objective:** As a developer, I want 評価メトリクスの型安全なインターフェース, so that PSDBやcollar-based評価が明確に実行できる

#### Acceptance Criteria

1. The SEBBs Wrapper shall evaluate_psds()メソッドで詳細なPSDS評価を提供する
2. The SEBBs Wrapper shall evaluate_psds1()メソッドでDCASE標準のPSDS1評価を提供する
3. The SEBBs Wrapper shall evaluate_psds2()メソッドでDCASE標準のPSDS2評価を提供する
4. The SEBBs Wrapper shall evaluate_collar_based_fscore()メソッドでcollar-based F-scoreを提供する
5. The SEBBs Wrapper shall find_best_collar_based_fscore()メソッドで最適F-scoreと閾値を探索する機能を提供する
6. When 評価メソッドが呼び出される, then the SEBBs Wrapper shall sed_scores_evalモジュールに適切にデリゲートする
7. The SEBBs Wrapper shall すべての評価メソッドに型アノテーションと詳細なdocstringを付与する

### Requirement 5: 既存コードの移行

**Objective:** As a developer, I want sed_trainer_pretrained.pyをラッパー経由に移行, so that プロジェクト全体で型安全性が向上する

#### Acceptance Criteria

1. The SEBBs Wrapper shall sed_trainer_pretrained.pyのimport文をラッパー経由に更新する
2. When csebbs.tune()が使用されていた箇所, then the SEBBs Wrapper shall SEBBsTuner.tune_for_psds()またはSEBBsTuner.tune()に置き換える
3. When csebbs.CSEBBsPredictor()が使用されていた箇所, then the SEBBs Wrapper shall SEBBsPredictor()に置き換える
4. The SEBBs Wrapper shall すべての既存機能の互換性を保持する
5. The SEBBs Wrapper shall sed_scores_from_sebbsなどのユーティリティ関数の直接importを維持する

### Requirement 6: ドキュメントの整備

**Objective:** As a developer, I want 包括的なドキュメント, so that ラッパーの使用方法と利点が明確になる

#### Acceptance Criteria

1. The SEBBs Wrapper shall README.mdで概要、アーキテクチャ、使用例を提供する
2. The SEBBs Wrapper shall __init__.pyで明確なパブリックAPIをエクスポートする
3. The SEBBs Wrapper shall 各モジュールのdocstringで目的と使用方法を説明する
4. The SEBBs Wrapper shall 型安全性の利点を示す例をドキュメントに含める
5. The SEBBs Wrapper shall submodule非編集でのリファクタリング手法を説明する
6. The SEBBs Wrapper shall 移行前後の比較例を提供する

### Requirement 7: テストカバレッジ

**Objective:** As a developer, I want ラッパーレイヤーのユニットテスト, so that リファクタリングの正確性が検証できる

#### Acceptance Criteria

1. The SEBBs Wrapper shall tests/ディレクトリにpytest準拠のテスト構造を提供する
2. The SEBBs Wrapper shall SEBBsPredictorの初期化、予測、検出、コピー機能のテストを含める
3. When テストが実行される, then the SEBBs Wrapper shall 基本機能の正常動作を検証する
4. The SEBBs Wrapper shall 簡潔なテストケースでラッパーの動作を検証する
5. The SEBBs Wrapper shall 合成データを使用した独立したテストを提供する

### Requirement 8: パッケージ構造

**Objective:** As a developer, I want 明確なモジュール構成, so that 保守性と拡張性が向上する

#### Acceptance Criteria

1. The SEBBs Wrapper shall local/sebbs_wrapper/ディレクトリ配下に配置される
2. The SEBBs Wrapper shall types.py、predictor.py、tuner.py、evaluator.pyの4つの主要モジュールを含む
3. The SEBBs Wrapper shall __init__.pyでパブリックAPIを明示的にエクスポートする
4. The SEBBs Wrapper shall README.mdとtests/サブディレクトリを含む
5. The SEBBs Wrapper shall 各ファイルが単一責任原則に従い、200-400行程度に収まる
6. The SEBBs Wrapper shall 合計約1,300行のコードで構成される

### Requirement 9: Submodule非依存性

**Objective:** As a developer, I want submoduleを変更せずにリファクタリング, so that submodule更新時の影響を局所化できる

#### Acceptance Criteria

1. The SEBBs Wrapper shall sebbs/ submodule配下のファイルを一切変更しない
2. The SEBBs Wrapper shall ラッパーパターンを使用してsubmodule機能を拡張する
3. When submoduleが更新される, then the SEBBs Wrapper shall ラッパーレイヤーのみの調整で対応できる構造とする
4. The SEBBs Wrapper shall submoduleの全機能へのアクセスを維持する
5. The SEBBs Wrapper shall 内部実装でsubmoduleのクラスにデリゲートする

### Requirement 10: 後方互換性

**Objective:** As a developer, I want 既存機能の完全な互換性, so that 既存のトレーニングパイプラインが影響を受けない

#### Acceptance Criteria

1. The SEBBs Wrapper shall CSEBBsPredictorのすべてのメソッドと同等の機能を提供する
2. The SEBBs Wrapper shall csebbs.tuneのすべてのパラメータをサポートする
3. When ラッパー経由で実行される, then the SEBBs Wrapper shall submodule直接使用と同一の結果を返す
4. The SEBBs Wrapper shall カスタム選択関数やコールバックのサポートを維持する
5. The SEBBs Wrapper shall sed_scores_from_sebbsなどの外部ユーティリティとの互換性を保持する

### Requirement 11: MAESTRO専用チューニングサポート

**Objective:** As a developer, I want MAESTRO dataset-specific tuning support, so that 都市音・屋内音イベントをDESED家庭内音と独立して最適化できる

#### Acceptance Criteria

1. The SEBBs Wrapper shall DESEDとMAESTROで独立したcSEBBsチューニングをサポートする
2. The SEBBs Wrapper shall MAESTRO専用のチューニングガイダンスをドキュメント化する
3. When SEBBsTuner.tune()が使用される, then the SEBBs Wrapper shall カスタム選択関数（mpAUC最適化など）をサポートする
4. The SEBBs Wrapper shall csebbs.select_best_psds, select_best_cbf, select_best_psds_and_cbfのすべてをサポートする
5. The SEBBs Wrapper shall MAESTRO/DESED分離の設計判断をドキュメント化する

### Requirement 12: mpAUC/mAUC評価メトリクス

**Objective:** As a developer, I want mpAUC (mean partial AUROC) evaluation support, so that MAESTRO評価指標を正確に計算できる

#### Acceptance Criteria

1. The SEBBs Wrapper shall evaluate_mpauc()メソッドでmpAUC評価を提供する
2. The SEBBs Wrapper shall evaluate_mauc()メソッドで通常のAUROC評価を提供する
3. When mpAUC評価が実行される, then the SEBBs Wrapper shall sed_scores_eval.segment_based.auroc(partial_auroc=True)を使用する
4. The SEBBs Wrapper shall クラス別mpAUC辞書の計算をサポートする
5. The SEBBs Wrapper shall obj_metric_maestro_type設定（"mpauc", "mauc", "fmo"）の使用方法をドキュメント化する
6. The SEBBs Wrapper shall すべての評価メソッドに型アノテーションと詳細なdocstringを付与する

### Requirement 13: チューニング進捗表示

**Objective:** As a developer, I want progress reporting during hyperparameter tuning, so that 長時間のグリッドサーチの進行状況を把握できる

#### Acceptance Criteria

1. The SEBBs Wrapper shall tune()、tune_for_psds()、tune_for_collar_based_f1()メソッドにverboseパラメータを追加する
2. When verbose=Trueが指定される, then the SEBBs Wrapper shall グリッドサーチ開始時に総試行回数を表示する
3. When verbose=Trueが指定される, then the SEBBs Wrapper shall 各パラメータ組み合わせの評価完了時に進捗情報を表示する
4. The SEBBs Wrapper shall 進捗情報に現在の試行番号/総試行数、評価スコア、経過時間を含める
5. The SEBBs Wrapper shall tqdmライブラリが利用可能な場合は進捗バーを表示する
6. The SEBBs Wrapper shall tqdmが利用不可の場合は標準出力による進捗ログを表示する
7. The SEBBs Wrapper shall verbose=False（デフォルト）の場合は既存動作を維持する（進捗表示なし）
8. The SEBBs Wrapper shall 進捗表示機能の使用例をREADME.mdに追加する


