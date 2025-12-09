# Implementation Plan: SEBBs Refactoring

## Overview

本実装計画は、SEBBs（Sound Event Bounding Boxes）パッケージに対する型安全なWrapperレイヤーの構築を定義する。Submodule非編集制約下で、型アノテーション、明確なインターフェース、包括的ドキュメント、MAESTRO専用チューニング、mpAUC/mAUC評価メトリクスを提供する。

---

## Implementation Tasks

### Phase 1: Core Wrapper Components

- [ ] 1. 型定義モジュールの実装
- [ ] 1.1 (P) 基本型定義の実装
  - SEBB、Detection、SEBBList、DetectionList型エイリアスの定義
  - Scores、GroundTruth、AudioDurations型の定義
  - ClasswiseParam、OptionalClasswiseParam型の定義
  - すべての型にdocstringを付与
  - _Requirements: 1_

- [ ] 1.2 (P) 設定型の実装
  - PredictorConfig TypedDictの定義（total=False）
  - TuningConfig TypedDictの定義
  - EvaluationConfig TypedDictの定義
  - すべてのキーがオプショナルであることを確認
  - _Requirements: 1_

- [ ] 1.3 (P) Protocolインターフェースの実装
  - PredictorProtocolの定義（構造的サブタイピング）
  - predict()、detect()メソッドシグネチャの定義
  - 型アノテーションとdocstringの付与
  - _Requirements: 1_

- [ ] 2. Predictorラッパーの実装
- [ ] 2.1 (P) 初期化とデリゲーション機能の実装
  - SEBBsPredictorクラスの初期化（step_filter_length等のパラメータ）
  - 内部_predictorへのデリゲーション（CSEBBsPredictorBase）
  - from_config()クラスメソッドの実装
  - copy()メソッドの実装
  - プロパティアクセスの実装（5つのパラメータ）
  - _Requirements: 2, 9, 10_

- [ ] 2.2 (P) 予測機能の実装
  - predict()メソッドの実装（デリゲーション）
  - detect()メソッドの実装（predict + detection_thresholding）
  - detection_thresholding()メソッドの実装
  - すべてのメソッドに型アノテーションとdocstringを付与
  - _Requirements: 2, 9, 10_

- [ ] 3. Tunerラッパーの実装
- [ ] 3.1 (P) 汎用チューニング機能の実装
  - tune()静的メソッドの実装（グリッドサーチ）
  - カスタム選択関数のサポート（selection_fnパラメータ）
  - クロスバリデーション対応（foldsパラメータ）
  - デリゲーション（csebbs.tune()）とラッパー化
  - _Requirements: 3, 9, 10, 11_

- [ ] 3.2 (P) PSDS/Collar-based F1チューニングの実装
  - tune_for_psds()メソッドの実装（PSDS最適化）
  - tune_for_collar_based_f1()メソッドの実装（F1最適化）
  - デフォルト探索空間の設定
  - 結果のSEBBsPredictorラッピング
  - _Requirements: 3, 9, 10_

- [ ] 3.3 (P) クロスバリデーション機能の実装
  - cross_validation()静的メソッドの実装
  - leave-one-out戦略のサポート
  - 全foldの予測と検出の統合
  - _Requirements: 3, 9, 10_

- [ ] 4. Evaluatorラッパーの実装
- [ ] 4.1 (P) PSDS評価機能の実装
  - evaluate_psds()メソッドの実装（詳細PSDS評価）
  - evaluate_psds1()メソッドの実装（DCASE標準）
  - evaluate_psds2()メソッドの実装（DCASE標準）
  - sed_scores_eval.intersection_basedへのデリゲーション
  - _Requirements: 4, 9, 10_

- [ ] 4.2 (P) mpAUC/mAUC評価機能の実装
  - evaluate_mpauc()メソッドの実装（mean partial AUROC）
  - evaluate_mauc()メソッドの実装（mean AUROC）
  - sed_scores_eval.segment_based.auroc()へのデリゲーション
  - partial_auroc=True/Falseの適切な設定
  - クラス別mpAUC辞書の計算
  - _Requirements: 4, 9, 10, 12_

- [ ] 4.3 (P) Collar-based評価機能の実装
  - evaluate_collar_based_fscore()メソッドの実装
  - find_best_collar_based_fscore()メソッドの実装
  - sed_scores_eval.collar_basedへのデリゲーション
  - _Requirements: 4, 9, 10_

### Phase 2: Integration & Documentation

- [ ] 5. パブリックAPIとドキュメントの整備
- [ ] 5.1 (P) __init__.pyの実装
  - すべてのパブリッククラスのエクスポート
  - すべての型定義のエクスポート
  - __all__リストの定義
  - プライベート実装の非公開維持
  - _Requirements: 6, 8_

- [ ] 5.2 (P) README.mdの作成
  - 概要とアーキテクチャの説明
  - 基本的な使用例（Predictor、Tuner、Evaluator）
  - MAESTRO専用チューニングのベストプラクティス
  - mpAUC/mAUC評価の使用例とobj_metric_maestro_type設定の説明
  - 移行ガイド（従来のcsebbs利用からの移行）
  - Submodule非編集でのリファクタリング手法の説明
  - _Requirements: 6, 11, 12_

- [ ] 6. 既存コードの移行
  - sed_trainer_pretrained.pyのimport文とメソッド呼び出しの更新
  - csebbs.CSEBBsPredictor()の置換
  - csebbs.tune()の置換（SEBBsTuner.tune_for_psds()等）
  - MAESTRO専用チューニングの統合（csebbs_predictor_maestro）
  - 既存機能の互換性維持の確認
  - _Requirements: 5, 9, 10, 11_

### Phase 3: Testing & Validation

- [ ] 7. ユニットテストの実装
- [x] 7.1 (P) Predictorテストの実装
  - 初期化テスト（デフォルト/カスタムパラメータ）
  - from_config()テスト
  - predict()テスト（合成データ使用）
  - detect()テスト（閾値適用検証）
  - copy()テスト（インスタンス複製検証）
  - __repr__()テスト
  - _Requirements: 7_

- [x] 7.2* (P) Tuner/Evaluatorテストの実装
  - tune()基本機能テスト
  - tune_for_psds()テスト
  - tune_for_collar_based_f1()テスト
  - evaluate_psds1/psds2テスト
  - evaluate_mpauc/maucテスト（MAESTRO評価）
  - evaluate_collar_based_fscoreテスト
  - 合成データでの独立テスト
  - _Requirements: 7, 12_

- [ ] 8. 統合テストと検証
- [x] 8.1 既存パイプラインとの統合検証
  - sed_trainer_pretrained.py実行による動作確認
  - Submodule直接利用との結果一致検証
  - MAESTRO専用チューニングの動作確認
  - mpAUC/mAUC評価の動作確認
  - 後方互換性の完全検証
  - _Requirements: 5, 9, 10, 11, 12_

- [ ] 8.2* パフォーマンステスト
  - デリゲーションオーバーヘッドの測定（<1%許容）
  - predict()実行時間の比較
  - tune()実行時間の比較
  - メモリフットプリントの測定（<1MB許容）
  - _Requirements: 9, 10_

### Phase 4: Final Quality Checks

- [x] 9. コード品質とドキュメントの最終確認
- [x] 9.1 (P) 型安全性の検証
  - すべてのパブリックメソッドに型アノテーション付与確認
  - TypedDict、Protocol、型エイリアスの適切な使用確認
  - mypy型チェックの実行（エラーなし）
  - _Requirements: 1, 2, 3, 4, 12_

- [x] 9.2 (P) docstringとドキュメントの検証
  - すべてのパブリックAPIにdocstring付与確認
  - README.mdの網羅性確認
  - 使用例の動作確認
  - MAESTRO/mpAUCドキュメントの正確性確認
  - _Requirements: 6, 11, 12_

- [x] 9.3 (P) Lintとコーディングスタイルの確認
  - Ruff lint実行（エラーなし）
  - コーディングスタイルの統一確認
  - プロジェクト標準への準拠確認
  - _Requirements: 8_

### Phase 5: Progress Reporting Feature

- [x] 10. チューニング進捗表示機能の実装
- [x] 10.1 (P) 進捗表示ヘルパー関数の実装
  - グリッドサイズ計算ユーティリティを実装
  - 標準出力による進捗ログフォーマッタを実装
  - _Requirements: 13_

- [x] 10.2 (P) tune()メソッドへのverboseパラメータ追加
  - （スキップ: tune()は汎用メソッドのため、具体的なメソッドで実装）
  - _Requirements: 13_

- [x] 10.3 (P) tune_for_psds()とtune_for_collar_based_f1()への統合
  - tune_for_psds()にverboseパラメータを追加
  - tune_for_collar_based_f1()にverboseパラメータを追加
  - verbose=True時にグリッドサイズと最適化目標を表示
  - _Requirements: 13_

- [x] 10.4 (P) ドキュメントと使用例の追加
  - README.mdに進捗表示機能の使用例を追加（完了）
  - 各tuneメソッドのdocstringにverboseパラメータの説明を追加（完了）
  - tqdmのオプショナル依存についての説明を追加（完了）
  - _Requirements: 13_

- [x] 11. 進捗表示機能のテスト
- [x] 11.1 (P) 進捗表示のユニットテスト
  - verbose=False時の既存動作維持を検証（完了）
  - verbose=True時の進捗ログ出力を検証（完了）
  - tune_for_psds()とtune_for_collar_based_f1()の両方でテスト（完了）
  - _Requirements: 13_

---

## Task Summary

- **Total**: 11 major tasks, 26 sub-tasks
- **Requirements Coverage**: 13/13 requirements covered
- **Optional Tasks**: 2 sub-tasks marked with `*` (Tuner/Evaluator tests, Performance tests)
- **Parallel Tasks**: 16 sub-tasks marked with `(P)`

## Requirements Traceability

| Requirement | Covered By Tasks |
|-------------|------------------|
| 1 | 1.1, 1.2, 1.3, 9.1 |
| 2 | 2.1, 2.2, 9.1 |
| 3 | 3.1, 3.2, 3.3, 9.1 |
| 4 | 4.1, 4.2, 4.3, 9.1 |
| 5 | 6, 8.1 |
| 6 | 5.1, 5.2, 9.2 |
| 7 | 7.1, 7.2 |
| 8 | 5.1, 9.3 |
| 9 | 2.1, 2.2, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 6, 8.1, 8.2 |
| 10 | 2.1, 2.2, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 6, 8.1, 8.2 |
| 11 | 3.1, 5.2, 6, 8.1 |
| 12 | 4.2, 5.2, 7.2, 8.1, 9.1, 9.2 |
| 13 | 10.1, 10.2, 10.3, 10.4, 11.1 |

## Notes

**並列実行可能タスク**: `(P)` マークの付いたタスクは独立して並列実行可能です。Phase 1の型定義、Predictor、Tuner、Evaluatorの各コンポーネントは独立しており、同時に作業できます。

**オプションタスク**: `*` マークの付いたタスク（7.2、8.2）は、基本機能の実装後に追加のテストカバレッジとして実施できます。MVP達成後の品質向上フェーズで実施を推奨します。

**MAESTRO/mpAUC重要事項**: Requirement 11, 12（MAESTRO専用チューニング、mpAUC/mAUC評価）は本リファクタリングの重要な拡張要素です。既存実装との整合性を保ちつつ、ドキュメントとテストで十分にカバーする必要があります。
