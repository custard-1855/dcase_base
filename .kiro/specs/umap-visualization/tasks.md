# Implementation Plan

## Overview
UMAP可視化システムの実装タスク。384次元RNN出力特徴量を2次元に削減し、クラス分離性、ドメイン比較、MixStyle効果の3種類の論文掲載品質プロットを生成する。

## Implementation Tasks

- [ ] 1. 特徴量読み込み・前処理レイヤーの実装
- [x] 1.1 (P) FeatureLoaderクラスの基本実装
  - .npzファイルから特徴量、ターゲット、ファイル名を読み込む
  - 形状検証(features: N×384, targets: N×27)を実行し、不正な場合はValueErrorを送出
  - 存在しないファイルに対してFileNotFoundErrorを送出
  - 既存のclasses_dictからクラス定義を読み込み、インスタンス変数として保持
  - _Requirements: 1, 8_

- [x] 1.2 (P) マルチラベルからの主要クラス抽出機能
  - argmax操作により(N, 27)のマルチラベル配列から(N,)の主要クラスインデックスを抽出
  - 全要素が0のサンプルを"Unknown"クラス(特殊インデックス)として処理
  - サンプルインデックスから全該当クラスへのマッピング辞書を生成し、ログ出力
  - _Requirements: 5_

- [x] 1.3 (P) ファイル名からのドメインラベル抽出機能
  - ファイル名またはパスから"desed_synthetic", "desed_real", "maestro_training", "maestro_validation"を識別
  - 4つのドメインに対応する整数インデックス(0-3)の配列を返す
  - 識別不可能なパターンに対してはValueErrorを送出し、有効なパターン例を提示
  - _Requirements: 3_

- [x] 1.4 (P) 複数データセットの結合機能
  - .npzファイルパスのリストを受け取り、全データセットを結合
  - 各データセットの特徴量、クラス、ターゲット、ファイル名を個別に読み込み
  - np.concatenateで全配列を結合し、統一した形状を保証
  - ログに各データセットのサンプル数と結合後の合計サンプル数を出力
  - _Requirements: 1_

- [x] 2. UMAP次元削減レイヤーの実装
- [x] 2.1 (P) UMAPReducerクラスの実装
  - umap.UMAPのラッパークラスとして、n_neighbors, min_dist, metric, random_stateをコンストラクタで受け取る
  - デフォルト値(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)を設定
  - fit_transform()メソッドで(N, 384)を(N, 2)に削減し、実行時間をINFOレベルでログ出力
  - transform()メソッドで学習済みモデルを使用した新規データ変換をサポート(未学習時はValueError)
  - _Requirements: 2, 4, 7, 8_

- [x] 2.2 (P) メモリ効率と大規模データ対応
  - サンプル数がn_neighbors未満の場合に警告を表示し、n_neighborsを自動調整
  - サンプル数が10,000を超える場合にWARNINGレベルのログを表示
  - MemoryErrorをキャッチし、サンプル数削減またはダウンサンプリング提案のメッセージを表示
  - _Requirements: 8_

- [ ] 3. プロット生成・出力レイヤーの実装
- [x] 3.1 (P) PlotGeneratorクラスの基本構造とスタイル適用
  - 出力ディレクトリ、DPI(300以上)、図サイズ、カラーパレット、フォントサイズをコンストラクタで受け取る
  - seabornのcolorblindパレットをデフォルトとして設定
  - 論文掲載用スタイル適用メソッド(_apply_publication_style)を実装
    - 軸ラベル"UMAP Dimension 1/2"を12pt以上で設定
    - タイトルを14pt以上で設定
    - 凡例を図外(bbox_to_anchor=(1.05, 1), loc='upper left')に10pt以上で配置
  - _Requirements: 6_

- [x] 3.2 (P) クラス分離性プロットの生成機能
  - (N, 2)の埋め込みと(N,)のクラスラベル(0-20)を受け取り、散布図を生成
  - 21クラスそれぞれに異なる色を割り当て(colorblindパレット使用)
  - 凡例に全21クラス名(DESED 10クラス + MAESTRO 11クラス)を表示
  - PNG(300 DPI以上)とPDF形式で出力し、ファイルパスを返す
  - ファイル名にプレフィックス("class_separation")、モデル名、タイムスタンプを含める
  - _Requirements: 2, 6_

- [x] 3.3 (P) ドメイン比較プロットの生成機能
  - (N, 2)の埋め込みと(N,)のドメインラベル(0-3)を受け取り、散布図を生成
  - 4つのドメインそれぞれに異なる色とマーカー形状を割り当て
  - 凡例にドメイン名とサンプル数を表示(例: "DESED Synthetic (N=1500)")
  - PNG/PDF両形式で出力し、ファイル名にプレフィックス("domain_comparison")を含める
  - _Requirements: 3, 6_

- [x] 3.4 (P) MixStyle効果比較サブプロットの生成機能
  - MixStyle適用前後の(N, 2)埋め込みとドメインラベルを受け取る
  - fig, (ax1, ax2) = plt.subplots(1, 2)で2つのサブプロットを生成
  - 両サブプロットでドメイン別に色分けし、同一のカラーマッピングを使用
  - 両サブプロットの軸スケール(x軸とy軸の範囲)を統一
  - PNG/PDF両形式で出力し、ファイル名にプレフィックス("mixstyle_effect")を含める
  - _Requirements: 4, 6_

- [x] 4. CLI統合とオーケストレーションレイヤーの実装
- [x] 4.1 UMAPVisualizerクラスとコマンドライン引数パーサー
  - argparseでサブコマンド("class_separation", "domain_comparison", "mixstyle_effect")を定義
  - 各モードに応じた必須引数(input, inputs, before/after)と共通引数(model_type, config, output_dir)を設定
  - UMAPパラメータ(n_neighbors, min_dist, metric, random_state)とプロット設定(dpi, figsize, palette)をオプション引数として追加
  - デフォルト設定を返す_get_default_config()メソッドを実装
  - _Requirements: 7_

- [x] 4.2 YAML設定ファイルの読み込みと設定マージ
  - PyYAMLで設定ファイルを読み込み、辞書として返すload_config()メソッドを実装
  - merge_config()メソッドで優先度(CLI引数 > YAML設定 > デフォルト)に従って設定を統合
  - ログレベル設定をlogging.basicConfigで適用
  - _Requirements: 7, 8_

- [x] 4.3 クラス分離性可視化の実行フロー
  - run_class_separation()メソッドを実装
  - FeatureLoader.load_features()で特徴量とターゲットを読み込み
  - FeatureLoader.extract_primary_class()で主要クラスを抽出
  - UMAPReducer.fit_transform()で2次元埋め込みを生成
  - PlotGenerator.plot_class_separation()でプロットを生成・保存
  - 成功メッセージと出力ファイルパスをINFOレベルでログ出力
  - _Requirements: 2, 8_

- [x] 4.4 ドメイン比較可視化の実行フロー
  - run_domain_comparison()メソッドを実装
  - FeatureLoader.load_multiple_datasets()で複数データセットを結合
  - FeatureLoader.extract_domain_labels()でドメインラベルを抽出
  - UMAPReducer.fit_transform()で2次元埋め込みを生成
  - PlotGenerator.plot_domain_comparison()でプロットを生成・保存
  - _Requirements: 3, 8_

- [x] 4.5 MixStyle効果比較可視化の実行フロー
  - run_mixstyle_comparison()メソッドを実装
  - MixStyle適用前後の2つの.npzファイルから特徴量とドメインラベルを読み込み
  - 両モデルの特徴量をnp.concatenateで結合し、UMAPReducer.fit_transform()で共通埋め込み空間を生成
  - 結合された埋め込みを分割し、PlotGenerator.plot_mixstyle_comparison()でサブプロット生成
  - _Requirements: 4, 8_

- [x] 5. エントリーポイントとスクリプト配置
- [x] 5.1 visualize_umap.pyスクリプトの作成
  - visualize/ディレクトリにvisualize_umap.pyを配置
  - main()関数でUMAPVisualizer.parse_args()、設定マージ、モード別実行を制御
  - if __name__ == "__main__": main()でスクリプト実行をサポート
  - 未知のモードが指定された場合にValueErrorを送出
  - _Requirements: 7, 8_

- [x] 5.2 統合テストとエラーハンドリングの検証
  - 正常系: 各モード(class_separation, domain_comparison, mixstyle_effect)で実際の.npzファイルを使用し、プロット生成を確認
  - 異常系: 存在しないファイル、形状不正、低分散特徴量、大規模データに対するエラーメッセージを確認
  - ログ出力: サンプル数、クラス分布、UMAP実行時間、保存ファイルパスがINFOレベルで出力されることを確認
  - _Requirements: 1, 8_

- [x] 6. ドキュメントとサンプル設定ファイルの作成
- [x] 6.1 (P) YAML設定ファイル例の作成
  - config/umap_visualization.yamlにサンプル設定を作成
  - umap, plot, output, loggingの4セクションを含める
  - デフォルト値と各パラメータのコメントを記載
  - _Requirements: 7_

- [x] 6.2 (P) 使用例とREADMEの追加
  - visualize/README.mdまたはプロジェクトREADMEに使用例セクションを追加
  - 3つのモードのコマンドライン例を記載(基本実行、カスタムパラメータ、YAML設定使用)
  - 21クラスリスト(DESED 10 + MAESTRO 11)を参照情報として記載
  - _Requirements: 6, 7_

## Task Summary
- **Total**: 6 major tasks, 17 sub-tasks
- **Parallel-capable**: 12 sub-tasks marked with (P)
- **Requirements Coverage**: All 8 requirements (1-8) mapped
- **Estimated Effort**: 1-3 hours per sub-task (total ~25-50 hours)

## Quality Validation
- ✅ All requirements (1-8) mapped to tasks
- ✅ Task dependencies verified (Data → Processing → Presentation → Orchestration → Integration)
- ✅ Testing tasks included (5.2: integration tests and error handling)
- ✅ Parallel tasks identified based on component boundaries (FeatureLoader, UMAPReducer, PlotGenerator are independent)
- ✅ Task progression is incremental (layer-by-layer implementation)

## Notes
- **Parallel Execution**: Tasks 1.1-1.4, 2.1-2.2, 3.1-3.4, 6.1-6.2 can be executed in parallel within their respective layers
- **Sequential Dependencies**: Layer 4 (Orchestration) depends on layers 1-3; Task 5.1 depends on layer 4; Task 5.2 depends on 5.1
- **Integration Point**: Task 5.1 wires all components together into the final CLI entry point
