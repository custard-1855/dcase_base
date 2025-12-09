# Requirements Document

## Project Description (Input)
UMAP visualization system for DCASE2024 sound event detection model analysis. This feature implements comprehensive visualization of learned feature representations to demonstrate: (1) Class separability - Show that DESED and MAESTRO event classes are well-separated in the feature space for both student and teacher models. (2) Domain generalization effectiveness - Compare features before and after MixStyle application across different domains (DESED synthetic vs real, DESED vs MAESTRO). Uses 384-dimensional RNN output features (pre-dropout, time-averaged) extracted from trained models, which analysis confirmed are optimal for UMAP (dense, high-variance, non-sparse). Handles multi-label ground truth by selecting primary class (argmax) for color coding in visualizations. Outputs publication-ready plots comparing student/teacher models, different domains, and MixStyle effects for academic paper inclusion.

## Introduction

本仕様書は、DCASE2024音響イベント検出モデルの学習済み特徴表現を可視化するUMAPシステムの要件を定義する。本システムは、(1) クラス間分離性の検証、(2) ドメイン汎化手法（MixStyle）の効果検証、の2つの主要目的を持つ。384次元のRNN出力特徴量（dropout前、時間平均済み）を用いて、論文掲載に適した高品質な可視化プロットを生成する。

## Requirements

### Requirement 1: 特徴量データの読み込み
**Objective:** As a 研究者, I want 事前抽出された特徴量ファイルを読み込む機能, so that UMAP可視化の入力データを準備できる

#### Acceptance Criteria
1. When ユーザーが.npzファイルパスを指定した場合, the UMAP可視化システム shall 以下のキーを含むデータを読み込む: `features_student`, `features_teacher`, `probs_student`, `probs_teacher`, `filenames`, `targets`
2. If 指定されたファイルが存在しない場合, then the UMAP可視化システム shall 明確なエラーメッセージを表示して処理を中断する
3. The UMAP可視化システム shall `features_student`と`features_teacher`が(N, 384)の形状であることを検証する
4. The UMAP可視化システム shall `targets`が(N, 27)の形状であることを検証する
5. When 複数のデータセット（desed_validation, desed_unlabeled, maestro_training, maestro_validation）を読み込む場合, the UMAP可視化システム shall 全データセットを結合して単一の特徴量行列を生成する

### Requirement 2: クラス分離性の可視化
**Objective:** As a 研究者, I want DESEDとMAESTROの21クラスが特徴空間で分離していることを示すUMAPプロット, so that モデルがクラス構造を学習できていることを論文で示せる

#### Acceptance Criteria
1. When ユーザーがクラス分離性可視化を要求した場合, the UMAP可視化システム shall UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean')を用いて384次元特徴量を2次元に削減する
2. The UMAP可視化システム shall マルチラベルGround Truthから主要クラス（argmax）を選択してクラスラベルとして使用する
3. When 生徒モデルの特徴量を可視化する場合, the UMAP可視化システム shall `features_student`を使用し、クラスごとに異なる色で散布図をプロットする
4. When 教師モデルの特徴量を可視化する場合, the UMAP可視化システム shall `features_teacher`を使用し、クラスごとに異なる色で散布図をプロットする
5. The UMAP可視化システム shall 凡例に全21クラス名（DESED 10クラス + MAESTRO 11クラス）を表示する
6. The UMAP可視化システム shall 生徒モデルと教師モデルの可視化を並列表示（subplot）する機能を提供する

### Requirement 3: ドメイン別の可視化
**Objective:** As a 研究者, I want ドメイン（DESED合成音/実録音、MAESTRO）ごとに色分けしたUMAPプロット, so that ドメイン間の特徴分布の違いを論文で示せる

#### Acceptance Criteria
1. When ユーザーがドメイン別可視化を要求した場合, the UMAP可視化システム shall ファイル名またはデータセット名からドメインラベル（desed_synthetic, desed_real, maestro_training, maestro_validation）を抽出する
2. The UMAP可視化システム shall ドメインごとに異なる色とマーカー形状で散布図をプロットする
3. When DESED合成音とDESED実録音を比較する場合, the UMAP可視化システム shall 両ドメインのデータポイントを同一プロット上に重ねて表示する
4. When DESEDとMAESTROを比較する場合, the UMAP可視化システム shall 両ドメインのデータポイントを同一プロット上に重ねて表示する
5. The UMAP可視化システム shall 凡例にドメイン名と各ドメインのサンプル数を表示する

### Requirement 4: MixStyle効果の検証可視化
**Objective:** As a 研究者, I want MixStyle適用前後のモデルの特徴分布を比較するUMAPプロット, so that ドメイン汎化手法の効果を論文で示せる

#### Acceptance Criteria
1. When ユーザーが2つのモデルcheckpoint（MixStyle適用前/後）を指定した場合, the UMAP可視化システム shall 両モデルの特徴量を読み込む
2. The UMAP可視化システム shall 両モデルの特徴量を結合した上でUMAP次元削減を実行する（共通の埋め込み空間を使用）
3. When MixStyle適用前後を比較する場合, the UMAP可視化システム shall 2つのsubplotを生成し、それぞれにドメインラベルで色分けしたプロットを表示する
4. The UMAP可視化システム shall MixStyle適用後のプロットで、ドメイン間の重なり度合いが増加していることを視覚的に示す
5. The UMAP可視化システム shall subplot間で軸スケールとUMAP埋め込み空間を統一する

### Requirement 5: マルチラベル処理
**Objective:** As a システム, I want マルチラベルGround Truthから可視化用の単一クラスラベルを抽出する機能, so that 散布図の色分けを可能にする

#### Acceptance Criteria
1. When Ground Truthが(N, 27)のマルチラベル形式である場合, the UMAP可視化システム shall argmax操作により各サンプルの主要クラス（最も確信度の高いクラス）を選択する
2. If 複数のクラスが同じ最大値を持つ場合, then the UMAP可視化システム shall インデックスが最小のクラスを選択する
3. The UMAP可視化システム shall クラスインデックスをクラス名（例: "Alarm_bell_ringing", "Blender"）にマッピングする
4. When サンプルがいずれのクラスにも属さない（全要素が0）場合, the UMAP可視化システム shall そのサンプルを"Unknown"クラスとしてラベル付けする
5. The UMAP可視化システム shall 選択されたクラスラベルと元のマルチラベルデータの対応関係をログ出力する

### Requirement 6: 論文掲載用プロット出力
**Objective:** As a 研究者, I want 高解像度・高品質な可視化プロットファイル, so that 論文に直接掲載できる

#### Acceptance Criteria
1. The UMAP可視化システム shall プロットをPNGとPDF形式で出力する
2. The UMAP可視化システム shall 出力解像度を300 DPI以上に設定する
3. The UMAP可視化システム shall フォントサイズを12pt以上（軸ラベル、タイトル）、10pt以上（凡例）に設定する
4. The UMAP可視化システム shall 軸ラベルに"UMAP Dimension 1"、"UMAP Dimension 2"を表示する
5. The UMAP可視化システム shall プロットタイトルにモデル名（Student/Teacher）とデータセット情報を表示する
6. When ファイルを保存する場合, the UMAP可視化システム shall ファイル名に可視化タイプ（class_separation, domain_comparison, mixstyle_effect）、モデル名、タイムスタンプを含める
7. The UMAP可視化システム shall カラーパレットに色覚多様性対応の配色（例: seaborn "colorblind"パレット）を使用する
8. The UMAP可視化システム shall 凡例を図の外側に配置し、プロット領域を最大化する

### Requirement 7: 設定のカスタマイズ
**Objective:** As a 研究者, I want UMAPパラメータと可視化設定をカスタマイズできる機能, so that 実験目的に応じた最適な可視化を生成できる

#### Acceptance Criteria
1. The UMAP可視化システム shall コマンドライン引数またはYAML設定ファイルで以下のパラメータを指定可能にする: `n_neighbors`, `min_dist`, `metric`, `random_state`
2. When ユーザーがカスタムカラーパレットを指定した場合, the UMAP可視化システム shall 指定されたパレットを使用する
3. When ユーザーが出力ディレクトリを指定した場合, the UMAP可視化システム shall 指定されたディレクトリにファイルを保存する
4. The UMAP可視化システム shall デフォルト設定値（n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42）を提供する
5. When ユーザーが図のサイズを指定した場合, the UMAP可視化システム shall 指定されたサイズ（幅×高さ）でプロットを生成する

### Requirement 8: エラーハンドリングとログ出力
**Objective:** As a システム, I want 堅牢なエラーハンドリングと詳細なログ出力, so that 問題発生時のデバッグを容易にする

#### Acceptance Criteria
1. If UMAP次元削減中にメモリ不足エラーが発生した場合, then the UMAP可視化システム shall サンプル数を削減するか、バッチ処理を提案するエラーメッセージを表示する
2. If 特徴量の次元数が384でない場合, then the UMAP可視化システム shall 警告を表示し、処理を続行するか確認する
3. The UMAP可視化システム shall 以下の情報をログ出力する: 読み込んだサンプル数、各データセットのサンプル数、クラス分布、UMAP実行時間
4. When 可視化が正常に完了した場合, the UMAP可視化システム shall 保存されたファイルパスとサンプル統計情報を表示する
5. The UMAP可視化システム shall Pythonの標準loggingモジュールを使用し、ログレベル（DEBUG, INFO, WARNING, ERROR）を設定可能にする


