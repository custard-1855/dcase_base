# UMAP 可視化ツール
このディレクトリには、DCASE2024 Task 4 サウンドイベント検出モデルをUMAP次元削減で解析するための可視化ツールが含まれています。
## 概要
UMAP可視化システムは、学習された特徴表現を解析するための3種類の論文品質のプロットを提供します。
1. **クラス分離の可視化**: DESEDとMAESTROのイベントクラスが特徴空間でどれだけ分離されているかを示します。
2. **ドメイン比較**: 異なるドメイン（DESED合成／実データ、MAESTRO間）での特徴分布を比較します。
3. **MixStyle効果の比較**: MixStyleドメイン一般化技術の有効性を、適用前後の特徴分布の比較で示します。
すべての可視化は、訓練済みモデルから抽出した384次元のRNN出力特徴量（ドロップアウト前、時間平均）を用いて行われます。
## イベントクラス
システムでは21種類のサウンドイベントクラスを可視化します。
**DESED（10クラス）:**
- Alarm_bell_ringing
- Blender
- Cat
- Dishes
- Dog
- Electric_shaver_toothbrush
- Frying
- Running_water
- Speech
- Vacuum_cleaner
**MAESTRO 実データ評価（11クラス）:**
- birds_singing
- car
- people talking
- footsteps
- children voices
- wind_blowing
- brakes_squeaking
- large_vehicle
- cutlery and dishes
- metro approaching
- metro leaving
## 使用例
### 1. クラス分離の可視化
モデルが特徴空間で異なるイベントクラスをどの程度分離できているかを可視化します。
# studentモデルでの基本的な実行例
python visualize/umap/visualize_umap.py \
  --mode class_separation \
  --input inference_outputs/baseline/desed_validation.npz \
  --model_type student \
  --output_dir output/class_separation

# teacherモデルの場合
python visualize/umap/visualize_umap.py \
  --mode class_separation \
  --input inference_outputs/baseline/desed_validation.npz \
  --model_type teacher
### 2. ドメイン比較
異なるデータセット／ドメイン間での特徴分布を比較します。
# DESEDとMAESTROドメインの比較
python visualize/umap/visualize_umap.py \
  --mode domain_comparison \
  --inputs inference_outputs/baseline/desed_validation.npz \
           inference_outputs/baseline/maestro_validation.npz \
  --model_type student

# 合成DESEDと実DESEDデータの比較
python visualize/umap/visualize_umap.py \
  --mode domain_comparison \
  --inputs inference_outputs/baseline/desed_synthetic.npz \
           inference_outputs/baseline/desed_real.npz \
  --model_type teacher
### 3. MixStyle効果の比較
MixStyle適用の前後で特徴分布がどう変化するかを可視化します。
# MixStyle有無で訓練したモデルを比較
python visualize/umap/visualize_umap.py \
  --mode mixstyle_effect \
  --before inference_outputs/baseline_no_mixstyle/desed_validation.npz \
  --after inference_outputs/baseline_with_mixstyle/desed_validation.npz \
  --model_type student
### 4. UMAPパラメータのカスタマイズ
異なる可視化ニーズのために、UMAP次元削減パラメータをカスタマイズ可能です。
python visualize/umap/visualize_umap.py \
  --mode class_separation \
  --input data.npz \
  --n_neighbors 30 \
  --min_dist 0.05 \
  --metric cosine \
  --random_state 123
### 5. YAML設定ファイルの利用
実験設定を一貫して適用するため、YAML設定ファイルが利用できます。
# デフォルト設定ファイルを使用
python visualize/umap/visualize_umap.py \
  --mode class_separation \
  --input data.npz \
  --config config/umap_visualization.yaml

# 特定パラメータの上書き
python visualize/umap/visualize_umap.py \
  --mode class_separation \
  --input data.npz \
  --config config/umap_visualization.yaml \
  --dpi 600 \
  --figsize 16 10
### 6. カスタムプロットスタイリング
論文向けにプロットの外観を調整できます。
python visualize/umap/visualize_umap.py \
  --mode class_separation \
  --input data.npz \
  --palette tab20 \
  --figsize 14 10 \
  --dpi 600 \
  --font_size_title 16 \
  --font_size_label 14
## 設定
システムはYAMLファイルで柔軟に設定できます。詳細は config/umap_visualization.yaml を参照してください。
### UMAPパラメータ
- n_neighbors（デフォルト: 15）：UMAPの近傍数。大きいほどグローバル構造を反映します。
- min_dist（デフォルト: 0.1）：埋め込み空間内の点間最小距離。小さいほどクラスタが密接します。
- metric（デフォルト: euclidean）：距離指標（euclidean, cosine, manhattanなど）。
- random_state（デフォルト: 42）：再現性確保の乱数シード
### プロットパラメータ
- dpi（デフォルト: 300）：解像度（DPI）。出版品質には300以上を推奨
- figsize（デフォルト: [12, 8]）：画像サイズ[幅, 高さ]
- palette（デフォルト: colorblind）：Seabornカラーパレット（カラーバリアフリー推奨）
- font_size_title（デフォルト: 14）：タイトルのフォントサイズ
- font_size_label（デフォルト: 12）：軸ラベルのフォントサイズ
- font_size_legend（デフォルト: 10）：凡例のフォントサイズ
### 出力パラメータ
- dir（デフォルト: output/umap_plots）：出力プロット保存ディレクトリ
### ログパラメータ
- level（デフォルト: INFO）：ログ出力レベル（DEBUG, INFO, WARNING, ERROR）
## 出力ファイル
すべての可視化でPNG形式とPDF形式の両方が出力されます。
- PNG：高解像度ラスター画像（プレゼンやWeb用）
- PDF：ベクター画像（論文・印刷用）
ファイル名のパターン：
{prefix}_{model_type}_{timestamp}.{extension}
例：
class_separation_student_20250108_143052.png
class_separation_student_20250108_143052.pdf
## コマンドライン引数
### 共通引数
22:54
- --mode: 可視化モード（class_separation, domain_comparison, mixstyle_effect）
- --model_type: 可視化対象モデル（studentまたはteacher、デフォルト: student）
- --output_dir: プロット出力先ディレクトリ
- --config: YAML設定ファイルのパス
### モード別引数
**class_separation モード:** 
- --input: 単一.npz特徴量ファイルのパス
**domain_comparison モード:** 
- --inputs: 複数の.npz特徴量ファイルのパス（空白区切り）
**mixstyle_effect モード:** 
- --before: MixStyle前の.npzファイルパス
- --after: MixStyle後の.npzファイルパス
### UMAPパラメータ（任意）
- --n_neighbors: 近傍数（デフォルト: 15）
- --min_dist: 最小距離（デフォルト: 0.1）
- --metric: 距離指標（デフォルト: euclidean）
- --random_state: 乱数シード（デフォルト: 42）
### プロットパラメータ（任意）
- --dpi: DPI解像度（デフォルト: 300）
- --figsize: 画像サイズ（デフォルト: 12 8）
- --palette: カラーパレット名（デフォルト: colorblind）
- --font_size_title: タイトルフォントサイズ（デフォルト: 14）
- --font_size_label: 軸ラベルフォントサイズ（デフォルト: 12）
- --font_size_legend: 凡例フォントサイズ（デフォルト: 10）
### ロギング（任意）
- --log_level: ログレベル（デフォルト: INFO）
## 特徴量の抽出
可視化の前に訓練済みモデルから特徴量を抽出しておく必要があります。特徴量は.npz形式で、以下のキーを持つべきです。
必須キー:
- features_student: studentモデル特徴量 (N, 384)
- features_teacher: teacherモデル特徴量 (N, 384)
- targets: 正解ラベル (N, 27) - マルチラベル
- filenames: 音声ファイル名 (N,)
任意キー:
- probs_student: studentモデル予測値 (N, 27)
- probs_teacher: teacherモデル予測値 (N, 27)
可視化では、マルチラベルのうち最大値（argmax）をクラスラベルとして色分けに使います。
## 注意事項
- **メモリ使用量**: サンプル数が1万を超える場合は警告が出ます。メモリエラー時はデータのダウンサンプリングを検討してください。
- **カラーパレット**: デフォルト colorblind パレットはアクセシビリティに配慮。21クラスでは類似色もあるので凡例参照を推奨します。
- **再現性**: random_state を設定してUMAPの埋め込み結果を安定化できます。
- **出版品質**: DPIは300以上、PDF出力を推奨。フォントサイズも論文向けに調整済みです。
## トラブルシューティング
**FileNotFoundError**: 入力 .npz ファイルの存在とパスを確認してください。
**ValueError（shape mismatch）**: 特徴量配列は(N, 384)、ラベルは(N, 27)の形状であることを確認してください。
**MemoryError**: サンプル数を減らすなどで対応してください。
**Low variance warning**: 一部特徴量次元の分散が非常に小さい場合、UMAP品質への影響があります。多くは致命的ではありませんが、必要に応じて確認してください。
## 関連スクリプト
- check_feature_properties.py: 可視化前の特徴量の統計解析
- feature_loader.py: 特徴量ローディング補助
- umap_reducer.py: UMAP次元削減の実装
- plot_generator.py: プロット生成およびスタイリング