# 可視化スクリプト使用ガイド

このディレクトリには、推論結果を分析・可視化するための4つのスクリプトが含まれています。

## 📁 ディレクトリ構成

```
dcase2024_task4_baseline/
├── inference_outputs/          # 推論結果（入力）
│   ├── baseline/
│   ├── cmt_normal/
│   └── cmt_neg/
├── visualization_outputs/      # 可視化結果（出力）
│   ├── umap/
│   ├── reliability/
│   ├── gradcam/
│   └── analysis_report.md
├── visualize_umap.py          # 1. UMAP可視化
├── visualize_reliability.py   # 2. Reliability Diagram
├── generate_analysis_report.py # 3. 統合分析レポート
└── visualize_gradcam.py       # 4. Grad-CAM分析
```

---

## 🎯 各スクリプトの概要

### 1. `visualize_umap.py` - UMAP可視化

**目的**: 特徴量空間を2次元に投影し、データセット・ドメイン・クラス・モデル別に可視化

**機能**:
- データセット別の比較（DESED vs MAESTRO）
- ドメイン別の比較（合成データ vs 実データ）
- クラス別の分布確認
- モデル間の特徴量空間の違い
- インタラクティブHTML版の生成

**使用法**:
```bash
python visualize_umap.py \
    --input_dirs inference_outputs/baseline inference_outputs/cmt_normal inference_outputs/cmt_neg \
    --output_dir visualization_outputs/umap \
    --feature_type student \
    --n_components 2 \
    --random_state 42
```

**オプション**:
- `--feature_type`: 使用する特徴量 (`student`, `teacher`, `both`)
- `--n_components`: UMAP出力次元数（デフォルト: 2）
- `--random_state`: 乱数シード（デフォルト: 42）

**出力**:
```
visualization_outputs/umap/
├── student/
│   ├── dataset_comparison.png
│   ├── domain_comparison.png
│   ├── model_comparison.png
│   ├── class_distribution_predicted.png
│   ├── class_distribution_true.png
│   ├── interactive_umap.html
│   ├── umap_embedding.npy
│   └── metadata.csv
└── teacher/
    └── (同様の構成)
```

---

### 2. `visualize_reliability.py` - Reliability Diagram

**目的**: モデルの予測確率が実際の正答率とどの程度一致しているか（キャリブレーション）を評価

**機能**:
- Reliability Diagram（予測確率 vs 実際の正答率）
- ECE（Expected Calibration Error）の計算
- データセット別・モデル別の比較
- Confidence Histogram
- クラス別のReliability Diagram（上位K個）

**使用法**:
```bash
python visualize_reliability.py \
    --input_dirs inference_outputs/baseline inference_outputs/cmt_normal inference_outputs/cmt_neg \
    --output_dir visualization_outputs/reliability \
    --pred_type student \
    --n_bins 10 \
    --top_k_classes 10
```

**オプション**:
- `--pred_type`: 使用する予測 (`student`, `teacher`, `both`)
- `--n_bins`: ビン数（デフォルト: 10）
- `--top_k_classes`: クラス別分析で表示する上位K個（デフォルト: 10）

**出力**:
```
visualization_outputs/reliability/
├── student/
│   ├── reliability_diagram_by_dataset.png
│   ├── ece_comparison.csv
│   ├── ece_model_comparison.csv
│   ├── confidence_histogram.png
│   └── per_class_reliability/
│       ├── Speech.png
│       ├── Dog.png
│       └── ...
└── teacher/
    └── (同様の構成)
```

**重要な指標**:
- **ECE (Expected Calibration Error)**: 小さいほど良い（0に近いほどキャリブレーションが良い）
- **Reliability Diagram**: 対角線（Perfect Calibration）に近いほど良い

---

### 3. `generate_analysis_report.py` - 統合分析レポート

**目的**: 全モデルの定量的な比較分析をMarkdown形式のレポートとして生成

**機能**:
- 全体的な性能比較（Accuracy, Precision, Recall, F1, mAP, ECE）
- データセット別の詳細分析
- クラス別性能分析
- モデル改善の分析（ベースラインとの比較）
- CSV形式での数値データ出力

**使用法**:
```bash
python generate_analysis_report.py \
    --input_dirs inference_outputs/baseline inference_outputs/cmt_normal inference_outputs/cmt_neg \
    --output visualization_outputs/analysis_report.md \
    --threshold 0.5
```

**オプション**:
- `--threshold`: 予測の閾値（デフォルト: 0.5）

**出力**:
```
visualization_outputs/
├── analysis_report.md          # Markdownレポート
└── csv_outputs/
    ├── overall_metrics.csv
    ├── per_class_student_desed_validation.csv
    ├── per_class_student_maestro_training.csv
    └── ...
```

**レポートの内容**:
1. モデル概要
2. 全体的な性能比較
3. データセット別の詳細分析
4. クラス別性能分析
5. モデル改善の分析
6. サマリーと考察

---

### 4. `visualize_gradcam.py` - Grad-CAM分析

**目的**: モデルが注目している時間-周波数領域を可視化し、境界事例や誤予測の原因を分析

**機能**:
- 境界事例の自動抽出（予測確率0.4-0.6）
- 誤予測サンプルの抽出
- メルスペクトログラムとGrad-CAMの重畳表示
- モデル間比較

**⚠️ 注意**: このスクリプトは音声ファイルとモデルチェックポイントへのアクセスが必要です。

**使用法**:
```bash
python visualize_gradcam.py \
    --input_dirs inference_outputs/baseline inference_outputs/cmt_normal \
    --checkpoints wandb/evaled_runs/run-XXX/files/checkpoints/best.ckpt \
                  wandb/evaled_runs/run-YYY/files/checkpoints/best.ckpt \
    --config confs/pretrained.yaml \
    --output_dir visualization_outputs/gradcam \
    --n_samples 10 \
    --device cuda \
    --pred_type student
```

**オプション**:
- `--checkpoints`: モデルチェックポイント（`--input_dirs`と同じ順序で指定）
- `--config`: モデル設定ファイル
- `--n_samples`: 可視化するサンプル数（デフォルト: 10）
- `--device`: 使用デバイス（`cuda` or `cpu`）
- `--pred_type`: 使用するモデル（`student` or `teacher`）

**出力**:
```
visualization_outputs/gradcam/
├── boundary_cases/
│   └── desed_validation/
│       ├── baseline_sample001_Speech_prob0.52.png
│       ├── cmt_normal_sample001_Speech_prob0.52.png
│       └── ...
└── misclassified/
    └── desed_validation/
        ├── baseline_sample001_predSpeech_trueDog.png
        └── ...
```

**Grad-CAMの解釈**:
- **赤色領域**: モデルが強く注目している領域（高い勾配）
- **青色領域**: モデルがあまり注目していない領域（低い勾配）

---

## 📊 推奨される実行順序

### Phase 1: 基本可視化（すぐ実行可能）

```bash
# 1. UMAP可視化
python visualize_umap.py \
    --input_dirs inference_outputs/baseline inference_outputs/cmt_normal inference_outputs/cmt_neg \
    --output_dir visualization_outputs/umap \
    --feature_type both

# 2. Reliability Diagram
python visualize_reliability.py \
    --input_dirs inference_outputs/baseline inference_outputs/cmt_normal inference_outputs/cmt_neg \
    --output_dir visualization_outputs/reliability \
    --pred_type both

# 3. 統合レポート
python generate_analysis_report.py \
    --input_dirs inference_outputs/baseline inference_outputs/cmt_normal inference_outputs/cmt_neg \
    --output visualization_outputs/analysis_report.md
```

### Phase 2: 詳細分析（音声データとモデルが必要）

```bash
# 4. Grad-CAM分析
python visualize_gradcam.py \
    --input_dirs inference_outputs/baseline inference_outputs/cmt_normal \
    --checkpoints <baseline_checkpoint_path> <cmt_normal_checkpoint_path> \
    --config confs/pretrained.yaml \
    --output_dir visualization_outputs/gradcam \
    --n_samples 20
```

---

## 🔧 必要なライブラリ

### 基本的な可視化（Phase 1）

```bash
pip install numpy pandas matplotlib seaborn umap-learn scikit-learn plotly
```

### Grad-CAM分析（Phase 2）

```bash
pip install librosa torch torchaudio pyyaml
```

---

## 📈 分析のヒント

### UMAP可視化で確認すべきポイント

1. **データセット間の分離度**
   - DESEDとMAESTROがどの程度分離しているか
   - 合成データ（DESED synth）と実データ（MAESTRO Real）の分布の違い

2. **モデル間の特徴量空間の違い**
   - CMT改良によって特徴量空間がどう変化したか
   - クラスタの分離がより明確になっているか

3. **クラスの混同パターン**
   - どのクラス同士が近くに配置されているか（混同しやすい）
   - 例: Speech と people talking, Dog と Dog barking など

### Reliability Diagramで確認すべきポイント

1. **ECEの値**
   - 0.1以下: 良好なキャリブレーション
   - 0.1-0.2: 許容範囲
   - 0.2以上: キャリブレーションが悪い

2. **Over-confidence vs Under-confidence**
   - 線が対角線より上: モデルが過信（実際より高い確率を出力）
   - 線が対角線より下: モデルが過小評価（実際より低い確率を出力）

3. **データセット間の違い**
   - DESED validationとMAESTRO validationでキャリブレーションが異なるか
   - 学習データ（MAESTRO training）で過学習の兆候はないか

### Grad-CAMで確認すべきポイント

1. **境界事例の注目領域**
   - 予測確率が中間的なサンプルで、モデルは何に注目しているか
   - 複数のクラスに該当しそうな音響イベントが混在しているか

2. **誤予測の原因**
   - 誤予測サンプルで、モデルが間違った領域に注目しているか
   - 背景雑音やノイズに反応しているか

3. **モデル間の注目領域の違い**
   - CMT改良によって、より適切な領域に注目するようになったか

---

## 🐛 トラブルシューティング

### メモリ不足エラー

UMAP計算時にメモリ不足が発生する場合:

```bash
# データセットを個別に処理
python visualize_umap.py \
    --input_dirs inference_outputs/baseline \
    --output_dir visualization_outputs/umap_baseline
```

### Grad-CAMで音声ファイルが見つからない

`inference_outputs/*/metadata.json` の `config_path` が正しいか確認してください。
音声ファイルのパスは、そこで指定されたYAMLファイルから解決されます。

### プロット生成が遅い

クラス別のReliability Diagramは時間がかかります。`--top_k_classes`を減らしてください:

```bash
python visualize_reliability.py \
    --input_dirs ... \
    --top_k_classes 5  # デフォルトは10
```

---

## 📝 結果の解釈例

### 例1: CMT改良の効果が明確な場合

**UMAP**:
- CMT改良モデルの特徴量が、ベースラインよりも明確にクラスタリングされている
- 境界が曖昧だったクラスペアが分離している

**Reliability Diagram**:
- CMT改良モデルのECEがベースラインより低い（0.15 → 0.08など）
- 対角線に近い形状になっている

**統合レポート**:
- F1スコアとmAPが向上している
- 特定のクラス（例: Speech, Dog）で顕著な改善

**Grad-CAM**:
- 境界事例で、より適切な時間-周波数領域に注目している
- 誤予測が減少し、正しいイベントに焦点が当たっている

### 例2: データセット間のドメインギャップが大きい場合

**UMAP**:
- DESEDとMAESTROが完全に分離している
- 合成データと実データのクラスタが異なる

**Reliability Diagram**:
- DESED validationとMAESTRO validationでECEが大きく異なる
- 学習時に見ていないドメインでキャリブレーションが悪い

**対策**:
- ドメイン適応手法の導入
- より多様なデータでの学習
- マルチタスク学習の検討

---

## 📚 参考文献

- **UMAP**: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.
- **Calibration**: Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks.
- **Grad-CAM**: Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.

---

**生成日**: 2025-12-04
**作成者**: Claude Code
**バージョン**: 1.0
