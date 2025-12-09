# 実装ギャップ分析: UMAP可視化システム

## 分析サマリー

**スコープ:** DCASE2024音響イベント検出モデルの特徴量をUMAPで2次元可視化するシステム。クラス分離性、ドメイン比較、MixStyle効果の3つの可視化タイプを提供。

**主な課題:**
- 既存の特徴抽出インフラは構築済みだが、UMAP可視化スクリプトは存在しない
- プロット生成の論文品質要件（DPI、フォント、配色）に対応する必要がある
- MixStyle効果の比較可視化では複数モデルcheckpointの統合処理が必要

**推奨アプローチ:** 既存の特徴抽出パイプラインを活用し、新規にUMAP可視化モジュールを作成（Option B）。`visualize/`ディレクトリに`visualize_umap.py`を配置し、既存の特徴抽出スクリプトとの統合を図る。

---

## 1. 現状の資産調査

### 1.1 既存のディレクトリ構造

```
DESED_task/dcase2024_task4_baseline/
├── visualize/
│   ├── check_feature_properties.py        # 特徴量の性質分析（UMAP適性評価済み）
│   └── get_features/
│       ├── extract_inference_features.py  # 特徴量抽出スクリプト（.npz出力）
│       └── create_fixed_dataset.py        # データセットサンプリング設定
├── desed_task/nnet/
│   ├── CRNN.py                            # モデル定義（return_features=True対応済み）
│   └── mixstyle.py                         # MixStyle実装
└── local/
    ├── sed_trainer_pretrained.py          # Lightning トレーナー
    └── classes_dict.py                     # クラス名定義（DESED 10 + MAESTRO 11）
```

### 1.2 再利用可能な資産

#### **特徴抽出パイプライン（既存）**
- `extract_inference_features.py`: Checkpoint→.npz形式での特徴量抽出が完全実装済み
  - 384次元RNN特徴量（時間平均済み）: `features_student`, `features_teacher`
  - 27次元Weak確率: `probs_student`, `probs_teacher`
  - Ground Truth: `targets` (マルチラベル 27次元)
  - ファイル名: `filenames`
  - 出力形式: `.npz`（numpy compressed）

#### **データセット情報（既存）**
- `create_fixed_dataset.py`: 4つのデータセットのサンプリング定義
  - `desed_validation`, `desed_unlabeled`, `maestro_training`, `maestro_validation`
  - クラス別固定サンプリング（各クラス500サンプル/データセット）

#### **クラス定義（既存）**
- `classes_dict.py`: 全21クラスの名前マッピング
  - DESED: 10クラス（Alarm_bell_ringing, Blender, Cat, ...）
  - MAESTRO Real（評価対象11クラス）: birds_singing, car, people talking, ...

#### **モデル構造（既存）**
- `CRNN.py`: `return_features=True`フラグで384次元RNN出力を取得可能（line 222, 318-326）
  - 特徴量は`(batch, frames, 384)`形状
  - 時間平均で`(batch, 384)`に削減済み（`extract_inference_features.py` line 271-272）

#### **MixStyle実装（既存）**
- `mixstyle.py`: `FrequencyAttentionMixStyle`クラスでドメイン汎化手法を実装
  - Attention Network 5種類のバリエーション（default, residual_deep, multiscale, se_deep, dilated_deep）

#### **依存関係（既存）**
- `umap-learn>=0.5.9.post2`: pyproject.tomlで既にインストール済み
- `seaborn>=0.13.2`: カラーパレット用
- `matplotlib`: seabornの依存として利用可能

---

## 2. 要件実現可能性の分析

### Requirement 1: 特徴量データの読み込み ✅ **既存資産で実現可能**

**技術的ニーズ:**
- .npzファイルからのデータ読み込み
- 形状検証（features: (N, 384), targets: (N, 27)）
- 複数データセットの結合

**ギャップ:**
- **なし** - `extract_inference_features.py`が既に.npz形式で出力済み
- numpy.load()での読み込みロジックのみ追加

**制約:**
- ファイルパスはコマンドライン引数または設定ファイルで指定

---

### Requirement 2: クラス分離性の可視化 ⚠️ **新規実装必要**

**技術的ニーズ:**
- UMAPによる2次元削減（`n_neighbors=15, min_dist=0.1, metric='euclidean'`）
- マルチラベル→単一クラスラベル変換（argmax）
- クラス別散布図プロット
- 生徒/教師モデルの並列表示

**ギャップ:**
- **Missing**: UMAP次元削減ロジック
- **Missing**: matplotlib/seabornを用いたプロット生成
- **Missing**: 21クラス分の色分け処理

**既存の参考資産:**
- `check_feature_properties.py`でUMAP適性評価済み（384次元特徴量は「非常に適している⭐⭐⭐」判定）
- クラス名マッピングは`classes_dict.py`で定義済み

**実装の複雑性:**
- シンプルなUMAP適用（標準的なサンプルコードで対応可能）
- 21クラスの色分けは`seaborn.color_palette("colorblind", 21)`で解決

---

### Requirement 3: ドメイン別の可視化 ⚠️ **新規実装必要**

**技術的ニーズ:**
- ファイル名からドメインラベル抽出（desed_synthetic, desed_real, maestro_training, maestro_validation）
- ドメイン別の色・マーカー形状分け

**ギャップ:**
- **Missing**: ファイル名解析ロジック（データセット名の推定）
- **Missing**: ドメイン別のプロット設定

**既存の参考資産:**
- `extract_inference_features.py`が`filenames`を保存済み
- データセット名は.npzファイル名で識別可能（`desed_validation.npz`, `maestro_training.npz`）

**実装の複雑性:**
- データセット名→ドメインラベルのマッピング辞書で対応
- マーカー形状の設定は`matplotlib.scatter(marker='o'/'s'/'D'/'v')`で実現

---

### Requirement 4: MixStyle効果の検証可視化 ⚠️ **新規実装 + 統合処理**

**技術的ニーズ:**
- 2つのモデルcheckpoint（MixStyle適用前/後）から特徴量を読み込み
- 両モデルの特徴量を結合してUMAP削減（共通埋め込み空間）
- 2つのsubplotで並列比較
- 軸スケールとUMAP埋め込み空間の統一

**ギャップ:**
- **Missing**: 複数.npzファイルの統合処理
- **Missing**: 共通UMAP空間での削減ロジック
- **Missing**: subplot生成とスケール統一

**既存の参考資産:**
- `mixstyle.py`で実装済み（モデル学習時に適用）
- 特徴抽出スクリプトはcheckpointを引数で受け取り可能

**実装の複雑性:**
- 中程度（numpy concatenateでデータ結合、UMAP.fit_transform()を共通で実行）
- subplot間の軸範囲統一は`ax.set_xlim()/set_ylim()`で設定

**Research Needed:**
- MixStyle適用前後のcheckpointの指定方法（ユーザーが2つのcheckpointパスを指定する想定）

---

### Requirement 5: マルチラベル処理 ✅ **既存資産で実現可能**

**技術的ニーズ:**
- Ground Truth (N, 27) → argmax → 単一クラスラベル
- 同値の場合は最小インデックス選択
- クラスインデックス→クラス名マッピング
- 全要素0の場合は"Unknown"

**ギャップ:**
- **なし** - numpy.argmax()で実現可能
- クラス名マッピングは`classes_dict.py`で定義済み

**実装の複雑性:**
- 低（標準的なnumpy操作）

---

### Requirement 6: 論文掲載用プロット出力 ⚠️ **新規実装必要**

**技術的ニーズ:**
- PNG/PDF形式出力（300 DPI以上）
- フォントサイズ設定（軸ラベル12pt+、凡例10pt+）
- カラーパレット：色覚多様性対応（seaborn "colorblind"）
- 凡例の外側配置

**ギャップ:**
- **Missing**: matplotlib高解像度出力設定
- **Missing**: フォントサイズ・凡例配置のカスタマイズ

**既存の参考資産:**
- seabornの"colorblind"パレットは既にインストール済み

**実装の複雑性:**
- 低（matplotlibの`rcParams`または`savefig(dpi=300)`で対応）

---

### Requirement 7: 設定のカスタマイズ ⚠️ **新規実装必要**

**技術的ニーズ:**
- コマンドライン引数またはYAML設定ファイルでパラメータ指定
- デフォルト値の提供

**ギャップ:**
- **Missing**: argparseまたはYAML読み込みロジック
- **Missing**: 設定のバリデーション

**既存の参考資産:**
- `extract_inference_features.py`がargparse実装（line 314-342）
- プロジェクトは`confs/`ディレクトリでYAML設定管理

**実装の複雑性:**
- 低（argparseのデフォルト値設定で対応）

---

### Requirement 8: エラーハンドリングとログ出力 ⚠️ **新規実装必要**

**技術的ニーズ:**
- メモリ不足時のエラーメッセージ
- 次元数不一致時の警告
- サンプル数・クラス分布・UMAP実行時間のログ出力
- Pythonロギングモジュールの使用

**ギャップ:**
- **Missing**: try-exceptによるエラーハンドリング
- **Missing**: loggingモジュールの設定

**既存の参考資産:**
- `src.library.logger.LOGGER`: カスタムロガーがRuff設定で定義済み（`pyproject.toml` line 58）

**実装の複雑性:**
- 低（標準的なロギング設定）

---

## 3. 実装アプローチの選択肢
### 新規コンポーネントの作成 ✅ **推奨**

**新規作成ファイル:**
- `visualize/visualize_umap.py`: UMAP可視化スクリプト（メインエントリポイント）
- `visualize/umap_utils.py`（オプション）: 共通ユーティリティ関数（プロット設定、色設定など）

**統合ポイント:**
- `extract_inference_features.py`が出力した.npzファイルを入力として受け取る
- `classes_dict.py`のクラス名定義をインポート
- `check_feature_properties.py`とは独立（必要に応じて参照）

**責任境界:**
- `extract_inference_features.py`: 特徴量抽出（Checkpoint→.npz）
- `check_feature_properties.py`: 特徴量の統計分析
- **`visualize_umap.py`（新規）**: UMAP次元削減と可視化プロット生成

**トレードオフ:**
- ✅ 明確な責任分離（単一責任原則）
- ✅ テストとデバッグの容易性
- ✅ 既存コンポーネントへの影響ゼロ
- ❌ 新規ファイルの追加（ただしプロジェクト構造上は適切な配置）

**ファイル構造:**
```python
# visualize/visualize_umap.py
def load_features(npz_paths: List[str]) -> Dict[str, np.ndarray]:
    """複数の.npzファイルを読み込み統合"""
    pass

def preprocess_labels(targets: np.ndarray, class_names: List[str]) -> np.ndarray:
    """マルチラベル→単一ラベル変換"""
    pass

def apply_umap(features: np.ndarray, **umap_params) -> np.ndarray:
    """UMAP次元削減"""
    pass

def plot_class_separation(embedding: np.ndarray, labels: np.ndarray, ...):
    """クラス分離性プロット生成"""
    pass

def plot_domain_comparison(embedding: np.ndarray, domain_labels: np.ndarray, ...):
    """ドメイン比較プロット生成"""
    pass

def plot_mixstyle_effect(embedding_before: np.ndarray, embedding_after: np.ndarray, ...):
    """MixStyle効果プロット生成（2 subplots）"""
    pass

def main():
    """argparseでCLI設定 → 処理実行 → ファイル保存"""
    pass
```

---

## 4. 実装複雑度とリスク評価

### 工数見積もり: **M（3-7日）**

**理由:**
- 既存の特徴抽出パイプラインが完成しているため、UMAP可視化ロジックの実装に集中できる
- 3つの可視化タイプ（クラス分離、ドメイン比較、MixStyle効果）の実装が必要
- 論文品質要件（DPI、フォント、配色、凡例配置）への対応が追加作業

**内訳:**
- Day 1-2: 基本的なUMAP可視化スクリプトの実装（データ読み込み、UMAP適用、基本プロット）
- Day 3-4: 3つの可視化タイプの実装（クラス分離、ドメイン比較、MixStyle効果）
- Day 5-6: 論文品質要件への対応（高解像度出力、フォント設定、色覚対応配色）
- Day 7: エラーハンドリング、ロギング、テスト、ドキュメント

### リスク: **Medium**

**技術的リスク:**
- **UMAP性能**: 大規模データセット（数千サンプル）での次元削減に時間がかかる可能性
  - **対策**: バッチサイズの調整、または事前にサンプル数を制限
- **メモリ使用量**: 複数データセットの統合時にメモリ不足
  - **対策**: データセット別にUMAP適用後、結果を結合（ただし埋め込み空間が異なるため注意）
- **色分けの視認性**: 21クラスの色分けが多すぎて視認性が低下
  - **対策**: seabornの"colorblind"パレットを使用、または凡例を工夫

**統合リスク:**
- **低** - 既存の特徴抽出スクリプトと独立した新規モジュールのため、既存コードへの影響なし

**パフォーマンスリスク:**
- **中** - UMAPの計算時間（数千サンプル×384次元で1-5分程度）
  - **対策**: `random_state`で再現性を確保し、結果をキャッシュ

---

## 5. 設計フェーズへの推奨事項

### 推奨アプローチ: **Option B（新規コンポーネント作成）**

**理由:**
1. **責任の明確化**: 特徴抽出（既存）と可視化（新規）を分離
2. **保守性**: 既存コンポーネントに影響を与えない
3. **拡張性**: 将来的に他の可視化手法（t-SNE, PCA等）を追加しやすい

### 主要な設計決定事項

1. **UMAP埋め込み空間の統一方法（MixStyle効果可視化）**
   - **Option 1**: 2つのモデルの特徴量を結合してUMAP.fit_transform()を1回実行
   - **Option 2**: 片方のモデルでUMAP.fit()し、もう片方にUMAP.transform()を適用
   - **推奨**: Option 1（共通空間で比較しやすい）

2. **データセット名→ドメインラベルのマッピング**
   - .npzファイル名を解析してドメインを推定
   - または`filenames`配列からファイル名パターンを分析

3. **論文品質プロットの設定**
   - matplotlib `rcParams`でグローバル設定
   - または各プロット関数で個別に設定
   - **推奨**: 各プロット関数で設定（柔軟性が高い）

### 設計フェーズで調査すべき項目

1. **MixStyle適用前後のcheckpoint識別方法**
   - ユーザーが2つのcheckpointパスを指定する仕様で問題ないか確認
   - wandbログやcheckpointファイル名からMixStyle設定を自動判別するか

2. **UMAPパラメータのチューニング**
   - デフォルト値（n_neighbors=15, min_dist=0.1）でクラス分離が十分か検証
   - データセットに応じた最適値の探索

3. **プロット形式の詳細**
   - 論文掲載時のレイアウト（1列2行 vs 2列1行）
   - カラーバーの必要性（連続値の場合）

---

## 6. 要件・資産対応表

| 要件 | 既存資産 | ギャップ | 実装アプローチ |
|------|----------|----------|----------------|
| R1: 特徴量読み込み | .npzファイル出力済み | なし | numpy.load()のみ追加 |
| R2: クラス分離可視化 | クラス名定義済み | UMAP + プロット | 新規実装（標準ライブラリ使用） |
| R3: ドメイン別可視化 | ファイル名保存済み | ドメイン推定ロジック | 新規実装（辞書マッピング） |
| R4: MixStyle効果検証 | MixStyle実装済み | 複数モデル統合処理 | 新規実装（numpy concatenate） |
| R5: マルチラベル処理 | Ground Truth保存済み | なし | numpy.argmax()のみ |
| R6: 論文品質出力 | seaborn/matplotlib | 高解像度・配色設定 | 新規実装（matplotlib設定） |
| R7: 設定カスタマイズ | argparse参考例あり | YAML/CLI設定 | 新規実装（argparse） |
| R8: エラーハンドリング | カスタムロガー定義済み | エラー処理ロジック | 新規実装（logging + try-except） |

---

## 7. 次のステップ

### 設計フェーズへ進む準備

**完了条件:**
- ✅ 既存資産の調査完了（特徴抽出パイプライン、クラス定義、依存関係）
- ✅ 要件の実現可能性確認（すべての要件が実装可能）
- ✅ 実装アプローチの選択（Option B: 新規コンポーネント作成）

**推奨コマンド:**
```bash
/kiro:spec-design umap-visualization
```

または、要件を承認してから設計フェーズへ進む:
```bash
/kiro:spec-design umap-visualization -y
```

**設計フェーズで作成されるもの:**
- `design.md`: 詳細な設計書（モジュール構造、関数シグネチャ、データフロー）
- API設計、エラーハンドリング戦略、テスト計画

---

## 付録: 既存コードの再利用可能性

### A. 特徴抽出パイプライン

**ファイル:** `visualize/get_features/extract_inference_features.py`

**再利用可能な関数:**
- `get_embeddings_name()`: 埋め込みファイルパス取得
- `get_encoder()`: CatManyHotEncoder作成
- `create_datasets()`: データセット作成（JSON定義ベース）
- `extract_features_from_dataset()`: モデル推論→特徴量抽出

**出力形式:**
```python
# .npzファイル構造
{
    'features_student': (N, 384),    # RNN出力特徴量（時間平均済み）
    'features_teacher': (N, 384),
    'probs_student': (N, 27),        # Weak確率（attention pooling済み）
    'probs_teacher': (N, 27),
    'filenames': (N,),               # ファイル名リスト
    'targets': (N, 27)               # Ground Truth（マルチラベル）
}
```

### B. クラス定義

**ファイル:** `local/classes_dict.py`

**利用可能な辞書:**
- `classes_labels_desed`: DESED 10クラス（OrderedDict）
- `classes_labels_maestro_real`: MAESTRO 17クラス（OrderedDict）
- `classes_labels_maestro_real_eval`: 評価対象11クラス（set）

**全21クラスのリスト:**
```python
all_classes = list(classes_labels_desed.keys()) + list(classes_labels_maestro_real_eval)
# ['Alarm_bell_ringing', 'Blender', 'Cat', 'Dishes', 'Dog', ..., 'metro_leaving']
```

### C. モデルアーキテクチャ

**ファイル:** `desed_task/nnet/CRNN.py`

**重要な機能:**
- `forward(..., return_features=True)`: 384次元RNN特徴量を返す
  - 返り値: `{'strong_probs': ..., 'weak_probs': ..., 'features': (batch, frames, 384)}`
  - `features`はdropout前のRNN出力

**MixStyle統合:**
- `desed_task/nnet/mixstyle.py`: `FrequencyAttentionMixStyle`クラス
- 学習時にCNNレイヤーに適用済み

### D. 依存関係

**既にインストール済み:**
- `umap-learn>=0.5.9.post2`: UMAP次元削減
- `seaborn>=0.13.2`: カラーパレット
- `matplotlib`: プロット生成（seabornの依存）
- `numpy`, `pandas`: データ処理

**追加インストール不要:** すべての依存関係が`pyproject.toml`で管理済み
