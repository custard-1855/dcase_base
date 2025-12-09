# Research & Design Decisions Template

---
**Purpose**: Capture discovery findings, architectural investigations, and rationale that inform the technical design.

**Usage**:
- Log research activities and outcomes during the discovery phase.
- Document design decision trade-offs that are too detailed for `design.md`.
- Provide references and evidence for future audits or reuse.
---

## Summary
- **Feature**: `umap-visualization`
- **Discovery Scope**: Extension (既存の可視化システムへの新機能追加)
- **Key Findings**:
  - 既存の特徴量抽出・分析コード(`check_feature_properties.py`, `extract_inference_features.py`)が存在し、統合可能
  - UMAP-learn 0.5.9が既にプロジェクトに含まれ、安定版APIが利用可能
  - 21クラス(DESED 10 + MAESTRO 11評価クラス)を扱う既存のエンコーダとクラス定義が利用可能
  - Seabornのcolorblindパレットが色覚多様性対応として標準で利用可能

## Research Log

### UMAP Library API and Parameters
- **Context**: 要件で指定されたUMAPパラメータ(`n_neighbors=15`, `min_dist=0.1`, `metric='euclidean'`)の妥当性を検証
- **Sources Consulted**:
  - [UMAP 0.5.8 公式ドキュメント - Basic Parameters](https://umap-learn.readthedocs.io/en/latest/parameters.html)
  - [UMAP API Guide](https://umap-learn.readthedocs.io/en/latest/api.html)
- **Findings**:
  - `n_neighbors`: 5-50の範囲が推奨、10-15がデフォルトとして適切。15は標準的な選択肢
  - `min_dist`: 0.001-0.5の範囲が推奨、0.1がデフォルト。クラスタの視覚的分離に適している
  - `metric`: euclidean距離が標準。384次元の密なRNN特徴量に適合
  - `random_state`: 再現性のため42を推奨(プロジェクト標準)
- **Implications**: 要件で指定されたデフォルト値は最新のベストプラクティスに準拠

### Existing Feature Extraction Infrastructure
- **Context**: 特徴量の読み込みと前処理の実装方針を決定
- **Sources Consulted**:
  - `visualize/check_feature_properties.py`: 特徴量分析の既存実装
  - `visualize/get_features/extract_inference_features.py`: 特徴量抽出パイプライン
- **Findings**:
  - `.npz`形式で特徴量が保存されており、以下のキーが標準化されている:
    - `features_student`, `features_teacher`: (N, 384) shape
    - `probs_student`, `probs_teacher`: (N, 27) shape (弱ラベル予測)
    - `targets`: (N, 27) shape (Ground Truthマルチラベル)
    - `filenames`: ファイル名リスト
  - データセット名(`desed_validation`, `desed_unlabeled`, `maestro_training`, `maestro_validation`)がファイル名に含まれる
  - 特徴量検証ロジック(次元数、分散、スパース性チェック)が既に実装済み
- **Implications**:
  - 既存のデータローダを拡張可能。新規実装は不要
  - 特徴量検証機能を可視化システムにも統合可能(エラーハンドリング要件8に対応)

### Class Label Encoding and Multi-label Handling
- **Context**: 要件5(マルチラベル処理)の実装方針
- **Sources Consulted**: `local/classes_dict.py`
- **Findings**:
  - DESED 10クラス: `classes_labels_desed` (OrderedDict)
  - MAESTRO 11評価クラス: `classes_labels_maestro_real_eval` (set)
  - MAESTROとDESEDのエイリアス定義: `maestro_desed_alias` (例: "people talking" → "Speech")
  - 既存のエンコーダ(`CatManyHotEncoder`)がDESED+MAESTRO統合エンコーディングをサポート
- **Implications**:
  - argmax操作後のクラスインデックスを既存のクラス辞書でマッピング可能
  - 統合21クラス(DESED 10 + MAESTRO real eval 11)のラベルリストを新規生成する必要あり
  - "Unknown"クラス処理(要件5.4)のための追加ロジックが必要

### Color Palette for Accessibility
- **Context**: 要件6.7(色覚多様性対応)の実装方針
- **Sources Consulted**:
  - [Seaborn Color Palettes Documentation](https://seaborn.pydata.org/tutorial/color_palettes.html)
  - [Color Universal Design GitHub](https://github.com/mbhall88/cud)
  - Medium記事: Finding the Best Color-blind Friendly Palette on Python Seaborn
- **Findings**:
  - Seaborn標準の`"colorblind"`パレット(6色)が利用可能
  - 21クラスには不足するため、`sns.color_palette("colorblind", n_colors=21)`で拡張が必要
  - または`tab20`パレット(20色)と組み合わせる選択肢も存在
  - 2024年時点でColor Universal Design(CUD)パケッジも利用可能だが、seabornで十分
- **Implications**:
  - デフォルトで`seaborn.color_palette("colorblind", 21)`を使用
  - カスタムパレット指定機能(要件7.2)で`tab20`, `tab20b`などの代替も許可
  - ドメイン別可視化(要件3)では4ドメインのみのため、標準colorblindパレット(6色)で対応可能

### Integration with Existing Project Structure
- **Context**: 新規スクリプトの配置場所とインポートパターンの決定
- **Sources Consulted**: `.kiro/steering/structure.md`, 既存の`visualize/`ディレクトリ構造
- **Findings**:
  - `DESED_task/dcase2024_task4_baseline/visualize/`: 分析ツール配置場所
  - 既存パターン: `visualize_*.py`形式のトップレベルスクリプト
  - `get_features/`: 特徴量抽出サブモジュール
  - `visualization_utils.py`: 共通プロット関数(ステアリング文書で言及されているが未確認)
- **Implications**:
  - 新規スクリプト名: `visualize_umap.py`(既存パターンに準拠)
  - 共通関数(プロット設定、カラーパレット生成)は`visualization_utils.py`に配置
  - 既存の`get_features/`モジュールからインポート可能

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| Monolithic Script | 単一の`visualize_umap.py`に全機能を実装 | シンプル、デバッグ容易 | 機能追加で肥大化、テスト困難 | 小規模な可視化ツールには妥当 |
| Modular Architecture | クラス分離: `UMAPVisualizer`, `FeatureLoader`, `PlotGenerator` | 保守性、再利用性、テスト容易 | 小規模機能には過剰設計 | **推奨**: 今後の拡張性を考慮 |
| Functional Decomposition | 関数ベース + ユーティリティモジュール | バランスが良い、既存パターンに準拠 | 状態管理が複雑になる可能性 | 既存の`check_feature_properties.py`と一貫性あり |

## Design Decisions

### Decision: Modular Class-Based Architecture
- **Context**: 8つの要件(特徴量読み込み、3種類の可視化、マルチラベル処理、出力、設定、エラー処理)を統合する必要がある
- **Alternatives Considered**:
  1. モノリシックスクリプト — 単一ファイルに全機能を実装
  2. 関数型分解 — 独立した関数群とユーティリティモジュール
  3. クラスベースモジュラー — 責任ごとにクラス分離
- **Selected Approach**: クラスベースモジュラーアーキテクチャ
  - `FeatureLoader`: 特徴量読み込み、検証、マルチラベル処理(要件1, 5, 8)
  - `UMAPReducer`: UMAP次元削減の実行とキャッシング(要件2-4, 7)
  - `PlotGenerator`: プロット生成と出力(要件6, 7)
  - `UMAPVisualizer`: エントリーポイント、コマンドライン引数処理(要件7)
- **Rationale**:
  - 各クラスが単一責任を持ち、テスト容易性が向上
  - 設定のカスタマイズ(要件7)が各コンポーネントで独立して実装可能
  - 今後の拡張(例: t-SNE、PCA)に対応しやすい
  - PyTorch Lightning等の既存プロジェクトパターンと一貫性
- **Trade-offs**:
  - Benefits: 保守性、再利用性、テストカバレッジ、型安全性
  - Compromises: 初期実装コストがやや増加(約20-30%のコード増)
- **Follow-up**: 単体テストの作成(pytestベース)、型アノテーションの徹底(mypy準拠)

### Decision: Shared UMAP Embedding for MixStyle Comparison
- **Context**: 要件4.2で「両モデルの特徴量を結合した上でUMAP次元削減を実行(共通の埋め込み空間を使用)」が指定されている
- **Alternatives Considered**:
  1. 個別埋め込み — 各モデルで独立してUMAP実行 → 比較不可能
  2. 共通埋め込み — 結合後にUMAP実行 → 同一座標系で比較可能
  3. Aligned UMAP — `umap.AlignedUMAP`を使用 → 複雑度増加
- **Selected Approach**: 共通埋め込み(特徴量結合後にUMAP実行)
  - `np.concatenate([features_before, features_after], axis=0)`で結合
  - UMAP fittingは結合データに対して1回のみ実行
  - プロット時にインデックスで分割して2つのsubplotに表示
- **Rationale**:
  - 要件4.5「subplot間で軸スケールとUMAP埋め込み空間を統一」を自然に満たす
  - MixStyle効果の定量的比較が可能(ドメイン間の距離変化など)
- **Trade-offs**:
  - Benefits: 視覚的整合性、比較可能性
  - Compromises: メモリ使用量が2倍(両モデルの特徴量を同時保持)
- **Follow-up**: 大規模データセット(サンプル数>10,000)の場合のバッチ処理検討(要件8.1)

### Decision: YAML Configuration File Support
- **Context**: 要件7.1で「コマンドライン引数またはYAML設定ファイル」が指定されている
- **Alternatives Considered**:
  1. コマンドライン引数のみ — シンプルだが長いコマンドになる
  2. YAML設定のみ — 柔軟性が低い
  3. 両方サポート(優先度: CLI > YAML > デフォルト) — 最も柔軟
- **Selected Approach**: 両方サポート、優先度付き
  - `argparse`でCLI引数をパース
  - `--config`オプションでYAMLファイルを読み込み
  - 優先度: CLI引数 > YAML設定 > デフォルト値
- **Rationale**:
  - 既存のプロジェクトパターン(Hydra/OmegaConf使用)と一貫性
  - 実験の再現性向上(YAML設定をバージョン管理)
  - CLI引数でクイック実験が可能
- **Trade-offs**:
  - Benefits: 柔軟性、再現性、既存パターンとの一貫性
  - Compromises: 設定マージロジックの実装が必要
- **Follow-up**: 設定スキーマのドキュメント化、YAML例の提供

### Decision: Output File Naming Convention
- **Context**: 要件6.6「ファイル名に可視化タイプ、モデル名、タイムスタンプを含める」
- **Selected Approach**:
  - フォーマット: `{vis_type}_{model_name}_{timestamp}.{ext}`
  - 例: `class_separation_student_20250101_120000.png`
  - `vis_type`: `class_separation`, `domain_comparison`, `mixstyle_effect`
  - `model_name`: checkpointファイル名から抽出またはユーザー指定
  - `timestamp`: `datetime.now().strftime("%Y%m%d_%H%M%S")`
- **Rationale**:
  - 一意性の保証(タイムスタンプ)
  - 自己文書化(ファイル名から内容が推測可能)
  - ソート容易(タイムスタンプベース)
- **Trade-offs**:
  - Benefits: 衝突回避、可読性
  - Compromises: ファイル名がやや長くなる
- **Follow-up**: ユーザー指定プレフィックスのサポート(オプション)

## Risks & Mitigations
- **Risk 1**: 大規模データセット(サンプル数>10,000)でUMAPのメモリ不足エラーが発生
  - Mitigation: 要件8.1に従い、サンプル数チェック + バッチ処理またはダウンサンプリングの提案
  - Implementation: サンプル数>10,000の場合に警告を出し、`--max_samples`オプションでランダムサンプリング
- **Risk 2**: マルチラベルのargmax処理で情報損失(複数クラスが同時に存在する場合)
  - Mitigation: 要件5.5に従い、選択されたクラスラベルと元のマルチラベルの対応関係をログ出力
  - Implementation: INFO levelログで「Sample X: Selected class Y, original labels [Y, Z]」を出力
- **Risk 3**: 21クラスのカラーパレットが視覚的に区別困難
  - Mitigation: デフォルトでSeaborn colorblindパレット(21色拡張)を使用、凡例の最適配置(要件6.8)
  - Implementation: `plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')`で図外配置
- **Risk 4**: 異なるモデルcheckpointの互換性(要件4)
  - Mitigation: 特徴量の次元数検証(要件8.2)、不一致時に警告表示
  - Implementation: `assert features1.shape[1] == features2.shape[1]`でチェック

## References
- [UMAP-learn Documentation 0.5.8](https://umap-learn.readthedocs.io/) — 公式API仕様
- [Seaborn Color Palettes](https://seaborn.pydata.org/tutorial/color_palettes.html) — 色覚多様性対応パレット
- [Matplotlib DPI Settings](https://matplotlib.org/stable/gallery/misc/print_stdout_sgskip.html) — 論文掲載用の解像度設定
- プロジェクト内部: `.kiro/steering/structure.md`, `.kiro/steering/tech.md` — コーディング規約とアーキテクチャパターン
