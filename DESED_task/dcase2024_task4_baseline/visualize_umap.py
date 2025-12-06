#!/usr/bin/env python3
"""
UMAP可視化スクリプト

推論結果から特徴量を読み込み、UMAPで2D/3D投影して可視化する。
データセット別、ドメイン別、クラス別、モデル別の比較が可能。

使用法:
    python visualize_umap.py \
        --input_dirs inference_outputs/baseline inference_outputs/cmt_normal \
        --output_dir visualization_outputs/umap
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import umap
from tqdm import tqdm

# 共通ユーティリティをインポート
from visualization_utils import (
    DESED_CLASSES, MAESTRO_REAL_ALL, ALL_CLASSES_27,
    USED_CLASS_INDICES, USED_CLASSES_21,
    load_inference_data,
    create_metadata,
    combine_multiple_models,
    filter_by_domain,
    filter_by_class,
    sample_balanced,
    compute_dataset_statistics,
    print_statistics
)


@dataclass
class UMAPConfig:
    """UMAP設定パラメータ"""
    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = 'euclidean'
    random_state: int = 42
    verbose: bool = True

    n_epochs: Optional[int] = None
    learning_rate: float = 1.0
    init: str = 'spectral'

    def to_dict(self) -> Dict:
        return {
            'n_components': self.n_components,
            'n_neighbors': self.n_neighbors,
            'min_dist': self.min_dist,
            'metric': self.metric,
            'random_state': self.random_state,
            'n_epochs': self.n_epochs,
            'learning_rate': self.learning_rate,
            'init': self.init,
            'verbose': self.verbose
        }


@dataclass
class PlotConfig:
    """プロット設定パラメータ"""
    figsize: Tuple[int, int] = (12, 10)
    dpi: int = 300
    alpha: float = 0.6
    point_size: int = 20
    color_palette: str = 'tab20'
    interactive_width: int = 1200
    interactive_height: int = 900
    show_legend: bool = True
    legend_position: str = 'upper right'


class UMAPVisualizer:
    """
    UMAP次元削減と可視化のメインクラス

    Attributes:
        umap_config: UMAP設定
        plot_config: プロット設定
        reducer: UMAP reducer instance
        embedding: 次元削減後の埋め込み
        metadata: データのメタデータ
    """

    def __init__(
        self,
        umap_config: Optional[UMAPConfig] = None,
        plot_config: Optional[PlotConfig] = None
    ):
        """
        Args:
            umap_config: UMAP設定（Noneの場合はデフォルト値を使用）
            plot_config: プロット設定（Noneの場合はデフォルト値を使用）
        """
        self.umap_config = umap_config or UMAPConfig()
        self.plot_config = plot_config or PlotConfig()
        self.reducer = None
        self.embedding = None
        self.metadata = None
        self._cache = {}

    def fit_transform(
        self,
        features: np.ndarray,
        cache_key: Optional[str] = None
    ) -> np.ndarray:
        """
        UMAP次元削減を適用

        Args:
            features: 入力特徴量 (N, D)
            cache_key: キャッシュキー（同じキーの結果を再利用）

        Returns:
            次元削減後の埋め込み (N, n_components)

        Raises:
            ValueError: 入力特徴量が不正な場合
        """
        # 入力検証
        if features.ndim != 2:
            raise ValueError(f"Features must be 2D array, got shape {features.shape}")

        if features.shape[0] < self.umap_config.n_neighbors:
            warnings.warn(
                f"サンプル数 ({features.shape[0]}) が n_neighbors ({self.umap_config.n_neighbors}) "
                f"より少ないため、n_neighbors を {features.shape[0] - 1} に調整します",
                UserWarning
            )
            # 一時的にn_neighborsを調整
            original_n_neighbors = self.umap_config.n_neighbors
            self.umap_config.n_neighbors = min(features.shape[0] - 1, 15)

        # キャッシュチェック
        if cache_key and cache_key in self._cache:
            print(f"キャッシュから埋め込みを読み込み: {cache_key}")
            self.embedding = self._cache[cache_key]
            return self.embedding

        print(f"UMAPを適用中... (入力次元: {features.shape[1]}, 出力次元: {self.umap_config.n_components})")

        # パラメータの準備
        umap_params = self.umap_config.to_dict()
        # Noneのパラメータは除外
        umap_params = {k: v for k, v in umap_params.items() if v is not None}

        self.reducer = umap.UMAP(**umap_params)
        self.embedding = self.reducer.fit_transform(features)

        print(f"UMAP完了: {self.embedding.shape}")

        # キャッシュに保存
        if cache_key:
            self._cache[cache_key] = self.embedding

        # n_neighborsを元に戻す
        if 'original_n_neighbors' in locals():
            self.umap_config.n_neighbors = original_n_neighbors

        return self.embedding

    def plot_static(
        self,
        embedding: np.ndarray,
        metadata: pd.DataFrame,
        color_by: str,
        output_path: Union[str, Path],
        title: Optional[str] = None,
        subset_mask: Optional[np.ndarray] = None
    ) -> None:
        """
        静的な散布図を作成（Matplotlib）

        Args:
            embedding: UMAP埋め込み (N, 2 or 3)
            metadata: メタデータDataFrame
            color_by: 色分けに使用するカラム名
            output_path: 出力ファイルパス
            title: プロットタイトル
            subset_mask: データのサブセット用マスク
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # サブセットの適用
        if subset_mask is not None:
            embedding = embedding[subset_mask]
            metadata = metadata[subset_mask].reset_index(drop=True)

        fig, ax = plt.subplots(figsize=self.plot_config.figsize)

        # 色分けの準備
        unique_values = metadata[color_by].unique()
        n_colors = len(unique_values)

        # カラーパレットの選択（クラス数に応じて）
        if n_colors <= 10:
            colors = sns.color_palette('tab10', n_colors=n_colors)
        elif n_colors <= 20:
            colors = sns.color_palette('tab20', n_colors=n_colors)
        else:
            colors = sns.color_palette('hls', n_colors=n_colors)

        color_map = {val: colors[i] for i, val in enumerate(unique_values)}

        # 各グループをプロット
        for value in unique_values:
            mask = metadata[color_by] == value
            if not mask.any():
                continue

            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[color_map[value]],
                label=value,
                alpha=self.plot_config.alpha,
                s=self.plot_config.point_size,
                edgecolors='none'
            )

        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14, pad=20)
        else:
            ax.set_title(f'UMAP Projection (colored by {color_by})', fontsize=14, pad=20)

        # グリッドの追加
        ax.grid(True, alpha=0.3, linestyle='--')

        # 凡例の配置
        if self.plot_config.show_legend:
            if n_colors > 15:
                # 多い場合は外側に複数列で配置
                ax.legend(
                    bbox_to_anchor=(1.05, 1),
                    loc='upper left',
                    ncol=2 if n_colors > 20 else 1,
                    fontsize=8,
                    frameon=True,
                    fancybox=True,
                    shadow=True
                )
            else:
                ax.legend(
                    loc=self.plot_config.legend_position,
                    fontsize=10,
                    frameon=True,
                    fancybox=True,
                    shadow=True
                )

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.plot_config.dpi, bbox_inches='tight')
        plt.close()

        print(f"  ✓ 保存: {output_path}")

    def plot_interactive(
        self,
        embedding: np.ndarray,
        metadata: pd.DataFrame,
        color_by: str,
        output_path: Union[str, Path],
        title: Optional[str] = None,
        hover_data: Optional[List[str]] = None
    ) -> None:
        """
        インタラクティブな散布図を作成（Plotly）

        Args:
            embedding: UMAP埋め込み (N, 2 or 3)
            metadata: メタデータDataFrame
            color_by: 色分けに使用するカラム名
            output_path: 出力ファイルパス
            title: プロットタイトル
            hover_data: ホバー時に表示するカラムのリスト
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # DataFrameに埋め込みを追加
        df = metadata.copy()

        if embedding.shape[1] == 2:
            df['UMAP_1'] = embedding[:, 0]
            df['UMAP_2'] = embedding[:, 1]
            z_col = None
        elif embedding.shape[1] == 3:
            df['UMAP_1'] = embedding[:, 0]
            df['UMAP_2'] = embedding[:, 1]
            df['UMAP_3'] = embedding[:, 2]
            z_col = 'UMAP_3'
        else:
            raise ValueError(f"Embedding must be 2D or 3D, got {embedding.shape[1]}D")

        # ホバーデータのデフォルト設定
        if hover_data is None:
            hover_data = ['model', 'dataset', 'domain']
            if 'predicted_class' in df.columns:
                hover_data.append('predicted_class')
            if 'true_class' in df.columns and not all(df['true_class'] == 'unlabeled'):
                hover_data.append('true_class')
            if 'max_prob' in df.columns:
                hover_data.append('max_prob')
            if 'entropy' in df.columns:
                hover_data.append('entropy')

        # 有効なカラムのみを使用
        hover_data = [col for col in hover_data if col in df.columns]

        # 2D or 3Dプロットの作成
        if z_col is None:
            # 2Dプロット
            fig = px.scatter(
                df,
                x='UMAP_1',
                y='UMAP_2',
                color=color_by,
                hover_data=hover_data,
                title=title or f'UMAP Projection (colored by {color_by})',
                width=self.plot_config.interactive_width,
                height=self.plot_config.interactive_height,
                opacity=0.7
            )
        else:
            # 3Dプロット
            fig = px.scatter_3d(
                df,
                x='UMAP_1',
                y='UMAP_2',
                z='UMAP_3',
                color=color_by,
                hover_data=hover_data,
                title=title or f'UMAP 3D Projection (colored by {color_by})',
                width=self.plot_config.interactive_width,
                height=self.plot_config.interactive_height,
                opacity=0.7
            )

        # レイアウトの調整
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(
            font=dict(size=12),
            legend=dict(
                font=dict(size=10),
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            hovermode='closest'
        )

        # HTMLファイルとして保存
        fig.write_html(output_path)
        print(f"  ✓ 保存: {output_path}")

    def plot_comparison(
        self,
        embedding: np.ndarray,
        metadata: pd.DataFrame,
        output_dir: Path,
        feature_type: str
    ) -> None:
        """
        複数の観点から比較プロットを生成

        Args:
            embedding: UMAP埋め込み
            metadata: メタデータ
            output_dir: 出力ディレクトリ
            feature_type: 特徴量タイプ ('student' or 'teacher')
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n可視化を生成中...")

        # 1. データセット別
        self.plot_static(
            embedding, metadata, 'dataset',
            output_dir / 'dataset_comparison.png',
            f'UMAP: Dataset Comparison ({feature_type})'
        )

        # 2. ドメイン別
        self.plot_static(
            embedding, metadata, 'domain',
            output_dir / 'domain_comparison.png',
            f'UMAP: Domain Comparison ({feature_type})'
        )

        # 3. モデル別（複数モデルの場合のみ）
        if metadata['model'].nunique() > 1:
            self.plot_static(
                embedding, metadata, 'model',
                output_dir / 'model_comparison.png',
                f'UMAP: Model Comparison ({feature_type})'
            )

        # 4. クラス別（ラベル付きデータのみ）
        if 'predicted_class' in metadata.columns:
            labeled_mask = metadata['predicted_class'] != 'unlabeled'
            if labeled_mask.sum() > 0:
                # 予測クラス分布
                self.plot_static(
                    embedding, metadata, 'predicted_class',
                    output_dir / 'class_distribution_predicted.png',
                    f'UMAP: Predicted Class Distribution ({feature_type})',
                    subset_mask=labeled_mask
                )

                # 真のクラス分布（存在する場合）
                if 'true_class' in metadata.columns:
                    true_labeled_mask = (metadata['true_class'] != 'unlabeled') & (metadata['true_class'] != 'none')
                    if true_labeled_mask.sum() > 0:
                        self.plot_static(
                            embedding, metadata, 'true_class',
                            output_dir / 'class_distribution_true.png',
                            f'UMAP: True Class Distribution ({feature_type})',
                            subset_mask=true_labeled_mask
                        )

        # 5. インタラクティブ版
        self.plot_interactive(
            embedding, metadata, 'dataset',
            output_dir / 'interactive_umap.html',
            f'UMAP: Interactive Visualization ({feature_type})'
        )

    def save_results(
        self,
        embedding: np.ndarray,
        metadata: pd.DataFrame,
        output_dir: Path,
        save_config: bool = True
    ) -> None:
        """
        埋め込みとメタデータを保存

        Args:
            embedding: UMAP埋め込み
            metadata: メタデータ
            output_dir: 出力ディレクトリ
            save_config: 設定も保存するかどうか
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 埋め込みの保存
        np.save(output_dir / 'umap_embedding.npy', embedding)
        print(f"  ✓ 埋め込み保存: {output_dir / 'umap_embedding.npy'}")

        # メタデータの保存
        metadata.to_csv(output_dir / 'metadata.csv', index=False)
        print(f"  ✓ メタデータ保存: {output_dir / 'metadata.csv'}")

        # 設定の保存
        if save_config:
            config = {
                'umap': self.umap_config.to_dict(),
                'plot': {
                    'figsize': list(self.plot_config.figsize),
                    'dpi': self.plot_config.dpi,
                    'alpha': self.plot_config.alpha,
                    'point_size': self.plot_config.point_size,
                    'color_palette': self.plot_config.color_palette
                }
            }
            with open(output_dir / 'config.json', 'w') as f:
                json.dump(config, f, indent=2)
            print(f"  ✓ 設定保存: {output_dir / 'config.json'}")

    def load_results(
        self,
        input_dir: Path
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        保存された結果を読み込む

        Args:
            input_dir: 入力ディレクトリ

        Returns:
            embedding: UMAP埋め込み
            metadata: メタデータ

        Raises:
            FileNotFoundError: 必要なファイルが見つからない場合
        """
        input_dir = Path(input_dir)

        # 埋め込みの読み込み
        embedding_path = input_dir / 'umap_embedding.npy'
        if not embedding_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        embedding = np.load(embedding_path)

        # メタデータの読み込み
        metadata_path = input_dir / 'metadata.csv'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        metadata = pd.read_csv(metadata_path)

        # 設定の読み込み（存在する場合）
        config_path = input_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"  ✓ 設定を読み込みました: {config_path}")

        return embedding, metadata


# --- メイン処理 ---
def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(
        description="UMAP可視化スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # 基本的な使用法
  python visualize_umap.py --input_dirs inference_outputs/baseline --output_dir visualization_outputs/umap

  # 複数モデルの比較
  python visualize_umap.py --input_dirs inference_outputs/baseline inference_outputs/cmt_normal --output_dir visualization_outputs/comparison

  # 3D UMAP with custom parameters
  python visualize_umap.py --input_dirs inference_outputs/baseline --output_dir visualization_outputs/umap_3d --n_components 3 --n_neighbors 30
        """
    )

    # 必須引数
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="推論結果ディレクトリ（複数指定可）"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="出力ディレクトリ"
    )

    # 特徴量オプション
    parser.add_argument(
        "--feature_type",
        choices=['student', 'teacher', 'both'],
        default='student',
        help="使用する特徴量タイプ (default: student)"
    )

    # UMAPパラメータ
    parser.add_argument(
        "--n_components",
        type=int,
        default=2,
        help="UMAP出力次元数 (default: 2)"
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors パラメータ (default: 15)"
    )
    parser.add_argument(
        "--min_dist",
        type=float,
        default=0.1,
        help="UMAP min_dist パラメータ (default: 0.1)"
    )
    parser.add_argument(
        "--metric",
        default='euclidean',
        help="UMAP距離メトリック (default: euclidean)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="乱数シード (default: 42)"
    )

    # プロットオプション
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="プロット生成をスキップ（埋め込みのみ計算）"
    )
    parser.add_argument(
        "--interactive_only",
        action="store_true",
        help="インタラクティブプロットのみ生成"
    )

    # サンプリングオプション
    parser.add_argument(
        "--max_samples_per_class",
        type=int,
        default=None,
        help="クラスごとの最大サンプル数（バランシング用）"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=['synthetic', 'real', 'unlabeled'],
        default=None,
        help="使用するドメイン（指定しない場合は全て）"
    )

    # その他
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細出力を有効化"
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="UMAP埋め込みのキャッシュを有効化"
    )

    return parser.parse_args()


def main():
    """メイン処理"""
    args = parse_arguments()

    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("UMAP可視化スクリプト")
    print("="*60)

    # UMAP設定の作成
    umap_config = UMAPConfig(
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
        verbose=args.verbose
    )

    # プロット設定
    plot_config = PlotConfig()

    # ビジュアライザーの初期化
    visualizer = UMAPVisualizer(umap_config, plot_config)

    # データ読み込み
    print("\n[1/4] データ読み込み中...")
    models_data = {}
    for input_dir in args.input_dirs:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"  警告: {input_path} が見つかりません。スキップします。")
            continue
        model_name = input_path.name
        print(f"  - {model_name}: {input_path}")
        models_data[model_name] = load_inference_data(input_path, verbose=args.verbose)

    if not models_data:
        print("エラー: 有効な入力データが見つかりませんでした。")
        return

    # 特徴量タイプの決定
    feature_types = ['student', 'teacher'] if args.feature_type == 'both' else [args.feature_type]

    for feature_type in feature_types:
        print(f"\n{'='*60}")
        print(f"特徴量タイプ: {feature_type}")
        print(f"{'='*60}")

        # データ準備
        print("\n[2/4] データ準備中...")
        features, metadata = combine_multiple_models(models_data, feature_type)

        # ドメインフィルタリング
        if args.domains:
            print(f"  ドメインフィルタリング: {args.domains}")
            features, metadata = filter_by_domain(features, metadata, args.domains)

        # クラスバランシング
        if args.max_samples_per_class:
            print(f"  クラスバランシング: 最大 {args.max_samples_per_class} サンプル/クラス")
            features, metadata = sample_balanced(
                features, metadata,
                args.max_samples_per_class,
                column='predicted_class',
                random_state=args.random_state
            )

        # 統計情報の表示
        stats = compute_dataset_statistics(metadata)
        print_statistics(stats, f"Dataset Statistics ({feature_type})")

        # UMAP適用
        print("\n[3/4] UMAP適用中...")
        cache_key = f"{feature_type}_{args.n_components}d" if args.cache else None
        embedding = visualizer.fit_transform(features, cache_key)

        # サブディレクトリ作成
        sub_dir = output_dir / feature_type
        sub_dir.mkdir(exist_ok=True)

        # 結果の保存
        print("\n[4/4] 結果の保存中...")
        visualizer.save_results(embedding, metadata, sub_dir, save_config=True)

        # プロット生成
        if not args.no_plots:
            if args.interactive_only:
                # インタラクティブプロットのみ
                visualizer.plot_interactive(
                    embedding, metadata, 'dataset',
                    sub_dir / 'interactive_umap.html',
                    f'UMAP: Interactive Visualization ({feature_type})'
                )
            else:
                # 全プロット生成
                visualizer.plot_comparison(embedding, metadata, sub_dir, feature_type)

    print("\n" + "="*60)
    print("完了！")
    print(f"出力ディレクトリ: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()