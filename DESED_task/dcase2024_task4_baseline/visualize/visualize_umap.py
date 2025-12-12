#!/usr/bin/env python3
"""UMAP可視化スクリプト

抽出された特徴量をUMAPで次元削減し、2D/3Dプロットで可視化します。

使用法:
    python visualize_umap.py \
        --input_dirs features/model1 features/model2 \
        --output_dir umap_visualizations \
        --n_components 2 \
        --feature_type student
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from umap import UMAP

# 共通ユーティリティをインポート
from visualization_utils import (
    ALL_CLASSES_27,
    combine_multiple_models,
    compute_dataset_statistics,
    filter_by_domain,
    load_inference_data,
    print_statistics,
    sample_balanced,
)


class UMAPVisualizer:
    """UMAP可視化クラス"""

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 42,
    ):
        """初期化

        Args:
            n_components: UMAP次元数 (2 or 3)
            n_neighbors: UMAPのn_neighbors
            min_dist: UMAPのmin_dist
            metric: 距離メトリック
            random_state: 乱数シード
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state

        self.umap_model = None
        self.embeddings = None

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """UMAPで次元削減

        Args:
            features: 特徴量 (N, D)

        Returns:
            embeddings: UMAP埋め込み (N, n_components)
        """
        print(f"\nUMAP次元削減中... (n_components={self.n_components})")
        print(f"  入力特徴量: {features.shape}")

        self.umap_model = UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
            verbose=True,
        )

        self.embeddings = self.umap_model.fit_transform(features)
        print(f"  UMAP埋め込み: {self.embeddings.shape}")

        return self.embeddings

    def plot_2d(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        output_path: Path,
        color_by: str = "model",
        title: str = "UMAP Visualization",
        figsize: Tuple[int, int] = (12, 10),
        alpha: float = 0.6,
        s: int = 20,
    ) -> None:
        """2Dプロット

        Args:
            embeddings: UMAP埋め込み (N, 2)
            metadata: メタデータDataFrame
            output_path: 出力パス
            color_by: 色分けに使用するカラム ('model', 'dataset', 'domain', 'predicted_class')
            title: タイトル
            figsize: 図のサイズ
            alpha: 透明度
            s: マーカーサイズ
        """
        if embeddings.shape[1] != 2:
            raise ValueError("2D plot requires embeddings with 2 components")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=figsize)

        # カラーマップ
        unique_labels = metadata[color_by].unique()
        n_labels = len(unique_labels)

        if n_labels <= 10:
            palette = sns.color_palette("tab10", n_labels)
        elif n_labels <= 20:
            palette = sns.color_palette("tab20", n_labels)
        else:
            palette = sns.color_palette("husl", n_labels)

        # プロット
        for i, label in enumerate(unique_labels):
            mask = metadata[color_by] == label
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=[palette[i]],
                label=label,
                alpha=alpha,
                s=s,
                edgecolors="none",
            )

        ax.set_xlabel("UMAP 1", fontsize=12)
        ax.set_ylabel("UMAP 2", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        # 凡例
        if n_labels <= 30:
            ax.legend(
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize=8,
                frameon=True,
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  ✓ 保存: {output_path}")

    def plot_3d(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        output_path: Path,
        color_by: str = "model",
        title: str = "UMAP Visualization (3D)",
        figsize: Tuple[int, int] = (14, 10),
        alpha: float = 0.6,
        s: int = 20,
    ) -> None:
        """3Dプロット

        Args:
            embeddings: UMAP埋め込み (N, 3)
            metadata: メタデータDataFrame
            output_path: 出力パス
            color_by: 色分けに使用するカラム
            title: タイトル
            figsize: 図のサイズ
            alpha: 透明度
            s: マーカーサイズ
        """
        if embeddings.shape[1] != 3:
            raise ValueError("3D plot requires embeddings with 3 components")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # カラーマップ
        unique_labels = metadata[color_by].unique()
        n_labels = len(unique_labels)

        if n_labels <= 10:
            palette = sns.color_palette("tab10", n_labels)
        elif n_labels <= 20:
            palette = sns.color_palette("tab20", n_labels)
        else:
            palette = sns.color_palette("husl", n_labels)

        # プロット
        for i, label in enumerate(unique_labels):
            mask = metadata[color_by] == label
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                embeddings[mask, 2],
                c=[palette[i]],
                label=label,
                alpha=alpha,
                s=s,
                edgecolors="none",
            )

        ax.set_xlabel("UMAP 1", fontsize=12)
        ax.set_ylabel("UMAP 2", fontsize=12)
        ax.set_zlabel("UMAP 3", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        # 凡例
        if n_labels <= 30:
            ax.legend(
                bbox_to_anchor=(1.15, 1),
                loc="upper left",
                fontsize=8,
                frameon=True,
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  ✓ 保存: {output_path}")

    def plot_multiple_views(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        output_dir: Path,
        prefix: str = "umap",
    ) -> None:
        """複数の視点でプロット

        Args:
            embeddings: UMAP埋め込み
            metadata: メタデータDataFrame
            output_dir: 出力ディレクトリ
            prefix: ファイル名のプレフィックス
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 色分け対象
        color_by_options = ["model", "dataset", "domain"]

        # predicted_classがある場合は追加
        if "predicted_class" in metadata.columns:
            color_by_options.append("predicted_class")

        # true_classがある場合は追加
        if "true_class" in metadata.columns and not all(metadata["true_class"] == "unlabeled"):
            color_by_options.append("true_class")

        for color_by in color_by_options:
            if color_by not in metadata.columns:
                continue

            print(f"\n色分け: {color_by}")

            if self.n_components == 2:
                output_path = output_dir / f"{prefix}_2d_{color_by}.png"
                self.plot_2d(
                    embeddings,
                    metadata,
                    output_path,
                    color_by=color_by,
                    title=f"UMAP 2D - Colored by {color_by}",
                )
            elif self.n_components == 3:
                output_path = output_dir / f"{prefix}_3d_{color_by}.png"
                self.plot_3d(
                    embeddings,
                    metadata,
                    output_path,
                    color_by=color_by,
                    title=f"UMAP 3D - Colored by {color_by}",
                )


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(
        description="UMAP可視化スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 必須引数
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="特徴量ディレクトリ（複数指定可）",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="出力ディレクトリ",
    )

    # オプション引数
    parser.add_argument(
        "--feature_type",
        choices=["student", "teacher"],
        default="student",
        help="使用する特徴量タイプ (default: student)",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=2,
        choices=[2, 3],
        help="UMAP次元数 (default: 2)",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=15,
        help="UMAPのn_neighbors (default: 15)",
    )
    parser.add_argument(
        "--min_dist",
        type=float,
        default=0.1,
        help="UMAPのmin_dist (default: 0.1)",
    )
    parser.add_argument(
        "--metric",
        default="euclidean",
        help="距離メトリック (default: euclidean)",
    )
    parser.add_argument(
        "--filter_domains",
        nargs="+",
        help="フィルタリングするドメイン (例: synthetic real)",
    )
    parser.add_argument(
        "--sample_per_class",
        type=int,
        help="クラスごとのサンプル数（バランシング用）",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="乱数シード (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細出力を有効化",
    )

    return parser.parse_args()


def main():
    """メイン処理"""
    args = parse_arguments()

    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UMAP可視化スクリプト")
    print("=" * 60)

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

    # 特徴量とメタデータの結合
    print("\n[2/4] 特徴量とメタデータを結合中...")
    features, metadata = combine_multiple_models(models_data, args.feature_type)
    print(f"  結合後: {features.shape}, メタデータ: {len(metadata)}")

    # フィルタリング
    if args.filter_domains:
        print(f"\n  ドメインフィルタリング: {args.filter_domains}")
        features, metadata = filter_by_domain(features, metadata, args.filter_domains)
        print(f"  フィルタ後: {features.shape}, メタデータ: {len(metadata)}")

    # バランシング
    if args.sample_per_class:
        print(f"\n  クラスバランシング: {args.sample_per_class} samples/class")
        features, metadata = sample_balanced(
            features,
            metadata,
            args.sample_per_class,
            random_state=args.random_state,
        )
        print(f"  バランシング後: {features.shape}, メタデータ: {len(metadata)}")

    # 統計情報
    stats = compute_dataset_statistics(metadata)
    print_statistics(stats, title="データセット統計")

    # UMAP可視化
    print("\n[3/4] UMAP可視化中...")
    visualizer = UMAPVisualizer(
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
    )

    # 次元削減
    embeddings = visualizer.fit_transform(features)

    # 複数視点でプロット
    visualizer.plot_multiple_views(embeddings, metadata, output_dir)

    # UMAP埋め込みとメタデータを保存
    print("\n[4/4] 結果保存中...")
    np.save(output_dir / "umap_embeddings.npy", embeddings)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    print(f"  ✓ UMAP埋め込み保存: {output_dir / 'umap_embeddings.npy'}")
    print(f"  ✓ メタデータ保存: {output_dir / 'metadata.csv'}")

    # 設定の保存
    config = {
        "n_components": args.n_components,
        "n_neighbors": args.n_neighbors,
        "min_dist": args.min_dist,
        "metric": args.metric,
        "feature_type": args.feature_type,
        "filter_domains": args.filter_domains,
        "sample_per_class": args.sample_per_class,
        "random_state": args.random_state,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ 設定保存: {output_dir / 'config.json'}")

    print("\n" + "=" * 60)
    print("完了！")
    print(f"出力ディレクトリ: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
