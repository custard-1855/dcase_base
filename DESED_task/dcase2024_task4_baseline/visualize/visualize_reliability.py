#!/usr/bin/env python3
"""Reliability Diagram (信頼性ダイアグラム) 可視化スクリプト

モデルの予測確率と実際の正答率を比較し、キャリブレーション（較正）の品質を評価する。
ECE (Expected Calibration Error) も計算する。

使用法:
    python visualize_reliability.py \
        --input_dirs inference_outputs/baseline inference_outputs/cmt_normal \
        --output_dir visualization_outputs/reliability
"""

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# 共通ユーティリティをインポート
from visualization_utils import (
    ALL_CLASSES_27,
    DESED_CLASSES,
    MAESTRO_REAL_ALL,
    USED_CLASS_INDICES,
    USED_CLASSES_21,
    combine_multiple_models,
    compute_dataset_statistics,
    create_metadata,
    load_inference_data,
    print_statistics,
)


@dataclass
class ReliabilityConfig:
    """信頼性ダイアグラム設定パラメータ"""

    n_bins: int = 10
    threshold: float = 0.5
    min_samples_per_class: int = 10
    top_k_classes: int = 10

    # プロット設定
    figsize_per_plot: tuple[int, int] = (5, 5)
    dpi: int = 300
    alpha: float = 0.7
    show_distribution: bool = True

    def to_dict(self) -> dict:
        """辞書形式に変換"""
        return {
            "n_bins": self.n_bins,
            "threshold": self.threshold,
            "min_samples_per_class": self.min_samples_per_class,
            "top_k_classes": self.top_k_classes,
        }


class CalibrationMetrics:
    """キャリブレーションメトリクス計算クラス"""

    def __init__(self, config: ReliabilityConfig | None = None):
        """Args:
        config: 信頼性設定（Noneの場合はデフォルト値を使用）

        """
        self.config = config or ReliabilityConfig()

    def compute_ece(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        n_bins: int | None = None,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Expected Calibration Error (ECE) を計算

        Args:
            probs: (N,) 予測確率
            targets: (N,) 正解ラベル (0 or 1)
            n_bins: ビン数（Noneの場合は設定値を使用）

        Returns:
            ece: ECE値
            bin_centers: ビンの中心値
            bin_accs: 各ビンの正答率
            bin_confs: 各ビンの平均予測確率
            bin_counts: 各ビンのサンプル数

        """
        n_bins = n_bins or self.config.n_bins

        # 入力検証
        if probs.ndim != 1 or targets.ndim != 1:
            raise ValueError("probs and targets must be 1D arrays")
        if len(probs) != len(targets):
            raise ValueError("probs and targets must have the same length")
        if len(probs) == 0:
            return 0.0, np.array([]), np.array([]), np.array([]), np.array([])

        # ビン境界の設定
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_accs = []
        bin_confs = []
        bin_counts = []

        ece = 0.0

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # このビンに属するサンプルを選択
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.sum() / len(probs)

            if prop_in_bin > 0:
                accuracy_in_bin = targets[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()

                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

                bin_accs.append(accuracy_in_bin)
                bin_confs.append(avg_confidence_in_bin)
                bin_counts.append(in_bin.sum())
            else:
                bin_accs.append(0)
                bin_confs.append(0)
                bin_counts.append(0)

        bin_centers = (bin_lowers + bin_uppers) / 2

        return ece, bin_centers, np.array(bin_accs), np.array(bin_confs), np.array(bin_counts)

    def compute_mce(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        n_bins: int | None = None,
    ) -> float:
        """Maximum Calibration Error (MCE) を計算

        Args:
            probs: (N,) 予測確率
            targets: (N,) 正解ラベル
            n_bins: ビン数

        Returns:
            MCE値

        """
        _, _, bin_accs, bin_confs, bin_counts = self.compute_ece(probs, targets, n_bins)

        if len(bin_accs) == 0:
            return 0.0

        # 空でないビンのみ考慮
        non_empty = bin_counts > 0
        if not non_empty.any():
            return 0.0

        calibration_errors = np.abs(bin_confs[non_empty] - bin_accs[non_empty])
        return calibration_errors.max()

    def prepare_binary_predictions(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """マルチラベル問題を複数の二値分類問題に変換

        Args:
            probs: (N, C) 予測確率
            targets: (N, C) 正解ラベル
            threshold: 二値化閾値

        Returns:
            all_probs: (N*C,) 全ての予測確率
            all_targets: (N*C,) 全ての正解ラベル

        """
        threshold = threshold or self.config.threshold

        # 全クラスの予測と正解をフラット化
        all_probs = probs.flatten()
        all_targets = (targets > threshold).astype(float).flatten()

        return all_probs, all_targets

    def compute_per_class_metrics(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
    ) -> dict[int, dict[str, float | np.ndarray]]:
        """クラス別にECEと信頼性データを計算

        Args:
            probs: (N, C) 予測確率
            targets: (N, C) 正解ラベル

        Returns:
            クラスインデックスをキーとする辞書

        """
        n_classes = probs.shape[1]
        results = {}

        for class_idx in range(n_classes):
            class_probs = probs[:, class_idx]
            class_targets = (targets[:, class_idx] > self.config.threshold).astype(float)

            # サンプル数が少ない場合はスキップ
            n_positive = class_targets.sum()
            n_samples = len(class_targets)

            if n_positive < self.config.min_samples_per_class:
                continue

            ece, bin_centers, bin_accs, bin_confs, bin_counts = self.compute_ece(
                class_probs,
                class_targets,
            )

            mce = self.compute_mce(class_probs, class_targets)

            results[class_idx] = {
                "ece": ece,
                "mce": mce,
                "bin_centers": bin_centers,
                "bin_accs": bin_accs,
                "bin_confs": bin_confs,
                "bin_counts": bin_counts,
                "n_samples": n_samples,
                "n_positive": int(n_positive),
            }

        return results


class ReliabilityVisualizer:
    """信頼性ダイアグラム可視化クラス"""

    def __init__(
        self,
        config: ReliabilityConfig | None = None,
        metrics: CalibrationMetrics | None = None,
    ):
        """Args:
        config: 信頼性設定
        metrics: キャリブレーションメトリクス計算インスタンス

        """
        self.config = config or ReliabilityConfig()
        self.metrics = metrics or CalibrationMetrics(self.config)

    def plot_reliability_diagram(
        self,
        results: dict[str, dict[str, dict[str, float | np.ndarray]]],
        output_path: str | Path,
        title: str = "Reliability Diagram",
    ) -> None:
        """Reliability Diagram を描画

        Args:
            results: {model_name: {dataset_name: metrics_dict}}
            output_path: 出力ファイルパス
            title: グラフタイトル

        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_models = len(results)
        n_datasets = len(next(iter(results.values())))

        # 図のサイズを計算
        fig_width = self.config.figsize_per_plot[0] * n_models
        fig_height = self.config.figsize_per_plot[1] * n_datasets

        fig, axes = plt.subplots(n_datasets, n_models, figsize=(fig_width, fig_height))

        # 軸配列の形状を調整
        if n_models == 1 and n_datasets == 1:
            axes = np.array([[axes]])
        elif n_models == 1:
            axes = axes.reshape(-1, 1)
        elif n_datasets == 1:
            axes = axes.reshape(1, -1)

        for j, (model_name, datasets) in enumerate(results.items()):
            for i, (dataset_name, metrics) in enumerate(datasets.items()):
                ax = axes[i, j]

                # メトリクスを取得
                if isinstance(metrics, tuple):
                    # 古い形式（後方互換性のため）
                    ece, bin_centers, bin_accs, bin_confs, bin_counts = metrics
                else:
                    # 新しい形式
                    ece = metrics["ece"]
                    bin_centers = metrics["bin_centers"]
                    bin_accs = metrics["bin_accs"]
                    bin_confs = metrics["bin_confs"]
                    bin_counts = metrics["bin_counts"]

                # Perfect calibration line
                ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=2)

                # Reliability curve
                # ビン内のサンプル数に応じてマーカーサイズを変更
                if bin_counts.max() > 0:
                    sizes = bin_counts / bin_counts.max() * 300
                else:
                    sizes = np.ones_like(bin_counts) * 50

                scatter = ax.scatter(
                    bin_confs,
                    bin_accs,
                    s=sizes,
                    alpha=self.config.alpha,
                    c=bin_centers,
                    cmap="coolwarm",
                    edgecolors="black",
                    linewidths=1.5,
                )

                # 信頼度分布のバー表示（オプション）
                if self.config.show_distribution and bin_counts.max() > 0:
                    ax.bar(
                        bin_centers,
                        bin_counts / bin_counts.max(),
                        width=0.08,
                        alpha=0.3,
                        color="gray",
                        label="Confidence Distribution",
                    )

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel("Confidence (Predicted Probability)", fontsize=11)
                ax.set_ylabel("Accuracy (Actual Correctness)", fontsize=11)
                ax.set_title(
                    f"{model_name} - {dataset_name}\nECE: {ece:.4f}", fontsize=12, fontweight="bold"
                )
                ax.grid(alpha=0.3)
                ax.legend(fontsize=9, loc="lower right")

        plt.suptitle(title, fontsize=16, fontweight="bold", y=1.00)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close()

        print(f"  ✓ 保存: {output_path}")

    def plot_confidence_histogram(
        self,
        models_data: dict[str, dict[str, dict[str, np.ndarray]]],
        output_path: str | Path,
        pred_type: str = "student",
    ) -> None:
        """予測確率の分布をヒストグラムで可視化

        Args:
            models_data: モデルデータ
            output_path: 出力パス
            pred_type: 予測タイプ

        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_models = len(models_data)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

        if n_models == 1:
            axes = [axes]

        for ax, (model_name, datasets) in zip(axes, models_data.items()):
            all_probs = []
            labels = []

            for dataset_name, data in datasets.items():
                if f"probs_{pred_type}" in data:
                    probs = data[f"probs_{pred_type}"]
                    all_probs.append(probs.flatten())
                    labels.append(dataset_name)

            # 各データセット別にヒストグラム
            for prob, label in zip(all_probs, labels):
                ax.hist(prob, bins=50, alpha=0.5, label=label, density=True)

            ax.set_xlabel("Confidence", fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            ax.set_title(
                f"{model_name} - Confidence Distribution ({pred_type})",
                fontsize=13,
                fontweight="bold",
            )
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close()

        print(f"  ✓ 保存: {output_path}")

    def plot_per_class_reliability(
        self,
        models_data: dict[str, dict[str, dict[str, np.ndarray]]],
        output_dir: str | Path,
        pred_type: str = "student",
        top_k: int | None = None,
    ) -> None:
        """クラス別のReliability Diagramを作成

        Args:
            models_data: モデルデータ
            output_dir: 出力ディレクトリ
            pred_type: 予測タイプ
            top_k: 上位K個のクラスのみ表示

        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        top_k = top_k or self.config.top_k_classes

        # 全モデル・全データセットのクラス別ECEを計算
        class_ece_summary = {i: [] for i in range(len(ALL_CLASSES_27))}

        for model_name, datasets in models_data.items():
            for dataset_name, data in datasets.items():
                if f"probs_{pred_type}" in data and "targets" in data:
                    probs = data[f"probs_{pred_type}"]
                    targets = data["targets"]

                    per_class = self.metrics.compute_per_class_metrics(probs, targets)

                    for class_idx, metrics in per_class.items():
                        class_ece_summary[class_idx].append(metrics["ece"])

        # 平均ECEが高い上位K個のクラスを選択
        avg_eces = {idx: np.mean(eces) if eces else 0 for idx, eces in class_ece_summary.items()}
        top_classes = sorted(avg_eces.items(), key=lambda x: x[1], reverse=True)[:top_k]

        print(f"\n  クラス別ECE上位{top_k}個:")
        for class_idx, avg_ece in top_classes:
            if avg_ece > 0:  # ECEが計算されたクラスのみ表示
                class_name = (
                    ALL_CLASSES_27[class_idx]
                    if class_idx < len(ALL_CLASSES_27)
                    else f"class_{class_idx}"
                )
                print(f"    {class_name}: {avg_ece:.4f}")

        # 各クラスのReliability Diagramを作成
        for class_idx, _ in tqdm(
            top_classes[: min(top_k, len([c for c, e in top_classes if e > 0]))],
            desc="  クラス別Reliability Diagram作成中",
        ):
            class_name = (
                ALL_CLASSES_27[class_idx]
                if class_idx < len(ALL_CLASSES_27)
                else f"class_{class_idx}"
            )

            results = {}
            for model_name, datasets in models_data.items():
                results[model_name] = {}
                for dataset_name, data in datasets.items():
                    if f"probs_{pred_type}" in data and "targets" in data:
                        probs = data[f"probs_{pred_type}"][:, class_idx]
                        targets = (data["targets"][:, class_idx] > self.config.threshold).astype(
                            float
                        )

                        ece, bin_centers, bin_accs, bin_confs, bin_counts = (
                            self.metrics.compute_ece(
                                probs,
                                targets,
                            )
                        )

                        results[model_name][dataset_name] = {
                            "ece": ece,
                            "bin_centers": bin_centers,
                            "bin_accs": bin_accs,
                            "bin_confs": bin_confs,
                            "bin_counts": bin_counts,
                        }

            # プロット
            output_path = output_dir / f"{class_name.replace(' ', '_').replace('/', '_')}.png"
            self.plot_reliability_diagram(
                results,
                output_path,
                title=f"Reliability Diagram: {class_name}",
            )

    def save_results(
        self,
        ece_records: list[dict],
        output_dir: Path,
        pred_type: str,
    ) -> None:
        """結果をCSVファイルとして保存

        Args:
            ece_records: ECE記録のリスト
            output_dir: 出力ディレクトリ
            pred_type: 予測タイプ

        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # ECE比較表の保存
        if ece_records:
            ece_df = pd.DataFrame(ece_records)
            ece_df.to_csv(output_dir / "ece_comparison.csv", index=False)
            print(f"  ✓ ECE比較表保存: {output_dir / 'ece_comparison.csv'}")

            # モデル間比較の要約（複数モデルの場合）
            if "model" in ece_df.columns and ece_df["model"].nunique() > 1:
                pivot_df = ece_df.pivot(index="dataset", columns="model", values="ece")
                pivot_df.to_csv(output_dir / "ece_model_comparison.csv")
                print(f"  ✓ モデル間ECE比較表: {output_dir / 'ece_model_comparison.csv'}")


# メイン処理
def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(
        description="Reliability Diagram可視化スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # 基本的な使用法
  python visualize_reliability.py --input_dirs inference_outputs/baseline --output_dir visualization_outputs/reliability

  # 複数モデルの比較
  python visualize_reliability.py --input_dirs inference_outputs/baseline inference_outputs/cmt_normal --output_dir visualization_outputs/comparison

  # 詳細設定付き
  python visualize_reliability.py --input_dirs inference_outputs/baseline --output_dir visualization_outputs/reliability --n_bins 15 --top_k_classes 5
        """,
    )

    # 必須引数
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="推論結果ディレクトリ（複数指定可）",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="出力ディレクトリ",
    )

    # オプション引数
    parser.add_argument(
        "--pred_type",
        choices=["student", "teacher", "both"],
        default="student",
        help="使用する予測タイプ (default: student)",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=10,
        help="ビン数 (default: 10)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="二値化閾値 (default: 0.5)",
    )
    parser.add_argument(
        "--top_k_classes",
        type=int,
        default=10,
        help="クラス別分析で表示する上位K個 (default: 10)",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=10,
        help="クラス別分析の最小サンプル数 (default: 10)",
    )
    parser.add_argument(
        "--no_distribution",
        action="store_true",
        help="信頼度分布のバーを表示しない",
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
    print("Reliability Diagram 可視化スクリプト")
    print("=" * 60)

    # 設定の作成
    config = ReliabilityConfig(
        n_bins=args.n_bins,
        threshold=args.threshold,
        min_samples_per_class=args.min_samples,
        top_k_classes=args.top_k_classes,
        show_distribution=not args.no_distribution,
    )

    # ビジュアライザーとメトリクスの初期化
    metrics = CalibrationMetrics(config)
    visualizer = ReliabilityVisualizer(config, metrics)

    # データ読み込み
    print("\n[1/5] データ読み込み中...")
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

    # 予測タイプの決定
    pred_types = ["student", "teacher"] if args.pred_type == "both" else [args.pred_type]

    for pred_type in pred_types:
        print(f"\n{'=' * 60}")
        print(f"予測タイプ: {pred_type}")
        print(f"{'=' * 60}")

        # サブディレクトリ
        sub_dir = output_dir / pred_type
        sub_dir.mkdir(exist_ok=True)

        # 全体のReliability Diagram
        print("\n[2/5] 全体のReliability Diagram作成中...")
        overall_results = {}
        ece_records = []

        for model_name, datasets in models_data.items():
            overall_results[model_name] = {}
            for dataset_name, data in datasets.items():
                if f"probs_{pred_type}" in data and "targets" in data:
                    probs = data[f"probs_{pred_type}"]
                    targets = data["targets"]

                    # マルチラベルを二値分類に変換
                    all_probs, all_targets = metrics.prepare_binary_predictions(probs, targets)

                    # ECE計算
                    ece, bin_centers, bin_accs, bin_confs, bin_counts = metrics.compute_ece(
                        all_probs,
                        all_targets,
                    )

                    # MCE計算
                    mce = metrics.compute_mce(all_probs, all_targets)

                    overall_results[model_name][dataset_name] = {
                        "ece": ece,
                        "mce": mce,
                        "bin_centers": bin_centers,
                        "bin_accs": bin_accs,
                        "bin_confs": bin_confs,
                        "bin_counts": bin_counts,
                    }

                    ece_records.append(
                        {
                            "model": model_name,
                            "dataset": dataset_name,
                            "pred_type": pred_type,
                            "ece": ece,
                            "mce": mce,
                        }
                    )

                    if args.verbose:
                        print(f"  {model_name} - {dataset_name}: ECE = {ece:.4f}, MCE = {mce:.4f}")

        # プロット
        if overall_results:
            visualizer.plot_reliability_diagram(
                overall_results,
                sub_dir / "reliability_diagram_by_dataset.png",
                title=f"Reliability Diagram by Dataset ({pred_type})",
            )

        # Confidence Histogram
        print("\n[3/5] Confidence Histogram作成中...")
        visualizer.plot_confidence_histogram(
            models_data,
            sub_dir / "confidence_histogram.png",
            pred_type,
        )

        # クラス別Reliability Diagram
        print(f"\n[4/5] クラス別Reliability Diagram作成中（上位{args.top_k_classes}個）...")
        per_class_dir = sub_dir / "per_class_reliability"
        visualizer.plot_per_class_reliability(
            models_data,
            per_class_dir,
            pred_type,
            args.top_k_classes,
        )

        # 結果の保存
        print("\n[5/5] 結果の保存中...")
        visualizer.save_results(ece_records, sub_dir, pred_type)

        # 設定の保存
        with open(sub_dir / "config.json", "w") as f:
            json.dump(config.to_dict(), f, indent=2)
        print(f"  ✓ 設定保存: {sub_dir / 'config.json'}")

    print("\n" + "=" * 60)
    print("完了！")
    print(f"出力ディレクトリ: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
