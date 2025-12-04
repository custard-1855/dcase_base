#!/usr/bin/env python3
"""
Reliability Diagram (信頼性ダイアグラム) 可視化スクリプト

モデルの予測確率と実際の正答率を比較し、キャリブレーション（較正）の品質を評価する。
ECE (Expected Calibration Error) も計算する。

使用法:
    python visualize_reliability.py \
        --input_dirs inference_outputs/baseline inference_outputs/cmt_normal \
        --output_dir visualization_outputs/reliability
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


# クラス定義
# モデル出力は27次元（DESED 10 + MAESTRO Real 17）だが、
# 固定データセットでは21クラスのみ使用

# DESED 10クラス（インデックス 0-9）
DESED_CLASSES = [
    "Alarm_bell_ringing", "Blender", "Cat", "Dishes", "Dog",
    "Electric_shaver_toothbrush", "Frying", "Running_water",
    "Speech", "Vacuum_cleaner"
]

# MAESTRO Real 17クラス（インデックス 10-26）
MAESTRO_REAL_ALL = [
    "cutlery and dishes",      # 10
    "furniture dragging",      # 11 (未使用)
    "people talking",          # 12
    "children voices",         # 13
    "coffee machine",          # 14 (未使用)
    "footsteps",               # 15
    "large_vehicle",           # 16
    "car",                     # 17
    "brakes_squeaking",        # 18
    "cash register beeping",   # 19 (未使用)
    "announcement",            # 20 (未使用)
    "shopping cart",           # 21 (未使用)
    "metro leaving",           # 22
    "metro approaching",       # 23
    "door opens/closes",       # 24 (未使用)
    "wind_blowing",            # 25
    "birds_singing",           # 26
]

# 全27クラス（モデル出力次元に対応）
ALL_CLASSES_27 = DESED_CLASSES + MAESTRO_REAL_ALL

# 固定データセットで使用している21クラス
USED_CLASS_INDICES = [
    # DESED 10クラス全て
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    # MAESTRO 11クラス
    10,  # cutlery and dishes
    12,  # people talking
    13,  # children voices
    15,  # footsteps
    16,  # large_vehicle
    17,  # car
    18,  # brakes_squeaking
    22,  # metro leaving
    23,  # metro approaching
    25,  # wind_blowing
    26,  # birds_singing
]

USED_CLASSES_21 = [ALL_CLASSES_27[i] for i in USED_CLASS_INDICES]


def load_inference_data(model_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """推論結果を読み込む"""
    datasets = {}
    npz_files = list(model_dir.glob("*.npz"))

    for npz_file in npz_files:
        dataset_name = npz_file.stem
        data = np.load(npz_file, allow_pickle=True)

        # ラベル付きデータのみ処理
        if 'targets' in data:
            datasets[dataset_name] = {
                'probs_student': data['probs_student'],
                'probs_teacher': data['probs_teacher'],
                'targets': data['targets'],
                'filenames': data['filenames'],
            }

    return datasets


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Expected Calibration Error (ECE) を計算

    Args:
        probs: (N,) 予測確率
        targets: (N,) 正解ラベル (0 or 1)
        n_bins: ビン数

    Returns:
        ece: ECE値
        bin_centers: ビンの中心値
        bin_accs: 各ビンの正答率
        bin_confs: 各ビンの平均予測確率
    """
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


def prepare_binary_predictions(probs: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    マルチラベル問題を複数の二値分類問題に変換

    Returns:
        all_probs: (N*C,) 全ての予測確率
        all_targets: (N*C,) 全ての正解ラベル
    """
    # 全クラスの予測と正解をフラット化
    all_probs = probs.flatten()
    all_targets = (targets > threshold).astype(float).flatten()

    return all_probs, all_targets


def plot_reliability_diagram(
    results: Dict[str, Dict[str, Tuple]],
    output_path: Path,
    title: str = "Reliability Diagram"
):
    """
    Reliability Diagram を描画

    Args:
        results: {model_name: {dataset_name: (ece, bin_centers, bin_accs, bin_confs, bin_counts)}}
    """
    n_models = len(results)
    n_datasets = len(next(iter(results.values())))

    fig, axes = plt.subplots(n_datasets, n_models, figsize=(5*n_models, 5*n_datasets))

    if n_models == 1:
        axes = axes.reshape(-1, 1)
    if n_datasets == 1:
        axes = axes.reshape(1, -1)

    for j, (model_name, datasets) in enumerate(results.items()):
        for i, (dataset_name, (ece, bin_centers, bin_accs, bin_confs, bin_counts)) in enumerate(datasets.items()):
            ax = axes[i, j]

            # Perfect calibration line
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)

            # Reliability curve
            # ビン内のサンプル数に応じてマーカーサイズを変更
            sizes = (bin_counts / bin_counts.max() * 300) if bin_counts.max() > 0 else np.ones_like(bin_counts) * 50
            scatter = ax.scatter(
                bin_confs, bin_accs,
                s=sizes,
                alpha=0.7,
                c=bin_centers,
                cmap='coolwarm',
                edgecolors='black',
                linewidths=1.5
            )

            # バーで各ビンの予測確率分布を表示
            ax.bar(
                bin_centers, bin_counts / bin_counts.max(),
                width=0.08, alpha=0.3, color='gray',
                label='Confidence Distribution'
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Confidence (Predicted Probability)', fontsize=11)
            ax.set_ylabel('Accuracy (Actual Correctness)', fontsize=11)
            ax.set_title(f'{model_name} - {dataset_name}\nECE: {ece:.4f}', fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"保存: {output_path}")


def plot_confidence_histogram(
    models_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    output_path: Path,
    pred_type: str = 'student'
):
    """予測確率の分布をヒストグラムで可視化"""
    n_models = len(models_data)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, datasets) in zip(axes, models_data.items()):
        all_probs = []
        labels = []

        for dataset_name, data in datasets.items():
            probs = data[f'probs_{pred_type}']
            all_probs.append(probs.flatten())
            labels.append(dataset_name)

        # 各データセット別にヒストグラム
        for prob, label in zip(all_probs, labels):
            ax.hist(prob, bins=50, alpha=0.5, label=label, density=True)

        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{model_name} - Confidence Distribution ({pred_type})', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"保存: {output_path}")


def compute_per_class_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10
) -> Dict[int, Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    クラス別にECEと信頼性データを計算

    Returns:
        {class_idx: (ece, bin_centers, bin_accs, bin_confs, bin_counts)}
    """
    n_classes = probs.shape[1]  # 27
    results = {}

    for class_idx in range(n_classes):
        class_probs = probs[:, class_idx]
        class_targets = (targets[:, class_idx] > 0.5).astype(float)

        # サンプル数が少ない場合はスキップ
        if class_targets.sum() < 10:
            continue

        ece, bin_centers, bin_accs, bin_confs, bin_counts = compute_ece(
            class_probs, class_targets, n_bins
        )

        results[class_idx] = (ece, bin_centers, bin_accs, bin_confs, bin_counts)

    return results


def plot_per_class_reliability(
    models_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    output_dir: Path,
    pred_type: str = 'student',
    top_k: int = 10
):
    """クラス別のReliability Diagramを作成（上位K個のクラス）"""
    output_dir.mkdir(exist_ok=True)

    # 全モデル・全データセットのクラス別ECEを計算
    class_ece_summary = {i: [] for i in range(len(ALL_CLASSES_27))}

    for model_name, datasets in models_data.items():
        for dataset_name, data in datasets.items():
            probs = data[f'probs_{pred_type}']
            targets = data['targets']

            per_class = compute_per_class_metrics(probs, targets)

            for class_idx, (ece, _, _, _, _) in per_class.items():
                class_ece_summary[class_idx].append(ece)

    # 平均ECEが高い（キャリブレーションが悪い）上位K個のクラスを選択
    avg_eces = {idx: np.mean(eces) if eces else 0 for idx, eces in class_ece_summary.items()}
    top_classes = sorted(avg_eces.items(), key=lambda x: x[1], reverse=True)[:top_k]

    print(f"\nクラス別ECE上位{top_k}個:")
    for class_idx, avg_ece in top_classes:
        class_name = ALL_CLASSES_27[class_idx] if class_idx < len(ALL_CLASSES_27) else f'class_{class_idx}'
        print(f"  {class_name}: {avg_ece:.4f}")

    # 各クラスのReliability Diagramを作成
    for class_idx, _ in tqdm(top_classes, desc="クラス別Reliability Diagram作成中"):
        class_name = ALL_CLASSES_27[class_idx] if class_idx < len(ALL_CLASSES_27) else f'class_{class_idx}'

        results = {}
        for model_name, datasets in models_data.items():
            results[model_name] = {}
            for dataset_name, data in datasets.items():
                probs = data[f'probs_{pred_type}'][:, class_idx]
                targets = (data['targets'][:, class_idx] > 0.5).astype(float)

                ece, bin_centers, bin_accs, bin_confs, bin_counts = compute_ece(probs, targets)
                results[model_name][dataset_name] = (ece, bin_centers, bin_accs, bin_confs, bin_counts)

        # プロット
        output_path = output_dir / f"{class_name.replace(' ', '_')}.png"
        plot_reliability_diagram(
            results,
            output_path,
            title=f"Reliability Diagram: {class_name}"
        )


def main():
    parser = argparse.ArgumentParser(description="Reliability Diagram可視化スクリプト")
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
    parser.add_argument(
        "--pred_type",
        choices=['student', 'teacher', 'both'],
        default='student',
        help="使用する予測"
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=10,
        help="ビン数"
    )
    parser.add_argument(
        "--top_k_classes",
        type=int,
        default=10,
        help="クラス別分析で表示する上位K個"
    )

    args = parser.parse_args()

    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Reliability Diagram 可視化スクリプト")
    print("="*60)

    # データ読み込み
    print("\n[1/5] データ読み込み中...")
    models_data = {}
    for input_dir in args.input_dirs:
        input_path = Path(input_dir)
        model_name = input_path.name
        print(f"  - {model_name}: {input_path}")
        models_data[model_name] = load_inference_data(input_path)

    # 予測タイプの決定
    pred_types = ['student', 'teacher'] if args.pred_type == 'both' else [args.pred_type]

    for pred_type in pred_types:
        print(f"\n{'='*60}")
        print(f"予測タイプ: {pred_type}")
        print(f"{'='*60}")

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
                probs = data[f'probs_{pred_type}']
                targets = data['targets']

                # マルチラベルを二値分類に変換
                all_probs, all_targets = prepare_binary_predictions(probs, targets)

                ece, bin_centers, bin_accs, bin_confs, bin_counts = compute_ece(
                    all_probs, all_targets, args.n_bins
                )

                overall_results[model_name][dataset_name] = (ece, bin_centers, bin_accs, bin_confs, bin_counts)

                ece_records.append({
                    'model': model_name,
                    'dataset': dataset_name,
                    'pred_type': pred_type,
                    'ece': ece
                })

                print(f"  {model_name} - {dataset_name}: ECE = {ece:.4f}")

        # プロット
        plot_reliability_diagram(
            overall_results,
            sub_dir / 'reliability_diagram_by_dataset.png',
            title=f'Reliability Diagram by Dataset ({pred_type})'
        )

        # ECE比較表の保存
        ece_df = pd.DataFrame(ece_records)
        ece_df.to_csv(sub_dir / 'ece_comparison.csv', index=False)
        print(f"\nECE比較表保存: {sub_dir / 'ece_comparison.csv'}")

        # Confidence Histogram
        print("\n[3/5] Confidence Histogram作成中...")
        plot_confidence_histogram(
            models_data,
            sub_dir / 'confidence_histogram.png',
            pred_type
        )

        # クラス別Reliability Diagram
        print(f"\n[4/5] クラス別Reliability Diagram作成中（上位{args.top_k_classes}個）...")
        per_class_dir = sub_dir / 'per_class_reliability'
        plot_per_class_reliability(
            models_data,
            per_class_dir,
            pred_type,
            args.top_k_classes
        )

        # モデル間比較の要約
        print("\n[5/5] モデル間比較の要約作成中...")
        if len(models_data) > 1:
            pivot_df = ece_df.pivot(index='dataset', columns='model', values='ece')
            pivot_df.to_csv(sub_dir / 'ece_model_comparison.csv')
            print(f"モデル間ECE比較表: {sub_dir / 'ece_model_comparison.csv'}")

    print("\n" + "="*60)
    print("完了！")
    print(f"出力ディレクトリ: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
