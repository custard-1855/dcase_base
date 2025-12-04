#!/usr/bin/env python3
"""
統合分析レポート生成スクリプト

複数モデルの推論結果から定量的な比較分析を行い、Markdownレポートを生成する。

使用法:
    python generate_analysis_report.py \
        --input_dirs inference_outputs/baseline inference_outputs/cmt_normal inference_outputs/cmt_neg \
        --output visualization_outputs/analysis_report.md
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, average_precision_score, roc_auc_score)


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
    """推論結果とメタデータを読み込む"""
    datasets = {}
    npz_files = list(model_dir.glob("*.npz"))

    for npz_file in npz_files:
        dataset_name = npz_file.stem
        data = np.load(npz_file, allow_pickle=True)

        datasets[dataset_name] = {
            'probs_student': data['probs_student'],
            'probs_teacher': data['probs_teacher'],
            'filenames': data['filenames'],
        }

        if 'targets' in data:
            datasets[dataset_name]['targets'] = data['targets']

    # メタデータの読み込み
    metadata_path = model_dir / 'inference_metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return datasets, metadata


def compute_metrics(probs: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    マルチラベル分類の各種メトリクスを計算

    Args:
        probs: (N, C) 予測確率
        targets: (N, C) 正解ラベル

    Returns:
        各種メトリクスの辞書
    """
    preds = (probs > threshold).astype(int)
    targets_binary = (targets > threshold).astype(int)

    # サンプル単位のメトリクス（micro-average）
    metrics = {
        'accuracy': accuracy_score(targets_binary.flatten(), preds.flatten()),
        'precision': precision_score(targets_binary.flatten(), preds.flatten(), zero_division=0),
        'recall': recall_score(targets_binary.flatten(), preds.flatten(), zero_division=0),
        'f1': f1_score(targets_binary.flatten(), preds.flatten(), zero_division=0),
    }

    # mAP (mean Average Precision)
    try:
        # クラスごとのAPの平均
        aps = []
        for i in range(targets.shape[1]):
            if targets_binary[:, i].sum() > 0:  # 正例が存在する場合のみ
                ap = average_precision_score(targets_binary[:, i], probs[:, i])
                aps.append(ap)
        metrics['mAP'] = np.mean(aps) if aps else 0.0
    except:
        metrics['mAP'] = 0.0

    return metrics


def compute_per_class_metrics(probs: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    """クラス別のメトリクスを計算"""
    preds = (probs > threshold).astype(int)
    targets_binary = (targets > threshold).astype(int)

    records = []

    for i, class_name in enumerate(ALL_CLASSES_27):
        if i >= probs.shape[1]:
            break

        class_preds = preds[:, i]
        class_targets = targets_binary[:, i]
        class_probs = probs[:, i]

        # 正例数
        n_positive = class_targets.sum()

        if n_positive == 0:
            continue

        # メトリクス計算
        precision = precision_score(class_targets, class_preds, zero_division=0)
        recall = recall_score(class_targets, class_preds, zero_division=0)
        f1 = f1_score(class_targets, class_preds, zero_division=0)

        try:
            ap = average_precision_score(class_targets, class_probs)
        except:
            ap = 0.0

        records.append({
            'class': class_name,
            'n_positive': int(n_positive),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'AP': ap
        })

    return pd.DataFrame(records)


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) を計算"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    all_probs = probs.flatten()
    all_targets = (targets > 0.5).astype(float).flatten()

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (all_probs > bin_lower) & (all_probs <= bin_upper)
        prop_in_bin = in_bin.sum() / len(all_probs)

        if prop_in_bin > 0:
            accuracy_in_bin = all_targets[in_bin].mean()
            avg_confidence_in_bin = all_probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def generate_markdown_report(
    models_results: Dict[str, Dict],
    output_path: Path
):
    """Markdownレポートを生成"""
    lines = []

    # ヘッダー
    lines.append("# 音響イベント検出モデル 統合分析レポート")
    lines.append("")
    lines.append(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. モデル概要
    lines.append("## 1. モデル概要")
    lines.append("")

    for model_name, results in models_results.items():
        metadata = results['metadata']
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append(f"- **チェックポイント**: `{metadata.get('checkpoint', 'N/A')}`")
        lines.append(f"- **設定ファイル**: `{metadata.get('config_path', 'N/A')}`")
        lines.append(f"- **推論時刻**: {metadata.get('timestamp', 'N/A')}")
        lines.append(f"- **デバイス**: {metadata.get('device', 'N/A')}")
        lines.append("")

    lines.append("---")
    lines.append("")

    # 2. 全体的な性能比較
    lines.append("## 2. 全体的な性能比較")
    lines.append("")

    # Student/Teacher別に表を作成
    for pred_type in ['student', 'teacher']:
        lines.append(f"### {pred_type.capitalize()} モデル")
        lines.append("")

        # データセット別の比較表
        comparison_data = []

        for model_name, results in models_results.items():
            for dataset_name, metrics in results['overall_metrics'][pred_type].items():
                comparison_data.append({
                    'Model': model_name,
                    'Dataset': dataset_name,
                    'Accuracy': f"{metrics['accuracy']:.4f}",
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1': f"{metrics['f1']:.4f}",
                    'mAP': f"{metrics['mAP']:.4f}",
                    'ECE': f"{metrics['ece']:.4f}"
                })

        df = pd.DataFrame(comparison_data)
        lines.append(df.to_markdown(index=False))
        lines.append("")

    lines.append("---")
    lines.append("")

    # 3. データセット別の詳細分析
    lines.append("## 3. データセット別の詳細分析")
    lines.append("")

    datasets = list(next(iter(models_results.values()))['overall_metrics']['student'].keys())

    for dataset_name in datasets:
        lines.append(f"### {dataset_name}")
        lines.append("")

        # モデル間比較（Student）
        comparison = []
        for model_name, results in models_results.items():
            metrics = results['overall_metrics']['student'][dataset_name]
            comparison.append({
                'Model': model_name,
                **{k: f"{v:.4f}" for k, v in metrics.items()}
            })

        df = pd.DataFrame(comparison)
        lines.append(df.to_markdown(index=False))
        lines.append("")

    lines.append("---")
    lines.append("")

    # 4. クラス別性能分析
    lines.append("## 4. クラス別性能分析 (Student)")
    lines.append("")

    # 最初のモデルのクラス別メトリクスを基準として表示
    first_model = list(models_results.keys())[0]

    for dataset_name in datasets:
        lines.append(f"### {dataset_name}")
        lines.append("")

        per_class_df = models_results[first_model]['per_class_metrics']['student'][dataset_name]

        # 各モデルのF1スコアを追加
        for model_name in models_results.keys():
            if model_name == first_model:
                continue

            other_df = models_results[model_name]['per_class_metrics']['student'][dataset_name]
            per_class_df = per_class_df.merge(
                other_df[['class', 'f1']],
                on='class',
                suffixes=('', f'_{model_name}'),
                how='outer'
            )

        # 上位10クラス（F1スコア順）
        if 'f1' in per_class_df.columns:
            top_classes = per_class_df.nlargest(10, 'f1')
            lines.append("#### Top 10 Classes (by F1 score)")
            lines.append("")
            lines.append(top_classes.to_markdown(index=False))
            lines.append("")

    lines.append("---")
    lines.append("")

    # 5. モデル改善の分析（ベースラインとの比較）
    if 'baseline' in models_results and len(models_results) > 1:
        lines.append("## 5. モデル改善の分析")
        lines.append("")

        baseline_results = models_results['baseline']

        for model_name, results in models_results.items():
            if model_name == 'baseline':
                continue

            lines.append(f"### {model_name} vs baseline")
            lines.append("")

            improvements = []

            for dataset_name in datasets:
                baseline_metrics = baseline_results['overall_metrics']['student'][dataset_name]
                current_metrics = results['overall_metrics']['student'][dataset_name]

                improvement = {
                    'Dataset': dataset_name,
                    'ΔAccuracy': f"{(current_metrics['accuracy'] - baseline_metrics['accuracy']):.4f}",
                    'ΔF1': f"{(current_metrics['f1'] - baseline_metrics['f1']):.4f}",
                    'ΔmAP': f"{(current_metrics['mAP'] - baseline_metrics['mAP']):.4f}",
                    'ΔECE': f"{(current_metrics['ece'] - baseline_metrics['ece']):.4f}",
                }
                improvements.append(improvement)

            df = pd.DataFrame(improvements)
            lines.append(df.to_markdown(index=False))
            lines.append("")
            lines.append("*Note: 正の値は改善、負の値は悪化を示す（ECEは例外：負の値が改善）*")
            lines.append("")

    lines.append("---")
    lines.append("")

    # 6. サマリーと考察
    lines.append("## 6. サマリーと考察")
    lines.append("")
    lines.append("### 主な発見")
    lines.append("")
    lines.append("- （自動生成されたメトリクスから主な傾向を記述）")
    lines.append("")

    # 最高性能モデルを特定
    best_models = {}
    for dataset_name in datasets:
        best_f1 = -1
        best_model = None

        for model_name, results in models_results.items():
            f1 = results['overall_metrics']['student'][dataset_name]['f1']
            if f1 > best_f1:
                best_f1 = f1
                best_model = model_name

        best_models[dataset_name] = (best_model, best_f1)

    lines.append("### データセット別最高性能モデル")
    lines.append("")
    for dataset_name, (best_model, f1) in best_models.items():
        lines.append(f"- **{dataset_name}**: {best_model} (F1: {f1:.4f})")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*このレポートは `generate_analysis_report.py` により自動生成されました。*")

    # ファイルに書き込み
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"レポート保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="統合分析レポート生成スクリプト")
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="推論結果ディレクトリ（複数指定可）"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="出力レポートファイル (.md)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="予測の閾値"
    )

    args = parser.parse_args()

    print("="*60)
    print("統合分析レポート生成スクリプト")
    print("="*60)

    # データ読み込みと分析
    print("\n[1/3] データ読み込み中...")
    models_results = {}

    for input_dir in args.input_dirs:
        input_path = Path(input_dir)
        model_name = input_path.name
        print(f"  - {model_name}: {input_path}")

        datasets, metadata = load_inference_data(input_path)

        # メトリクス計算
        overall_metrics = {'student': {}, 'teacher': {}}
        per_class_metrics = {'student': {}, 'teacher': {}}

        for dataset_name, data in datasets.items():
            # ラベル付きデータのみ処理
            if 'targets' not in data:
                continue

            targets = data['targets']

            for pred_type in ['student', 'teacher']:
                probs = data[f'probs_{pred_type}']

                # 全体のメトリクス
                metrics = compute_metrics(probs, targets, args.threshold)
                metrics['ece'] = compute_ece(probs, targets)
                overall_metrics[pred_type][dataset_name] = metrics

                # クラス別メトリクス
                per_class_df = compute_per_class_metrics(probs, targets, args.threshold)
                per_class_metrics[pred_type][dataset_name] = per_class_df

        models_results[model_name] = {
            'metadata': metadata,
            'overall_metrics': overall_metrics,
            'per_class_metrics': per_class_metrics
        }

    # レポート生成
    print("\n[2/3] レポート生成中...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_markdown_report(models_results, output_path)

    # CSV出力も作成
    print("\n[3/3] CSV出力作成中...")
    csv_dir = output_path.parent / 'csv_outputs'
    csv_dir.mkdir(exist_ok=True)

    # 全体メトリクスのCSV
    all_records = []
    for model_name, results in models_results.items():
        for pred_type in ['student', 'teacher']:
            for dataset_name, metrics in results['overall_metrics'][pred_type].items():
                record = {
                    'model': model_name,
                    'pred_type': pred_type,
                    'dataset': dataset_name,
                    **metrics
                }
                all_records.append(record)

    df_overall = pd.DataFrame(all_records)
    df_overall.to_csv(csv_dir / 'overall_metrics.csv', index=False)
    print(f"  - {csv_dir / 'overall_metrics.csv'}")

    # クラス別メトリクスのCSV
    for pred_type in ['student', 'teacher']:
        for dataset_name in datasets.keys():
            if 'targets' not in datasets[dataset_name]:
                continue

            all_class_records = []
            for model_name, results in models_results.items():
                per_class_df = results['per_class_metrics'][pred_type][dataset_name].copy()
                per_class_df['model'] = model_name
                all_class_records.append(per_class_df)

            if all_class_records:
                df_class = pd.concat(all_class_records, ignore_index=True)
                output_csv = csv_dir / f'per_class_{pred_type}_{dataset_name}.csv'
                df_class.to_csv(output_csv, index=False)
                print(f"  - {output_csv}")

    print("\n" + "="*60)
    print("完了！")
    print(f"レポート: {output_path}")
    print(f"CSV出力: {csv_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
