#!/usr/bin/env python3
"""
統合分析レポート生成スクリプト（リファクタリング版）

複数モデルの推論結果から定量的な比較分析を行い、Markdownレポートを生成する。

使用法:
    python generate_analysis_report.py \
        --input_dirs inference_outputs/baseline inference_outputs/cmt_normal inference_outputs/cmt_neg \
        --output visualization_outputs/analysis_report.md
"""

import argparse
import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, average_precision_score, roc_auc_score
)

# 共通ユーティリティをインポート
from visualization_utils import (
    DESED_CLASSES, MAESTRO_REAL_ALL, ALL_CLASSES_27,
    USED_CLASS_INDICES, USED_CLASSES_21
)


# --- 設定クラス ---
@dataclass
class AnalysisConfig:
    """分析レポート設定パラメータ"""
    threshold: float = 0.5
    n_ece_bins: int = 10
    top_k_classes: int = 10
    generate_csv: bool = True
    metrics_to_compute: List[str] = None

    def __post_init__(self):
        if self.metrics_to_compute is None:
            self.metrics_to_compute = [
                'accuracy', 'precision', 'recall', 'f1', 'mAP', 'ece'
            ]

    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            'threshold': self.threshold,
            'n_ece_bins': self.n_ece_bins,
            'top_k_classes': self.top_k_classes,
            'generate_csv': self.generate_csv,
            'metrics_to_compute': self.metrics_to_compute
        }


# --- データローダークラス ---
class InferenceDataLoader:
    """推論結果データのローディングを管理"""

    @staticmethod
    def load_inference_data(
        model_dir: Union[str, Path],
        verbose: bool = True
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict]:
        """
        推論結果とメタデータを読み込む

        Args:
            model_dir: モデルディレクトリ
            verbose: 詳細出力

        Returns:
            datasets: データセット辞書
            metadata: メタデータ辞書
        """
        model_dir = Path(model_dir)

        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        datasets = {}
        npz_files = list(model_dir.glob("*.npz"))

        if not npz_files and verbose:
            print(f"  警告: {model_dir} にNPZファイルが見つかりません")

        for npz_file in npz_files:
            dataset_name = npz_file.stem
            try:
                data = np.load(npz_file, allow_pickle=True)

                datasets[dataset_name] = {
                    'probs_student': data['probs_student'],
                    'probs_teacher': data['probs_teacher'],
                    'filenames': data['filenames'],
                }

                if 'targets' in data:
                    datasets[dataset_name]['targets'] = data['targets']

            except Exception as e:
                if verbose:
                    print(f"  エラー: {npz_file} の読み込みに失敗: {e}")
                continue

        # メタデータの読み込み
        metadata_path = model_dir / 'inference_metadata.json'
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                metadata = {}
                if verbose:
                    print(f"  警告: メタデータの読み込みに失敗: {e}")
        else:
            metadata = {}

        return datasets, metadata


# --- メトリクス計算クラス ---
class MetricsCalculator:
    """各種メトリクスの計算を管理"""

    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Args:
            config: 分析設定
        """
        self.config = config or AnalysisConfig()

    def compute_metrics(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        マルチラベル分類の各種メトリクスを計算

        Args:
            probs: (N, C) 予測確率
            targets: (N, C) 正解ラベル
            threshold: 二値化閾値

        Returns:
            各種メトリクスの辞書
        """
        threshold = threshold or self.config.threshold

        # 入力検証
        if probs.shape != targets.shape:
            raise ValueError("probs and targets must have the same shape")

        preds = (probs > threshold).astype(int)
        targets_binary = (targets > threshold).astype(int)

        metrics = {}

        # 基本メトリクス
        if 'accuracy' in self.config.metrics_to_compute:
            metrics['accuracy'] = accuracy_score(
                targets_binary.flatten(), preds.flatten()
            )

        if 'precision' in self.config.metrics_to_compute:
            metrics['precision'] = precision_score(
                targets_binary.flatten(), preds.flatten(), zero_division=0
            )

        if 'recall' in self.config.metrics_to_compute:
            metrics['recall'] = recall_score(
                targets_binary.flatten(), preds.flatten(), zero_division=0
            )

        if 'f1' in self.config.metrics_to_compute:
            metrics['f1'] = f1_score(
                targets_binary.flatten(), preds.flatten(), zero_division=0
            )

        # mAP (mean Average Precision)
        if 'mAP' in self.config.metrics_to_compute:
            metrics['mAP'] = self._compute_map(probs, targets_binary)

        # ECE
        if 'ece' in self.config.metrics_to_compute:
            metrics['ece'] = self.compute_ece(probs, targets)

        return metrics

    def _compute_map(
        self,
        probs: np.ndarray,
        targets_binary: np.ndarray
    ) -> float:
        """
        mean Average Precisionを計算

        Args:
            probs: 予測確率
            targets_binary: 二値化された正解ラベル

        Returns:
            mAP値
        """
        try:
            aps = []
            for i in range(targets_binary.shape[1]):
                if targets_binary[:, i].sum() > 0:  # 正例が存在する場合のみ
                    ap = average_precision_score(
                        targets_binary[:, i], probs[:, i]
                    )
                    aps.append(ap)
            return np.mean(aps) if aps else 0.0
        except Exception:
            return 0.0

    def compute_ece(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        n_bins: Optional[int] = None
    ) -> float:
        """
        Expected Calibration Error (ECE) を計算

        Args:
            probs: 予測確率
            targets: 正解ラベル
            n_bins: ビン数

        Returns:
            ECE値
        """
        n_bins = n_bins or self.config.n_ece_bins

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        all_probs = probs.flatten()
        all_targets = (targets > self.config.threshold).astype(float).flatten()

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (all_probs > bin_lower) & (all_probs <= bin_upper)
            prop_in_bin = in_bin.sum() / len(all_probs) if len(all_probs) > 0 else 0

            if prop_in_bin > 0:
                accuracy_in_bin = all_targets[in_bin].mean()
                avg_confidence_in_bin = all_probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def compute_per_class_metrics(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        クラス別のメトリクスを計算

        Args:
            probs: 予測確率
            targets: 正解ラベル
            threshold: 二値化閾値

        Returns:
            クラス別メトリクスのDataFrame
        """
        threshold = threshold or self.config.threshold

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
            except Exception:
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


# --- レポート生成クラス ---
class ReportGenerator:
    """分析レポートの生成を管理"""

    def __init__(
        self,
        config: Optional[AnalysisConfig] = None,
        calculator: Optional[MetricsCalculator] = None
    ):
        """
        Args:
            config: 分析設定
            calculator: メトリクス計算インスタンス
        """
        self.config = config or AnalysisConfig()
        self.calculator = calculator or MetricsCalculator(self.config)

    def generate_markdown_report(
        self,
        models_results: Dict[str, Dict],
        output_path: Union[str, Path]
    ) -> None:
        """
        Markdownレポートを生成

        Args:
            models_results: モデル結果の辞書
            output_path: 出力ファイルパス
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []

        # ヘッダー
        self._add_header(lines)

        # 1. モデル概要
        self._add_model_overview(lines, models_results)

        # 2. 全体的な性能比較
        self._add_overall_performance(lines, models_results)

        # 3. データセット別の詳細分析
        self._add_dataset_analysis(lines, models_results)

        # 4. クラス別性能分析
        self._add_class_analysis(lines, models_results)

        # 5. モデル改善の分析
        if 'baseline' in models_results:
            self._add_improvement_analysis(lines, models_results)

        # 6. サマリーと考察
        self._add_summary(lines, models_results)

        # フッター
        self._add_footer(lines)

        # ファイルに書き込み
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"  ✓ レポート保存: {output_path}")

    def _add_header(self, lines: List[str]) -> None:
        """ヘッダーを追加"""
        lines.extend([
            "# 音響イベント検出モデル 統合分析レポート",
            "",
            f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            ""
        ])

    def _add_model_overview(
        self,
        lines: List[str],
        models_results: Dict[str, Dict]
    ) -> None:
        """モデル概要セクションを追加"""
        lines.extend(["## 1. モデル概要", ""])

        for model_name, results in models_results.items():
            metadata = results.get('metadata', {})
            lines.extend([
                f"### {model_name}",
                "",
                f"- **チェックポイント**: `{metadata.get('checkpoint', 'N/A')}`",
                f"- **設定ファイル**: `{metadata.get('config_path', 'N/A')}`",
                f"- **推論時刻**: {metadata.get('timestamp', 'N/A')}",
                f"- **デバイス**: {metadata.get('device', 'N/A')}",
                ""
            ])

        lines.extend(["---", ""])

    def _add_overall_performance(
        self,
        lines: List[str],
        models_results: Dict[str, Dict]
    ) -> None:
        """全体的な性能比較セクションを追加"""
        lines.extend(["## 2. 全体的な性能比較", ""])

        for pred_type in ['student', 'teacher']:
            lines.extend([f"### {pred_type.capitalize()} モデル", ""])

            comparison_data = []
            for model_name, results in models_results.items():
                if pred_type not in results.get('overall_metrics', {}):
                    continue

                for dataset_name, metrics in results['overall_metrics'][pred_type].items():
                    comparison_data.append({
                        'Model': model_name,
                        'Dataset': dataset_name,
                        'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
                        'Precision': f"{metrics.get('precision', 0):.4f}",
                        'Recall': f"{metrics.get('recall', 0):.4f}",
                        'F1': f"{metrics.get('f1', 0):.4f}",
                        'mAP': f"{metrics.get('mAP', 0):.4f}",
                        'ECE': f"{metrics.get('ece', 0):.4f}"
                    })

            if comparison_data:
                df = pd.DataFrame(comparison_data)
                lines.append(df.to_markdown(index=False))
                lines.append("")

        lines.extend(["---", ""])

    def _add_dataset_analysis(
        self,
        lines: List[str],
        models_results: Dict[str, Dict]
    ) -> None:
        """データセット別分析セクションを追加"""
        lines.extend(["## 3. データセット別の詳細分析", ""])

        # データセット名を取得
        datasets = set()
        for results in models_results.values():
            if 'overall_metrics' in results and 'student' in results['overall_metrics']:
                datasets.update(results['overall_metrics']['student'].keys())

        for dataset_name in sorted(datasets):
            lines.extend([f"### {dataset_name}", ""])

            comparison = []
            for model_name, results in models_results.items():
                if 'overall_metrics' not in results:
                    continue
                if 'student' not in results['overall_metrics']:
                    continue
                if dataset_name not in results['overall_metrics']['student']:
                    continue

                metrics = results['overall_metrics']['student'][dataset_name]
                comparison.append({
                    'Model': model_name,
                    **{k: f"{v:.4f}" for k, v in metrics.items()}
                })

            if comparison:
                df = pd.DataFrame(comparison)
                lines.append(df.to_markdown(index=False))
                lines.append("")

        lines.extend(["---", ""])

    def _add_class_analysis(
        self,
        lines: List[str],
        models_results: Dict[str, Dict]
    ) -> None:
        """クラス別分析セクションを追加"""
        lines.extend(["## 4. クラス別性能分析 (Student)", ""])

        # データセット名を取得
        datasets = set()
        for results in models_results.values():
            if 'per_class_metrics' in results and 'student' in results['per_class_metrics']:
                datasets.update(results['per_class_metrics']['student'].keys())

        first_model = list(models_results.keys())[0] if models_results else None

        for dataset_name in sorted(datasets):
            lines.extend([f"### {dataset_name}", ""])

            if first_model and 'per_class_metrics' in models_results[first_model]:
                per_class_data = models_results[first_model]['per_class_metrics']
                if 'student' in per_class_data and dataset_name in per_class_data['student']:
                    per_class_df = per_class_data['student'][dataset_name]

                    if not per_class_df.empty:
                        # 上位クラス（F1スコア順）
                        top_k = min(self.config.top_k_classes, len(per_class_df))
                        if 'f1' in per_class_df.columns:
                            top_classes = per_class_df.nlargest(top_k, 'f1')
                            lines.extend([
                                f"#### Top {top_k} Classes (by F1 score)",
                                "",
                                top_classes.to_markdown(index=False),
                                ""
                            ])

        lines.extend(["---", ""])

    def _add_improvement_analysis(
        self,
        lines: List[str],
        models_results: Dict[str, Dict]
    ) -> None:
        """モデル改善分析セクションを追加"""
        if len(models_results) <= 1:
            return

        lines.extend(["## 5. モデル改善の分析", ""])

        baseline_results = models_results.get('baseline', {})
        if not baseline_results:
            return

        for model_name, results in models_results.items():
            if model_name == 'baseline':
                continue

            lines.extend([f"### {model_name} vs baseline", ""])

            improvements = []
            datasets = set()

            # 両モデルに共通するデータセットを取得
            if 'overall_metrics' in baseline_results and 'overall_metrics' in results:
                baseline_datasets = set(baseline_results['overall_metrics'].get('student', {}).keys())
                current_datasets = set(results['overall_metrics'].get('student', {}).keys())
                datasets = baseline_datasets.intersection(current_datasets)

            for dataset_name in sorted(datasets):
                baseline_metrics = baseline_results['overall_metrics']['student'][dataset_name]
                current_metrics = results['overall_metrics']['student'][dataset_name]

                improvement = {
                    'Dataset': dataset_name,
                    'ΔAccuracy': f"{(current_metrics.get('accuracy', 0) - baseline_metrics.get('accuracy', 0)):.4f}",
                    'ΔF1': f"{(current_metrics.get('f1', 0) - baseline_metrics.get('f1', 0)):.4f}",
                    'ΔmAP': f"{(current_metrics.get('mAP', 0) - baseline_metrics.get('mAP', 0)):.4f}",
                    'ΔECE': f"{(current_metrics.get('ece', 0) - baseline_metrics.get('ece', 0)):.4f}",
                }
                improvements.append(improvement)

            if improvements:
                df = pd.DataFrame(improvements)
                lines.extend([
                    df.to_markdown(index=False),
                    "",
                    "*Note: 正の値は改善、負の値は悪化を示す（ECEは例外：負の値が改善）*",
                    ""
                ])

        lines.extend(["---", ""])

    def _add_summary(
        self,
        lines: List[str],
        models_results: Dict[str, Dict]
    ) -> None:
        """サマリーセクションを追加"""
        lines.extend([
            "## 6. サマリーと考察",
            "",
            "### 主な発見",
            ""
        ])

        # 最高性能モデルを特定
        best_models = self._find_best_models(models_results)

        if best_models:
            lines.extend(["### データセット別最高性能モデル", ""])
            for dataset_name, (best_model, f1) in best_models.items():
                lines.append(f"- **{dataset_name}**: {best_model} (F1: {f1:.4f})")
            lines.append("")

        lines.extend(["---", ""])

    def _find_best_models(
        self,
        models_results: Dict[str, Dict]
    ) -> Dict[str, Tuple[str, float]]:
        """各データセットの最高性能モデルを特定"""
        datasets = set()
        for results in models_results.values():
            if 'overall_metrics' in results and 'student' in results['overall_metrics']:
                datasets.update(results['overall_metrics']['student'].keys())

        best_models = {}
        for dataset_name in datasets:
            best_f1 = -1
            best_model = None

            for model_name, results in models_results.items():
                if 'overall_metrics' not in results:
                    continue
                if 'student' not in results['overall_metrics']:
                    continue
                if dataset_name not in results['overall_metrics']['student']:
                    continue

                f1 = results['overall_metrics']['student'][dataset_name].get('f1', 0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model_name

            if best_model:
                best_models[dataset_name] = (best_model, best_f1)

        return best_models

    def _add_footer(self, lines: List[str]) -> None:
        """フッターを追加"""
        lines.append("*このレポートは `generate_analysis_report.py` により自動生成されました。*")

    def generate_csv_outputs(
        self,
        models_results: Dict[str, Dict],
        output_dir: Union[str, Path]
    ) -> None:
        """
        CSV形式の出力を生成

        Args:
            models_results: モデル結果
            output_dir: 出力ディレクトリ
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 全体メトリクスのCSV
        self._save_overall_metrics_csv(models_results, output_dir)

        # クラス別メトリクスのCSV
        self._save_per_class_metrics_csv(models_results, output_dir)

    def _save_overall_metrics_csv(
        self,
        models_results: Dict[str, Dict],
        output_dir: Path
    ) -> None:
        """全体メトリクスをCSVとして保存"""
        all_records = []

        for model_name, results in models_results.items():
            if 'overall_metrics' not in results:
                continue

            for pred_type in ['student', 'teacher']:
                if pred_type not in results['overall_metrics']:
                    continue

                for dataset_name, metrics in results['overall_metrics'][pred_type].items():
                    record = {
                        'model': model_name,
                        'pred_type': pred_type,
                        'dataset': dataset_name,
                        **metrics
                    }
                    all_records.append(record)

        if all_records:
            df = pd.DataFrame(all_records)
            output_path = output_dir / 'overall_metrics.csv'
            df.to_csv(output_path, index=False)
            print(f"  ✓ 保存: {output_path}")

    def _save_per_class_metrics_csv(
        self,
        models_results: Dict[str, Dict],
        output_dir: Path
    ) -> None:
        """クラス別メトリクスをCSVとして保存"""
        # データセットとpred_typeの組み合わせを収集
        combinations = set()
        for results in models_results.values():
            if 'per_class_metrics' in results:
                for pred_type in results['per_class_metrics']:
                    for dataset_name in results['per_class_metrics'][pred_type]:
                        combinations.add((pred_type, dataset_name))

        for pred_type, dataset_name in combinations:
            all_class_records = []

            for model_name, results in models_results.items():
                if 'per_class_metrics' not in results:
                    continue
                if pred_type not in results['per_class_metrics']:
                    continue
                if dataset_name not in results['per_class_metrics'][pred_type]:
                    continue

                per_class_df = results['per_class_metrics'][pred_type][dataset_name].copy()
                per_class_df['model'] = model_name
                all_class_records.append(per_class_df)

            if all_class_records:
                df = pd.concat(all_class_records, ignore_index=True)
                output_path = output_dir / f'per_class_{pred_type}_{dataset_name}.csv'
                df.to_csv(output_path, index=False)
                print(f"  ✓ 保存: {output_path}")


# --- メイン処理 ---
def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(
        description="統合分析レポート生成スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # 基本的な使用法
  python generate_analysis_report.py --input_dirs inference_outputs/baseline --output report.md

  # 複数モデルの比較
  python generate_analysis_report.py --input_dirs outputs/baseline outputs/cmt_normal outputs/cmt_neg --output comparison_report.md

  # CSVなしで高速生成
  python generate_analysis_report.py --input_dirs outputs/baseline --output report.md --no_csv
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
        "--output",
        required=True,
        help="出力レポートファイル (.md)"
    )

    # オプション引数
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="予測の閾値 (default: 0.5)"
    )
    parser.add_argument(
        "--n_ece_bins",
        type=int,
        default=10,
        help="ECE計算のビン数 (default: 10)"
    )
    parser.add_argument(
        "--top_k_classes",
        type=int,
        default=10,
        help="表示する上位クラス数 (default: 10)"
    )
    parser.add_argument(
        "--no_csv",
        action="store_true",
        help="CSV出力を生成しない"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細出力を有効化"
    )

    return parser.parse_args()


def main():
    """メイン処理"""
    args = parse_arguments()

    print("="*60)
    print("統合分析レポート生成スクリプト")
    print("="*60)

    # 設定の作成
    config = AnalysisConfig(
        threshold=args.threshold,
        n_ece_bins=args.n_ece_bins,
        top_k_classes=args.top_k_classes,
        generate_csv=not args.no_csv
    )

    # 各種インスタンスの初期化
    loader = InferenceDataLoader()
    calculator = MetricsCalculator(config)
    generator = ReportGenerator(config, calculator)

    # データ読み込みと分析
    print("\n[1/3] データ読み込み中...")
    models_results = {}

    for input_dir in args.input_dirs:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"  警告: {input_path} が見つかりません。スキップします。")
            continue

        model_name = input_path.name
        print(f"  - {model_name}: {input_path}")

        try:
            datasets, metadata = loader.load_inference_data(input_path, verbose=args.verbose)

            # メトリクス計算
            overall_metrics = {'student': {}, 'teacher': {}}
            per_class_metrics = {'student': {}, 'teacher': {}}

            for dataset_name, data in datasets.items():
                # ラベル付きデータのみ処理
                if 'targets' not in data:
                    if args.verbose:
                        print(f"    {dataset_name}: ラベルなし、スキップ")
                    continue

                targets = data['targets']

                for pred_type in ['student', 'teacher']:
                    probs = data[f'probs_{pred_type}']

                    # 全体のメトリクス
                    metrics = calculator.compute_metrics(probs, targets)
                    overall_metrics[pred_type][dataset_name] = metrics

                    # クラス別メトリクス
                    per_class_df = calculator.compute_per_class_metrics(probs, targets)
                    per_class_metrics[pred_type][dataset_name] = per_class_df

            models_results[model_name] = {
                'metadata': metadata,
                'overall_metrics': overall_metrics,
                'per_class_metrics': per_class_metrics
            }

        except Exception as e:
            print(f"  エラー: {model_name} の処理に失敗: {e}")
            continue

    if not models_results:
        print("エラー: 有効なモデル結果がありません。")
        return

    # レポート生成
    print("\n[2/3] レポート生成中...")
    output_path = Path(args.output)
    generator.generate_markdown_report(models_results, output_path)

    # CSV出力
    if config.generate_csv:
        print("\n[3/3] CSV出力作成中...")
        csv_dir = output_path.parent / 'csv_outputs'
        generator.generate_csv_outputs(models_results, csv_dir)
    else:
        print("\n[3/3] CSV出力をスキップ")

    # 設定の保存
    config_path = output_path.parent / 'report_config.json'
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"\n  ✓ 設定保存: {config_path}")

    print("\n" + "="*60)
    print("完了！")
    print(f"レポート: {output_path}")
    if config.generate_csv:
        print(f"CSV出力: {output_path.parent / 'csv_outputs'}")
    print("="*60)


if __name__ == "__main__":
    main()