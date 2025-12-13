"""クリップレベルでクラスの混同パターンを分析するスクリプト

使用方法:
    1. 学習済みモデルから予測スコアを取得
    2. このスクリプトでスコアとground truthを比較
    3. 特定のクラスペア（例: Running_water vs footsteps）の混同を可視化
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sed_scores_eval.base_modules.scores import validate_score_dataframe


def load_ground_truth(
    tsv_path: str | Path,
    target_classes: list[str] | None = None,
) -> dict[str, set[str]]:
    """Ground truthファイルから各クリップの正解クラスを取得

    Args:
        tsv_path: ground truthのTSVファイルパス
        target_classes: 分析対象のクラスリスト。Noneの場合は全クラス

    Returns:
        {audio_id: {class1, class2, ...}}形式の辞書

    """
    df = pd.read_csv(tsv_path, sep="\t")
    ground_truth = {}

    for audio_id in df["filename"].unique():
        clip_df = df[df["filename"] == audio_id]
        classes = set(clip_df["event_label"].unique())

        # target_classesが指定されている場合はフィルタ
        if target_classes is not None:
            classes = classes & set(target_classes)

        if classes:  # 対象クラスが含まれる場合のみ追加
            ground_truth[Path(audio_id).stem] = classes

    return ground_truth


def compute_clip_level_scores(
    scores_dict: dict[str, pd.DataFrame],
    aggregation: str = "max",
) -> dict[str, dict[str, float]]:
    """フレームレベルのスコアからクリップレベルのスコアを計算

    Args:
        scores_dict: {audio_id: score_dataframe}形式の辞書
        aggregation: 集約方法 ("max", "mean", "median")

    Returns:
        {audio_id: {class_name: score}}形式の辞書

    """
    clip_scores = {}

    for audio_id, score_df in scores_dict.items():
        # score_dfはDataFrameで、列名がクラス名、値がスコア
        if aggregation == "max":
            clip_score = score_df.max(axis=0).to_dict()
        elif aggregation == "mean":
            clip_score = score_df.mean(axis=0).to_dict()
        elif aggregation == "median":
            clip_score = score_df.median(axis=0).to_dict()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        clip_scores[audio_id] = clip_score

    return clip_scores


def analyze_confusion_for_class_pair(
    ground_truth: dict[str, set[str]],
    clip_scores: dict[str, dict[str, float]],
    class1: str,
    class2: str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """2つのクラス間の混同パターンを分析

    Args:
        ground_truth: {audio_id: {class1, class2, ...}}
        clip_scores: {audio_id: {class_name: score}}
        class1: 分析対象クラス1（例: "Running_water"）
        class2: 分析対象クラス2（例: "footsteps"）
        threshold: 予測の閾値

    Returns:
        分析結果の辞書

    """
    results = {
        "true_class1_pred_class1": [],  # 正解class1, 予測class1
        "true_class1_pred_class2": [],  # 正解class1, 予測class2
        "true_class2_pred_class1": [],  # 正解class2, 予測class1
        "true_class2_pred_class2": [],  # 正解class2, 予測class2
        "true_class1_pred_both": [],  # 正解class1, 両方予測
        "true_class2_pred_both": [],  # 正解class2, 両方予測
    }

    for audio_id in ground_truth:
        if audio_id not in clip_scores:
            continue

        gt_classes = ground_truth[audio_id]
        scores = clip_scores[audio_id]

        # 予測クラスを決定
        pred_class1 = scores.get(class1, 0.0) > threshold
        pred_class2 = scores.get(class2, 0.0) > threshold

        # class1が正解の場合
        if class1 in gt_classes and class2 not in gt_classes:
            if pred_class1 and not pred_class2:
                results["true_class1_pred_class1"].append(
                    {
                        "audio_id": audio_id,
                        f"{class1}_score": scores.get(class1, 0.0),
                        f"{class2}_score": scores.get(class2, 0.0),
                    }
                )
            elif pred_class2 and not pred_class1:
                results["true_class1_pred_class2"].append(
                    {
                        "audio_id": audio_id,
                        f"{class1}_score": scores.get(class1, 0.0),
                        f"{class2}_score": scores.get(class2, 0.0),
                    }
                )
            elif pred_class1 and pred_class2:
                results["true_class1_pred_both"].append(
                    {
                        "audio_id": audio_id,
                        f"{class1}_score": scores.get(class1, 0.0),
                        f"{class2}_score": scores.get(class2, 0.0),
                    }
                )

        # class2が正解の場合
        if class2 in gt_classes and class1 not in gt_classes:
            if pred_class2 and not pred_class1:
                results["true_class2_pred_class2"].append(
                    {
                        "audio_id": audio_id,
                        f"{class1}_score": scores.get(class1, 0.0),
                        f"{class2}_score": scores.get(class2, 0.0),
                    }
                )
            elif pred_class1 and not pred_class2:
                results["true_class2_pred_class1"].append(
                    {
                        "audio_id": audio_id,
                        f"{class1}_score": scores.get(class1, 0.0),
                        f"{class2}_score": scores.get(class2, 0.0),
                    }
                )
            elif pred_class1 and pred_class2:
                results["true_class2_pred_both"].append(
                    {
                        "audio_id": audio_id,
                        f"{class1}_score": scores.get(class1, 0.0),
                        f"{class2}_score": scores.get(class2, 0.0),
                    }
                )

    return results


def plot_confusion_pattern(
    analysis_results: dict[str, Any],
    class1: str,
    class2: str,
    save_path: str | Path | None = None,
) -> None:
    """混同パターンを可視化

    Args:
        analysis_results: analyze_confusion_for_class_pairの結果
        class1: クラス1の名前
        class2: クラス2の名前
        save_path: 保存先パス（Noneの場合は表示のみ）

    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Confusion Analysis: {class1} vs {class2}", fontsize=16, fontweight="bold")

    # 混同行列
    confusion_matrix = np.array(
        [
            [
                len(analysis_results["true_class1_pred_class1"]),
                len(analysis_results["true_class1_pred_class2"]),
            ],
            [
                len(analysis_results["true_class2_pred_class1"]),
                len(analysis_results["true_class2_pred_class2"]),
            ],
        ]
    )

    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"Pred {class1}", f"Pred {class2}"],
        yticklabels=[f"True {class1}", f"True {class2}"],
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Confusion Matrix (Single Class Predictions)")

    # スコア分布：正解class1
    ax = axes[0, 1]
    for result_list, label, color in [
        (analysis_results["true_class1_pred_class1"], "Correct", "green"),
        (analysis_results["true_class1_pred_class2"], "Confused", "red"),
    ]:
        if result_list:
            class1_scores = [r[f"{class1}_score"] for r in result_list]
            class2_scores = [r[f"{class2}_score"] for r in result_list]
            ax.scatter(class1_scores, class2_scores, alpha=0.6, label=label, color=color)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel(f"{class1} Score")
    ax.set_ylabel(f"{class2} Score")
    ax.set_title(f"Score Distribution (True: {class1})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # スコア分布：正解class2
    ax = axes[1, 0]
    for result_list, label, color in [
        (analysis_results["true_class2_pred_class2"], "Correct", "green"),
        (analysis_results["true_class2_pred_class1"], "Confused", "red"),
    ]:
        if result_list:
            class1_scores = [r[f"{class1}_score"] for r in result_list]
            class2_scores = [r[f"{class2}_score"] for r in result_list]
            ax.scatter(class1_scores, class2_scores, alpha=0.6, label=label, color=color)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel(f"{class1} Score")
    ax.set_ylabel(f"{class2} Score")
    ax.set_title(f"Score Distribution (True: {class2})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # サマリー統計
    ax = axes[1, 1]
    ax.axis("off")

    total_class1 = (
        len(analysis_results["true_class1_pred_class1"])
        + len(analysis_results["true_class1_pred_class2"])
        + len(analysis_results["true_class1_pred_both"])
    )
    total_class2 = (
        len(analysis_results["true_class2_pred_class2"])
        + len(analysis_results["true_class2_pred_class1"])
        + len(analysis_results["true_class2_pred_both"])
    )

    class1_accuracy = (
        len(analysis_results["true_class1_pred_class1"]) / total_class1 * 100
        if total_class1 > 0
        else 0
    )
    class2_accuracy = (
        len(analysis_results["true_class2_pred_class2"]) / total_class2 * 100
        if total_class2 > 0
        else 0
    )

    class1_confusion_rate = (
        len(analysis_results["true_class1_pred_class2"]) / total_class1 * 100
        if total_class1 > 0
        else 0
    )
    class2_confusion_rate = (
        len(analysis_results["true_class2_pred_class1"]) / total_class2 * 100
        if total_class2 > 0
        else 0
    )

    summary_text = f"""
    Summary Statistics:

    {class1}:
    - Total clips: {total_class1}
    - Correctly predicted: {len(analysis_results["true_class1_pred_class1"])} ({class1_accuracy:.1f}%)
    - Confused with {class2}: {len(analysis_results["true_class1_pred_class2"])} ({class1_confusion_rate:.1f}%)
    - Both predicted: {len(analysis_results["true_class1_pred_both"])}

    {class2}:
    - Total clips: {total_class2}
    - Correctly predicted: {len(analysis_results["true_class2_pred_class2"])} ({class2_accuracy:.1f}%)
    - Confused with {class1}: {len(analysis_results["true_class2_pred_class1"])} ({class2_confusion_rate:.1f}%)
    - Both predicted: {len(analysis_results["true_class2_pred_both"])}
    """

    ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment="center", fontfamily="monospace")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def main_example():
    """使用例

    実際の使用時は、以下のようにスコア辞書を取得して使用します：
    1. トレーニング済みモデルをロード
    2. validation_stepまたはtest_stepで取得したスコア辞書を使用
    """
    # 例：スコア辞書のフォーマット（実際には validation_step から取得）
    # scores_dict = {
    #     "audio1": pd.DataFrame({
    #         "Running_water": [0.8, 0.7, 0.9, ...],  # フレームごとのスコア
    #         "footsteps": [0.2, 0.3, 0.1, ...],
    #         ... other classes ...
    #     }),
    #     "audio2": pd.DataFrame({ ... }),
    #     ...
    # }

    # Ground truthのロード
    # gt = load_ground_truth(
    #     "/path/to/validation.tsv",
    #     target_classes=["Running_water", "footsteps"],
    # )

    # クリップレベルスコアの計算
    # clip_scores = compute_clip_level_scores(scores_dict, aggregation="max")

    # 混同分析
    # results = analyze_confusion_for_class_pair(
    #     gt,
    #     clip_scores,
    #     class1="Running_water",
    #     class2="footsteps",
    #     threshold=0.5,
    # )

    # 可視化
    # plot_confusion_pattern(
    #     results,
    #     class1="Running_water",
    #     class2="footsteps",
    #     save_path="confusion_running_water_vs_footsteps.png",
    # )

    print("このスクリプトを使用するには:")
    print("1. トレーニング済みモデルから予測スコアを取得")
    print(
        "2. load_ground_truth, compute_clip_level_scores, analyze_confusion_for_class_pair を使用"
    )
    print("3. plot_confusion_pattern で可視化")


if __name__ == "__main__":
    main_example()
