#!/usr/bin/env python3
"""データセット数調査スクリプト

train_pretrained.pyの実際のデータセット構築ロジックに基づいて、
学習/検証/評価に使用されているデータ数を調査します。
クラスごとのイベント数も表示します。

DESED, MAESTRO Real, MAESTRO Synth など全てのデータセットを網羅します。
"""

import os
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# クラス定義（local/classes_dict.pyから）
classes_labels_desed = OrderedDict(
    {
        "Alarm_bell_ringing": 0,
        "Blender": 1,
        "Cat": 2,
        "Dishes": 3,
        "Dog": 4,
        "Electric_shaver_toothbrush": 5,
        "Frying": 6,
        "Running_water": 7,
        "Speech": 8,
        "Vacuum_cleaner": 9,
    },
)

classes_labels_maestro_real = OrderedDict(
    {
        "cutlery and dishes": 0,
        "furniture dragging": 1,
        "people talking": 2,
        "children voices": 3,
        "coffee machine": 4,
        "footsteps": 5,
        "large_vehicle": 6,
        "car": 7,
        "brakes_squeaking": 8,
        "cash register beeping": 9,
        "announcement": 10,
        "shopping cart": 11,
        "metro leaving": 12,
        "metro approaching": 13,
        "door opens/closes": 14,
        "wind_blowing": 15,
        "birds_singing": 16,
    },
)

maestro_desed_alias = {
    "people talking": "Speech",
    "children voices": "Speech",
    "announcement": "Speech",
    "cutlery and dishes": "Dishes",
    "dog_bark": "Dog",
}


def count_audio_files(folder_path: str) -> int:
    """指定されたフォルダ内のオーディオファイル数をカウント"""
    if not os.path.exists(folder_path):
        return 0

    audio_extensions = [".wav", ".mp3", ".flac", ".ogg"]
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                count += 1
    return count


def count_tsv_unique_files(tsv_path: str) -> tuple[int, int, pd.DataFrame]:
    """TSVファイルの一意のファイル数と総サンプル数をカウントし、DataFrameを返す

    Returns:
        unique_files: ユニークなファイル数
        total_samples: 総サンプル数（行数）
        df: DataFrame

    """
    if not os.path.exists(tsv_path):
        return 0, 0, pd.DataFrame()

    try:
        df = pd.read_csv(tsv_path, sep="\t")
        # 最初のカラムがファイル名と仮定
        if len(df.columns) > 0:
            filename_column = df.columns[0]
            unique_count = df[filename_column].nunique()
            total_samples = len(df)  # 総行数（実際の学習サンプル数）
            return unique_count, total_samples, df
        return 0, 0, df
    except Exception as e:
        print(f"Warning: TSVファイル読み込みエラー ({tsv_path}): {e}")
        return 0, 0, pd.DataFrame()


def split_maestro_count(
    maestro_df: pd.DataFrame,
    maestro_split: float,
    seed: int,
) -> tuple[int, int, int, int]:
    """MAESTROデータのsplit後のtrain/valid数をカウント

    Returns:
        train_samples: 訓練用サンプル数（行数）
        valid_samples: 検証用サンプル数（行数）
        train_files: 訓練用ユニークファイル数
        valid_files: 検証用ユニークファイル数

    """
    np.random.seed(seed)

    train_samples = 0
    valid_samples = 0
    train_files = 0
    valid_files = 0

    for scene_name in [
        "cafe_restaurant",
        "city_center",
        "grocery_store",
        "metro_station",
        "residential_area",
    ]:
        mask = maestro_df["filename"].apply(lambda x: "_".join(x.split("_")[:-1])) == scene_name
        filenames = maestro_df[mask]["filename"].apply(lambda x: x.split("-")[0]).unique()
        np.random.shuffle(filenames)

        pivot = int(maestro_split * len(filenames))
        filenames_train = filenames[:pivot]
        filenames_valid = filenames[pivot:]

        mask_train = maestro_df["filename"].apply(lambda x: x.split("-")[0]).isin(filenames_train)
        mask_valid = maestro_df["filename"].apply(lambda x: x.split("-")[0]).isin(filenames_valid)

        train_samples += maestro_df[mask_train].shape[0]
        valid_samples += maestro_df[mask_valid].shape[0]
        train_files += pivot
        valid_files += len(filenames) - pivot

    return train_samples, valid_samples, train_files, valid_files


def count_class_events(df: pd.DataFrame, dataset_type: str = "desed") -> dict[str, int]:
    """TSVのDataFrameからクラスごとのイベント数をカウント

    Args:
        df: TSV DataFrameデータ（event_labelカラムが必要）
        dataset_type: "desed", "maestro_real", "maestro_synth"のいずれか

    Returns:
        クラス名をキー、イベント数を値とする辞書

    """
    if df.empty or "event_label" not in df.columns:
        return {}

    # event_labelカラムのカウント
    class_counts = Counter(df["event_label"].dropna())

    # 期待されるクラスリストを取得
    if dataset_type == "desed":
        expected_classes = list(classes_labels_desed.keys())
    elif dataset_type == "maestro_real":
        expected_classes = list(classes_labels_maestro_real.keys())
    else:
        expected_classes = []

    # 期待されるクラスで0のものも含める
    result = OrderedDict()
    for cls in expected_classes:
        result[cls] = class_counts.get(cls, 0)

    # 期待されないクラスも追加
    for cls, count in class_counts.items():
        if cls not in result:
            result[cls] = count

    return result


def print_class_statistics(class_counts: dict[str, int], indent: str = "   ", top_n: int = None):
    """クラスごとの統計を見やすく表示

    Args:
        class_counts: クラス名をキー、イベント数を値とする辞書
        indent: インデント文字列
        top_n: 上位N個のみ表示（Noneの場合は全て表示）

    """
    if not class_counts:
        print(f"{indent}(データなし)")
        return

    # カウントでソート
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

    if top_n:
        sorted_counts = sorted_counts[:top_n]

    total = sum(class_counts.values())
    print(f"{indent}Total events: {total}")

    for cls, count in sorted_counts:
        if count > 0:
            percentage = (count / total * 100) if total > 0 else 0
            print(f"{indent}  {cls}: {count} ({percentage:.1f}%)")


def convert_path(config_path: str, base_path: str = "ssh") -> str:
    """設定ファイルのパスを実際のパスに変換

    Args:
        config_path: 設定ファイルに記載されているパス
        base_path:
            - "ssh": SSH環境用（設定ファイルのパスをそのまま使用）
            - "local": ローカル環境用（./data/data/dcase/dataset/に変換）
            - その他の文字列: カスタムベースパスに変換

    Returns:
        実際のパス

    """
    if base_path == "ssh":
        # SSH環境: 設定ファイルのパスをそのまま使用
        return config_path
    if base_path == "local":
        # ローカル環境: 相対パスに変換
        return config_path.replace("/mnt/data/data/dcase/dataset/", "./data/data/dcase/dataset/")
    # カスタムパス: 指定されたベースパスに変換
    return config_path.replace("/mnt/data/data/dcase/dataset/", f"{base_path}/")


def count_dataset_detail(
    name: str,
    audio_folder: str,
    audio_folder_44k: str | None,
    tsv_path: str | None,
    dataset_type: str = "desed",
    show_class_distribution: bool = True,
) -> dict:
    """データセットの詳細情報を取得

    Args:
        name: データセット名
        audio_folder: 16kHz音声フォルダのパス
        audio_folder_44k: 44.1kHz音声フォルダのパス（オプション）
        tsv_path: アノテーションTSVファイルのパス（オプション）
        dataset_type: "desed", "maestro_real", "unlabeled"のいずれか
        show_class_distribution: クラス分布を表示するかどうか

    Returns:
        データセット情報の辞書

    """
    info = {"name": name}

    # 16kHz フォルダのカウント
    audio_count_16k = count_audio_files(audio_folder)
    info["audio_files_16k"] = audio_count_16k
    info["audio_folder_16k"] = audio_folder
    info["folder_exists_16k"] = os.path.exists(audio_folder)

    # 44.1kHz フォルダのカウント
    if audio_folder_44k:
        audio_count_44k = count_audio_files(audio_folder_44k)
        info["audio_files_44k"] = audio_count_44k
        info["audio_folder_44k"] = audio_folder_44k
        info["folder_exists_44k"] = os.path.exists(audio_folder_44k)
    else:
        info["audio_files_44k"] = 0
        info["audio_folder_44k"] = None
        info["folder_exists_44k"] = False

    # TSVファイルのカウント
    if tsv_path:
        unique_files, total_samples, df = count_tsv_unique_files(tsv_path)
        info["unique_files"] = unique_files
        info["total_samples"] = total_samples
        info["tsv_path"] = tsv_path
        info["tsv_exists"] = os.path.exists(tsv_path)

        if show_class_distribution and not df.empty:
            class_counts = count_class_events(df, dataset_type)
            info["class_counts"] = class_counts
        else:
            info["class_counts"] = {}
    else:
        info["unique_files"] = 0
        info["total_samples"] = 0
        info["tsv_path"] = None
        info["tsv_exists"] = False
        info["class_counts"] = {}

    return info


def print_dataset_info(info: dict, indent: str = "  "):
    """データセット情報を表示"""
    print(f"\n{info['name']}:")

    # フォルダ存在確認
    if not info.get("folder_exists_16k", False) and info.get("audio_folder_16k"):
        print(f"{indent}⚠ 16kHz フォルダが見つかりません")
    if not info.get("folder_exists_44k", False) and info.get("audio_folder_44k"):
        print(f"{indent}⚠ 44.1kHz フォルダが見つかりません")

    # 音声ファイル数
    print(f"{indent}Audio files (16kHz): {info['audio_files_16k']}")
    if info.get("audio_files_44k", 0) > 0:
        print(f"{indent}Audio files (44.1kHz): {info['audio_files_44k']}")

    # TSV情報
    if info.get("tsv_path"):
        if not info.get("tsv_exists", False):
            print(f"{indent}⚠ TSVファイルが見つかりません")
        else:
            print(f"{indent}Unique files (TSV): {info['unique_files']}")
            print(f"{indent}Total samples (TSV rows): {info['total_samples']}")

    # クラス分布
    if info.get("class_counts"):
        print(f"{indent}Class distribution:")
        print_class_statistics(info["class_counts"], indent=f"{indent}  ")


def main():
    """メイン処理"""
    print("=" * 80)
    print("DCASE 2024 Task 4 データセット数調査（詳細版）")
    print("train_pretrained.pyのロジックに基づく実際の使用データ数")
    print("DESED, MAESTRO Real, MAESTRO Synth など全てのデータセットを網羅")
    print("=" * 80)

    # 設定ファイルを読み込む
    config_path = "confs/pretrained.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 環境設定（自動検出）
    # "ssh"    : SSH環境（/mntのパスをそのまま使用）
    # "local"  : ローカル環境（./data/data/dcase/dataset/に変換）
    # その他   : カスタムパスを指定
    import os

    if os.path.exists("/mnt/data/data/dcase/dataset/"):
        base_path = "ssh"  # SSH環境を自動検出
    else:
        base_path = "local"  # ローカル環境

    print(f"\n設定ファイル: {config_path}")
    print(
        f"環境: {'SSH (/mntのパスを使用)' if base_path == 'ssh' else 'ローカル (./data/data/dcase/dataset/を使用)'}"
    )
    print(f"Weak split: {config['training']['weak_split']}")
    print(f"MAESTRO split: {config['training']['maestro_split']}")
    print(f"Seed: {config['training']['seed']}")

    data_config = config["data"]
    seed = config["training"]["seed"]
    weak_split = config["training"]["weak_split"]
    maestro_split = config["training"]["maestro_split"]

    # すべてのデータセット情報を格納
    all_datasets = {}

    # ==========================================================================
    # DESED データセット
    # ==========================================================================
    print("\n" + "=" * 80)
    print("【DESED データセット】")
    print("=" * 80)

    # 1. DESED Synthetic Train
    print("\n--- DESED Synthetic (Train) ---")
    synth_info = count_dataset_detail(
        name="DESED Synthetic Train",
        audio_folder=convert_path(data_config["synth_folder"], base_path),
        audio_folder_44k=convert_path(data_config["synth_folder_44k"], base_path),
        tsv_path=convert_path(data_config["synth_tsv"], base_path),
        dataset_type="desed",
    )
    print_dataset_info(synth_info)
    all_datasets["desed_synth_train"] = synth_info

    # 2. DESED Synthetic Validation
    print("\n--- DESED Synthetic (Validation) ---")
    synth_val_info = count_dataset_detail(
        name="DESED Synthetic Validation",
        audio_folder=convert_path(data_config["synth_val_folder"], base_path),
        audio_folder_44k=convert_path(data_config["synth_val_folder_44k"], base_path),
        tsv_path=convert_path(data_config["synth_val_tsv"], base_path),
        dataset_type="desed",
    )
    print_dataset_info(synth_val_info)
    all_datasets["desed_synth_val"] = synth_val_info

    # 3. DESED Strong Real
    print("\n--- DESED Strong Real ---")
    strong_info = count_dataset_detail(
        name="DESED Strong Real (AudioSet Strong)",
        audio_folder=convert_path(data_config["strong_folder"], base_path),
        audio_folder_44k=convert_path(data_config["strong_folder_44k"], base_path),
        tsv_path=convert_path(data_config["strong_tsv"], base_path),
        dataset_type="desed",
    )
    print_dataset_info(strong_info)
    all_datasets["desed_strong_real"] = strong_info

    # 4. DESED Weak
    print("\n--- DESED Weak ---")
    weak_info = count_dataset_detail(
        name="DESED Weak",
        audio_folder=convert_path(data_config["weak_folder"], base_path),
        audio_folder_44k=convert_path(data_config["weak_folder_44k"], base_path),
        tsv_path=convert_path(data_config["weak_tsv"], base_path),
        dataset_type="desed",
    )
    print_dataset_info(weak_info)
    # Split情報を追加
    if weak_info["total_samples"] > 0:
        train_weak_samples = int(weak_info["total_samples"] * weak_split)
        valid_weak_samples = weak_info["total_samples"] - train_weak_samples
        print(f"  Split ({weak_split * 100:.0f}% train / {(1 - weak_split) * 100:.0f}% valid):")
        print(f"    Train samples: {train_weak_samples}")
        print(f"    Valid samples: {valid_weak_samples}")
        weak_info["train_samples"] = train_weak_samples
        weak_info["valid_samples"] = valid_weak_samples
    all_datasets["desed_weak"] = weak_info

    # 5. DESED Unlabeled
    print("\n--- DESED Unlabeled ---")
    unlabeled_info = count_dataset_detail(
        name="DESED Unlabeled (In-domain)",
        audio_folder=convert_path(data_config["unlabeled_folder"], base_path),
        audio_folder_44k=convert_path(data_config["unlabeled_folder_44k"], base_path),
        tsv_path=None,
        dataset_type="unlabeled",
        show_class_distribution=False,
    )
    print_dataset_info(unlabeled_info)
    all_datasets["desed_unlabeled"] = unlabeled_info

    # 6. DESED DevTest (Validation)
    print("\n--- DESED DevTest (Validation) ---")
    devtest_info = count_dataset_detail(
        name="DESED DevTest",
        audio_folder=convert_path(data_config["test_folder"], base_path),
        audio_folder_44k=convert_path(data_config["test_folder_44k"], base_path),
        tsv_path=convert_path(data_config["test_tsv"], base_path),
        dataset_type="desed",
    )
    print_dataset_info(devtest_info)
    all_datasets["desed_devtest"] = devtest_info

    # 7. DESED Evaluation
    print("\n--- DESED Evaluation (DCASE 2024) ---")
    eval_info = count_dataset_detail(
        name="DESED Evaluation 2024",
        audio_folder=convert_path(data_config["eval_folder"], base_path),
        audio_folder_44k=convert_path(data_config["eval_folder_44k"], base_path),
        tsv_path=None,
        dataset_type="unlabeled",
        show_class_distribution=False,
    )
    print_dataset_info(eval_info)
    all_datasets["desed_eval"] = eval_info

    # ==========================================================================
    # MAESTRO Real データセット
    # ==========================================================================
    print("\n" + "=" * 80)
    print("【MAESTRO Real データセット】")
    print("=" * 80)

    # 8. MAESTRO Real Train
    print("\n--- MAESTRO Real Train ---")
    maestro_real_train_info = count_dataset_detail(
        name="MAESTRO Real Train",
        audio_folder=convert_path(data_config["real_maestro_train_folder"], base_path),
        audio_folder_44k=convert_path(data_config["real_maestro_train_folder_44k"], base_path),
        tsv_path=convert_path(data_config["real_maestro_train_tsv"], base_path),
        dataset_type="maestro_real",
    )
    print_dataset_info(maestro_real_train_info)

    # Split情報を追加
    maestro_real_tsv = convert_path(data_config["real_maestro_train_tsv"], base_path)
    if os.path.exists(maestro_real_tsv):
        maestro_real_df = pd.read_csv(maestro_real_tsv, sep="\t")
        if not maestro_real_df.empty:
            (
                maestro_train_samples,
                maestro_valid_samples,
                maestro_train_files,
                maestro_valid_files,
            ) = split_maestro_count(maestro_real_df, maestro_split, seed)
            print(
                f"  Split ({maestro_split * 100:.0f}% train / {(1 - maestro_split) * 100:.0f}% valid):",
            )
            print(f"    Train samples: {maestro_train_samples}")
            print(f"    Train files: {maestro_train_files}")
            print(f"    Valid samples: {maestro_valid_samples}")
            print(f"    Valid files: {maestro_valid_files}")
            maestro_real_train_info["train_samples"] = maestro_train_samples
            maestro_real_train_info["valid_samples"] = maestro_valid_samples
            maestro_real_train_info["train_files"] = maestro_train_files
            maestro_real_train_info["valid_files"] = maestro_valid_files
    all_datasets["maestro_real_train"] = maestro_real_train_info

    # 9. MAESTRO Real Validation
    print("\n--- MAESTRO Real Validation ---")
    maestro_real_val_info = count_dataset_detail(
        name="MAESTRO Real Validation",
        audio_folder=convert_path(data_config["real_maestro_val_folder"], base_path),
        audio_folder_44k=convert_path(data_config["real_maestro_val_folder_44k"], base_path),
        tsv_path=convert_path(data_config["real_maestro_val_tsv"], base_path),
        dataset_type="maestro_real",
    )
    print_dataset_info(maestro_real_val_info)
    all_datasets["maestro_real_val"] = maestro_real_val_info

    # ==========================================================================
    # MAESTRO Synthetic データセット
    # ==========================================================================
    print("\n" + "=" * 80)
    print("【MAESTRO Synthetic データセット】")
    print("=" * 80)

    # 10. MAESTRO Synthetic Train
    print("\n--- MAESTRO Synthetic Train ---")
    maestro_synth_train_info = count_dataset_detail(
        name="MAESTRO Synthetic Train",
        audio_folder=convert_path(data_config["synth_maestro_train"], base_path),
        audio_folder_44k=convert_path(data_config["synth_maestro_train_44k"], base_path),
        tsv_path=convert_path(data_config["synth_maestro_tsv"], base_path),
        dataset_type="maestro_real",  # MAESTRO Synthも同じクラスラベル
    )
    print_dataset_info(maestro_synth_train_info)
    all_datasets["maestro_synth_train"] = maestro_synth_train_info

    # ==========================================================================
    # サマリー
    # ==========================================================================
    print("\n" + "=" * 80)
    print("【サマリー】")
    print("=" * 80)

    print("\n■ DESED データセット (10クラス)")
    print("-" * 60)
    desed_total_files = 0
    desed_total_samples = 0

    desed_datasets = [
        ("Synthetic Train", all_datasets["desed_synth_train"]),
        ("Synthetic Validation", all_datasets["desed_synth_val"]),
        ("Strong Real", all_datasets["desed_strong_real"]),
        ("Weak", all_datasets["desed_weak"]),
        ("Unlabeled", all_datasets["desed_unlabeled"]),
        ("DevTest", all_datasets["desed_devtest"]),
        ("Evaluation 2024", all_datasets["desed_eval"]),
    ]

    for name, info in desed_datasets:
        files = info.get("audio_files_16k", 0)
        samples = info.get("total_samples", 0)
        desed_total_files += files
        desed_total_samples += samples
        print(f"  {name:25s} | Files: {files:6d} | Samples: {samples:6d}")

    print("-" * 60)
    print(
        f"  {'DESED Total':25s} | Files: {desed_total_files:6d} | Samples: {desed_total_samples:6d}",
    )

    print("\n■ MAESTRO Real データセット (17クラス)")
    print("-" * 60)
    maestro_real_total_files = 0
    maestro_real_total_samples = 0

    maestro_real_datasets = [
        ("Real Train", all_datasets["maestro_real_train"]),
        ("Real Validation", all_datasets["maestro_real_val"]),
    ]

    for name, info in maestro_real_datasets:
        files = info.get("audio_files_16k", 0)
        samples = info.get("total_samples", 0)
        maestro_real_total_files += files
        maestro_real_total_samples += samples
        print(f"  {name:25s} | Files: {files:6d} | Samples: {samples:6d}")

    print("-" * 60)
    print(
        f"  {'MAESTRO Real Total':25s} | Files: {maestro_real_total_files:6d} | Samples: {maestro_real_total_samples:6d}",
    )

    print("\n■ MAESTRO Synthetic データセット")
    print("-" * 60)
    maestro_synth_info = all_datasets["maestro_synth_train"]
    maestro_synth_files = maestro_synth_info.get("audio_files_16k", 0)
    maestro_synth_samples = maestro_synth_info.get("total_samples", 0)
    print(
        f"  {'Synthetic Train':25s} | Files: {maestro_synth_files:6d} | Samples: {maestro_synth_samples:6d}",
    )
    print("-" * 60)
    print(
        f"  {'MAESTRO Synth Total':25s} | Files: {maestro_synth_files:6d} | Samples: {maestro_synth_samples:6d}",
    )

    # 全体合計
    total_files = desed_total_files + maestro_real_total_files + maestro_synth_files
    total_samples = desed_total_samples + maestro_real_total_samples + maestro_synth_samples

    print("\n" + "=" * 60)
    print(f"  {'全データセット合計':25s} | Files: {total_files:6d} | Samples: {total_samples:6d}")
    print("=" * 60)

    # ==========================================================================
    # Training用データ構成
    # ==========================================================================
    print("\n" + "=" * 80)
    print("【Training用データ構成】")
    print(
        "tot_train_data = [maestro_real_train, synth_set, strong_full_set, weak_set, unlabeled_set]",
    )
    print("batch_size = [12, 6, 6, 12, 24]")
    print("=" * 80)

    # strong_full_set = strong_set + synth_set
    strong_samples = all_datasets["desed_strong_real"].get("total_samples", 0)
    synth_samples = all_datasets["desed_synth_train"].get("total_samples", 0)
    strong_full_samples = strong_samples + synth_samples

    print(
        f"\n  Index 0 - maestro_real_train:  {all_datasets['maestro_real_train'].get('train_samples', 0):6d} samples (split後)",
    )
    print(f"  Index 1 - synth_set:           {synth_samples:6d} samples")
    print(f"  Index 2 - strong_full_set:     {strong_full_samples:6d} samples")
    print(f"           ├─ strong_real:       {strong_samples:6d}")
    print(f"           └─ synth_set:         {synth_samples:6d}")
    print(
        f"  Index 3 - weak_set:            {all_datasets['desed_weak'].get('train_samples', 0):6d} samples (split後)",
    )
    print(
        f"  Index 4 - unlabeled_set:       {all_datasets['desed_unlabeled'].get('audio_files_16k', 0):6d} files",
    )

    # ==========================================================================
    # 詳細情報をファイルに保存
    # ==========================================================================
    output_file = "dataset_count_detailed.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("DCASE 2024 Task 4 データセット詳細情報\n")
        f.write("train_pretrained.pyのロジックに基づく実際の使用データ数\n")
        f.write("DESED, MAESTRO Real, MAESTRO Synth など全てのデータセットを網羅\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"設定ファイル: {config_path}\n")
        f.write(f"Weak split: {config['training']['weak_split']}\n")
        f.write(f"MAESTRO split: {config['training']['maestro_split']}\n")
        f.write(f"Seed: {config['training']['seed']}\n\n")

        # DESED
        f.write("=" * 60 + "\n")
        f.write("【DESED データセット (10クラス)】\n")
        f.write("=" * 60 + "\n")
        for key in [
            "desed_synth_train",
            "desed_synth_val",
            "desed_strong_real",
            "desed_weak",
            "desed_unlabeled",
            "desed_devtest",
            "desed_eval",
        ]:
            if key in all_datasets:
                info = all_datasets[key]
                f.write(f"\n{info.get('name', key)}:\n")
                f.write(f"  Audio files (16kHz): {info.get('audio_files_16k', 0)}\n")
                if info.get("audio_files_44k", 0) > 0:
                    f.write(f"  Audio files (44.1kHz): {info.get('audio_files_44k', 0)}\n")
                if info.get("tsv_path"):
                    f.write(f"  Unique files (TSV): {info.get('unique_files', 0)}\n")
                    f.write(f"  Total samples: {info.get('total_samples', 0)}\n")
                if info.get("train_samples"):
                    f.write(f"  Train samples (split): {info.get('train_samples', 0)}\n")
                    f.write(f"  Valid samples (split): {info.get('valid_samples', 0)}\n")
                if info.get("class_counts"):
                    f.write("  Class distribution:\n")
                    for cls, count in info["class_counts"].items():
                        if count > 0:
                            f.write(f"    {cls}: {count}\n")

        # MAESTRO Real
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("【MAESTRO Real データセット (17クラス)】\n")
        f.write("=" * 60 + "\n")
        for key in ["maestro_real_train", "maestro_real_val"]:
            if key in all_datasets:
                info = all_datasets[key]
                f.write(f"\n{info.get('name', key)}:\n")
                f.write(f"  Audio files (16kHz): {info.get('audio_files_16k', 0)}\n")
                if info.get("audio_files_44k", 0) > 0:
                    f.write(f"  Audio files (44.1kHz): {info.get('audio_files_44k', 0)}\n")
                if info.get("tsv_path"):
                    f.write(f"  Unique files (TSV): {info.get('unique_files', 0)}\n")
                    f.write(f"  Total samples: {info.get('total_samples', 0)}\n")
                if info.get("train_samples"):
                    f.write(f"  Train samples (split): {info.get('train_samples', 0)}\n")
                    f.write(f"  Valid samples (split): {info.get('valid_samples', 0)}\n")
                    f.write(f"  Train files (split): {info.get('train_files', 0)}\n")
                    f.write(f"  Valid files (split): {info.get('valid_files', 0)}\n")
                if info.get("class_counts"):
                    f.write("  Class distribution:\n")
                    for cls, count in info["class_counts"].items():
                        if count > 0:
                            f.write(f"    {cls}: {count}\n")

        # MAESTRO Synth
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("【MAESTRO Synthetic データセット】\n")
        f.write("=" * 60 + "\n")
        if "maestro_synth_train" in all_datasets:
            info = all_datasets["maestro_synth_train"]
            f.write(f"\n{info.get('name', 'maestro_synth_train')}:\n")
            f.write(f"  Audio files (16kHz): {info.get('audio_files_16k', 0)}\n")
            if info.get("audio_files_44k", 0) > 0:
                f.write(f"  Audio files (44.1kHz): {info.get('audio_files_44k', 0)}\n")
            if info.get("tsv_path"):
                f.write(f"  Unique files (TSV): {info.get('unique_files', 0)}\n")
                f.write(f"  Total samples: {info.get('total_samples', 0)}\n")
            if info.get("class_counts"):
                f.write("  Class distribution:\n")
                for cls, count in info["class_counts"].items():
                    if count > 0:
                        f.write(f"    {cls}: {count}\n")

        # サマリー
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("【サマリー】\n")
        f.write("=" * 60 + "\n\n")
        f.write("■ DESED データセット (10クラス)\n")
        f.write("-" * 50 + "\n")
        for name, info in desed_datasets:
            files = info.get("audio_files_16k", 0)
            samples = info.get("total_samples", 0)
            f.write(f"  {name:25s} | Files: {files:6d} | Samples: {samples:6d}\n")
        f.write("-" * 50 + "\n")
        f.write(
            f"  {'DESED Total':25s} | Files: {desed_total_files:6d} | Samples: {desed_total_samples:6d}\n\n",
        )

        f.write("■ MAESTRO Real データセット (17クラス)\n")
        f.write("-" * 50 + "\n")
        for name, info in maestro_real_datasets:
            files = info.get("audio_files_16k", 0)
            samples = info.get("total_samples", 0)
            f.write(f"  {name:25s} | Files: {files:6d} | Samples: {samples:6d}\n")
        f.write("-" * 50 + "\n")
        f.write(
            f"  {'MAESTRO Real Total':25s} | Files: {maestro_real_total_files:6d} | Samples: {maestro_real_total_samples:6d}\n\n",
        )

        f.write("■ MAESTRO Synthetic データセット\n")
        f.write("-" * 50 + "\n")
        f.write(
            f"  {'Synthetic Train':25s} | Files: {maestro_synth_files:6d} | Samples: {maestro_synth_samples:6d}\n",
        )
        f.write("-" * 50 + "\n")
        f.write(
            f"  {'MAESTRO Synth Total':25s} | Files: {maestro_synth_files:6d} | Samples: {maestro_synth_samples:6d}\n\n",
        )

        f.write("=" * 50 + "\n")
        f.write(
            f"  {'全データセット合計':25s} | Files: {total_files:6d} | Samples: {total_samples:6d}\n",
        )
        f.write("=" * 50 + "\n")

    print(f"\n\n詳細情報を {output_file} に保存しました。")


if __name__ == "__main__":
    main()
