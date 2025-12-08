#!/usr/bin/env python3
"""データセット数調査スクリプト

train_pretrained.pyの実際のデータセット構築ロジックに基づいて、
学習/検証/評価に使用されているデータ数を調査します。
クラスごとのイベント数も表示します。
"""

import os
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

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
        print(f"Warning: Folder not found: {folder_path}")
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
        print(f"Warning: TSV not found: {tsv_path}")
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
    maestro_df: pd.DataFrame, maestro_split: float, seed: int
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


def convert_path(config_path: str, base_path: str = None) -> str:
    """設定ファイルのパスを実際のパスに変換（SSH環境ではそのまま返す）"""
    if base_path is None:
        # SSH環境: 設定ファイルのパスをそのまま使用
        return config_path
    # ローカル環境: パスを変換
    return config_path.replace("/mnt/data/data/dcase/dataset/", f"{base_path}/")


def main():
    """メイン処理"""
    print("=" * 80)
    print("DCASE 2024 Task 4 データセット数調査")
    print("train_pretrained.pyのロジックに基づく実際の使用データ数")
    print("=" * 80)

    # 設定ファイルを読み込む
    config_path = "confs/pretrained.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # データベースパスを設定
    # SSH環境: None (設定ファイルのパスをそのまま使用)
    # ローカル環境: "../../../sed-fl/fl/data/dcase/dataset"
    base_path = None  # SSH環境用

    print(f"\n設定ファイル: {config_path}")
    if base_path is None:
        print("環境: SSH (confs/pretrained.yamlのパスをそのまま使用)")
    else:
        print(f"環境: ローカル (データベースパス: {base_path})")
    print(f"Weak split: {config['training']['weak_split']}")
    print(f"MAESTRO split: {config['training']['maestro_split']}")
    print(f"Seed: {config['training']['seed']}")

    data_config = config["data"]

    # === Training データの調査 ===
    print("\n" + "=" * 80)
    print("TRAINING データ (train_pretrained.py line 471)")
    print(
        "tot_train_data = [maestro_real_train, synth_set, strong_full_set, weak_set, unlabeled_set]"
    )
    print("=" * 80)

    train_info = {}

    # 1. synth_set (Synthetic training)
    synth_folder = convert_path(data_config["synth_folder"], base_path)
    synth_tsv = convert_path(data_config["synth_tsv"], base_path)
    synth_audio_count = count_audio_files(synth_folder)
    synth_unique_files, synth_tsv_count, synth_df = count_tsv_unique_files(synth_tsv)
    synth_class_counts = count_class_events(synth_df, "desed")
    train_info["synth_set"] = {
        "audio_files": synth_audio_count,
        "unique_files": synth_unique_files,
        "samples": synth_tsv_count,
        "description": "Synthetic training (index 1 in tot_train_data)",
        "class_counts": synth_class_counts,
    }
    print("\n1. synth_set:")
    print(f"   Audio files: {synth_audio_count}")
    print(f"   Unique files: {synth_unique_files}")
    print(f"   Training samples (TSV rows): {synth_tsv_count}")
    print("   Class distribution (DESED 10 classes):")
    print_class_statistics(synth_class_counts, indent="     ")

    # 2. strong_set (Strong real)
    strong_folder = convert_path(data_config["strong_folder"], base_path)
    strong_tsv = convert_path(data_config["strong_tsv"], base_path)
    strong_audio_count = count_audio_files(strong_folder)
    strong_unique_files, strong_tsv_count, strong_df = count_tsv_unique_files(strong_tsv)
    strong_class_counts = count_class_events(strong_df, "desed")
    train_info["strong_set"] = {
        "audio_files": strong_audio_count,
        "unique_files": strong_unique_files,
        "samples": strong_tsv_count,
        "description": "Strong real (part of strong_full_set)",
        "class_counts": strong_class_counts,
    }
    print("\n2. strong_set (Strong real):")
    print(f"   Audio files: {strong_audio_count}")
    print(f"   Unique files: {strong_unique_files}")
    print(f"   Training samples (TSV rows): {strong_tsv_count}")
    print("   Class distribution (DESED 10 classes):")
    print_class_statistics(strong_class_counts, indent="     ")

    # 3. strong_full_set = strong_set + synth_set
    strong_full_samples = strong_tsv_count + synth_tsv_count
    train_info["strong_full_set"] = {
        "audio_files": strong_audio_count + synth_audio_count,
        "unique_files": strong_unique_files + synth_unique_files,
        "samples": strong_full_samples,
        "description": "strong_set + synth_set (index 2 in tot_train_data)",
    }
    print("\n3. strong_full_set (strong_set + synth_set):")
    print(f"   Combined samples: {strong_full_samples}")

    # 4. weak_set (Weak training) - split考慮
    weak_folder = convert_path(data_config["weak_folder"], base_path)
    weak_tsv = convert_path(data_config["weak_tsv"], base_path)
    weak_audio_count = count_audio_files(weak_folder)
    weak_unique_files, weak_tsv_count, weak_df = count_tsv_unique_files(weak_tsv)
    weak_class_counts = count_class_events(weak_df, "desed")

    # weak_splitを適用
    weak_split = config["training"]["weak_split"]
    seed = config["training"]["seed"]
    np.random.seed(seed)
    train_weak_samples = int(weak_tsv_count * weak_split)
    valid_weak_samples = weak_tsv_count - train_weak_samples

    train_info["weak_set"] = {
        "audio_files": weak_audio_count,
        "unique_files": weak_unique_files,
        "samples_total": weak_tsv_count,
        "samples_train": train_weak_samples,
        "samples_valid": valid_weak_samples,
        "description": f"Weak training ({weak_split * 100:.0f}% of weak data, index 3 in tot_train_data)",
        "class_counts": weak_class_counts,
    }
    print("\n4. weak_set (Weak training):")
    print(f"   Total audio files: {weak_audio_count}")
    print(f"   Total unique files: {weak_unique_files}")
    print(f"   Total samples: {weak_tsv_count}")
    print(f"   Train split ({weak_split * 100:.0f}%): {train_weak_samples}")
    print(f"   Valid split ({(1 - weak_split) * 100:.0f}%): {valid_weak_samples}")
    print("   Class distribution (DESED 10 classes):")
    print_class_statistics(weak_class_counts, indent="     ")

    # 5. unlabeled_set (Unlabeled)
    unlabeled_folder = convert_path(data_config["unlabeled_folder"], base_path)
    unlabeled_audio_count = count_audio_files(unlabeled_folder)
    train_info["unlabeled_set"] = {
        "audio_files": unlabeled_audio_count,
        "description": "Unlabeled (index 4 in tot_train_data)",
    }
    print("\n5. unlabeled_set:")
    print(f"   Audio files: {unlabeled_audio_count}")

    # 6. maestro_real_train - split考慮
    maestro_real_folder = convert_path(data_config["real_maestro_train_folder"], base_path)
    maestro_real_tsv = convert_path(data_config["real_maestro_train_tsv"], base_path)
    maestro_real_audio_count = count_audio_files(maestro_real_folder)
    maestro_real_unique_files, maestro_real_tsv_count, maestro_real_df = count_tsv_unique_files(
        maestro_real_tsv
    )
    maestro_class_counts = count_class_events(maestro_real_df, "maestro_real")

    # maestro_splitを適用
    maestro_split = config["training"]["maestro_split"]
    if not maestro_real_df.empty:
        maestro_train_samples, maestro_valid_samples, maestro_train_files, maestro_valid_files = (
            split_maestro_count(maestro_real_df, maestro_split, seed)
        )
    else:
        maestro_train_samples = int(maestro_real_tsv_count * maestro_split)
        maestro_valid_samples = maestro_real_tsv_count - maestro_train_samples
        maestro_train_files = int(maestro_real_unique_files * maestro_split)
        maestro_valid_files = maestro_real_unique_files - maestro_train_files

    train_info["maestro_real_train"] = {
        "audio_files": maestro_real_audio_count,
        "unique_files_total": maestro_real_unique_files,
        "unique_files_train": maestro_train_files,
        "unique_files_valid": maestro_valid_files,
        "samples_total": maestro_real_tsv_count,
        "samples_train": maestro_train_samples,
        "samples_valid": maestro_valid_samples,
        "description": f"MAESTRO real training ({maestro_split * 100:.0f}% split, index 0 in tot_train_data)",
        "class_counts": maestro_class_counts,
    }
    print("\n6. maestro_real_train (MAESTRO real):")
    print(f"   Total audio files: {maestro_real_audio_count}")
    print(f"   Total unique files: {maestro_real_unique_files}")
    print(f"   Total samples: {maestro_real_tsv_count}")
    print(f"   Train split ({maestro_split * 100:.0f}%):")
    print(f"     - Samples: {maestro_train_samples}")
    print(f"     - Unique files: {maestro_train_files}")
    print(f"   Valid split ({(1 - maestro_split) * 100:.0f}%):")
    print(f"     - Samples: {maestro_valid_samples}")
    print(f"     - Unique files: {maestro_valid_files}")
    print("   Class distribution (MAESTRO Real 17 classes):")
    print_class_statistics(maestro_class_counts, indent="     ")

    # === Validation データの調査 ===
    print("\n" + "=" * 80)
    print("VALIDATION データ (train_pretrained.py line 395)")
    print("valid_dataset = [synth_val, weak_val, maestro_real_valid]")
    print("=" * 80)

    valid_info = {}

    # 1. synth_val (Synthetic validation)
    synth_val_folder = convert_path(data_config["synth_val_folder"], base_path)
    synth_val_tsv = convert_path(data_config["synth_val_tsv"], base_path)
    synth_val_audio_count = count_audio_files(synth_val_folder)
    synth_val_unique_files, synth_val_tsv_count, synth_val_df = count_tsv_unique_files(
        synth_val_tsv
    )
    synth_val_class_counts = count_class_events(synth_val_df, "desed")
    valid_info["synth_val"] = {
        "audio_files": synth_val_audio_count,
        "unique_files": synth_val_unique_files,
        "samples": synth_val_tsv_count,
        "description": "Synthetic validation",
        "class_counts": synth_val_class_counts,
    }
    print("\n1. synth_val:")
    print(f"   Audio files: {synth_val_audio_count}")
    print(f"   Unique files: {synth_val_unique_files}")
    print(f"   Validation samples: {synth_val_tsv_count}")
    print("   Class distribution (DESED 10 classes):")
    print_class_statistics(synth_val_class_counts, indent="     ")

    # 2. weak_val (from weak_set valid split)
    valid_info["weak_val"] = {
        "audio_files": weak_audio_count,  # 同じフォルダ
        "samples": valid_weak_samples,
        "description": f"Weak validation ({(1 - weak_split) * 100:.0f}% of weak data)",
    }
    print("\n2. weak_val:")
    print(f"   Validation samples: {valid_weak_samples}")

    # 3. maestro_real_valid (from maestro_real_train valid split)
    valid_info["maestro_real_valid"] = {
        "audio_files": maestro_real_audio_count,  # 同じフォルダ
        "unique_files": maestro_valid_files,
        "samples": maestro_valid_samples,
        "description": f"MAESTRO real validation ({(1 - maestro_split) * 100:.0f}% split)",
    }
    print("\n3. maestro_real_valid:")
    print(f"   Validation samples: {maestro_valid_samples}")
    print(f"   Unique files: {maestro_valid_files}")

    # === Test データの調査 ===
    print("\n" + "=" * 80)
    print("TEST データ (train_pretrained.py line 320)")
    print("devtest_dataset = [desed_devtest_dataset, maestro_real_devtest]")
    print("=" * 80)

    test_info = {}

    # 1. desed_devtest_dataset (DESED test/validation)
    test_folder = convert_path(data_config["test_folder"], base_path)
    test_tsv = convert_path(data_config["test_tsv"], base_path)
    test_audio_count = count_audio_files(test_folder)
    test_unique_files, test_tsv_count, test_df = count_tsv_unique_files(test_tsv)
    test_class_counts = count_class_events(test_df, "desed")
    test_info["desed_devtest"] = {
        "audio_files": test_audio_count,
        "unique_files": test_unique_files,
        "samples": test_tsv_count,
        "description": "DESED test/validation",
        "class_counts": test_class_counts,
    }
    print("\n1. desed_devtest_dataset:")
    print(f"   Audio files: {test_audio_count}")
    print(f"   Unique files: {test_unique_files}")
    print(f"   Test samples: {test_tsv_count}")
    print("   Class distribution (DESED 10 classes):")
    print_class_statistics(test_class_counts, indent="     ")

    # 2. maestro_real_devtest (MAESTRO real validation - 全体)
    maestro_val_folder = convert_path(data_config["real_maestro_val_folder"], base_path)
    maestro_val_tsv = convert_path(data_config["real_maestro_val_tsv"], base_path)
    maestro_val_audio_count = count_audio_files(maestro_val_folder)
    maestro_val_unique_files, maestro_val_tsv_count, maestro_val_df = count_tsv_unique_files(
        maestro_val_tsv
    )
    maestro_val_class_counts = count_class_events(maestro_val_df, "maestro_real")
    test_info["maestro_real_devtest"] = {
        "audio_files": maestro_val_audio_count,
        "unique_files": maestro_val_unique_files,
        "samples": maestro_val_tsv_count,
        "description": "MAESTRO real validation (full)",
        "class_counts": maestro_val_class_counts,
    }
    print("\n2. maestro_real_devtest:")
    print(f"   Audio files: {maestro_val_audio_count}")
    print(f"   Unique files: {maestro_val_unique_files}")
    print(f"   Test samples: {maestro_val_tsv_count}")
    print("   Class distribution (MAESTRO Real 17 classes):")
    print_class_statistics(maestro_val_class_counts, indent="     ")

    # === Evaluation データの調査 ===
    print("\n" + "=" * 80)
    print("EVALUATION データ")
    print("=" * 80)

    eval_info = {}
    eval_folder = convert_path(data_config["eval_folder"], base_path)
    eval_audio_count = count_audio_files(eval_folder)
    eval_info["eval"] = {
        "audio_files": eval_audio_count,
        "description": "Evaluation set (unlabeled)",
    }
    print("\neval set:")
    print(f"   Audio files: {eval_audio_count}")

    # === サマリー出力 ===
    print("\n" + "=" * 80)
    print("サマリー")
    print("=" * 80)

    print("\n【TRAINING データ (batch_size = [12, 6, 6, 12, 24])】")
    print(f"  Index 0 - maestro_real_train:  {maestro_train_samples} samples (batch_size: 12)")
    print(f"           └─ Unique files:      {maestro_train_files}")
    print(f"  Index 1 - synth_set:            {synth_tsv_count} samples (batch_size: 6)")
    print(f"  Index 2 - strong_full_set:      {strong_full_samples} samples (batch_size: 6)")
    print(f"           └─ strong_set:         {strong_tsv_count}")
    print(f"           └─ synth_set:          {synth_tsv_count}")
    print(f"  Index 3 - weak_set:             {train_weak_samples} samples (batch_size: 12)")
    print(f"  Index 4 - unlabeled_set:        {unlabeled_audio_count} audio files (batch_size: 24)")

    # Total samples (synthが2回カウントされないように調整)
    train_total_samples = (
        maestro_train_samples
        + synth_tsv_count
        + strong_tsv_count
        + train_weak_samples
        + unlabeled_audio_count
    )
    print(f"\n  Total training samples: {train_total_samples}")

    print("\n【VALIDATION データ】")
    print(f"  synth_val:            {synth_val_tsv_count} samples")
    print(f"  weak_val:             {valid_weak_samples} samples")
    print(f"  maestro_real_valid:   {maestro_valid_samples} samples")
    print(f"                        ({maestro_valid_files} unique files)")
    val_total = synth_val_tsv_count + valid_weak_samples + maestro_valid_samples
    print(f"\n  Total validation samples: {val_total}")

    print("\n【TEST データ】")
    print(f"  desed_devtest:        {test_tsv_count} samples")
    print(f"  maestro_real_devtest: {maestro_val_tsv_count} samples")
    test_total = test_tsv_count + maestro_val_tsv_count
    print(f"\n  Total test samples: {test_total}")

    print("\n【EVALUATION データ】")
    print(f"  eval set:             {eval_audio_count} audio files")

    print("\n" + "=" * 80)
    print(
        f"総計（サンプル数ベース）: {train_total_samples + val_total + test_total + eval_audio_count}"
    )
    print("=" * 80)

    # 詳細情報を保存
    print("\n詳細情報をdataset_count_detailed.txtに保存しました。")
    with open("dataset_count_detailed.txt", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("DCASE 2024 Task 4 データセット詳細情報\n")
        f.write("train_pretrained.pyのロジックに基づく実際の使用データ数\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"設定ファイル: {config_path}\n")
        if base_path is None:
            f.write("環境: SSH (confs/pretrained.yamlのパスをそのまま使用)\n")
        else:
            f.write(f"環境: ローカル (データベースパス: {base_path})\n")
        f.write(f"Weak split: {config['training']['weak_split']}\n")
        f.write(f"MAESTRO split: {config['training']['maestro_split']}\n")
        f.write(f"Seed: {config['training']['seed']}\n\n")

        f.write("【TRAINING データ】\n")
        for key, info in train_info.items():
            f.write(f"\n{key}:\n")
            for k, v in info.items():
                if k == "class_counts" and v:
                    f.write(f"  {k}:\n")
                    for cls, count in v.items():
                        if count > 0:
                            f.write(f"    {cls}: {count}\n")
                else:
                    f.write(f"  {k}: {v}\n")

        f.write("\n【VALIDATION データ】\n")
        for key, info in valid_info.items():
            f.write(f"\n{key}:\n")
            for k, v in info.items():
                if k == "class_counts" and v:
                    f.write(f"  {k}:\n")
                    for cls, count in v.items():
                        if count > 0:
                            f.write(f"    {cls}: {count}\n")
                else:
                    f.write(f"  {k}: {v}\n")

        f.write("\n【TEST データ】\n")
        for key, info in test_info.items():
            f.write(f"\n{key}:\n")
            for k, v in info.items():
                if k == "class_counts" and v:
                    f.write(f"  {k}:\n")
                    for cls, count in v.items():
                        if count > 0:
                            f.write(f"    {cls}: {count}\n")
                else:
                    f.write(f"  {k}: {v}\n")

        f.write("\n【EVALUATION データ】\n")
        for key, info in eval_info.items():
            f.write(f"\n{key}:\n")
            for k, v in info.items():
                f.write(f"  {k}: {v}\n")


if __name__ == "__main__":
    main()
