#!/usr/bin/env python3
"""
データセット数調査スクリプト

train_pretrained.pyの実際のデータセット構築ロジックに基づいて、
学習/検証/評価に使用されているデータ数を調査します。
"""

import os
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple


def count_audio_files(folder_path: str) -> int:
    """指定されたフォルダ内のオーディオファイル数をカウント"""
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}")
        return 0

    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                count += 1
    return count


def count_tsv_unique_files(tsv_path: str) -> Tuple[int, pd.DataFrame]:
    """TSVファイルの一意のファイル数をカウントし、DataFrameを返す"""
    if not os.path.exists(tsv_path):
        print(f"Warning: TSV not found: {tsv_path}")
        return 0, pd.DataFrame()

    try:
        df = pd.read_csv(tsv_path, sep='\t')
        # 最初のカラムがファイル名と仮定
        if len(df.columns) > 0:
            filename_column = df.columns[0]
            unique_count = df[filename_column].nunique()
            return unique_count, df
        return 0, df
    except Exception as e:
        print(f"Warning: TSVファイル読み込みエラー ({tsv_path}): {e}")
        return 0, pd.DataFrame()


def split_maestro_count(maestro_df: pd.DataFrame, maestro_split: float, seed: int) -> Tuple[int, int]:
    """MAESTROデータのsplit後のtrain/valid数をカウント"""
    np.random.seed(seed)

    train_count = 0
    valid_count = 0

    for scene_name in ["cafe_restaurant", "city_center", "grocery_store", "metro_station", "residential_area"]:
        mask = maestro_df["filename"].apply(lambda x: "_".join(x.split("_")[:-1])) == scene_name
        filenames = maestro_df[mask]["filename"].apply(lambda x: x.split("-")[0]).unique()
        np.random.shuffle(filenames)

        pivot = int(maestro_split * len(filenames))
        filenames_train = filenames[:pivot]
        filenames_valid = filenames[pivot:]

        mask_train = maestro_df["filename"].apply(lambda x: x.split("-")[0]).isin(filenames_train)
        mask_valid = maestro_df["filename"].apply(lambda x: x.split("-")[0]).isin(filenames_valid)

        train_count += maestro_df[mask_train].shape[0]
        valid_count += maestro_df[mask_valid].shape[0]

    # 一意のファイル名をカウント
    train_files = 0
    valid_files = 0
    for scene_name in ["cafe_restaurant", "city_center", "grocery_store", "metro_station", "residential_area"]:
        mask = maestro_df["filename"].apply(lambda x: "_".join(x.split("_")[:-1])) == scene_name
        filenames = maestro_df[mask]["filename"].apply(lambda x: x.split("-")[0]).unique()
        np.random.shuffle(filenames)

        pivot = int(maestro_split * len(filenames))
        train_files += pivot
        valid_files += len(filenames) - pivot

    return train_files, valid_files


def convert_path(config_path: str, base_path: str) -> str:
    """設定ファイルのパスを実際のパスに変換"""
    return config_path.replace("/mnt/data/data/dcase/dataset/", f"{base_path}/")


def main():
    """メイン処理"""
    print("="*80)
    print("DCASE 2024 Task 4 データセット数調査")
    print("train_pretrained.pyのロジックに基づく実際の使用データ数")
    print("="*80)

    # 設定ファイルを読み込む
    config_path = "confs/pretrained.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # データベースパスを設定
    base_path = "../../../sed-fl/fl/data/dcase/dataset"

    print(f"\n設定ファイル: {config_path}")
    print(f"データベースパス: {base_path}")
    print(f"Weak split: {config['training']['weak_split']}")
    print(f"MAESTRO split: {config['training']['maestro_split']}")
    print(f"Seed: {config['training']['seed']}")

    data_config = config['data']

    # === Training データの調査 ===
    print("\n" + "="*80)
    print("TRAINING データ (train_pretrained.py line 471)")
    print("tot_train_data = [maestro_real_train, synth_set, strong_full_set, weak_set, unlabeled_set]")
    print("="*80)

    train_info = {}

    # 1. synth_set (Synthetic training)
    synth_folder = convert_path(data_config['synth_folder'], base_path)
    synth_tsv = convert_path(data_config['synth_tsv'], base_path)
    synth_audio_count = count_audio_files(synth_folder)
    synth_tsv_count, synth_df = count_tsv_unique_files(synth_tsv)
    train_info['synth_set'] = {
        'audio_files': synth_audio_count,
        'tsv_entries': synth_tsv_count,
        'description': 'Synthetic training (index 1 in tot_train_data)'
    }
    print(f"\n1. synth_set:")
    print(f"   Audio files: {synth_audio_count}")
    print(f"   TSV entries: {synth_tsv_count}")

    # 2. strong_set (Strong real)
    strong_folder = convert_path(data_config['strong_folder'], base_path)
    strong_tsv = convert_path(data_config['strong_tsv'], base_path)
    strong_audio_count = count_audio_files(strong_folder)
    strong_tsv_count, strong_df = count_tsv_unique_files(strong_tsv)
    train_info['strong_set'] = {
        'audio_files': strong_audio_count,
        'tsv_entries': strong_tsv_count,
        'description': 'Strong real (part of strong_full_set)'
    }
    print(f"\n2. strong_set (Strong real):")
    print(f"   Audio files: {strong_audio_count}")
    print(f"   TSV entries: {strong_tsv_count}")

    # 3. strong_full_set = strong_set + synth_set
    strong_full_count = strong_tsv_count + synth_tsv_count
    train_info['strong_full_set'] = {
        'audio_files': strong_audio_count + synth_audio_count,
        'tsv_entries': strong_full_count,
        'description': 'strong_set + synth_set (index 2 in tot_train_data)'
    }
    print(f"\n3. strong_full_set (strong_set + synth_set):")
    print(f"   Combined entries: {strong_full_count}")

    # 4. weak_set (Weak training) - split考慮
    weak_folder = convert_path(data_config['weak_folder'], base_path)
    weak_tsv = convert_path(data_config['weak_tsv'], base_path)
    weak_audio_count = count_audio_files(weak_folder)
    weak_tsv_count, weak_df = count_tsv_unique_files(weak_tsv)

    # weak_splitを適用
    weak_split = config['training']['weak_split']
    seed = config['training']['seed']
    np.random.seed(seed)
    train_weak_count = int(weak_tsv_count * weak_split)
    valid_weak_count = weak_tsv_count - train_weak_count

    train_info['weak_set'] = {
        'audio_files': weak_audio_count,
        'tsv_total': weak_tsv_count,
        'tsv_train': train_weak_count,
        'tsv_valid': valid_weak_count,
        'description': f'Weak training ({weak_split*100:.0f}% of weak data, index 3 in tot_train_data)'
    }
    print(f"\n4. weak_set (Weak training):")
    print(f"   Total audio files: {weak_audio_count}")
    print(f"   Total TSV entries: {weak_tsv_count}")
    print(f"   Train split ({weak_split*100:.0f}%): {train_weak_count}")
    print(f"   Valid split ({(1-weak_split)*100:.0f}%): {valid_weak_count}")

    # 5. unlabeled_set (Unlabeled)
    unlabeled_folder = convert_path(data_config['unlabeled_folder'], base_path)
    unlabeled_audio_count = count_audio_files(unlabeled_folder)
    train_info['unlabeled_set'] = {
        'audio_files': unlabeled_audio_count,
        'description': 'Unlabeled (index 4 in tot_train_data)'
    }
    print(f"\n5. unlabeled_set:")
    print(f"   Audio files: {unlabeled_audio_count}")

    # 6. maestro_real_train - split考慮
    maestro_real_folder = convert_path(data_config['real_maestro_train_folder'], base_path)
    maestro_real_tsv = convert_path(data_config['real_maestro_train_tsv'], base_path)
    maestro_real_audio_count = count_audio_files(maestro_real_folder)
    maestro_real_tsv_count, maestro_real_df = count_tsv_unique_files(maestro_real_tsv)

    # maestro_splitを適用
    maestro_split = config['training']['maestro_split']
    if not maestro_real_df.empty:
        maestro_train_files, maestro_valid_files = split_maestro_count(maestro_real_df, maestro_split, seed)
    else:
        maestro_train_files = int(maestro_real_tsv_count * maestro_split)
        maestro_valid_files = maestro_real_tsv_count - maestro_train_files

    train_info['maestro_real_train'] = {
        'audio_files': maestro_real_audio_count,
        'tsv_total': maestro_real_tsv_count,
        'tsv_train': maestro_train_files,
        'tsv_valid': maestro_valid_files,
        'description': f'MAESTRO real training ({maestro_split*100:.0f}% split, index 0 in tot_train_data)'
    }
    print(f"\n6. maestro_real_train (MAESTRO real):")
    print(f"   Total audio files: {maestro_real_audio_count}")
    print(f"   Total TSV entries: {maestro_real_tsv_count}")
    print(f"   Train split ({maestro_split*100:.0f}%): {maestro_train_files}")
    print(f"   Valid split ({(1-maestro_split)*100:.0f}%): {maestro_valid_files}")

    # === Validation データの調査 ===
    print("\n" + "="*80)
    print("VALIDATION データ (train_pretrained.py line 395)")
    print("valid_dataset = [synth_val, weak_val, maestro_real_valid]")
    print("="*80)

    valid_info = {}

    # 1. synth_val (Synthetic validation)
    synth_val_folder = convert_path(data_config['synth_val_folder'], base_path)
    synth_val_tsv = convert_path(data_config['synth_val_tsv'], base_path)
    synth_val_audio_count = count_audio_files(synth_val_folder)
    synth_val_tsv_count, _ = count_tsv_unique_files(synth_val_tsv)
    valid_info['synth_val'] = {
        'audio_files': synth_val_audio_count,
        'tsv_entries': synth_val_tsv_count,
        'description': 'Synthetic validation'
    }
    print(f"\n1. synth_val:")
    print(f"   Audio files: {synth_val_audio_count}")
    print(f"   TSV entries: {synth_val_tsv_count}")

    # 2. weak_val (from weak_set valid split)
    valid_info['weak_val'] = {
        'audio_files': weak_audio_count,  # 同じフォルダ
        'tsv_entries': valid_weak_count,
        'description': f'Weak validation ({(1-weak_split)*100:.0f}% of weak data)'
    }
    print(f"\n2. weak_val:")
    print(f"   TSV entries: {valid_weak_count}")

    # 3. maestro_real_valid (from maestro_real_train valid split)
    valid_info['maestro_real_valid'] = {
        'audio_files': maestro_real_audio_count,  # 同じフォルダ
        'tsv_entries': maestro_valid_files,
        'description': f'MAESTRO real validation ({(1-maestro_split)*100:.0f}% split)'
    }
    print(f"\n3. maestro_real_valid:")
    print(f"   TSV entries: {maestro_valid_files}")

    # === Test データの調査 ===
    print("\n" + "="*80)
    print("TEST データ (train_pretrained.py line 320)")
    print("devtest_dataset = [desed_devtest_dataset, maestro_real_devtest]")
    print("="*80)

    test_info = {}

    # 1. desed_devtest_dataset (DESED test/validation)
    test_folder = convert_path(data_config['test_folder'], base_path)
    test_tsv = convert_path(data_config['test_tsv'], base_path)
    test_audio_count = count_audio_files(test_folder)
    test_tsv_count, _ = count_tsv_unique_files(test_tsv)
    test_info['desed_devtest'] = {
        'audio_files': test_audio_count,
        'tsv_entries': test_tsv_count,
        'description': 'DESED test/validation'
    }
    print(f"\n1. desed_devtest_dataset:")
    print(f"   Audio files: {test_audio_count}")
    print(f"   TSV entries: {test_tsv_count}")

    # 2. maestro_real_devtest (MAESTRO real validation - 全体)
    maestro_val_folder = convert_path(data_config['real_maestro_val_folder'], base_path)
    maestro_val_tsv = convert_path(data_config['real_maestro_val_tsv'], base_path)
    maestro_val_audio_count = count_audio_files(maestro_val_folder)
    maestro_val_tsv_count, _ = count_tsv_unique_files(maestro_val_tsv)
    test_info['maestro_real_devtest'] = {
        'audio_files': maestro_val_audio_count,
        'tsv_entries': maestro_val_tsv_count,
        'description': 'MAESTRO real validation (full)'
    }
    print(f"\n2. maestro_real_devtest:")
    print(f"   Audio files: {maestro_val_audio_count}")
    print(f"   TSV entries: {maestro_val_tsv_count}")

    # === Evaluation データの調査 ===
    print("\n" + "="*80)
    print("EVALUATION データ")
    print("="*80)

    eval_info = {}
    eval_folder = convert_path(data_config['eval_folder'], base_path)
    eval_audio_count = count_audio_files(eval_folder)
    eval_info['eval'] = {
        'audio_files': eval_audio_count,
        'description': 'Evaluation set (unlabeled)'
    }
    print(f"\neval set:")
    print(f"   Audio files: {eval_audio_count}")

    # === サマリー出力 ===
    print("\n" + "="*80)
    print("サマリー")
    print("="*80)

    print("\n【TRAINING データ (batch_size = [12, 6, 6, 12, 24])】")
    print(f"  Index 0 - maestro_real_train:  {maestro_train_files} files (batch_size: 12)")
    print(f"  Index 1 - synth_set:            {synth_tsv_count} files (batch_size: 6)")
    print(f"  Index 2 - strong_full_set:      {strong_full_count} files (batch_size: 6)")
    print(f"           └─ strong_set:         {strong_tsv_count}")
    print(f"           └─ synth_set:          {synth_tsv_count}")
    print(f"  Index 3 - weak_set:             {train_weak_count} files (batch_size: 12)")
    print(f"  Index 4 - unlabeled_set:        {unlabeled_audio_count} files (batch_size: 24)")

    # Total unique files (synthが2回カウントされないように調整)
    train_total = maestro_train_files + synth_tsv_count + strong_tsv_count + train_weak_count + unlabeled_audio_count
    print(f"\n  Total unique training files: {train_total}")

    print("\n【VALIDATION データ】")
    print(f"  synth_val:            {synth_val_tsv_count} files")
    print(f"  weak_val:             {valid_weak_count} files")
    print(f"  maestro_real_valid:   {maestro_valid_files} files")
    val_total = synth_val_tsv_count + valid_weak_count + maestro_valid_files
    print(f"\n  Total validation files: {val_total}")

    print("\n【TEST データ】")
    print(f"  desed_devtest:        {test_tsv_count} files")
    print(f"  maestro_real_devtest: {maestro_val_tsv_count} files")
    test_total = test_tsv_count + maestro_val_tsv_count
    print(f"\n  Total test files: {test_total}")

    print("\n【EVALUATION データ】")
    print(f"  eval set:             {eval_audio_count} files")

    print("\n" + "="*80)
    print(f"総計: {train_total + val_total + test_total + eval_audio_count} files")
    print("="*80)

    # 詳細情報を保存
    print("\n詳細情報をdataset_count_detailed.txtに保存しました。")
    with open("dataset_count_detailed.txt", "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("DCASE 2024 Task 4 データセット詳細情報\n")
        f.write("train_pretrained.pyのロジックに基づく実際の使用データ数\n")
        f.write("="*80 + "\n\n")

        f.write(f"設定ファイル: {config_path}\n")
        f.write(f"データベースパス: {base_path}\n")
        f.write(f"Weak split: {config['training']['weak_split']}\n")
        f.write(f"MAESTRO split: {config['training']['maestro_split']}\n")
        f.write(f"Seed: {config['training']['seed']}\n\n")

        f.write("【TRAINING データ】\n")
        for key, info in train_info.items():
            f.write(f"\n{key}:\n")
            for k, v in info.items():
                f.write(f"  {k}: {v}\n")

        f.write("\n【VALIDATION データ】\n")
        for key, info in valid_info.items():
            f.write(f"\n{key}:\n")
            for k, v in info.items():
                f.write(f"  {k}: {v}\n")

        f.write("\n【TEST データ】\n")
        for key, info in test_info.items():
            f.write(f"\n{key}:\n")
            for k, v in info.items():
                f.write(f"  {k}: {v}\n")

        f.write("\n【EVALUATION データ】\n")
        for key, info in eval_info.items():
            f.write(f"\n{key}:\n")
            for k, v in info.items():
                f.write(f"  {k}: {v}\n")


if __name__ == "__main__":
    main()
