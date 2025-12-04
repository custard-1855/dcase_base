#!/usr/bin/env python3
"""
固定データセットの作成スクリプト

各クラスから指定数のサンプルをランダムサンプリングし、
サンプルインデックスをJSON形式で保存する。

使用法:
    python create_fixed_dataset.py \
        --conf_file confs/pretrained.yaml \
        --samples_per_class 500 \
        --output inference_configs/fixed_21classes_500per.json \
        --seed 42
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from local.classes_dict import (
    classes_labels_desed,
    classes_labels_maestro_real,
    classes_labels_maestro_real_eval,
    maestro_desed_alias,
)
from local.utils import process_tsvs
from desed_task.dataio.datasets import StronglyAnnotatedSet, UnlabeledSet
from desed_task.utils.encoder import CatManyHotEncoder, ManyHotEncoder


def get_encoder(config):
    """CatManyHotEncoderを作成"""
    desed_encoder = ManyHotEncoder(
        list(classes_labels_desed.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )
    maestro_real_encoder = ManyHotEncoder(
        list(classes_labels_maestro_real.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )
    encoder = CatManyHotEncoder((desed_encoder, maestro_real_encoder))
    return encoder


def split_maestro(config, maestro_dev_df):
    """MAESTROデータセットをtrainとvalidに分割"""
    np.random.seed(config["training"]["seed"])
    split_f = config["training"]["maestro_split"]

    for indx, scene_name in enumerate([
        "cafe_restaurant",
        "city_center",
        "grocery_store",
        "metro_station",
        "residential_area",
    ]):
        mask = (
            maestro_dev_df["filename"].apply(lambda x: "_".join(x.split("_")[:-1]))
            == scene_name
        )
        filenames = (
            maestro_dev_df[mask]["filename"].apply(lambda x: x.split("-")[0]).unique()
        )
        np.random.shuffle(filenames)

        pivot = int(split_f * len(filenames))
        filenames_train = filenames[:pivot]
        filenames_valid = filenames[pivot:]

        if indx == 0:
            mask_train = (
                maestro_dev_df["filename"]
                .apply(lambda x: x.split("-")[0])
                .isin(filenames_train)
            )
            mask_valid = (
                maestro_dev_df["filename"]
                .apply(lambda x: x.split("-")[0])
                .isin(filenames_valid)
            )
        else:
            mask_train = mask_train | (
                maestro_dev_df["filename"]
                .apply(lambda x: x.split("-")[0])
                .isin(filenames_train)
            )
            mask_valid = mask_valid | (
                maestro_dev_df["filename"]
                .apply(lambda x: x.split("-")[0])
                .isin(filenames_valid)
            )

    maestro_train_df = maestro_dev_df[mask_train].reset_index(drop=True)
    maestro_valid_df = maestro_dev_df[mask_valid].reset_index(drop=True)

    return maestro_train_df, maestro_valid_df


def count_samples_per_class(dataset, encoder, dataset_name):
    """データセット内の各クラスのサンプル数をカウント

    Noneサンプル（欠損ファイル）は自動的にスキップされる
    """
    class_counts = defaultdict(int)
    class_to_samples = defaultdict(list)
    skipped_count = 0

    print(f"\n[{dataset_name}] サンプル数のカウント中... (total: {len(dataset)})")

    for idx in range(len(dataset)):
        try:
            sample = dataset[idx]

            # 明示的なNoneチェック（欠損ファイル対応）
            if sample is None:
                skipped_count += 1
                continue

            # labels shape: (27, frames) or (27,)
            labels = sample[1]  # (mixture, labels, padded_indx, filename, ...)

            # フレームレベルラベルの場合は時間方向でOR
            if labels.ndim == 2:
                labels = labels.max(dim=1)[0]  # (27,)

            # 各クラスについてカウント
            for class_idx in range(labels.shape[0]):
                if labels[class_idx] > 0:
                    class_counts[class_idx] += 1
                    class_to_samples[class_idx].append(idx)
        except Exception as e:
            print(f"Warning: Error processing index {idx}: {e}")
            skipped_count += 1
            continue

    # スキップ情報を表示
    valid_count = len(dataset) - skipped_count
    print(f"  有効サンプル: {valid_count} / {len(dataset)} ({skipped_count} skipped)")

    return class_counts, class_to_samples


def sample_fixed_subset(class_to_samples, samples_per_class, seed, class_names):
    """各クラスから固定数のサンプルをランダムサンプリング

    データが不足する場合は、利用可能数の半分を取得（フォールバック）
    """
    np.random.seed(seed)
    selected_indices = []
    class_distribution = {}

    for class_idx, sample_indices in class_to_samples.items():
        class_name = class_names[class_idx]
        available = len(sample_indices)

        # データ不足の場合はフォールバック: available // 2
        if available < samples_per_class:
            target = available // 2
            print(f"  {class_name}: 不足のため {target}/{available} サンプル選択 (目標: {samples_per_class})")
        else:
            target = samples_per_class
            print(f"  {class_name}: {target}/{available} サンプル選択")

        # ランダムサンプリング
        if target > 0:
            sampled = np.random.choice(sample_indices, size=target, replace=False)
            selected_indices.extend(sampled.tolist())
            class_distribution[class_name] = target

    # 重複を除去
    selected_indices = sorted(list(set(selected_indices)))

    return selected_indices, class_distribution


def generate_reports(result, output_path, samples_per_class):
    """CSV とMarkdown形式でレポートを生成"""
    output_path = Path(output_path)
    base_name = output_path.stem
    output_dir = output_path.parent

    csv_path = output_dir / f"{base_name}_report.csv"
    md_path = output_dir / f"{base_name}_report.md"

    # クラス名リスト
    desed_classes = result["desed_classes"]
    maestro_classes = result["maestro_classes"]
    all_classes = desed_classes + maestro_classes

    # データセット情報の取得
    datasets = result["datasets"]

    # クラス別データの集計
    class_data = {}
    for class_name in all_classes:
        class_data[class_name] = {
            "desed_validation": 0,
            "desed_unlabeled": 0,
            "maestro_training": 0,
            "maestro_validation": 0,
            "fallback": False
        }

    # 各データセットからクラス分布を集計
    for dataset_name, dataset_info in datasets.items():
        if "class_distribution" in dataset_info:
            for class_name, count in dataset_info["class_distribution"].items():
                if class_name in class_data:
                    class_data[class_name][dataset_name] = count
                    # フォールバック判定（目標数に達していない）
                    if count < samples_per_class:
                        class_data[class_name]["fallback"] = True

    # ===== CSV生成 =====
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_name",
            "desed_validation",
            "desed_unlabeled",
            "maestro_training",
            "maestro_validation",
            "total",
            "fallback"
        ])

        for class_name in all_classes:
            data = class_data[class_name]
            total = (data["desed_validation"] + data["desed_unlabeled"] +
                    data["maestro_training"] + data["maestro_validation"])
            writer.writerow([
                class_name,
                data["desed_validation"] if data["desed_validation"] > 0 else "",
                data["desed_unlabeled"] if data["desed_unlabeled"] > 0 else "",
                data["maestro_training"] if data["maestro_training"] > 0 else "",
                data["maestro_validation"] if data["maestro_validation"] > 0 else "",
                total,
                "Yes" if data["fallback"] else "No"
            ])

    # ===== Markdown生成 =====
    with open(md_path, "w") as f:
        f.write("# データセット固定化レポート\n\n")

        # 設定情報
        f.write("## 設定\n\n")
        f.write(f"- サンプル数/クラス: {samples_per_class}\n")
        f.write(f"- ランダムシード: {result['seed']}\n")
        f.write(f"- 生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 設定ファイル: {result['config_path']}\n\n")

        # データセット別サマリー
        f.write("## データセット別サマリー\n\n")
        f.write("| Dataset | Total Samples | Target per Class | Classes |\n")
        f.write("|---------|--------------|------------------|--------|\n")

        total_samples_all = 0
        for dataset_name in ["desed_validation", "desed_unlabeled", "maestro_training", "maestro_validation"]:
            if dataset_name in datasets:
                dataset_info = datasets[dataset_name]
                total_samples = dataset_info["total_samples"]
                total_samples_all += total_samples

                display_name = dataset_name.replace("_", " ").title()
                num_classes = len(desed_classes) if "desed" in dataset_name else len(maestro_classes)

                # フォールバックがあるか確認
                has_fallback = any(
                    count < samples_per_class
                    for count in dataset_info.get("class_distribution", {}).values()
                )
                target_note = f"{samples_per_class}" + (" (with fallback)" if has_fallback else "")

                f.write(f"| {display_name} | {total_samples} | {target_note} | {num_classes} |\n")

        f.write(f"| **Total** | **{total_samples_all}** | - | **{len(all_classes)}** |\n\n")

        # フォールバック発生クラス
        fallback_classes = [(name, data) for name, data in class_data.items() if data["fallback"]]
        if fallback_classes:
            f.write("## フォールバック発生クラス\n\n")
            f.write("| Class | Dataset | Target | Actual |\n")
            f.write("|-------|---------|--------|--------|\n")

            for class_name, data in fallback_classes:
                for ds_name in ["desed_validation", "desed_unlabeled", "maestro_training", "maestro_validation"]:
                    if data[ds_name] > 0 and data[ds_name] < samples_per_class:
                        display_ds = ds_name.replace("_", " ").title()
                        f.write(f"| {class_name} | {display_ds} | {samples_per_class} | {data[ds_name]} |\n")
            f.write("\n")

        # DESED詳細
        f.write("## クラス別詳細\n\n")
        f.write("### DESED (10 classes)\n\n")
        f.write("| Class | Validation | Unlabeled | Total |\n")
        f.write("|-------|-----------|-----------|-------|\n")

        for class_name in desed_classes:
            data = class_data[class_name]
            total = data["desed_validation"] + data["desed_unlabeled"]
            f.write(f"| {class_name} | {data['desed_validation']} | {data['desed_unlabeled']} | {total} |\n")

        f.write("\n")

        # MAESTRO詳細
        f.write("### MAESTRO (11 classes)\n\n")
        f.write("| Class | Training | Validation | Total | Fallback |\n")
        f.write("|-------|----------|-----------|-------|----------|\n")

        for class_name in maestro_classes:
            data = class_data[class_name]
            total = data["maestro_training"] + data["maestro_validation"]
            fallback = "Yes" if data["fallback"] else "No"
            f.write(f"| {class_name} | {data['maestro_training']} | {data['maestro_validation']} | {total} | {fallback} |\n")

    return csv_path, md_path


def main():
    parser = argparse.ArgumentParser(description="固定データセットの作成")
    parser.add_argument(
        "--conf_file",
        default="confs/pretrained.yaml",
        help="設定ファイルのパス"
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=500,
        help="各クラスから取得するサンプル数"
    )
    parser.add_argument(
        "--output",
        default="inference_configs/fixed_21classes_500per.json",
        help="出力JSONファイルのパス"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="ランダムシード"
    )
    args = parser.parse_args()

    # 出力ディレクトリの作成
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 設定ファイルの読み込み
    with open(args.conf_file, "r") as f:
        config = yaml.safe_load(f)

    print(f"設定ファイル: {args.conf_file}")
    print(f"サンプル数/クラス: {args.samples_per_class}")
    print(f"ランダムシード: {args.seed}")

    # Encoderの準備
    encoder = get_encoder(config)

    # クラス名のマッピング
    all_class_names = list(classes_labels_desed.keys()) + list(classes_labels_maestro_real.keys())
    desed_class_names = list(classes_labels_desed.keys())
    maestro_eval_classes = list(classes_labels_maestro_real_eval)

    print(f"\n評価対象クラス: {len(desed_class_names)} (DESED) + {len(maestro_eval_classes)} (MAESTRO) = {len(desed_class_names) + len(maestro_eval_classes)}")

    result = {
        "seed": args.seed,
        "samples_per_class": args.samples_per_class,
        "config_path": args.conf_file,
        "total_classes": len(desed_class_names) + len(maestro_eval_classes),
        "desed_classes": desed_class_names,
        "maestro_classes": maestro_eval_classes,
        "datasets": {}
    }

    # ===== 1. DESED Validation (synth_val) =====
    print("\n" + "="*60)
    print("DESED Validation (synth_val) の処理")
    print("="*60)

    synth_df_val = pd.read_csv(config["data"]["synth_val_tsv"], sep="\t")
    synth_val_dataset = StronglyAnnotatedSet(
        config["data"]["synth_val_folder"],
        synth_df_val,
        encoder,
        pad_to=config["data"]["audio_max_len"],
        return_filename=True,
    )

    counts, class_samples = count_samples_per_class(synth_val_dataset, encoder, "DESED Validation")

    # DESEDクラス（0-9）のみを対象
    desed_class_samples = {k: v for k, v in class_samples.items() if k < 10}
    selected_indices, class_dist = sample_fixed_subset(
        desed_class_samples, args.samples_per_class, args.seed, all_class_names
    )

    result["datasets"]["desed_validation"] = {
        "total_samples": len(selected_indices),
        "has_targets": True,
        "sample_indices": selected_indices,
        "class_distribution": class_dist
    }

    print(f"選択されたサンプル数: {len(selected_indices)}")

    # ===== 2. DESED Unlabeled =====
    print("\n" + "="*60)
    print("DESED Unlabeled の処理")
    print("="*60)

    unlabeled_dataset = UnlabeledSet(
        config["data"]["unlabeled_folder"],
        encoder,
        pad_to=config["data"]["audio_max_len"],
    )

    # Unlabeledはラベルがないため、全サンプルから均等にサンプリング
    total_unlabeled = len(unlabeled_dataset)
    target_per_class = args.samples_per_class
    total_target = target_per_class * len(desed_class_names)
    total_target = min(total_target, total_unlabeled)

    np.random.seed(args.seed)
    unlabeled_indices = np.random.choice(
        total_unlabeled,
        size=total_target,
        replace=False
    ).tolist()

    result["datasets"]["desed_unlabeled"] = {
        "total_samples": len(unlabeled_indices),
        "has_targets": False,
        "sample_indices": unlabeled_indices,
    }

    print(f"選択されたサンプル数: {len(unlabeled_indices)} / {total_unlabeled}")

    # ===== 3. MAESTRO Training + Validation =====
    print("\n" + "="*60)
    print("MAESTRO Training + Validation の処理")
    print("="*60)

    maestro_real_train_df = pd.read_csv(config["data"]["real_maestro_train_tsv"], sep="\t")
    maestro_train_df, maestro_valid_df = split_maestro(config, maestro_real_train_df)

    # MAESTRO Training
    print("\n[MAESTRO Training]")
    maestro_train_processed = process_tsvs(maestro_train_df, alias_map=maestro_desed_alias)
    maestro_train_dataset = StronglyAnnotatedSet(
        config["data"]["real_maestro_train_folder"],
        maestro_train_processed,
        encoder,
        pad_to=config["data"]["audio_max_len"],
        return_filename=True,
    )

    counts_train, class_samples_train = count_samples_per_class(
        maestro_train_dataset, encoder, "MAESTRO Training"
    )

    # MAESTRO Validation
    print("\n[MAESTRO Validation]")
    maestro_valid_dataset = StronglyAnnotatedSet(
        config["data"]["real_maestro_train_folder"],
        maestro_valid_df,
        encoder,
        pad_to=config["data"]["audio_max_len"],
        return_filename=True,
    )

    counts_valid, class_samples_valid = count_samples_per_class(
        maestro_valid_dataset, encoder, "MAESTRO Validation"
    )

    # MAESTROクラス（10-26）かつevalクラスのみを対象
    maestro_class_indices = []
    for class_name in maestro_eval_classes:
        if class_name in classes_labels_maestro_real:
            class_idx = 10 + classes_labels_maestro_real[class_name]  # offset 10
            maestro_class_indices.append(class_idx)

    # 各データセットから samples_per_class ずつサンプリング
    # Training
    maestro_train_class_samples = {k: v for k, v in class_samples_train.items() if k in maestro_class_indices}
    train_selected_indices, train_class_dist = sample_fixed_subset(
        maestro_train_class_samples, args.samples_per_class, args.seed, all_class_names
    )

    result["datasets"]["maestro_training"] = {
        "total_samples": len(train_selected_indices),
        "has_targets": True,
        "sample_indices": train_selected_indices,
        "class_distribution": train_class_dist
    }

    # Validation
    maestro_valid_class_samples = {k: v for k, v in class_samples_valid.items() if k in maestro_class_indices}
    valid_selected_indices, valid_class_dist = sample_fixed_subset(
        maestro_valid_class_samples, args.samples_per_class, args.seed + 1, all_class_names
    )

    result["datasets"]["maestro_validation"] = {
        "total_samples": len(valid_selected_indices),
        "has_targets": True,
        "sample_indices": valid_selected_indices,
        "class_distribution": valid_class_dist
    }

    # 全体のクラス分布を集計
    overall_class_dist = defaultdict(int)
    for dataset_info in result["datasets"].values():
        if "class_distribution" in dataset_info:
            for class_name, count in dataset_info["class_distribution"].items():
                overall_class_dist[class_name] += count

    result["overall_class_distribution"] = dict(overall_class_dist)

    # JSON保存
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    # レポート生成（CSV, Markdown）
    csv_path, md_path = generate_reports(result, args.output, args.samples_per_class)

    print("\n" + "="*60)
    print("完了")
    print("="*60)
    print(f"出力ファイル: {args.output}")
    print(f"レポート (CSV): {csv_path}")
    print(f"レポート (MD): {md_path}")
    print(f"\n全体のクラス分布:")
    for class_name, count in sorted(overall_class_dist.items()):
        print(f"  {class_name}: {count}")

    total_samples = sum(d["total_samples"] for d in result["datasets"].values())
    print(f"\n合計サンプル数: {total_samples}")


if __name__ == "__main__":
    main()
