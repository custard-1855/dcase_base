#!/usr/bin/env python3
"""
固定データセットから推論を実行し、特徴量を抽出するスクリプト

使用法:
    python extract_inference_features.py \
        --checkpoint exp/baseline/version_0/best.ckpt \
        --dataset_config inference_configs/fixed_21classes_500per.json \
        --output_dir inference_outputs/baseline
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from local.classes_dict import (
    classes_labels_desed,
    classes_labels_maestro_real,
    maestro_desed_alias,
)
from local.sed_trainer_pretrained import SEDTask4
from local.utils import process_tsvs
from desed_task.dataio.datasets import StronglyAnnotatedSet, UnlabeledSet
from desed_task.nnet.CRNN import CRNN
from desed_task.utils.encoder import CatManyHotEncoder, ManyHotEncoder


def get_embeddings_name(config, name):
    """埋め込みファイルのパスを取得"""
    embeddings_path = (
        None
        if config.get("pretrained", {}).get("e2e", False)
        else os.path.join(
            config["pretrained"]["extracted_embeddings_dir"],
            config["pretrained"]["model"],
            f"{name}.hdf5",
        )
    )
    return embeddings_path


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


def create_datasets(config, encoder, dataset_config):
    """JSON定義に基づいてデータセットを作成"""
    datasets = {}

    # クラスマスクの準備
    mask_events_desed = set(classes_labels_desed.keys())
    mask_events_maestro_real = set(classes_labels_maestro_real.keys()).union(
        set(["Speech", "Dog", "Dishes"])
    )

    # ===== DESED Validation =====
    synth_df_val = pd.read_csv(config["data"]["synth_val_tsv"], sep="\t")
    synth_val_full = StronglyAnnotatedSet(
        config["data"]["synth_val_folder"],
        synth_df_val,
        encoder,
        pad_to=config["data"]["audio_max_len"],
        return_filename=True,
        embeddings_hdf5_file=get_embeddings_name(config, "synth_val"),
        embedding_type=config["net"]["embedding_type"],
        mask_events_other_than=mask_events_desed,
    )

    indices = dataset_config["datasets"]["desed_validation"]["sample_indices"]
    datasets["desed_validation"] = torch.utils.data.Subset(synth_val_full, indices)
    print(f"DESED Validation: {len(datasets['desed_validation'])} サンプル")

    # ===== DESED Unlabeled =====
    unlabeled_full = UnlabeledSet(
        config["data"]["unlabeled_folder"],
        encoder,
        pad_to=config["data"]["audio_max_len"],
        return_filename=True,
        embeddings_hdf5_file=get_embeddings_name(config, "unlabeled_train"),
        embedding_type=config["net"]["embedding_type"],
        mask_events_other_than=mask_events_desed,
    )

    indices = dataset_config["datasets"]["desed_unlabeled"]["sample_indices"]
    datasets["desed_unlabeled"] = torch.utils.data.Subset(unlabeled_full, indices)
    print(f"DESED Unlabeled: {len(datasets['desed_unlabeled'])} サンプル")

    # ===== MAESTRO Training + Validation =====
    maestro_real_train_df = pd.read_csv(config["data"]["real_maestro_train_tsv"], sep="\t")
    maestro_train_df, maestro_valid_df = split_maestro(config, maestro_real_train_df)

    # Training
    maestro_train_processed = process_tsvs(maestro_train_df, alias_map=maestro_desed_alias)
    maestro_train_full = StronglyAnnotatedSet(
        config["data"]["real_maestro_train_folder"],
        maestro_train_processed,
        encoder,
        pad_to=config["data"]["audio_max_len"],
        return_filename=True,
        embeddings_hdf5_file=get_embeddings_name(config, "maestro_real_train"),
        embedding_type=config["net"]["embedding_type"],
        mask_events_other_than=mask_events_maestro_real,
    )

    indices = dataset_config["datasets"]["maestro_training"]["sample_indices"]
    datasets["maestro_training"] = torch.utils.data.Subset(maestro_train_full, indices)
    print(f"MAESTRO Training: {len(datasets['maestro_training'])} サンプル")

    # Validation
    maestro_valid_full = StronglyAnnotatedSet(
        config["data"]["real_maestro_train_folder"],
        maestro_valid_df,
        encoder,
        pad_to=config["data"]["audio_max_len"],
        return_filename=True,
        embeddings_hdf5_file=get_embeddings_name(config, "maestro_real_train"),
        embedding_type=config["net"]["embedding_type"],
        mask_events_other_than=mask_events_maestro_real,
    )

    indices = dataset_config["datasets"]["maestro_validation"]["sample_indices"]
    datasets["maestro_validation"] = torch.utils.data.Subset(maestro_valid_full, indices)
    print(f"MAESTRO Validation: {len(datasets['maestro_validation'])} サンプル")

    return datasets


def extract_features_from_dataset(
    model,
    dataset,
    dataset_name,
    has_targets,
    device,
    batch_size=16
):
    """データセットから特徴量を抽出"""
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    # バッファ
    all_features_student = []
    all_features_teacher = []
    all_probs_student = []
    all_probs_teacher = []
    all_targets = [] if has_targets else None
    all_filenames = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"推論中 [{dataset_name}]"):
            # バッチのアンパック
            # 構造: [audio, labels, padded_indx, filename, embeddings, valid_class_mask]
            audio = batch[0].to(device)
            labels = batch[1] if has_targets else None
            padded_indx = batch[2]  # 使用しないが、インデックスを維持するため取得
            filenames = batch[3]  # return_filename=True なので常に存在
            embeddings = batch[4]  # embeddings_hdf5_file が設定されているため常に存在
            mask = batch[5]  # mask_events_other_than が設定されているため常に存在

            # Mel spectrogram計算
            mels = model.mel_spec(audio)

            # 前処理
            mels_preprocessed = model.scaler(model.take_log(mels))

            # GPU/CPUに転送
            embeddings = embeddings.to(device)
            mask = mask.to(device)

            # Student推論
            out_student = model.sed_student(
                mels_preprocessed,
                embeddings=embeddings,
                classes_mask=mask,
                return_features=True
            )

            # Teacher推論
            out_teacher = model.sed_teacher(
                mels_preprocessed,
                embeddings=embeddings,
                classes_mask=mask,
                return_features=True
            )

            # 特徴量を時間平均（RNN出力: (batch, frames, 384) -> (batch, 384)）
            features_student = out_student['features'].mean(dim=1).cpu().numpy()
            features_teacher = out_teacher['features'].mean(dim=1).cpu().numpy()

            # Weak probs（attention pooling済み: (batch, 27)）
            probs_student = out_student['weak_probs'].cpu().numpy()
            probs_teacher = out_teacher['weak_probs'].cpu().numpy()

            # バッファに追加
            all_features_student.append(features_student)
            all_features_teacher.append(features_teacher)
            all_probs_student.append(probs_student)
            all_probs_teacher.append(probs_teacher)

            if has_targets and labels is not None:
                # フレームレベルラベルの場合は時間方向で最大値
                if labels.ndim == 3:  # (batch, 27, frames)
                    labels = labels.max(dim=2)[0]  # (batch, 27)
                all_targets.append(labels.cpu().numpy())

            if filenames is not None:
                all_filenames.extend(filenames)

    # 連結
    features_student = np.concatenate(all_features_student, axis=0)
    features_teacher = np.concatenate(all_features_teacher, axis=0)
    probs_student = np.concatenate(all_probs_student, axis=0)
    probs_teacher = np.concatenate(all_probs_teacher, axis=0)


    result = {
        'features_student': features_student,
        'features_teacher': features_teacher,
        'probs_student': probs_student,
        'probs_teacher': probs_teacher,
        'filenames': np.array(all_filenames) if all_filenames else None,
    }

    if has_targets:
        result['targets'] = np.concatenate(all_targets, axis=0)

    return result


def main():
    parser = argparse.ArgumentParser(description="固定データセットから特徴量を抽出")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="チェックポイントファイルのパス"
    )
    parser.add_argument(
        "--dataset_config",
        required=True,
        help="固定データセット定義JSONのパス"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="出力ディレクトリ"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="バッチサイズ"
    )
    parser.add_argument(
        "--device",
        default=str("cuda" if torch.cuda.is_available() else "cpu"),
        help="デバイス (cuda/cpu)"
    )
    args = parser.parse_args()

    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"チェックポイント: {args.checkpoint}")
    print(f"データセット定義: {args.dataset_config}")
    print(f"出力ディレクトリ: {args.output_dir}")
    print(f"デバイス: {args.device}")

    # JSON定義の読み込み
    with open(args.dataset_config, "r") as f:
        dataset_config = json.load(f)

    # Checkpointからハイパーパラメータとstate_dictを抽出
    print("\nCheckpointをロード中...")
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location=args.device)
    config = checkpoint["hyper_parameters"]
    state_dict = checkpoint["state_dict"]
    print(f"Checkpoint情報: epoch={checkpoint.get('epoch', 'N/A')}")

    # データパス情報を上書き (dataset_config["config_path"]から)
    config_path = dataset_config["config_path"]
    with open(config_path, "r") as f:
        data_config = yaml.safe_load(f)
    config["data"] = data_config["data"]
    print(f"データパス設定を {config_path} から読み込み")

    # Encoderの準備
    encoder = get_encoder(config)

    # モデルインスタンスを作成
    print("\nモデルを作成中...")
    sed_student = CRNN(**config["net"])

    # SEDTask4インスタンスを作成
    model = SEDTask4(
        config,
        encoder=encoder,
        sed_student=sed_student,
        pretrained_model=None,  # 推論時は事前計算された埋め込みを使用
        opt=None,
        train_data=None,
        valid_data=None,
        test_data=None,
        train_sampler=None,
        scheduler=None,
        fast_dev_run=False,
        evaluation=True  # 推論モード
    )

    # State dictをロード
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()
    print("モデルのロード完了")

    # データセットの作成
    print("\nデータセットを作成中...")
    datasets = create_datasets(config, encoder, dataset_config)

    # メタデータ
    metadata = {
        "checkpoint": args.checkpoint,
        "dataset_config": args.dataset_config,
        "config_path": config_path,
        "timestamp": datetime.now().isoformat(),
        "device": args.device,
        "batch_size": args.batch_size,
    }

    # 各データセットから特徴量を抽出
    for dataset_name, dataset in datasets.items():
        print(f"\n{'='*60}")
        print(f"処理中: {dataset_name}")
        print(f"{'='*60}")

        has_targets = dataset_config["datasets"][dataset_name]["has_targets"]

        result = extract_features_from_dataset(
            model,
            dataset,
            dataset_name,
            has_targets,
            args.device,
            args.batch_size
        )

        # .npz形式で保存
        output_path = output_dir / f"{dataset_name}.npz"
        np.savez_compressed(output_path, **result)

        print(f"保存完了: {output_path}")
        print(f"  - features_student: {result['features_student'].shape}")
        print(f"  - features_teacher: {result['features_teacher'].shape}")
        print(f"  - probs_student: {result['probs_student'].shape}")
        print(f"  - probs_teacher: {result['probs_teacher'].shape}")
        if has_targets:
            print(f"  - targets: {result['targets'].shape}")

    # メタデータの保存
    metadata_path = output_dir / "inference_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print("完了")
    print(f"{'='*60}")
    print(f"出力ディレクトリ: {args.output_dir}")


if __name__ == "__main__":
    main()
