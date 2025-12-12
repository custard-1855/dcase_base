#!/usr/bin/env python3
"""特徴量抽出スクリプト

モデルのチェックポイントから特徴量を抽出し、UMAP可視化用のデータを生成します。

使用法:
    python extract_features.py \
        --checkpoints exp1/checkpoint.ckpt exp2/checkpoint.ckpt \
        --config confs/pretrained.yaml \
        --output_dir features_output \
        --model_names model1 model2
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from desed_task.dataio.datasets import StronglyAnnotatedSet, WeakSet
from desed_task.nnet.CRNN import CRNN
from desed_task.utils.encoder import CatManyHotEncoder, ManyHotEncoder
from local.classes_dict import classes_labels_desed, classes_labels_maestro_real
from local.sed_trainer_pretrained import SEDTask4
from torch.utils.data import DataLoader
from tqdm import tqdm


class FeatureExtractor:
    """特徴量抽出クラス"""

    def __init__(self, config: dict, device: str = "cpu"):
        """初期化

        Args:
            config: モデル設定辞書
            device: 使用デバイス
        """
        self.config = config
        self.device = device
        self.encoder = self._create_encoder()

    def _create_encoder(self) -> CatManyHotEncoder:
        """CatManyHotEncoderを作成"""
        desed_encoder = ManyHotEncoder(
            list(classes_labels_desed.keys()),
            audio_len=self.config["data"]["audio_max_len"],
            frame_len=self.config["feats"]["n_filters"],
            frame_hop=self.config["feats"]["hop_length"],
            net_pooling=self.config["data"]["net_subsample"],
            fs=self.config["data"]["fs"],
        )
        maestro_real_encoder = ManyHotEncoder(
            list(classes_labels_maestro_real.keys()),
            audio_len=self.config["data"]["audio_max_len"],
            frame_len=self.config["feats"]["n_filters"],
            frame_hop=self.config["feats"]["hop_length"],
            net_pooling=self.config["data"]["net_subsample"],
            fs=self.config["data"]["fs"],
        )
        return CatManyHotEncoder((desed_encoder, maestro_real_encoder))

    def load_model(self, checkpoint_path: Path) -> SEDTask4:
        """モデルをロード

        Args:
            checkpoint_path: チェックポイントファイルパス

        Returns:
            ロードされたSEDTask4モデル
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"  モデルをロード中: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=self.device)
        state_dict = checkpoint["state_dict"]

        # モデルインスタンスを作成
        sed_student = CRNN(**self.config["net"])

        # SEDTask4インスタンスを作成
        model = SEDTask4(
            self.config,
            encoder=self.encoder,
            sed_student=sed_student,
            pretrained_model=None,
            opt=None,
            train_data=None,
            valid_data=None,
            test_data=None,
            train_sampler=None,
            scheduler=None,
            fast_dev_run=False,
            evaluation=True,
        )

        # State dictをロード
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        return model

    def create_dataloader(
        self,
        dataset_type: str = "validation",
        batch_size: int = 32,
    ) -> DataLoader:
        """データローダーを作成

        Args:
            dataset_type: データセットタイプ ('validation' or 'test')
            batch_size: バッチサイズ

        Returns:
            DataLoader
        """
        if dataset_type == "validation":
            # DESED Validation (Synthetic)
            dataset = StronglyAnnotatedSet(
                audio_folder=self.config["data"]["synth_val_folder"],
                tsv_entries=self.config["data"]["synth_val_tsv"],
                encoder=self.encoder,
                return_filename=True,
                pad_to=self.config["data"]["audio_max_len"],
            )
        elif dataset_type == "maestro_validation":
            # MAESTRO Real Validation
            dataset = StronglyAnnotatedSet(
                audio_folder=self.config["data"]["real_maestro_val_folder"],
                tsv_entries=self.config["data"]["real_maestro_val_tsv"],
                encoder=self.encoder,
                return_filename=True,
                pad_to=self.config["data"]["audio_max_len"],
            )
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config["training"]["num_workers"],
        )

    @torch.no_grad()
    def extract_features(
        self,
        model: SEDTask4,
        dataloader: DataLoader,
        use_teacher: bool = False,
    ) -> Dict[str, np.ndarray]:
        """特徴量を抽出

        Args:
            model: SEDTask4モデル
            dataloader: データローダー
            use_teacher: Teacherモデルを使用するか

        Returns:
            特徴量辞書
                - features_student: 生徒モデルの特徴量 (N, 384)
                - features_teacher: 教師モデルの特徴量 (N, 384)
                - probs_student: 生徒モデルの予測確率 (N, 27)
                - probs_teacher: 教師モデルの予測確率 (N, 27)
                - filenames: ファイル名のリスト (N,)
                - targets: 正解ラベル (N, 27)
        """
        all_features_student = []
        all_features_teacher = []
        all_probs_student = []
        all_probs_teacher = []
        all_filenames = []
        all_targets = []

        model.eval()

        for batch in tqdm(dataloader, desc="特徴量抽出中"):
            audio = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            filenames = batch[-1]

            # Student特徴量・予測
            features_student = model.sed_student.extract_features(audio)
            strong_preds_student, weak_preds_student = model.detect(
                audio,
                return_weak=True,
                use_teacher=False,
            )

            # Teacher特徴量・予測
            if model.sed_teacher is not None:
                features_teacher = model.sed_teacher.extract_features(audio)
                strong_preds_teacher, weak_preds_teacher = model.detect(
                    audio,
                    return_weak=True,
                    use_teacher=True,
                )
            else:
                features_teacher = features_student
                weak_preds_teacher = weak_preds_student

            # バッチ結果を保存
            all_features_student.append(features_student.cpu().numpy())
            all_features_teacher.append(features_teacher.cpu().numpy())
            all_probs_student.append(weak_preds_student.cpu().numpy())
            all_probs_teacher.append(weak_preds_teacher.cpu().numpy())
            all_filenames.extend(filenames)
            all_targets.append(targets.max(dim=1)[0].cpu().numpy())  # (B, C)

        # 結合
        features_student = np.concatenate(all_features_student, axis=0)
        features_teacher = np.concatenate(all_features_teacher, axis=0)
        probs_student = np.concatenate(all_probs_student, axis=0)
        probs_teacher = np.concatenate(all_probs_teacher, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        filenames = np.array(all_filenames)

        return {
            "features_student": features_student,
            "features_teacher": features_teacher,
            "probs_student": probs_student,
            "probs_teacher": probs_teacher,
            "filenames": filenames,
            "targets": targets,
        }


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(
        description="特徴量抽出スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 必須引数
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="モデルチェックポイント（複数指定可）",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="モデル設定ファイル (YAML)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="出力ディレクトリ",
    )

    # オプション引数
    parser.add_argument(
        "--model_names",
        nargs="+",
        help="モデル名（checkpointsと同じ数）",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["validation", "maestro_validation"],
        help="抽出するデータセット",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="バッチサイズ (default: 32)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="使用デバイス (default: cuda if available else cpu)",
    )

    return parser.parse_args()


def main():
    """メイン処理"""
    args = parse_arguments()

    # モデル名の設定
    if args.model_names is None:
        args.model_names = [f"model_{i}" for i in range(len(args.checkpoints))]
    elif len(args.model_names) != len(args.checkpoints):
        raise ValueError("model_namesの数とcheckpointsの数が一致しません")

    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("特徴量抽出スクリプト")
    print("=" * 60)

    # 設定読み込み
    print("\n[1/3] 設定読み込み中...")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 特徴量抽出器の初期化
    extractor = FeatureExtractor(config, args.device)

    # 各モデルで特徴量を抽出
    print("\n[2/3] 特徴量抽出中...")
    for checkpoint_path, model_name in zip(args.checkpoints, args.model_names):
        print(f"\n処理中: {model_name}")

        # モデルロード
        model = extractor.load_model(Path(checkpoint_path))

        # 各データセットで特徴量抽出
        for dataset_type in args.datasets:
            print(f"  データセット: {dataset_type}")

            # データローダー作成
            dataloader = extractor.create_dataloader(
                dataset_type=dataset_type,
                batch_size=args.batch_size,
            )

            # 特徴量抽出
            features = extractor.extract_features(model, dataloader)

            # 保存
            output_path = output_dir / f"{model_name}_{dataset_type}.npz"
            np.savez_compressed(output_path, **features)
            print(f"  ✓ 保存: {output_path}")

    print("\n" + "=" * 60)
    print("完了！")
    print(f"出力ディレクトリ: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
