#!/usr/bin/env python3
"""
Grad-CAM可視化スクリプト

境界事例や誤予測サンプルに対してGrad-CAMを適用し、
モデルがどの時間-周波数領域に注目しているかを可視化する。

使用法:
    python visualize_gradcam.py \
        --input_dirs inference_outputs/baseline inference_outputs/cmt_normal \
        --checkpoints path/to/baseline.ckpt path/to/cmt.ckpt \
        --config confs/pretrained.yaml \
        --output_dir visualization_outputs/gradcam \
        --n_samples 20
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from desed_task.nnet.CRNN import CRNN
from desed_task.utils.encoder import CatManyHotEncoder, ManyHotEncoder
from local.classes_dict import (classes_labels_desed,
                                classes_labels_maestro_real)
from local.sed_trainer_pretrained import SEDTask4


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


def load_model(checkpoint_path: Path, config: Dict, device: str) -> SEDTask4:
    """モデルをロード"""
    print(f"モデルをロード中: {checkpoint_path}")

    # Checkpointから読み込み
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    state_dict = checkpoint["state_dict"]

    # Encoderの準備
    encoder = get_encoder(config)

    # モデルインスタンスを作成
    sed_student = CRNN(**config["net"])

    # SEDTask4インスタンスを作成
    model = SEDTask4(
        config,
        encoder=encoder,
        sed_student=sed_student,
        pretrained_model=None,
        opt=None,
        train_data=None,
        valid_data=None,
        test_data=None,
        train_sampler=None,
        scheduler=None,
        fast_dev_run=False,
        evaluation=True
    )

    # State dictをロード
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def compute_gradcam(
    model: SEDTask4,
    audio: torch.Tensor,
    target_class: int,
    use_teacher: bool = False
) -> np.ndarray:
    """
    Grad-CAMを計算

    Args:
        model: SEDTask4モデル
        audio: (n_samples,) 音声テンソル
        target_class: 注目するクラスのインデックス
        use_teacher: Teacherモデルを使用するか

    Returns:
        gradcam: (time, freq) Grad-CAMヒートマップ
    """
    # 音声の次元を確認して適切に整形
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # (1, n_samples)

    # 使用するモデルを選択
    sed_model = model.sed_teacher if use_teacher else model.sed_student

    # Grad-CAM計算のため、一時的にトレーニングモードに切り替え
    # （RNNのbackwardにはtraining modeが必要）
    was_training = sed_model.training
    sed_model.train()

    try:
        # Mel spectrogram計算
        mels = model.mel_spec(audio)
        mels_preprocessed = model.scaler(model.take_log(mels))

        # CNN部分の特徴マップを取得
        mels_preprocessed.requires_grad = True

        x = mels_preprocessed.transpose(1, 2).unsqueeze(1)  # (B, 1, T, F)

        # CNNのforward（特徴マップを取得）
        cnn_out = sed_model.cnn(x)  # (B, C, T', F')

        # プーリングとRNN
        bs, chan, frames, freq = cnn_out.size()

        if freq != 1:
            x = cnn_out.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = cnn_out.squeeze(-1)
            x = x.permute(0, 2, 1)

        # RNN (training modeで実行)
        x = sed_model.rnn(x)[0]

        # Attention pooling（弱予測）
        # dropoutは無効化（推論時の挙動にする）
        x_no_dropout = x  # dropoutをスキップ
        weak_pred = sed_model.dense(x_no_dropout)
        weak_pred = weak_pred.mean(dim=1)  # (B, n_classes)

        # ターゲットクラスのスコア
        # weak_predの形状を確認してから適切にインデックス
        if weak_pred.dim() == 2:
            target_score = weak_pred[0, target_class]
        else:
            # バッチサイズが1でsqueezeされた場合
            target_score = weak_pred[target_class]

        # 勾配計算
        sed_model.zero_grad()
        target_score.backward()

        # CNN最終層の勾配を取得
        gradients = mels_preprocessed.grad.data  # (B, F, T)

        # 勾配の絶対値を使用（positive influence）
        gradients = torch.abs(gradients)

        # チャンネル方向（周波数方向）で平均
        if gradients.dim() == 3:
            cam = gradients.squeeze(0).cpu().numpy()  # (F, T)
        else:
            # 既に2次元の場合
            cam = gradients.cpu().numpy()  # (F, T)

        # 正規化
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.T  # (T, F) に転置

    finally:
        # 元のモードに戻す
        if not was_training:
            sed_model.eval()


def find_boundary_cases(
    probs: np.ndarray,
    targets: np.ndarray,
    filenames: np.ndarray,
    confidence_range: Tuple[float, float] = (0.4, 0.6),
    n_samples: int = 10
) -> List[Dict]:
    """
    境界事例（予測確率が中間的なサンプル）を抽出

    Returns:
        List of {
            'filename': str,
            'class_idx': int,
            'class_name': str,
            'prob': float,
            'target': float,
            'sample_idx': int
        }
    """
    boundary_cases = []

    for sample_idx in range(probs.shape[0]):
        for class_idx in range(probs.shape[1]):
            prob = probs[sample_idx, class_idx]
            target = targets[sample_idx, class_idx]

            # 境界条件: 予測確率が指定範囲内
            if confidence_range[0] < prob < confidence_range[1]:
                class_name = ALL_CLASSES_27[class_idx] if class_idx < len(ALL_CLASSES_27) else f'class_{class_idx}'

                boundary_cases.append({
                    'filename': str(filenames[sample_idx]),
                    'class_idx': class_idx,
                    'class_name': class_name,
                    'prob': float(prob),
                    'target': float(target),
                    'sample_idx': sample_idx
                })

    # 予測確率が0.5に近い順にソート
    boundary_cases.sort(key=lambda x: abs(x['prob'] - 0.5))

    return boundary_cases[:n_samples]


def find_misclassified_cases(
    probs: np.ndarray,
    targets: np.ndarray,
    filenames: np.ndarray,
    threshold: float = 0.5,
    n_samples: int = 10
) -> List[Dict]:
    """
    誤予測サンプルを抽出

    Returns:
        List of {
            'filename': str,
            'predicted_class_idx': int,
            'predicted_class_name': str,
            'true_class_idx': int,
            'true_class_name': str,
            'predicted_prob': float,
            'sample_idx': int
        }
    """
    misclassified_cases = []

    for sample_idx in range(probs.shape[0]):
        pred_class = np.argmax(probs[sample_idx])
        true_classes = np.where(targets[sample_idx] > threshold)[0]

        # 誤予測の場合
        if len(true_classes) > 0 and pred_class not in true_classes:
            pred_prob = probs[sample_idx, pred_class]

            misclassified_cases.append({
                'filename': str(filenames[sample_idx]),
                'predicted_class_idx': pred_class,
                'predicted_class_name': ALL_CLASSES_27[pred_class] if pred_class < len(ALL_CLASSES_27) else f'class_{pred_class}',
                'true_class_idx': true_classes[0],
                'true_class_name': ALL_CLASSES_27[true_classes[0]] if true_classes[0] < len(ALL_CLASSES_27) else f'class_{true_classes[0]}',
                'predicted_prob': float(pred_prob),
                'sample_idx': sample_idx
            })

    # 予測確率が高い順（確信を持った誤予測）
    misclassified_cases.sort(key=lambda x: x['predicted_prob'], reverse=True)

    return misclassified_cases[:n_samples]


def plot_gradcam_overlay(
    mel_spec: np.ndarray,
    gradcam: np.ndarray,
    output_path: Path,
    title: str = "Grad-CAM",
    class_name: str = "",
    prob: float = None
):
    """
    メルスペクトログラムとGrad-CAMを重ねて表示

    Args:
        mel_spec: (time, freq) メルスペクトログラム
        gradcam: (time, freq) Grad-CAM
        output_path: 出力パス
        title: タイトル
        class_name: クラス名
        prob: 予測確率
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. メルスペクトログラム
    im1 = axes[0].imshow(
        mel_spec.T,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        interpolation='nearest'
    )
    axes[0].set_title('Mel Spectrogram', fontsize=12)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Frequency (Mel bins)')
    plt.colorbar(im1, ax=axes[0])

    # 2. Grad-CAM
    im2 = axes[1].imshow(
        gradcam.T,
        aspect='auto',
        origin='lower',
        cmap='jet',
        interpolation='bilinear',
        alpha=0.8
    )
    axes[1].set_title('Grad-CAM', fontsize=12)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Frequency (Mel bins)')
    plt.colorbar(im2, ax=axes[1])

    # 3. オーバーレイ
    axes[2].imshow(
        mel_spec.T,
        aspect='auto',
        origin='lower',
        cmap='gray',
        interpolation='nearest',
        alpha=0.6
    )
    im3 = axes[2].imshow(
        gradcam.T,
        aspect='auto',
        origin='lower',
        cmap='jet',
        interpolation='bilinear',
        alpha=0.5
    )
    axes[2].set_title('Overlay', fontsize=12)
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Frequency (Mel bins)')
    plt.colorbar(im3, ax=axes[2])

    # 全体タイトル
    title_str = f'{title}'
    if class_name:
        title_str += f' - {class_name}'
    if prob is not None:
        title_str += f' (prob: {prob:.3f})'

    fig.suptitle(title_str, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM可視化スクリプト")
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="推論結果ディレクトリ（複数指定可）"
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="モデルチェックポイント（input_dirsと同じ順序）"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="モデル設定ファイル (YAML)"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="出力ディレクトリ"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="可視化するサンプル数"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="デバイス"
    )
    parser.add_argument(
        "--pred_type",
        choices=['student', 'teacher'],
        default='student',
        help="使用するモデル"
    )

    args = parser.parse_args()

    if len(args.checkpoints) != len(args.input_dirs):
        raise ValueError("checkpointsの数とinput_dirsの数が一致しません")

    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Grad-CAM可視化スクリプト")
    print("="*60)

    # 設定読み込み
    print("\n設定読み込み中...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # データ読み込み
    print("\nデータ読み込み中...")
    models_data = {}
    for input_dir in args.input_dirs:
        input_path = Path(input_dir)
        model_name = input_path.name
        models_data[model_name] = load_inference_data(input_path)

    # モデルロード
    print("\nモデルロード中...")
    models = {}
    for input_dir, checkpoint in zip(args.input_dirs, args.checkpoints):
        model_name = Path(input_dir).name
        models[model_name] = load_model(Path(checkpoint), config, args.device)

    # 境界事例と誤予測の抽出
    print("\n境界事例と誤予測を抽出中...")

    # 最初のモデルから抽出
    first_model = list(models_data.keys())[0]

    for dataset_name, data in models_data[first_model].items():
        print(f"\n処理中: {dataset_name}")

        probs = data[f'probs_{args.pred_type}']
        targets = data['targets']
        filenames = data['filenames']

        # 境界事例
        boundary_cases = find_boundary_cases(probs, targets, filenames, n_samples=args.n_samples)
        print(f"  境界事例: {len(boundary_cases)}個")

        # 誤予測
        misclassified_cases = find_misclassified_cases(probs, targets, filenames, n_samples=args.n_samples)
        print(f"  誤予測: {len(misclassified_cases)}個")

        # 各ケースについてGrad-CAMを計算・可視化
        # 境界事例
        boundary_dir = output_dir / 'boundary_cases' / dataset_name
        boundary_dir.mkdir(parents=True, exist_ok=True)

        for i, case in enumerate(tqdm(boundary_cases, desc="境界事例のGrad-CAM計算")):
            # 音声ファイルのパスを解決
            audio_path = Path(case['filename'])
            if not audio_path.exists():
                print(f"  警告: ファイルが見つかりません: {audio_path}")
                continue

            # 音声読み込み
            audio, sr = librosa.load(audio_path, sr=config['data']['fs'], mono=True)
            audio_tensor = torch.from_numpy(audio).float().to(args.device)

            # メルスペクトログラム計算（可視化用）
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=config['feats']['n_window'],
                hop_length=config['feats']['hop_length'],
                n_mels=config['feats']['n_filters']
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Grad-CAM計算
            for model_name, model in models.items():
                try:
                    gradcam = compute_gradcam(
                        model,
                        audio_tensor,
                        case['class_idx'],
                        use_teacher=(args.pred_type == 'teacher')
                    )

                    # プロット
                    output_path = boundary_dir / f"{model_name}_sample{i:03d}_{case['class_name']}_prob{case['prob']:.2f}.png"
                    plot_gradcam_overlay(
                        mel_spec_db.T,
                        gradcam,
                        output_path,
                        title=f"{model_name} - Boundary Case",
                        class_name=case['class_name'],
                        prob=case['prob']
                    )
                except Exception as e:
                    print(f"  エラー（境界事例 {i}）: {e}")

        # 誤予測
        misclass_dir = output_dir / 'misclassified' / dataset_name
        misclass_dir.mkdir(parents=True, exist_ok=True)

        for i, case in enumerate(tqdm(misclassified_cases, desc="誤予測のGrad-CAM計算")):
            audio_path = Path(case['filename'])
            if not audio_path.exists():
                print(f"  警告: ファイルが見つかりません: {audio_path}")
                continue

            audio, sr = librosa.load(audio_path, sr=config['data']['fs'], mono=True)
            audio_tensor = torch.from_numpy(audio).float().to(args.device)

            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=config['feats']['n_window'],
                hop_length=config['feats']['hop_length'],
                n_mels=config['feats']['n_filters']
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            for model_name, model in models.items():
                try:
                    gradcam = compute_gradcam(
                        model,
                        audio_tensor,
                        case['predicted_class_idx'],
                        use_teacher=(args.pred_type == 'teacher')
                    )

                    output_path = misclass_dir / f"{model_name}_sample{i:03d}_pred{case['predicted_class_name']}_true{case['true_class_name']}.png"
                    plot_gradcam_overlay(
                        mel_spec_db.T,
                        gradcam,
                        output_path,
                        title=f"{model_name} - Misclassified",
                        class_name=f"Pred: {case['predicted_class_name']} / True: {case['true_class_name']}",
                        prob=case['predicted_prob']
                    )
                except Exception as e:
                    print(f"  エラー（誤予測 {i}）: {e}")

    print("\n" + "="*60)
    print("完了！")
    print(f"出力ディレクトリ: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
