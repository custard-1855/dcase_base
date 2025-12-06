#!/usr/bin/env python3
"""
Grad-CAM可視化スクリプト（リファクタリング版）

境界事例や誤予測サンプルに対してGrad-CAMを適用し、
モデルがどの時間-周波数領域に注目しているかを可視化

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
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from desed_task.nnet.CRNN import CRNN
from desed_task.utils.encoder import CatManyHotEncoder, ManyHotEncoder
from local.classes_dict import (
    classes_labels_desed,
    classes_labels_maestro_real
)
from local.sed_trainer_pretrained import SEDTask4

# 共通ユーティリティをインポート
from visualization_utils import (
    DESED_CLASSES, MAESTRO_REAL_ALL, ALL_CLASSES_27,
    USED_CLASS_INDICES, USED_CLASSES_21,
    load_inference_data
)


# --- 設定クラス ---
@dataclass
class GradCAMConfig:
    """Grad-CAM設定パラメータ"""
    target_layer: str = "attention"  # 注目する層
    use_cuda: bool = True
    batch_size: int = 1
    confidence_range: Tuple[float, float] = (0.4, 0.6)  # 境界事例の閾値範囲
    misclass_threshold: float = 0.5
    n_samples: int = 10

    # プロット設定
    figsize: Tuple[int, int] = (18, 5)
    dpi: int = 200
    colormap_spec: str = 'viridis'
    colormap_gradcam: str = 'jet'
    overlay_alpha: float = 0.5

    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            'target_layer': self.target_layer,
            'use_cuda': self.use_cuda,
            'batch_size': self.batch_size,
            'confidence_range': list(self.confidence_range),
            'misclass_threshold': self.misclass_threshold,
            'n_samples': self.n_samples
        }


# --- モデルローダークラス ---
class ModelLoader:
    """モデルローディングを管理するクラス"""

    def __init__(self, config: Dict, device: str = 'cpu'):
        """
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

    def load_model(self, checkpoint_path: Union[str, Path]) -> SEDTask4:
        """
        モデルをロード

        Args:
            checkpoint_path: チェックポイントファイルパス

        Returns:
            ロードされたSEDTask4モデル

        Raises:
            FileNotFoundError: チェックポイントファイルが見つからない場合
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"  モデルをロード中: {checkpoint_path}")

        # Checkpointから読み込み
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
            evaluation=True
        )

        # State dictをロード
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        return model


# --- Grad-CAM計算クラス ---
class GradCAMComputer:
    """Grad-CAM計算を管理するクラス"""

    def __init__(self, config: Optional[GradCAMConfig] = None):
        """
        Args:
            config: Grad-CAM設定
        """
        self.config = config or GradCAMConfig()

    def compute_gradcam(
        self,
        model: SEDTask4,
        audio: torch.Tensor,
        target_class: int,
        use_teacher: bool = False
    ) -> np.ndarray:
        """
        Grad-CAMを計算

        Args:
            model: SEDTask4モデル
            audio: (n_samples,) または (batch, n_samples) 音声テンソル
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
            x_no_dropout = x  # dropoutをスキップ
            weak_pred = sed_model.dense(x_no_dropout)
            weak_pred = weak_pred.mean(dim=1)  # (B, n_classes)

            # ターゲットクラスのスコア
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
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        filenames: np.ndarray,
        confidence_range: Optional[Tuple[float, float]] = None,
        n_samples: Optional[int] = None
    ) -> List[Dict]:
        """
        境界事例（予測確率が中間的なサンプル）を抽出

        Args:
            probs: (N, C) 予測確率
            targets: (N, C) 正解ラベル
            filenames: (N,) ファイル名
            confidence_range: 境界とみなす確率範囲
            n_samples: 抽出するサンプル数

        Returns:
            境界事例のリスト
        """
        confidence_range = confidence_range or self.config.confidence_range
        n_samples = n_samples or self.config.n_samples

        boundary_cases = []

        for sample_idx in range(probs.shape[0]):
            for class_idx in range(probs.shape[1]):
                prob = probs[sample_idx, class_idx]
                target = targets[sample_idx, class_idx]

                # 境界条件: 予測確率が指定範囲内
                if confidence_range[0] < prob < confidence_range[1]:
                    class_name = (ALL_CLASSES_27[class_idx] if class_idx < len(ALL_CLASSES_27)
                                else f'class_{class_idx}')

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
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        filenames: np.ndarray,
        threshold: Optional[float] = None,
        n_samples: Optional[int] = None
    ) -> List[Dict]:
        """
        誤予測サンプルを抽出

        Args:
            probs: (N, C) 予測確率
            targets: (N, C) 正解ラベル
            filenames: (N,) ファイル名
            threshold: 二値化閾値
            n_samples: 抽出するサンプル数

        Returns:
            誤予測サンプルのリスト
        """
        threshold = threshold or self.config.misclass_threshold
        n_samples = n_samples or self.config.n_samples

        misclassified_cases = []

        for sample_idx in range(probs.shape[0]):
            pred_class = np.argmax(probs[sample_idx])
            true_classes = np.where(targets[sample_idx] > threshold)[0]

            # 誤予測の場合
            if len(true_classes) > 0 and pred_class not in true_classes:
                pred_prob = probs[sample_idx, pred_class]

                pred_name = (ALL_CLASSES_27[pred_class] if pred_class < len(ALL_CLASSES_27)
                           else f'class_{pred_class}')
                true_name = (ALL_CLASSES_27[true_classes[0]] if true_classes[0] < len(ALL_CLASSES_27)
                           else f'class_{true_classes[0]}')

                misclassified_cases.append({
                    'filename': str(filenames[sample_idx]),
                    'predicted_class_idx': int(pred_class),
                    'predicted_class_name': pred_name,
                    'true_class_idx': int(true_classes[0]),
                    'true_class_name': true_name,
                    'predicted_prob': float(pred_prob),
                    'sample_idx': sample_idx
                })

        # 予測確率が高い順（確信を持った誤予測）
        misclassified_cases.sort(key=lambda x: x['predicted_prob'], reverse=True)

        return misclassified_cases[:n_samples]


# --- 可視化クラス ---
class GradCAMVisualizer:
    """Grad-CAM可視化を管理するクラス"""

    def __init__(
        self,
        config: Optional[GradCAMConfig] = None,
        computer: Optional[GradCAMComputer] = None
    ):
        """
        Args:
            config: Grad-CAM設定
            computer: Grad-CAM計算インスタンス
        """
        self.config = config or GradCAMConfig()
        self.computer = computer or GradCAMComputer(self.config)

    def plot_gradcam_overlay(
        self,
        mel_spec: np.ndarray,
        gradcam: np.ndarray,
        output_path: Union[str, Path],
        title: str = "Grad-CAM",
        class_name: str = "",
        prob: Optional[float] = None
    ) -> None:
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
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=self.config.figsize)

        # 1. メルスペクトログラム
        im1 = axes[0].imshow(
            mel_spec.T,
            aspect='auto',
            origin='lower',
            cmap=self.config.colormap_spec,
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
            cmap=self.config.colormap_gradcam,
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
            cmap=self.config.colormap_gradcam,
            interpolation='bilinear',
            alpha=self.config.overlay_alpha
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
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()

        print(f"  ✓ 保存: {output_path}")

    def process_samples(
        self,
        cases: List[Dict],
        models: Dict[str, SEDTask4],
        model_config: Dict,
        output_dir: Path,
        case_type: str,
        pred_type: str = 'student'
    ) -> None:
        """
        サンプルを処理してGrad-CAMを生成

        Args:
            cases: 処理するケースのリスト
            models: モデル辞書
            model_config: モデル設定
            output_dir: 出力ディレクトリ
            case_type: ケースタイプ（'boundary' or 'misclassified'）
            pred_type: 使用するモデルタイプ
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, case in enumerate(tqdm(cases, desc=f"{case_type}のGrad-CAM計算")):
            # 音声ファイルのパスを解決
            audio_path = Path(case['filename'])
            if not audio_path.exists():
                warnings.warn(f"ファイルが見つかりません: {audio_path}")
                continue

            try:
                # 音声読み込み
                audio, sr = librosa.load(audio_path, sr=model_config['data']['fs'], mono=True)

                # デバイスを取得
                device = next(iter(models.values())).device
                audio_tensor = torch.from_numpy(audio).float().to(device)

                # メルスペクトログラム計算（可視化用）
                mel_spec = librosa.feature.melspectrogram(
                    y=audio,
                    sr=sr,
                    n_fft=model_config['feats']['n_window'],
                    hop_length=model_config['feats']['hop_length'],
                    n_mels=model_config['feats']['n_filters']
                )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

                # 各モデルでGrad-CAM計算
                for model_name, model in models.items():
                    if case_type == 'boundary':
                        target_class = case['class_idx']
                        class_info = case['class_name']
                        prob_info = case['prob']
                        filename_suffix = f"_{case['class_name']}_prob{case['prob']:.2f}"
                    else:  # misclassified
                        target_class = case['predicted_class_idx']
                        class_info = f"Pred: {case['predicted_class_name']} / True: {case['true_class_name']}"
                        prob_info = case['predicted_prob']
                        filename_suffix = f"_pred{case['predicted_class_name']}_true{case['true_class_name']}"

                    gradcam = self.computer.compute_gradcam(
                        model,
                        audio_tensor,
                        target_class,
                        use_teacher=(pred_type == 'teacher')
                    )

                    # プロット
                    output_path = output_dir / f"{model_name}_sample{i:03d}{filename_suffix}.png"
                    self.plot_gradcam_overlay(
                        mel_spec_db.T,
                        gradcam,
                        output_path,
                        title=f"{model_name} - {case_type.title()} Case",
                        class_name=class_info,
                        prob=prob_info
                    )

            except Exception as e:
                print(f"  エラー（{case_type} {i}）: {e}")


# --- メイン処理 ---
def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(
        description="Grad-CAM可視化スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # 基本的な使用法
  python visualize_gradcam.py --input_dirs inference_outputs/baseline --checkpoints path/to/baseline.ckpt --config confs/pretrained.yaml --output_dir visualization_outputs/gradcam

  # 複数モデルの比較
  python visualize_gradcam.py --input_dirs inference_outputs/baseline inference_outputs/cmt_normal --checkpoints baseline.ckpt cmt.ckpt --config confs/pretrained.yaml --output_dir outputs

  # サンプル数とデバイスを指定
  python visualize_gradcam.py --input_dirs outputs/baseline --checkpoints baseline.ckpt --config confs/pretrained.yaml --output_dir outputs --n_samples 20 --device cpu
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

    # オプション引数
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="可視化するサンプル数 (default: 10)"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="使用デバイス (default: cuda if available else cpu)"
    )
    parser.add_argument(
        "--pred_type",
        choices=['student', 'teacher'],
        default='student',
        help="使用するモデル (default: student)"
    )
    parser.add_argument(
        "--confidence_range",
        nargs=2,
        type=float,
        default=[0.4, 0.6],
        help="境界事例の確率範囲 (default: 0.4 0.6)"
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

    if len(args.checkpoints) != len(args.input_dirs):
        raise ValueError("checkpointsの数とinput_dirsの数が一致しません")

    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Grad-CAM可視化スクリプト")
    print("="*60)

    # 設定読み込み
    print("\n[1/4] 設定読み込み中...")
    with open(args.config, 'r') as f:
        model_config = yaml.safe_load(f)

    # Grad-CAM設定の作成
    gradcam_config = GradCAMConfig(
        confidence_range=tuple(args.confidence_range),
        n_samples=args.n_samples,
        use_cuda=(args.device == 'cuda')
    )

    # ビジュアライザーとコンピュータの初期化
    computer = GradCAMComputer(gradcam_config)
    visualizer = GradCAMVisualizer(gradcam_config, computer)

    # モデルローダーの初期化
    model_loader = ModelLoader(model_config, args.device)

    # データ読み込み
    print("\n[2/4] データ読み込み中...")
    models_data = {}
    for input_dir in args.input_dirs:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"  警告: {input_path} が見つかりません。スキップします。")
            continue
        model_name = input_path.name
        print(f"  - {model_name}: {input_path}")
        models_data[model_name] = load_inference_data(input_path, verbose=args.verbose)

    if not models_data:
        print("エラー: 有効な入力データが見つかりませんでした。")
        return

    # モデルロード
    print("\n[3/4] モデルロード中...")
    models = {}
    for input_dir, checkpoint in zip(args.input_dirs, args.checkpoints):
        model_name = Path(input_dir).name
        try:
            models[model_name] = model_loader.load_model(checkpoint)
        except Exception as e:
            print(f"  エラー: {model_name} のロードに失敗: {e}")
            continue

    if not models:
        print("エラー: モデルをロードできませんでした。")
        return

    # 境界事例と誤予測の抽出・可視化
    print("\n[4/4] 境界事例と誤予測を処理中...")

    # 最初のモデルから抽出
    first_model = list(models_data.keys())[0]

    for dataset_name, data in models_data[first_model].items():
        print(f"\n処理中: {dataset_name}")

        if f'probs_{args.pred_type}' not in data or 'targets' not in data:
            print(f"  警告: 必要なデータがありません。スキップします。")
            continue

        probs = data[f'probs_{args.pred_type}']
        targets = data['targets']
        filenames = data['filenames']

        # 境界事例
        boundary_cases = computer.find_boundary_cases(
            probs, targets, filenames,
            confidence_range=tuple(args.confidence_range),
            n_samples=args.n_samples
        )
        print(f"  境界事例: {len(boundary_cases)}個")

        # 誤予測
        misclassified_cases = computer.find_misclassified_cases(
            probs, targets, filenames,
            n_samples=args.n_samples
        )
        print(f"  誤予測: {len(misclassified_cases)}個")

        # 境界事例のGrad-CAM生成
        if boundary_cases:
            boundary_dir = output_dir / 'boundary_cases' / dataset_name
            visualizer.process_samples(
                boundary_cases,
                models,
                model_config,
                boundary_dir,
                'boundary',
                args.pred_type
            )

        # 誤予測のGrad-CAM生成
        if misclassified_cases:
            misclass_dir = output_dir / 'misclassified' / dataset_name
            visualizer.process_samples(
                misclassified_cases,
                models,
                model_config,
                misclass_dir,
                'misclassified',
                args.pred_type
            )

    # 設定の保存
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(gradcam_config.to_dict(), f, indent=2)
    print(f"\n  ✓ 設定保存: {output_dir / 'config.json'}")

    print("\n" + "="*60)
    print("完了！")
    print(f"出力ディレクトリ: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()