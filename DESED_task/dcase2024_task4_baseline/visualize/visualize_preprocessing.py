#!/usr/bin/env python3
"""音声前処理の可視化スクリプト

このスクリプトは、DCASE 2024 Task 4の音声前処理パイプラインを可視化します：
1. 音声波形
2. STFT結果（スペクトログラム）
3. メルスペクトログラム
4. Mixup前後の比較

使用例:
    python visualize/visualize_preprocessing.py \
        --audio-path data/sample.wav \
        --output-dir visualizations/ \
        --config confs/pretrained.yaml
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import yaml
from matplotlib.figure import Figure
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))


class AudioPreprocessingVisualizer:
    """音声前処理の可視化クラス"""

    def __init__(self, config_path: str):
        """
        Args:
            config_path: 設定ファイル（YAML）のパス
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # 特徴量パラメータ
        feat_params = self.config["feats"]
        self.sample_rate = feat_params["sample_rate"]
        self.n_fft = feat_params["n_window"]
        self.hop_length = feat_params["hop_length"]
        self.n_mels = feat_params["n_mels"]
        self.f_min = feat_params["f_min"]
        self.f_max = feat_params["f_max"]

        # メルスペクトログラム変換を初期化
        self.mel_spec = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            window_fn=torch.hamming_window,
            wkwargs={"periodic": False},
            power=1,  # マグニチュードスペクトログラム
        )

        # dB変換
        self.amplitude_to_db = AmplitudeToDB(stype="magnitude")

    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """音声ファイルを読み込む

        Args:
            audio_path: 音声ファイルのパス

        Returns:
            audio: 音声波形 (1, length)
            sr: サンプリングレート
        """
        audio, sr = torchaudio.load(audio_path)

        # モノラルに変換
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # サンプリングレートの確認
        if sr != self.sample_rate:
            warnings.warn(
                f"Sample rate mismatch: {sr} Hz -> {self.sample_rate} Hz. Resampling..."
            )
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
            sr = self.sample_rate

        return audio, sr

    def compute_stft(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """STFTを計算する

        Args:
            audio: 音声波形 (1, length)

        Returns:
            magnitude: マグニチュードスペクトログラム (freq, time)
            phase: 位相スペクトログラム (freq, time)
        """
        # STFTを計算
        stft = torch.stft(
            audio.squeeze(0),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=torch.hamming_window(self.n_fft),
            return_complex=True,
        )

        # マグニチュードと位相を計算
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)

        return magnitude, phase

    def compute_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """メルスペクトログラムを計算する

        Args:
            audio: 音声波形 (1, length)

        Returns:
            mel_spec_db: メルスペクトログラム（dBスケール） (n_mels, time)
        """
        # メルスペクトログラムを計算
        mel_spec = self.mel_spec(audio)

        # dBスケールに変換
        mel_spec_db = self.amplitude_to_db(mel_spec)

        return mel_spec_db.squeeze(0)

    def visualize_waveform(
        self, audio: torch.Tensor, sr: int, ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """音声波形を可視化する

        Args:
            audio: 音声波形 (1, length)
            sr: サンプリングレート
            ax: matplotlib axes（Noneの場合は新規作成）

        Returns:
            ax: matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))

        # 時間軸を作成
        time = np.arange(audio.shape[1]) / sr

        # 波形をプロット
        ax.plot(time, audio.squeeze(0).numpy(), linewidth=0.5)
        ax.set_xlabel("Time [s]", fontsize=12)
        ax.set_ylabel("Amplitude", fontsize=12)
        ax.set_title("Audio Waveform", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, time[-1])

        return ax

    def visualize_stft(
        self,
        magnitude: torch.Tensor,
        sr: int,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """STFT結果を可視化する

        Args:
            magnitude: マグニチュードスペクトログラム (freq, time)
            sr: サンプリングレート
            ax: matplotlib axes（Noneの場合は新規作成）

        Returns:
            ax: matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        # dBスケールに変換
        magnitude_db = 20 * torch.log10(magnitude + 1e-10)

        # 時間軸と周波数軸
        time = np.arange(magnitude.shape[1]) * self.hop_length / sr
        freq = np.linspace(0, sr / 2, magnitude.shape[0])

        # マグニチュードスペクトログラム
        im = ax.pcolormesh(
            time,
            freq,
            magnitude_db.numpy(),
            shading="auto",
            cmap="viridis",
        )
        ax.set_xlabel("Time [s]", fontsize=12)
        ax.set_ylabel("Frequency [Hz]", fontsize=12)
        ax.set_title("STFT Magnitude", fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Amplitude [dB]")

        return ax

    def visualize_mel_spectrogram(
        self,
        mel_spec_db: torch.Tensor,
        sr: int,
        ax: Optional[plt.Axes] = None,
        title: str = "Mel Spectrogram",
    ) -> plt.Axes:
        """メルスペクトログラムを可視化する

        Args:
            mel_spec_db: メルスペクトログラム（dBスケール） (n_mels, time)
            sr: サンプリングレート
            ax: matplotlib axes（Noneの場合は新規作成）
            title: プロットのタイトル

        Returns:
            ax: matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        # 時間軸
        time = np.arange(mel_spec_db.shape[1]) * self.hop_length / sr

        # メルスペクトログラムをプロット
        im = ax.pcolormesh(
            time,
            np.arange(self.n_mels),
            mel_spec_db.numpy(),
            shading="auto",
            cmap="viridis",
        )
        ax.set_xlabel("Time [s]", fontsize=12)
        ax.set_ylabel("Mel Filter Bank", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Amplitude [dB]")

        return ax

    def visualize_mixup_comparison(
        self,
        audio1: torch.Tensor,
        audio2: torch.Tensor,
        sr: int,
        alpha: float = 0.2,
        beta: float = 0.2,
    ) -> Figure:
        """Mixup前後の比較を可視化する

        Args:
            audio1: 音声波形1 (1, length)
            audio2: 音声波形2 (1, length)
            sr: サンプリングレート
            alpha: mixupのalphaパラメータ
            beta: mixupのbetaパラメータ

        Returns:
            fig: matplotlib figure
        """
        # メルスペクトログラムを計算
        mel1 = self.compute_mel_spectrogram(audio1)
        mel2 = self.compute_mel_spectrogram(audio2)

        # 手動でmixupを適用（確実にmel1とmel2をミックス）
        lam = np.random.beta(alpha, beta)
        mixed_mel = lam * mel1 + (1 - lam) * mel2

        # プロット
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        self.visualize_mel_spectrogram(
            mel1, sr, ax=axes[0], title="Original Audio 1 - Mel Spectrogram"
        )
        self.visualize_mel_spectrogram(
            mel2, sr, ax=axes[1], title="Original Audio 2 - Mel Spectrogram"
        )
        self.visualize_mel_spectrogram(
            mixed_mel,
            sr,
            ax=axes[2],
            title=f"After Mixup (mix ratio λ={lam:.2f})",
        )

        plt.tight_layout()
        return fig

    def visualize_all(
        self, audio_path: str, output_path: Optional[str] = None
    ) -> Figure:
        """すべての可視化を一度に実行する

        Args:
            audio_path: 音声ファイルのパス
            output_path: 保存先パス（Noneの場合は保存しない）

        Returns:
            fig: matplotlib figure
        """
        # 音声を読み込む
        audio, sr = self.load_audio(audio_path)

        # 各種特徴量を計算
        magnitude, phase = self.compute_stft(audio)
        mel_spec_db = self.compute_mel_spectrogram(audio)

        # プロット
        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(3, 1, hspace=0.5, left=0.1, right=0.9)

        # 1. 音声波形
        ax1 = fig.add_subplot(gs[0])
        self.visualize_waveform(audio, sr, ax=ax1)

        # 2. STFT結果（マグニチュードのみ）
        ax2 = fig.add_subplot(gs[1])
        self.visualize_stft(magnitude, sr, ax=ax2)

        # 3. メルスペクトログラム
        ax3 = fig.add_subplot(gs[2])
        self.visualize_mel_spectrogram(mel_spec_db, sr, ax=ax3)

        # 保存
        if output_path is not None:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Visualization saved: {output_path}")

        return fig


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(
        description="音声前処理パイプラインの可視化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 単一音声ファイルの可視化
  python visualize/visualize_preprocessing.py \\
      --audio-path data/sample.wav \\
      --output-dir visualizations/

  # Mixup比較も含める
  python visualize/visualize_preprocessing.py \\
      --audio-path data/sample1.wav \\
      --audio-path2 data/sample2.wav \\
      --output-dir visualizations/ \\
      --mixup
        """,
    )

    parser.add_argument(
        "--audio-path",
        type=str,
        required=True,
        help="可視化する音声ファイルのパス",
    )

    parser.add_argument(
        "--audio-path2",
        type=str,
        default=None,
        help="Mixup比較用の2つ目の音声ファイルのパス（--mixupと併用）",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="confs/pretrained.yaml",
        help="設定ファイル（YAML）のパス（デフォルト: confs/pretrained.yaml）",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations/preprocessing",
        help="可視化結果の保存先ディレクトリ（デフォルト: visualizations/preprocessing）",
    )

    parser.add_argument(
        "--mixup",
        action="store_true",
        help="Mixup比較の可視化を実行する",
    )

    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.2,
        help="Mixupのalphaパラメータ（デフォルト: 0.2）",
    )

    parser.add_argument(
        "--mixup-beta",
        type=float,
        default=0.2,
        help="Mixupのbetaパラメータ（デフォルト: 0.2）",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="ファイルに保存せず、表示のみ行う",
    )

    return parser.parse_args()


def main():
    """メイン処理"""
    args = parse_arguments()

    # 出力ディレクトリを作成
    output_dir = Path(args.output_dir)
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 可視化クラスを初期化
    visualizer = AudioPreprocessingVisualizer(args.config)

    # 1. すべての前処理ステップを可視化
    print(f"Processing audio file: {args.audio_path}")
    audio_name = Path(args.audio_path).stem
    output_path = None if args.no_save else output_dir / f"{audio_name}_preprocessing.png"

    fig1 = visualizer.visualize_all(args.audio_path, output_path)

    # 2. Mixup比較（オプション）
    if args.mixup:
        if args.audio_path2 is None:
            print("Warning: --mixup specified but --audio-path2 not provided.")
            print("Skipping Mixup comparison.")
        else:
            print(f"Creating Mixup comparison: {args.audio_path} + {args.audio_path2}")
            audio1, sr1 = visualizer.load_audio(args.audio_path)
            audio2, sr2 = visualizer.load_audio(args.audio_path2)

            # 長さを揃える
            min_len = min(audio1.shape[1], audio2.shape[1])
            audio1 = audio1[:, :min_len]
            audio2 = audio2[:, :min_len]

            fig2 = visualizer.visualize_mixup_comparison(
                audio1, audio2, sr1, alpha=args.mixup_alpha, beta=args.mixup_beta
            )

            if not args.no_save:
                mixup_output = output_dir / f"{audio_name}_mixup_comparison.png"
                fig2.savefig(mixup_output, dpi=300, bbox_inches="tight")
                print(f"Mixup comparison saved: {mixup_output}")

    # プロットを表示
    if args.no_save:
        plt.show()

    print("\nProcessing completed.")


if __name__ == "__main__":
    main()
