#!/usr/bin/env python3
"""
リアルタイム音響イベント検出（SED）システム
10秒間録音し、音響イベントを検出して可視化します
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')  # インタラクティブバックエンドを使用
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import torch
import torchaudio
import yaml
from matplotlib.animation import FuncAnimation

# プロジェクトのパスを追加
sys.path.insert(0, str(Path(__file__).parent / "DESED_task" / "dcase2024_task4_baseline"))

from DESED_task.dcase2024_task4_baseline.desed_task.nnet.CRNN import CRNN
from DESED_task.dcase2024_task4_baseline.local.classes_dict import classes_labels_desed, classes_labels_maestro_real

warnings.filterwarnings("ignore")

class RealtimeSED:
    """リアルタイムSEDシステム"""

    def __init__(self, checkpoint_path, config_path, device='cpu'):
        """
        Args:
            checkpoint_path: モデルチェックポイントのパス
            config_path: 設定ファイルのパス
            device: 使用するデバイス ('cpu' or 'cuda')
        """
        self.device = torch.device(device)

        # 設定の読み込み
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # クラスラベル（DESED + MAESTRO）
        self.class_labels = list(classes_labels_desed.keys()) + list(classes_labels_maestro_real.keys())
        self.n_classes = len(self.class_labels)

        # 音声パラメータ
        self.sample_rate = self.config['data']['fs']
        self.audio_max_len = self.config['data']['audio_max_len']
        self.duration = self.audio_max_len  # 10秒

        # 特徴抽出パラメータ
        self.n_mels = self.config['feats']['n_mels']
        self.n_fft = self.config['feats']['n_filters']
        self.hop_length = self.config['feats']['hop_length']
        self.f_min = self.config['feats']['f_min']
        self.f_max = self.config['feats']['f_max']

        # Mel spectrogram変換
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max
        )

        # モデルのロード
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        # 録音バッファ
        self.audio_buffer = None
        self.is_recording = False

        # 検出結果
        self.detection_results = None

    def _load_model(self, checkpoint_path):
        """モデルをロード"""
        print(f"モデルをロード中: {checkpoint_path}")

        # モデルの初期化
        net_config = self.config['net']
        model = CRNN(
            n_in_channel=net_config['n_in_channel'],
            nclass=self.n_classes,
            attention=net_config['attention'],
            n_RNN_cell=net_config['n_RNN_cell'],
            n_layers_RNN=net_config['rnn_layers'],
            activation=net_config['activation'],
            dropout=net_config['dropout'],
            kernel_size=net_config['kernel_size'],
            padding=net_config['padding'],
            stride=net_config['stride'],
            nb_filters=net_config['nb_filters'],
            pooling=net_config['pooling'],
            use_embeddings=False,  # リアルタイムではBEATs埋め込みを使用しない
            embedding_size=net_config.get('embedding_size', 768),
            embedding_type=net_config.get('embedding_type', 'frame')
        )

        # チェックポイントのロード
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # PyTorch Lightningのチェックポイントの場合
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # 'sed_student.' プレフィックスを削除
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('sed_student.'):
                        new_state_dict[k.replace('sed_student.', '')] = v
                    else:
                        new_state_dict[k] = v
                model.load_state_dict(new_state_dict, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print("チェックポイントのロードに成功しました")
        else:
            print(f"警告: チェックポイントが見つかりません: {checkpoint_path}")
            print("ランダムに初期化されたモデルを使用します")

        return model.to(self.device)

    def record_audio(self):
        """10秒間の音声を録音"""
        print(f"\n{self.duration}秒間録音を開始します...")
        print("録音中...")

        self.is_recording = True

        # 録音
        recording = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )

        sd.wait()  # 録音が完了するまで待機

        self.is_recording = False
        print("録音完了!")

        # バッファに保存
        self.audio_buffer = recording.squeeze()

        return self.audio_buffer

    def extract_features(self, audio):
        """音声からメル・スペクトログラムを抽出"""
        # numpy配列をTensorに変換
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        # チャネル次元を追加
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        # Mel spectrogram計算
        mel_spec = self.mel_transform(audio)

        # log scaleに変換
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)

        # 正規化（簡易版）
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)

        return mel_spec_db

    def detect_events(self, audio):
        """音響イベントを検出"""
        print("イベント検出中...")

        # 特徴抽出
        features = self.extract_features(audio)

        # バッチ次元を追加
        features = features.unsqueeze(0).to(self.device)

        # 推論
        with torch.no_grad():
            # CRNNモデルは (strong_preds, weak_preds) を返す
            strong_preds, weak_preds = self.model(features)

            # シグモイド適用
            strong_preds = torch.sigmoid(strong_preds)
            weak_preds = torch.sigmoid(weak_preds)

        # CPU に移動してnumpy配列に変換
        strong_preds = strong_preds.cpu().numpy()[0]  # (T, n_classes)
        weak_preds = weak_preds.cpu().numpy()[0]  # (n_classes,)

        self.detection_results = {
            'strong': strong_preds,
            'weak': weak_preds,
            'features': features.cpu().numpy()[0, 0]  # メル・スペクトログラム
        }

        print("検出完了!")
        return self.detection_results

    def visualize_results(self, save_path=None):
        """検出結果を可視化"""
        if self.detection_results is None:
            print("検出結果がありません")
            return

        strong_preds = self.detection_results['strong']
        weak_preds = self.detection_results['weak']
        mel_spec = self.detection_results['features']

        # 閾値
        threshold = 0.5

        # フレーム数から時間軸を計算
        n_frames = strong_preds.shape[0]
        time_axis = np.linspace(0, self.duration, n_frames)

        # 可視化
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 2, 1], hspace=0.3)

        # 1. メル・スペクトログラム
        ax1 = fig.add_subplot(gs[0])
        img = ax1.imshow(
            mel_spec,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            extent=[0, self.duration, 0, self.n_mels]
        )
        ax1.set_ylabel('Mel bins')
        ax1.set_xlabel('Time (s)')
        ax1.set_title('Mel Spectrogram')
        plt.colorbar(img, ax=ax1, label='Amplitude (dB)')

        # 2. フレームレベル予測（ヒートマップ）
        ax2 = fig.add_subplot(gs[1])

        # 閾値以上の予測のみを表示
        strong_preds_thresholded = strong_preds.T.copy()
        strong_preds_thresholded[strong_preds_thresholded < threshold] = 0

        img2 = ax2.imshow(
            strong_preds_thresholded,
            aspect='auto',
            origin='lower',
            cmap='hot',
            interpolation='nearest',
            extent=[0, self.duration, 0, self.n_classes],
            vmin=0,
            vmax=1
        )
        ax2.set_yticks(np.arange(self.n_classes) + 0.5)
        ax2.set_yticklabels(self.class_labels, fontsize=8)
        ax2.set_xlabel('Time (s)')
        ax2.set_title(f'Frame-level Predictions (threshold={threshold})')
        plt.colorbar(img2, ax=ax2, label='Confidence')

        # グリッド線を追加
        ax2.set_yticks(np.arange(self.n_classes + 1), minor=True)
        ax2.grid(which='minor', axis='y', color='white', linewidth=0.5)

        # 3. クリップレベル予測（棒グラフ）
        ax3 = fig.add_subplot(gs[2])

        # 閾値以上のクラスのみを抽出
        detected_indices = np.where(weak_preds >= threshold)[0]

        if len(detected_indices) > 0:
            detected_labels = [self.class_labels[i] for i in detected_indices]
            detected_scores = weak_preds[detected_indices]

            colors = ['red' if score >= 0.7 else 'orange' if score >= 0.5 else 'yellow'
                     for score in detected_scores]

            bars = ax3.barh(detected_labels, detected_scores, color=colors, alpha=0.7)
            ax3.set_xlabel('Confidence')
            ax3.set_xlim([0, 1])
            ax3.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold={threshold}')
            ax3.set_title('Clip-level Predictions (Detected Events)')
            ax3.legend()

            # 値をバーに表示
            for bar, score in zip(bars, detected_scores):
                ax3.text(score + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{score:.2f}', va='center', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'No events detected',
                    ha='center', va='center', fontsize=14, transform=ax3.transAxes)
            ax3.set_xlim([0, 1])
            ax3.set_title('Clip-level Predictions (No Events Detected)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可視化結果を保存しました: {save_path}")

        plt.show()

    def print_detection_summary(self, threshold=0.5):
        """検出結果のサマリーを表示"""
        if self.detection_results is None:
            print("検出結果がありません")
            return

        weak_preds = self.detection_results['weak']
        strong_preds = self.detection_results['strong']

        print("\n" + "="*60)
        print("検出結果サマリー")
        print("="*60)

        # クリップレベルで検出されたイベント
        detected_indices = np.where(weak_preds >= threshold)[0]

        if len(detected_indices) > 0:
            print(f"\n検出されたイベント ({len(detected_indices)}個):")
            print("-" * 60)

            for idx in detected_indices:
                label = self.class_labels[idx]
                score = weak_preds[idx]

                # フレームレベルで検出された時間範囲を計算
                frame_detections = strong_preds[:, idx] >= threshold
                if frame_detections.any():
                    n_frames = strong_preds.shape[0]
                    time_per_frame = self.duration / n_frames

                    # 検出された期間を計算
                    detected_frames = np.where(frame_detections)[0]
                    start_time = detected_frames[0] * time_per_frame
                    end_time = (detected_frames[-1] + 1) * time_per_frame
                    duration = end_time - start_time

                    print(f"  {label:30s} | Score: {score:.3f} | "
                          f"Time: {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}s)")
                else:
                    print(f"  {label:30s} | Score: {score:.3f}")
        else:
            print("\nイベントは検出されませんでした")

        print("="*60)

    def run_continuous(self, save_dir=None):
        """継続的にリアルタイム検出を実行"""
        print("\n" + "="*60)
        print("リアルタイムSEDシステム")
        print("="*60)
        print(f"サンプルレート: {self.sample_rate} Hz")
        print(f"録音時間: {self.duration} 秒")
        print(f"クラス数: {self.n_classes}")
        print("="*60)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        try:
            iteration = 0
            while True:
                iteration += 1
                print(f"\n[反復 {iteration}]")

                # 録音
                audio = self.record_audio()

                # イベント検出
                results = self.detect_events(audio)

                # 結果表示
                self.print_detection_summary()

                # 可視化
                save_path = None
                if save_dir:
                    save_path = os.path.join(save_dir, f'detection_{iteration:03d}.png')

                self.visualize_results(save_path)

                # 続行確認
                response = input("\n続けますか? (y/n): ").strip().lower()
                if response != 'y':
                    break

        except KeyboardInterrupt:
            print("\n\n中断されました")

        print("\nシステムを終了します")


def main():
    parser = argparse.ArgumentParser(description='リアルタイム音響イベント検出システム')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default="epoch=259-step=30680.ckpt",
        help='モデルチェックポイントのパス (.ckpt)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='DESED_task/dcase2024_task4_baseline/confs/pretrained.yaml',
        help='設定ファイルのパス'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='使用デバイス'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default=None,
        help='可視化結果の保存ディレクトリ'
    )
    parser.add_argument(
        '--continuous',
        action='store_true',
        help='継続的に検出を実行'
    )

    args = parser.parse_args()

    # デバイスの設定
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDAが利用できないため、CPUを使用します")
        args.device = 'cpu'

    # システムの初期化
    sed_system = RealtimeSED(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )

    if args.continuous:
        # 継続モード
        sed_system.run_continuous(save_dir=args.save_dir)
    else:
        # 1回のみ実行
        audio = sed_system.record_audio()
        results = sed_system.detect_events(audio)
        sed_system.print_detection_summary()

        save_path = None
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, 'detection.png')

        sed_system.visualize_results(save_path)


if __name__ == '__main__':
    main()
