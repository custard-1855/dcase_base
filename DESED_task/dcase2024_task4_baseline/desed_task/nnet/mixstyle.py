import torch
import torch.nn as nn
import torch.nn.functional as F
# import random
from torch.distributions.beta import Beta


# --- MixStyleのコア機能 ---
# 2つの特徴量の統計量（スタイル）を混ぜ合わせる関数
def mix_style(content_feature): # ok
    """
    Args:
        content_feature (Tensor): スタイルを適用したい元の特徴量
        style_feature (Tensor): スタイルを提供するための特徴量
    Returns:
        Tensor: スタイルが適用された新しい特徴量
    """
    # チャンネル次元で統計量を計算
    # 若干実装がRFNに寄っている

    # input size : (batch_size, n_channels, n_frames, n_freq)
    # 3は周波数
    # 時間を割るよう,3 > 2に修正

    content_mean = content_feature.mean(dim=(1,2), keepdim=True)
    content_var = content_feature.var(dim=(1,2), keepdim=True)
    content_std = (content_var + 1e-6).sqrt()

    content_mean, content_std = content_mean.detach(), content_std.detach()
    content_normed = (content_feature - content_mean) / (content_std)

    # x_styleを内部で用意
    B = content_feature.size(0)
    perm = torch.randperm(B, device=content_feature.device) # ランダムにシャッフル 本当はチャンクごとに区切るのが良い??? 現実的に適用する場合はランダムになりそうだが...

    mu2, sig2 = content_mean[perm], content_std[perm]

    # ランダムな比率で統計量を混ぜる
    alpha = 0.1
    lam = Beta(alpha, alpha).sample((B, 1, 1, 1))
    lam = lam.to(device=content_feature.device)

    mixed_mean = lam * content_mean + (1 - lam) * mu2
    mixed_std = lam * content_std + (1 - lam) *  sig2

    # 元の特徴量を新しい統計量で正規化・スケールし直す
    return  content_normed * mixed_std + mixed_mean

# --- AttentionとMixStyleを組み合わせたモジュール ---
class FrequencyAttentionMixStyle(nn.Module):
    def __init__(self, channels):
        """
        Args:
            channels (int): 入力特徴量のチャンネル数
            freq_bins (int): 周波数次元のサイズ (e.g., メルスペクトログラムの次元数)
        """
        super().__init__()
        self.channels = channels
        # self.freq_bins = freq_bins

        # 周波数ごとの重要度を学習するための小さなAttentionネットワーク
        # ここでは単純なConv1dを使用
        specific_channels = channels if channels == 1 else channels // 2


        self.attention_network = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=specific_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=specific_channels, out_channels=1, kernel_size=1)
        )

    def forward(self, x_content):
        """
        Args:
            x_content (Tensor): スタイル適用対象の特徴量 (Batch, Channels, Frame, Frequency)
            # x_style (Tensor): スタイルソースの特徴量 (Batch, Channels, Frame, Frequency)
        """
        # --- 1. 時間次元の情報を集約 ---
        # 時間方向の平均をとって、各周波数の静的な特徴を取得
        # input size : (batch_size, n_channels, n_frames, n_freq)

        x_content_avg = x_content.mean(dim=2) # Time(Frames)方向で平均を取得
        # print("[DEBUG]", x_content.size()) # [DEBUG] torch.Size([59, 1, 626, 128])
        # print("[DEBUG]", x_content_avg.size()) # [DEBUG] torch.Size([59, 1, 128])

        # --- 2. Attentionで周波数ごとの重みを計算 ---
        # (B, C, F) -> (B, 1, F)
        attn_logits = self.attention_network(x_content_avg)
        # print("[DEBUG]", attn_logits.size()) # [DEBUG] torch.Size([59, 1, 128])


        # Sigmoid関数で重みを0〜1の範囲に正規化
        # 各周波数が独立して重要かどうかを判断するため、SoftmaxよりSigmoidが適している場合が多い
        # (B, 1, F) -> (B, 1, 1, F) に変形してブロードキャスト可能にする
        attn_weights = torch.sigmoid(attn_logits).unsqueeze(-2)
        # print("[DEBUG]", attn_weights.size()) #[DEBUG] torch.Size([59, 1, 128, 1]) < 本来Time(Frame)が入る部分がFrequencyになっている


        # --- 3. MixStyleを適用 ---
        # スタイルを混ぜた特徴量を生成
        x_mixed = mix_style(x_content)
        # print("[DEBUG]", x_mixed.size()) # [DEBUG] torch.Size([59, 1, 626, 128])


        # --- 4. 計算した重みで元の特徴量と混ぜ合わせる ---

        output = attn_weights * x_mixed

        return output
