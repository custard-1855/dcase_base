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

    content_mean = content_feature.mean(dim=(1,3), keepdim=True)
    content_var = content_feature.var(dim=(1,3), keepdim=True)
    content_std = (content_var + 1e-6).sqrt()

    content_mean, content_std = content_mean.detach(), content_std.detach()
    content_normed = (content_feature - content_mean) / (content_std)

    # x_styleを内部で用意
    B = content_feature.size(0)
    perm = torch.randperm(B, device=content_feature.device) # ランダムにシャッフル 本当はチャンクごとに区切るのが良い??? 現実的に適用する場合はランダムになりそうだが...

    mu2, sig2 = content_mean[perm], content_std[perm]
    # style_mean, style_std = content_mean[perm].mean(dim=1, keepdim=True), content_mean[perm].std(dim=1, keepdim=True)

    # ランダムな比率で統計量を混ぜる
    # lam = random.uniform(0, 1)
    alpha = 0.1
    lam = Beta(alpha, alpha).sample((B, 1, 1, 1))
    lam = lam.to(device=content_feature.device)

    # mixed_mean = lam * style_mean + (1 - lam) * content_mean
    # mixed_std = lam * style_std + (1 - lam) * content_std
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
            x_content (Tensor): スタイル適用対象の特徴量 (Batch, Channels, Freq, Time)
            # x_style (Tensor): スタイルソースの特徴量 (Batch, Channels, Freq, Time)
        """
        # --- 1. 時間次元の情報を集約 ---
        # 時間方向の平均をとって、各周波数の静的な特徴を取得
        # (B, C, F, T) -> (B, C, F)
        x_content_avg = x_content.mean(dim=3)

        # --- 2. Attentionで周波数ごとの重みを計算 ---
        # (B, C, F) -> (B, 1, F)
        attn_logits = self.attention_network(x_content_avg)

        # Sigmoid関数で重みを0〜1の範囲に正規化
        # 各周波数が独立して重要かどうかを判断するため、SoftmaxよりSigmoidが適している場合が多い
        # (B, 1, F) -> (B, 1, F, 1) に変形してブロードキャスト可能にする
        attn_weights = torch.sigmoid(attn_logits).unsqueeze(-1)

        # --- 3. MixStyleを適用 ---
        # スタイルを混ぜた特徴量を生成
        x_mixed = mix_style(x_content)

        # --- 4. 計算した重みで元の特徴量と混ぜ合わせる ---
        # 重みが大きい周波数帯ほど、スタイルが混ざった特徴量(x_mixed)の比率が高くなる
        # x_out = (重み * スタイル適用後) + ((1 - 重み) * 元の特徴量)
        output = attn_weights * x_mixed + (1 - attn_weights) * x_content

        return output

# # --- 使用例 ---
# if __name__ == '__main__':
#     # パラメータ設定
#     BATCH_SIZE = 4
#     CHANNELS = 256
#     FREQ_BINS = 80  # e.g., 80-dim melspectrogram
#     TIME_STEPS = 100

#     # ダミーの入力データを作成
#     content_input = torch.randn(BATCH_SIZE, CHANNELS, FREQ_BINS, TIME_STEPS)
#     style_input = torch.randn(BATCH_SIZE, CHANNELS, FREQ_BINS, TIME_STEPS) # 別の話者や感情のデータ

#     # モデルを初期化
#     attn_mixstyle_layer = FrequencyAttentionMixStyle(channels=CHANNELS, freq_bins=FREQ_BINS)

#     # 実行
#     output_feature = attn_mixstyle_layer(content_input, style_input)

#     # 結果の形状を確認
#     print("Input shape:", content_input.shape)
#     print("Output shape:", output_feature.shape)
#     # >>> Input shape: torch.Size([4, 256, 80, 100])
#     # >>> Output shape: torch.Size([4, 256, 80, 100])