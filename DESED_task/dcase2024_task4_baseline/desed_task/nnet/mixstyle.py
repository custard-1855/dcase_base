import torch
import torch.nn.functional as F
from torch import nn

# import random
from torch.distributions.beta import Beta


class ResidualConvBlock(nn.Module):
    """残差接続付き畳み込みブロック"""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.conv(x))  # 残差接続


class MultiScaleConvBlock(nn.Module):
    """マルチスケール畳み込みブロック（異なる受容野を並列処理）"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 異なるkernel sizeで周波数パターンを捉える
        out_per_branch = out_channels // 3
        remainder = out_channels - (out_per_branch * 3)

        self.conv1 = nn.Conv1d(in_channels, out_per_branch + remainder, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels, out_per_branch, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_per_branch, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        out = torch.cat([x1, x3, x5], dim=1)  # チャネル方向に結合
        return self.relu(self.bn(out))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block (チャネルattention)"""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.excitation = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 1)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 1), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        # Squeeze: グローバルな情報を集約
        y = self.squeeze(x).view(b, c)
        # Excitation: チャネル重要度を計算
        y = self.excitation(y).view(b, c, 1)
        # 元の特徴量に重み付け
        return x * y.expand_as(x)


def mix_style(content_feature):
    """Args:
        content_feature (Tensor): スタイルを適用したい元の特徴量
            (batch_size, n_channels, n_frames, n_freq)

    Returns:
        Tensor: スタイルが適用された新しい特徴量

    """
    p = 0.5
    if torch.rand(1).item() > p:
        return content_feature

    content_mean = content_feature.mean(dim=(2), keepdim=True)
    content_var = content_feature.var(dim=(2), keepdim=True)
    content_std = (content_var + 1e-6).sqrt()

    content_mean, content_std = content_mean.detach(), content_std.detach()
    content_normed = (content_feature - content_mean) / (content_std)

    # x_styleを内部で用意
    B = content_feature.size(0)
    perm = torch.randperm(B, device=content_feature.device)

    mu2, sig2 = content_mean[perm], content_std[perm]

    # ランダムな比率で統計量を混ぜる
    alpha = 0.1  # デフォルトは0.1
    lam = Beta(alpha, alpha).sample((B, 1, 1, 1))
    lam = lam.to(device=content_feature.device)

    mixed_mean = lam * content_mean + (1 - lam) * mu2
    mixed_std = lam * content_std + (1 - lam) * sig2

    # 元の特徴量を新しい統計量で正規化・スケールし直す
    return content_normed * mixed_std + mixed_mean


class FrequencyAttentionMixStyle(nn.Module):
    def __init__(self, channels, **kwargs):
        """Args:
        channels (int): 入力特徴量のチャンネル数
        kwargs:
            attn_type (str): attention network type
                - "default": 浅い2層CNN
                - "residual_deep": 残差接続で深いネットワーク
                - "multiscale": マルチスケール畳み込み
                - "se_deep": SE-Block組み込み深層ネットワーク
                - "dilated_deep": Dilated Convolution
            attn_deepen (int): 深さ（層数）
            mixstyle_type (str): MixStyleとの組み合わせ方

        """
        super().__init__()
        self.channels = channels

        # 実験設定変更用
        self.attention_type = kwargs.get("attn_type", "default")
        self.deepen = kwargs.get("attn_deepen", 2)
        self.mixstyle_type = kwargs["mixstyle_type"]

        # チャネル数の計算
        specific_channels = channels if channels == 1 else channels // 2

        # Attention Networkの構築（attention_typeに応じて切り替え）
        self.attention_network = self._build_attention_network(
            channels,
            specific_channels,
            self.attention_type,
            int(self.deepen),
        )

    def _build_attention_network(self, in_channels, mid_channels, attn_type, depth):
        """Attention Networkを構築.

        Args:
            in_channels: 入力チャネル数
            mid_channels: 中間層のチャネル数
            attn_type: attentionのタイプ
            depth: ネットワークの深さ

        Returns:
            nn.Module: 構築されたattention network

        """
        if attn_type == "default":
            # デフォルト: 浅い2層CNN
            return nn.Sequential(
                nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(mid_channels, in_channels, kernel_size=1),
            )

        if attn_type == "residual_deep":
            # 残差接続で深いネットワーク
            layers = [
                nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(),
            ]

            # 残差ブロックを複数積み重ね
            for _ in range(depth):
                layers.append(ResidualConvBlock(mid_channels, kernel_size=3))

            # 出力変換
            layers.append(nn.Conv1d(mid_channels, in_channels, kernel_size=1))

            return nn.Sequential(*layers)

        if attn_type == "multiscale":
            # マルチスケール畳み込み
            layers = []
            layers.append(MultiScaleConvBlock(in_channels, mid_channels))

            # 追加の層
            for _ in range(depth - 1):
                layers.extend(
                    [
                        nn.Conv1d(mid_channels, mid_channels, kernel_size=3, padding=1),
                        nn.BatchNorm1d(mid_channels),
                        nn.ReLU(),
                    ],
                )

            layers.append(nn.Conv1d(mid_channels, in_channels, kernel_size=1))

            return nn.Sequential(*layers)

        if attn_type == "se_deep":
            # SE-Block組み込み深層ネットワーク
            layers = [
                nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(),
            ]

            # SE Blockを挟みながら層を深くする
            for _ in range(depth):
                layers.extend(
                    [
                        nn.Conv1d(mid_channels, mid_channels, kernel_size=3, padding=1),
                        nn.BatchNorm1d(mid_channels),
                        nn.ReLU(),
                        SEBlock(mid_channels, reduction=4),
                    ],
                )

            layers.append(nn.Conv1d(mid_channels, in_channels, kernel_size=1))

            return nn.Sequential(*layers)

        if attn_type == "dilated_deep":
            # Dilated Convolution（広い受容野）
            layers = [
                nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(),
            ]

            # Dilationを増やしながら層を深くする
            for i in range(depth):
                dilation = 2**i  # 1, 2, 4, 8, ...
                padding = dilation  # dilationに応じてpaddingを調整
                layers.extend(
                    [
                        nn.Conv1d(
                            mid_channels,
                            mid_channels,
                            kernel_size=3,
                            padding=padding,
                            dilation=dilation,
                        ),
                        nn.BatchNorm1d(mid_channels),
                        nn.ReLU(),
                    ],
                )

            layers.append(nn.Conv1d(mid_channels, in_channels, kernel_size=1))

            return nn.Sequential(*layers)

        raise ValueError(
            f"Unknown attention_type: {attn_type}. "
            f"Choose from ['default', 'residual_deep', 'multiscale', 'se_deep', 'dilated_deep']",
        )

    def forward(self, x_content):
        """x_content (Tensor): スタイル適用対象の特徴量 (Batch, Channels, Frame, Frequency)."""
        x_mixed = mix_style(x_content)

        # 時間方向の平均をとり、周波数の静的な特徴を取得
        x_avg = x_mixed.mean(dim=2)  # 実際に使用するMixedの周波数で重要度を得る

        # 周波数動的注意 重みを作成
        attn_logits = self.attention_network(x_avg)  # (B, C, F)

        # Sigmoid関数で重みを0〜1の範囲に正規化
        # (B, C, F) -> (B, C, 1, F) に変形してブロードキャスト可能に
        attn_weights = torch.sigmoid(attn_logits).unsqueeze(-2)

        output = attn_weights * x_mixed + (1 - attn_weights) * x_content
        return output
