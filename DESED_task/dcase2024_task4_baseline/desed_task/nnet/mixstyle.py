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
            blend_type (str): 出力の合成方法
                - "linear": attn * x_mixed + (1 - attn) * x_content (default)
                - "residual": x_content + attn * (x_mixed - x_content)
            attn_input (str): attention計算の入力選択
                - "mixed": x_mixed の周波数特徴から計算 (default)
                - "content": x_content の周波数特徴から計算
                - "dual_stream": 両方から計算して統合

        """
        super().__init__()
        self.channels = channels

        # 実験設定変更用
        self.attention_type = kwargs.get("attn_type", "default")
        self.deepen = kwargs.get("attn_deepen", 2)
        self.mixstyle_type = kwargs["mixstyle_type"]
        self.blend_type = kwargs.get("blend_type", "linear")
        self.attn_input = kwargs.get("attn_input", "mixed")

        # チャネル数の計算
        specific_channels = channels if channels == 1 else channels // 2

        # Attention Networkの構築（attention_typeに応じて切り替え）
        self.attention_network = self._build_attention_network(
            channels,
            specific_channels,
            self.attention_type,
            int(self.deepen),
        )

        # Dual-streamの場合は追加のネットワークを構築
        if self.attn_input == "dual_stream":
            self.attention_network_content = self._build_attention_network(
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

        # Attention計算の入力を選択
        if self.attn_input == "mixed":
            # 時間方向の平均をとり、周波数の静的な特徴を取得
            x_avg = x_mixed.mean(dim=2)  # (B, C, F)
            attn_logits = self.attention_network(x_avg)

        elif self.attn_input == "content":
            # x_contentの周波数特徴から計算（ドメイン中立）
            x_avg = x_content.mean(dim=2)  # (B, C, F)
            attn_logits = self.attention_network(x_avg)

        elif self.attn_input == "dual_stream":
            # 両方から計算して統合
            x_avg_mixed = x_mixed.mean(dim=2)  # (B, C, F)
            x_avg_content = x_content.mean(dim=2)  # (B, C, F)

            attn_logits_mixed = self.attention_network(x_avg_mixed)
            attn_logits_content = self.attention_network_content(x_avg_content)

            # 2つのattention logitsを統合
            attn_logits = attn_logits_mixed + attn_logits_content

        else:
            raise ValueError(
                f"Unknown attn_input: {self.attn_input}. "
                f"Choose from ['mixed', 'content', 'dual_stream']",
            )

        # Sigmoid関数で重みを0〜1の範囲に正規化
        # (B, C, F) -> (B, C, 1, F) に変形してブロードキャスト可能に
        attn_weights = torch.sigmoid(attn_logits).unsqueeze(-2)

        # 出力の合成方法を選択
        if self.blend_type == "linear":
            # 線形補間（現在の実装）
            output = attn_weights * x_mixed + (1 - attn_weights) * x_content

        elif self.blend_type == "residual":
            # 残差的な定式化
            delta = x_mixed - x_content  # MixStyleによる変化量
            output = x_content + attn_weights * delta

        else:
            raise ValueError(
                f"Unknown blend_type: {self.blend_type}. "
                f"Choose from ['linear', 'residual']",
            )

        return output


class FrequencyTransformerMixStyle(nn.Module):
    """Transformer-based Frequency Attention MixStyle.

    周波数軸にself-attentionを適用して、周波数間の長距離依存をモデル化します。
    """

    def __init__(self, channels, n_freq=None, **kwargs):
        """Args:
            channels (int): 入力特徴量のチャンネル数
            n_freq (int, optional): 周波数次元のサイズ（Positional encodingで使用）
            kwargs:
                mixstyle_type (str): MixStyleとの組み合わせ方
                blend_type (str): 出力の合成方法
                    - "linear": attn * x_mixed + (1 - attn) * x_content (default)
                    - "residual": x_content + attn * (x_mixed - x_content)
                attn_input (str): attention計算の入力選択
                    - "mixed": x_mixed の周波数特徴から計算 (default)
                    - "content": x_content の周波数特徴から計算
                n_heads (int): Multi-head attentionのヘッド数 (default: 4)
                ff_dim (int): Feed-forwardの中間次元 (default: 256)
                n_layers (int): Transformerブロックの層数 (default: 1)
                mixstyle_dropout (float): Dropoutの確率 (default: 0.1)

        """
        super().__init__()
        self.channels = channels
        self.mixstyle_type = kwargs["mixstyle_type"]
        self.blend_type = kwargs.get("blend_type", "linear")
        self.attn_input = kwargs.get("attn_input", "mixed")

        # Transformer設定
        self.n_heads = kwargs.get("n_heads", 4)
        self.ff_dim = kwargs.get("ff_dim", 256)
        self.n_layers = kwargs.get("n_layers", 1)
        self.dropout = kwargs.get("mixstyle_dropout", 0.1)

        # Positional encoding（周波数位置の情報）
        if n_freq is not None:
            self.pos_encoding = nn.Parameter(torch.randn(1, 1, n_freq))
        else:
            self.pos_encoding = None

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    channels,
                    self.n_heads,
                    self.ff_dim,
                    self.dropout,
                )
                for _ in range(self.n_layers)
            ],
        )

        # 出力投影（周波数重みを生成）
        self.output_projection = nn.Linear(channels, channels)

    def forward(self, x_content):
        """x_content (Tensor): スタイル適用対象の特徴量 (Batch, Channels, Frame, Frequency)."""
        x_mixed = mix_style(x_content)

        # Attention計算の入力を選択
        if self.attn_input == "mixed":
            x_avg = x_mixed.mean(dim=2)  # (B, C, F)
        elif self.attn_input == "content":
            x_avg = x_content.mean(dim=2)  # (B, C, F)
        else:
            raise ValueError(
                f"Unknown attn_input: {self.attn_input}. "
                f"Choose from ['mixed', 'content']",
            )

        # Positional encoding
        if self.pos_encoding is not None:
            x_avg = x_avg + self.pos_encoding

        # (B, C, F) → (B, F, C) for attention
        x_avg = x_avg.transpose(1, 2)

        # Transformer blocks
        for block in self.transformer_blocks:
            x_avg = block(x_avg)

        # (B, F, C) → (B, C, F)
        x_avg = x_avg.transpose(1, 2)

        # 周波数重みを生成
        freq_weights = torch.sigmoid(self.output_projection(x_avg.transpose(1, 2)))
        freq_weights = freq_weights.transpose(1, 2).unsqueeze(2)  # (B, C, 1, F)

        # 出力の合成
        if self.blend_type == "linear":
            output = freq_weights * x_mixed + (1 - freq_weights) * x_content
        elif self.blend_type == "residual":
            delta = x_mixed - x_content
            output = x_content + freq_weights * delta
        else:
            raise ValueError(
                f"Unknown blend_type: {self.blend_type}. "
                f"Choose from ['linear', 'residual']",
            )

        return output


class TransformerBlock(nn.Module):
    """Transformer Block with Multi-head Self-attention and Feed-forward."""

    def __init__(self, embed_dim, n_heads, ff_dim, dropout=0.1):
        super().__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """x (Tensor): (Batch, Seq_len, Embed_dim)."""
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_out))

        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))

        return x


class CrossAttentionMixStyle(nn.Module):
    """Cross-Attention based Frequency Attention MixStyle.

    x_content (Query) と x_mixed (Key/Value) 間でcross-attentionを行い、
    MixStyleの効果を明示的に比較します。
    """

    def __init__(self, channels, n_freq=None, **kwargs):
        """Args:
            channels (int): 入力特徴量のチャンネル数
            n_freq (int, optional): 周波数次元のサイズ（Positional encodingで使用）
            kwargs:
                mixstyle_type (str): MixStyleとの組み合わせ方
                blend_type (str): 出力の合成方法
                    - "linear": attn * x_mixed + (1 - attn) * x_content (default)
                    - "residual": x_content + attn * (x_mixed - x_content)
                n_heads (int): Multi-head attentionのヘッド数 (default: 4)
                ff_dim (int): Feed-forwardの中間次元 (default: 256)
                n_layers (int): Transformerブロックの層数 (default: 1)
                mixstyle_dropout (float): Dropoutの確率 (default: 0.1)

        """
        super().__init__()
        self.channels = channels
        self.mixstyle_type = kwargs["mixstyle_type"]
        self.blend_type = kwargs.get("blend_type", "linear")

        # Transformer設定
        self.n_heads = kwargs.get("n_heads", 4)
        self.ff_dim = kwargs.get("ff_dim", 256)
        self.n_layers = kwargs.get("n_layers", 1)
        self.dropout = kwargs.get("mixstyle_dropout", 0.1)

        # Positional encoding（周波数位置の情報）
        if n_freq is not None:
            self.pos_encoding = nn.Parameter(torch.randn(1, 1, n_freq))
        else:
            self.pos_encoding = None

        # Cross-attention blocks
        self.cross_attn_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    channels,
                    self.n_heads,
                    self.ff_dim,
                    self.dropout,
                )
                for _ in range(self.n_layers)
            ],
        )

        # 出力投影（周波数重みを生成）
        self.output_projection = nn.Linear(channels, channels)

    def forward(self, x_content):
        """x_content (Tensor): スタイル適用対象の特徴量 (Batch, Channels, Frame, Frequency)."""
        x_mixed = mix_style(x_content)

        # 時間方向の平均
        q = x_content.mean(dim=2)  # (B, C, F) - Query
        kv = x_mixed.mean(dim=2)  # (B, C, F) - Key/Value

        # Positional encoding
        if self.pos_encoding is not None:
            q = q + self.pos_encoding
            kv = kv + self.pos_encoding

        # (B, C, F) → (B, F, C) for attention
        q = q.transpose(1, 2)
        kv = kv.transpose(1, 2)

        # Cross-attention blocks
        for block in self.cross_attn_blocks:
            q = block(q, kv)

        # (B, F, C) → (B, C, F)
        q = q.transpose(1, 2)

        # 周波数重みを生成
        freq_weights = torch.sigmoid(self.output_projection(q.transpose(1, 2)))
        freq_weights = freq_weights.transpose(1, 2).unsqueeze(2)  # (B, C, 1, F)

        # 出力の合成
        if self.blend_type == "linear":
            output = freq_weights * x_mixed + (1 - freq_weights) * x_content
        elif self.blend_type == "residual":
            delta = x_mixed - x_content
            output = x_content + freq_weights * delta
        else:
            raise ValueError(
                f"Unknown blend_type: {self.blend_type}. "
                f"Choose from ['linear', 'residual']",
            )

        return output


class CrossAttentionBlock(nn.Module):
    """Cross-Attention Block with Feed-forward."""

    def __init__(self, embed_dim, n_heads, ff_dim, dropout=0.1):
        super().__init__()

        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key_value):
        """Cross-attention forward.

        Args:
            query (Tensor): (Batch, Seq_len, Embed_dim)
            key_value (Tensor): (Batch, Seq_len, Embed_dim)

        """
        # Cross-attention with residual connection
        attn_out, _ = self.cross_attn(query, key_value, key_value)
        x = self.norm1(query + self.dropout1(attn_out))

        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))

        return x
