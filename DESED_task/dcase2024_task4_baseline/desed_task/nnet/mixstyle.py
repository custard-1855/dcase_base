"""MixStyle with Frequency Attention for Sound Event Detection.

This module implements various frequency-wise MixStyle augmentation techniques:
1. Basic MixStyle augmentation (mix_style)
2. CNN-based frequency attention MixStyle (FrequencyAttentionMixStyle)
3. Transformer-based frequency attention MixStyle (FrequencyTransformerMixStyle)
"""

import torch
from torch import nn
from torch.distributions.beta import Beta


# Auxiliary Modules for Attention Networks
class ResidualConvBlock(nn.Module):
    """Residual convolutional block with skip connection."""

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
        return self.relu(x + self.conv(x))


class MultiScaleConvBlock(nn.Module):
    """Multi-scale convolutional block with parallel branches of different kernel sizes."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
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
        out = torch.cat([x1, x3, x5], dim=1)
        return self.relu(self.bn(out))


# Core MixStyle Function
def calc_weighted_stats(x, attn_map):
    """Attentionマップを重みとして平均と標準偏差を計算.

    x: [Batch, Channel, Time, Freq]
    attn_map: [Batch, Channel, 1, Freq]
    """
    EPSILON = 1e-6
    attn_sum = attn_map.sum(dim=(2), keepdim=True) + EPSILON

    # 1. 重み付き平均 (Weighted Mean)
    # \mu = \sum(x * w) / \sum(w)
    mean = (x * attn_map).sum(dim=(2), keepdim=True) / attn_sum

    # 2. 重み付き分散 (Weighted Variance)
    # \sigma^2 = \sum(w * (x - \mu)^2) / \sum(w)
    var = ((x - mean) ** 2 * attn_map).sum(dim=(2), keepdim=True) / attn_sum
    std = torch.sqrt(var + EPSILON)

    return mean, std


def mix_style(x, attn_map, use_attn):
    """Apply MixStyle augmentation by mixing statistics across the batch.

    Args:
        x: Input features (batch_size, n_channels, n_frames, n_freq)

    Returns:
        Augmented features with mixed statistics

    """
    # Probability of applying augmentation
    AUGMENTATION_PROB = 0.5
    if torch.rand(1).item() > AUGMENTATION_PROB:
        return x

    # Compute mean and std along time dimension
    if use_attn == "able":
        x_mean, x_std = calc_weighted_stats(x, attn_map)
    else:
        EPSILON = 1e-6
        x_mean = x.mean(dim=(2), keepdim=True)
        x_var = x.var(dim=(2), keepdim=True)
        x_std = (x_var + EPSILON).sqrt()

    x_normed = (x - x_mean) / x_std

    # Shuffle batch to get different style statistics
    B = x.size(0)
    perm = torch.randperm(B, device=x.device)
    mu2, sig2 = x_mean[perm], x_std[perm]

    # Mix statistics with random ratio
    BETA_ALPHA = 0.1
    lam = Beta(BETA_ALPHA, BETA_ALPHA).sample((B, 1, 1, 1))
    lam = lam.to(device=x.device)

    mixed_mean = lam * x_mean + (1 - lam) * mu2
    mixed_std = lam * x_std + (1 - lam) * sig2

    return x_normed * mixed_std + mixed_mean


# Main MixStyle Classes
class BasicMixStyleWrapper(nn.Module):
    """Wrapper for pure MixStyle without attention mechanism.

    This is used for baseline experiments (B0) that only apply
    basic MixStyle augmentation without frequency-aware attention.
    """

    def __init__(self, channels=None, **kwargs):
        """Initialize BasicMixStyleWrapper.

        Args:
            channels: Not used, but kept for API compatibility
            **kwargs: Not used, but kept for API compatibility

        """
        super().__init__()
        # No parameters needed - just wraps mix_style function

    def forward(self, x_content):
        """Forward pass applying pure MixStyle.

        Args:
            x_content: Input features (Batch, Channels, Frame, Frequency)

        Returns:
            Output features with pure MixStyle applied

        """
        return mix_style(x_content, None, None)


class FrequencyAttentionMixStyle(nn.Module):
    """CNN-based frequency attention MixStyle.

    Args:
        channels: Number of input channels
        **kwargs: Configuration options
            - attn_type: Attention network architecture
                * "default": Shallow 2-layer CNN
                * "residual_deep": Deep network with residual connections
                * "multiscale": Multi-scale convolution
            - attn_deepen: Network depth (number of layers)

    """

    def __init__(self, channels, **kwargs):
        super().__init__()
        self.channels = channels

        # Configuration options
        self.attention_type = kwargs.get("attn_type", "default")
        self.deepen = kwargs.get("attn_deepen", 2)

        # Calculate intermediate channel dimensions
        specific_channels = channels if channels == 1 else channels // 2

        # Build attention network
        self.attention_network = self._build_attention_network(
            channels,
            specific_channels,
            self.attention_type,
            int(self.deepen),
        )

    def _build_attention_network(self, in_channels, mid_channels, attn_type, depth):
        """Build attention network based on specified architecture.

        Args:
            in_channels: Input channel dimension
            mid_channels: Intermediate channel dimension
            attn_type: Attention network type
            depth: Network depth

        Returns:
            Constructed attention network module

        """
        if attn_type == "default":
            network = nn.Sequential(
                nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(mid_channels, in_channels, kernel_size=1),
            )
            # Initialize final layer bias to 1.0 to avoid sigmoid saturation at 0.5
            # sigmoid(1.0) = 0.73, providing bias towards mixed features
            nn.init.constant_(network[-1].bias, 1.0)
            return network

        if attn_type == "residual_deep":
            layers = [
                nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(),
            ]
            for _ in range(depth):
                layers.append(ResidualConvBlock(mid_channels, kernel_size=3))
            layers.append(nn.Conv1d(mid_channels, in_channels, kernel_size=1))
            network = nn.Sequential(*layers)
            # Initialize final layer bias to 1.0
            nn.init.constant_(network[-1].bias, 1.0)
            return network

        if attn_type == "multiscale":
            layers = [MultiScaleConvBlock(in_channels, mid_channels)]
            for _ in range(depth - 1):
                layers.extend(
                    [
                        nn.Conv1d(mid_channels, mid_channels, kernel_size=3, padding=1),
                        nn.BatchNorm1d(mid_channels),
                        nn.ReLU(),
                    ],
                )

            layers.append(nn.Conv1d(mid_channels, in_channels, kernel_size=1))
            network = nn.Sequential(*layers)
            # Initialize final layer bias to 1.0
            nn.init.constant_(network[-1].bias, 1.0)
            return network

        msg = (
            f"Unknown attention_type: {attn_type}. "
            f"Choose from ['default', 'residual_deep', 'multiscale', 'se_deep', 'dilated_deep']"
        )
        raise ValueError(msg)

    def forward(self, x_content):
        """Forward pass.

        Args:
            x_content: Input features (Batch, Channels, Frame, Frequency)

        Returns:
            Output features with frequency-aware MixStyle applied

        """
        x_avg = x_content.mean(dim=2)  # (B, C, F)
        attn_logits = self.attention_network(x_avg)

        attn_weights = torch.sigmoid(attn_logits).unsqueeze(-2)  # (B, C, 1, F)
        x_mixed = mix_style(x_content, attn_weights, "able")
        return x_content + x_mixed


class FrequencyTransformerMixStyle(nn.Module):
    """Transformer-based frequency attention MixStyle with self-attention.

    Args:
        channels: Number of input channels
        n_freq: Frequency dimension size (for positional encoding)
        **kwargs: Configuration options
            - n_heads: Number of attention heads (default: 4)
            - ff_dim: Feed-forward intermediate dimension (default: 256)
            - n_layers: Number of transformer blocks (default: 1)
            - mixstyle_dropout: Dropout probability (default: 0.1)

    """

    def __init__(self, channels, n_freq=None, **kwargs):
        super().__init__()
        self.channels = channels

        # Transformer configuration
        self.n_heads = kwargs.get("n_heads", 4)
        # Auto-adjust n_heads if channels < n_heads (embed_dim must be divisible by n_heads)
        self.n_heads = min(self.n_heads, channels)
        self.ff_dim = kwargs.get("ff_dim", 256)
        self.n_layers = kwargs.get("n_layers", 1)
        self.dropout = kwargs.get("mixstyle_dropout", 0.1)

        # Positional encoding for frequency dimension
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, n_freq)) if n_freq else None

        # Build transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(channels, self.n_heads, self.ff_dim, self.dropout)
                for _ in range(self.n_layers)
            ],
        )

        # Output projection to generate frequency weights
        self.output_projection = nn.Linear(channels, channels)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input features (Batch, Channels, Frame, Frequency)

        Returns:
            Output features with transformer-based frequency attention

        """
        x_avg = x.mean(dim=2)  # (B, C, F)

        # Add positional encoding
        if self.pos_encoding is not None:
            x_avg = x_avg + self.pos_encoding

        # Transpose for transformer: (B, C, F) -> (B, F, C)
        x_avg = x_avg.transpose(1, 2)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x_avg = block(x_avg)

        # Transpose back: (B, F, C) -> (B, C, F)
        x_avg = x_avg.transpose(1, 2)

        # Generate frequency weights
        freq_weights = torch.sigmoid(self.output_projection(x_avg.transpose(1, 2)))
        freq_weights = freq_weights.transpose(1, 2).unsqueeze(2)  # (B, C, 1, F)

        x_mixed = mix_style(x, freq_weights, "able")

        return x + x_mixed


# Transformer Components
class TransformerBlock(nn.Module):
    """Standard transformer block with multi-head self-attention and feed-forward."""

    def __init__(self, embed_dim, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor (Batch, Seq_len, Embed_dim)

        Returns:
            Output tensor with same shape

        """
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x
