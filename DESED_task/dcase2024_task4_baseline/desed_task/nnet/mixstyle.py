"""MixStyle with Frequency Attention for Sound Event Detection.

This module implements various frequency-aware MixStyle augmentation techniques:
1. Basic MixStyle augmentation (mix_style)
2. CNN-based frequency attention MixStyle (FrequencyAttentionMixStyle)
3. Transformer-based frequency attention MixStyle (FrequencyTransformerMixStyle)
4. Cross-attention based MixStyle (CrossAttentionMixStyle)
"""

import torch
from torch import nn
from torch.distributions.beta import Beta

# =============================================================================
# Auxiliary Modules for Attention Networks
# =============================================================================


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


# =============================================================================
# Core MixStyle Function
# =============================================================================


def mix_style(content_feature):
    """Apply MixStyle augmentation by mixing statistics across the batch.

    Args:
        content_feature: Input features (batch_size, n_channels, n_frames, n_freq)

    Returns:
        Augmented features with mixed statistics

    """
    # Probability of applying augmentation
    AUGMENTATION_PROB = 0.5
    if torch.rand(1).item() > AUGMENTATION_PROB:
        return content_feature

    # Compute mean and std along time dimension
    EPSILON = 1e-6
    content_mean = content_feature.mean(dim=2, keepdim=True)
    content_var = content_feature.var(dim=2, keepdim=True)
    content_std = (content_var + EPSILON).sqrt()

    content_mean, content_std = content_mean.detach(), content_std.detach()
    content_normed = (content_feature - content_mean) / content_std

    # Shuffle batch to get different style statistics
    B = content_feature.size(0)
    perm = torch.randperm(B, device=content_feature.device)
    mu2, sig2 = content_mean[perm], content_std[perm]

    # Mix statistics with random ratio
    BETA_ALPHA = 0.1
    lam = Beta(BETA_ALPHA, BETA_ALPHA).sample((B, 1, 1, 1))
    lam = lam.to(device=content_feature.device)

    mixed_mean = lam * content_mean + (1 - lam) * mu2
    mixed_std = lam * content_std + (1 - lam) * sig2

    return content_normed * mixed_std + mixed_mean


# =============================================================================
# Main MixStyle Classes
# =============================================================================


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
        return mix_style(x_content)


class FrequencyAttentionMixStyle(nn.Module):
    """CNN-based frequency attention MixStyle.

    Args:
        channels: Number of input channels
        **kwargs: Configuration options
            - attn_type: Attention network architecture
                * "default": Shallow 2-layer CNN
                * "residual_deep": Deep network with residual connections
                * "multiscale": Multi-scale convolution
                * "se_deep": Deep network with SE-Blocks
                * "dilated_deep": Dilated convolution
            - attn_deepen: Network depth (number of layers)
            - mixstyle_type: How to combine with MixStyle
            - blend_type: Output blending method
                * "linear": attn * x_mixed + (1 - attn) * x_content (default)
                * "residual": x_content + attn * (x_mixed - x_content)
            - attn_input: Input for attention computation
                * "mixed": Compute from x_mixed frequency features (default)
                * "content": Compute from x_content frequency features
                * "dual_stream": Compute from both and integrate

    """

    def __init__(self, channels, **kwargs):
        super().__init__()
        self.channels = channels

        # Configuration options
        self.attention_type = kwargs.get("attn_type", "default")
        self.deepen = kwargs.get("attn_deepen", 2)
        self.mixstyle_type = kwargs["mixstyle_type"]
        self.blend_type = kwargs.get("blend_type", "linear")
        self.attn_input = kwargs.get("attn_input", "mixed")

        # Calculate intermediate channel dimensions
        specific_channels = channels if channels == 1 else channels // 2

        # Build attention network
        self.attention_network = self._build_attention_network(
            channels,
            specific_channels,
            self.attention_type,
            int(self.deepen),
        )

        # Build additional network for dual-stream mode
        if self.attn_input == "dual_stream":
            self.attention_network_content = self._build_attention_network(
                channels,
                specific_channels,
                self.attention_type,
                int(self.deepen),
            )
            # Learnable gate for combining mixed and content streams
            # Each channel can learn its own mixing weight
            self.dual_stream_gate = nn.Parameter(torch.ones(channels, 1) * 0.5)

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
        x_mixed = mix_style(x_content)

        # Select input for attention computation
        if self.attn_input == "mixed":
            x_avg = x_mixed.mean(dim=2)  # (B, C, F)
            attn_logits = self.attention_network(x_avg)

        elif self.attn_input == "content":
            x_avg = x_content.mean(dim=2)  # (B, C, F)
            attn_logits = self.attention_network(x_avg)

        elif self.attn_input == "dual_stream":
            x_avg_mixed = x_mixed.mean(dim=2)  # (B, C, F)
            x_avg_content = x_content.mean(dim=2)  # (B, C, F)

            attn_logits_mixed = self.attention_network(x_avg_mixed)
            attn_logits_content = self.attention_network_content(x_avg_content)

            # Learnable weighted combination instead of simple addition
            # gate is (C, 1), expand to (1, C, 1) for broadcasting
            gate = torch.sigmoid(self.dual_stream_gate).unsqueeze(0)  # (1, C, 1)
            attn_logits = gate * attn_logits_mixed + (1 - gate) * attn_logits_content

        else:
            msg = (
                f"Unknown attn_input: {self.attn_input}. "
                f"Choose from ['mixed', 'content', 'dual_stream']"
            )
            raise ValueError(msg)

        # Compute attention weights and reshape for broadcasting
        attn_weights = torch.sigmoid(attn_logits).unsqueeze(-2)  # (B, C, 1, F)

        # Blend outputs
        if self.blend_type == "linear":
            output = attn_weights * x_mixed + (1 - attn_weights) * x_content

        elif self.blend_type == "residual":
            delta = x_mixed - x_content
            output = x_content + attn_weights * delta

        else:
            msg = f"Unknown blend_type: {self.blend_type}. Choose from ['linear', 'residual']"
            raise ValueError(msg)

        return output


class FrequencyTransformerMixStyle(nn.Module):
    """Transformer-based frequency attention MixStyle with self-attention.

    Args:
        channels: Number of input channels
        n_freq: Frequency dimension size (for positional encoding)
        **kwargs: Configuration options
            - mixstyle_type: How to combine with MixStyle
            - blend_type: Output blending method
                * "linear": attn * x_mixed + (1 - attn) * x_content (default)
                * "residual": x_content + attn * (x_mixed - x_content)
            - attn_input: Input for attention computation
                * "mixed": Compute from x_mixed (default)
                * "content": Compute from x_content
            - n_heads: Number of attention heads (default: 4)
            - ff_dim: Feed-forward intermediate dimension (default: 256)
            - n_layers: Number of transformer blocks (default: 1)
            - mixstyle_dropout: Dropout probability (default: 0.1)

    """

    def __init__(self, channels, n_freq=None, **kwargs):
        super().__init__()
        self.channels = channels
        self.mixstyle_type = kwargs["mixstyle_type"]
        self.blend_type = kwargs.get("blend_type", "linear")
        self.attn_input = kwargs.get("attn_input", "mixed")

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

    def forward(self, x_content):
        """Forward pass.

        Args:
            x_content: Input features (Batch, Channels, Frame, Frequency)

        Returns:
            Output features with transformer-based frequency attention

        """
        x_mixed = mix_style(x_content)

        # Select input for attention computation
        if self.attn_input == "mixed":
            x_avg = x_mixed.mean(dim=2)  # (B, C, F)
        elif self.attn_input == "content":
            x_avg = x_content.mean(dim=2)  # (B, C, F)
        else:
            msg = f"Unknown attn_input: {self.attn_input}. Choose from ['mixed', 'content']"
            raise ValueError(msg)

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

        # Blend outputs
        if self.blend_type == "linear":
            output = freq_weights * x_mixed + (1 - freq_weights) * x_content
        elif self.blend_type == "residual":
            delta = x_mixed - x_content
            output = x_content + freq_weights * delta
        else:
            msg = f"Unknown blend_type: {self.blend_type}. Choose from ['linear', 'residual']"
            raise ValueError(msg)

        return output


# =============================================================================
# Transformer Components
# =============================================================================


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


class CrossAttentionMixStyle(nn.Module):
    """Cross-attention based frequency attention MixStyle.

    Uses cross-attention between x_content (Query) and x_mixed (Key/Value)
    to explicitly compare MixStyle effects.

    Args:
        channels: Number of input channels
        n_freq: Frequency dimension size (for positional encoding)
        **kwargs: Configuration options
            - mixstyle_type: How to combine with MixStyle
            - blend_type: Output blending method
                * "linear": attn * x_mixed + (1 - attn) * x_content (default)
                * "residual": x_content + attn * (x_mixed - x_content)
            - n_heads: Number of attention heads (default: 4)
            - ff_dim: Feed-forward intermediate dimension (default: 256)
            - n_layers: Number of transformer blocks (default: 1)
            - mixstyle_dropout: Dropout probability (default: 0.1)

    """

    def __init__(self, channels, n_freq=None, **kwargs):
        super().__init__()
        self.channels = channels
        self.mixstyle_type = kwargs["mixstyle_type"]
        self.blend_type = kwargs.get("blend_type", "linear")

        # Transformer configuration
        self.n_heads = kwargs.get("n_heads", 4)
        # Auto-adjust n_heads if channels < n_heads (embed_dim must be divisible by n_heads)
        self.n_heads = min(self.n_heads, channels)
        self.ff_dim = kwargs.get("ff_dim", 256)
        self.n_layers = kwargs.get("n_layers", 1)
        self.dropout = kwargs.get("mixstyle_dropout", 0.1)

        # Positional encoding for frequency dimension
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, n_freq)) if n_freq else None

        # Build cross-attention blocks
        self.cross_attn_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(channels, self.n_heads, self.ff_dim, self.dropout)
                for _ in range(self.n_layers)
            ],
        )

        # Output projection to generate frequency weights
        self.output_projection = nn.Linear(channels, channels)

    def forward(self, x_content):
        """Forward pass.

        Args:
            x_content: Input features (Batch, Channels, Frame, Frequency)

        Returns:
            Output features with cross-attention based frequency attention

        """
        x_mixed = mix_style(x_content)

        # Average over time dimension for Query and Key/Value
        q = x_content.mean(dim=2)  # (B, C, F) - Query
        kv = x_mixed.mean(dim=2)  # (B, C, F) - Key/Value

        # Add positional encoding
        if self.pos_encoding is not None:
            q = q + self.pos_encoding
            kv = kv + self.pos_encoding

        # Transpose for attention: (B, C, F) -> (B, F, C)
        q = q.transpose(1, 2)
        kv = kv.transpose(1, 2)

        # Apply cross-attention blocks
        for block in self.cross_attn_blocks:
            q = block(q, kv)

        # Transpose back: (B, F, C) -> (B, C, F)
        q = q.transpose(1, 2)

        # Generate frequency weights
        freq_weights = torch.sigmoid(self.output_projection(q.transpose(1, 2)))
        freq_weights = freq_weights.transpose(1, 2).unsqueeze(2)  # (B, C, 1, F)

        # Blend outputs
        if self.blend_type == "linear":
            output = freq_weights * x_mixed + (1 - freq_weights) * x_content
        elif self.blend_type == "residual":
            delta = x_mixed - x_content
            output = x_content + freq_weights * delta
        else:
            msg = f"Unknown blend_type: {self.blend_type}. Choose from ['linear', 'residual']"
            raise ValueError(msg)

        return output


class CrossAttentionBlock(nn.Module):
    """Cross-attention block with feed-forward network."""

    def __init__(self, embed_dim, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
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

    def forward(self, query, key_value):
        """Forward pass with cross-attention.

        Args:
            query: Query tensor (Batch, Seq_len, Embed_dim)
            key_value: Key/Value tensor (Batch, Seq_len, Embed_dim)

        Returns:
            Output tensor with same shape as query

        """
        attn_out, _ = self.cross_attn(query, key_value, key_value)
        x = self.norm1(query + self.dropout1(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x
