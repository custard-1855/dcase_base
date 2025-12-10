import torch
from torch import nn

from .mixstyle import (
    BasicMixStyleWrapper,
    FrequencyAttentionMixStyle,
    FrequencyTransformerMixStyle,
    CrossAttentionMixStyle,
)


class GLU(nn.Module):
    def __init__(self, input_num):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, input_num):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(lin)
        res = x * sig
        return res


class CNN(nn.Module):
    def __init__(
        self,
        n_in_channel,
        activation="Relu",
        conv_dropout=0,
        kernel_size=[3, 3, 3],
        padding=[1, 1, 1],
        stride=[1, 1, 1],
        nb_filters=[64, 64, 64],
        pooling=[(1, 4), (1, 4), (1, 4)],
        normalization="batch",
        **kwargs,
    ):
        """Initialization of CNN network s

        Args:
            n_in_channel: int, number of input channel
            activation: str, activation function
            conv_dropout: float, dropout
            kernel_size: kernel size
            padding: padding
            stride: list, stride
            nb_filters: number of filters
            pooling: list of tuples, time and frequency pooling
            normalization: choose between "batch" for BatchNormalization and "layer" for LayerNormalization.

        """
        super(CNN, self).__init__()

        self.nb_filters = nb_filters
        self.mixstyle_type = kwargs.get("mixstyle_type", "disabled")
        self.pooling = pooling

        # Get initial frequency dimension (default: 128 mel bins)
        self.n_freq_bins = kwargs.get("n_freq_bins", 128)

        # --- 畳み込みブロックを個別に定義 ---
        # 単一のnn.Sequentialではなく、各ブロックをリストで管理
        self.conv_blocks = nn.ModuleList()
        for i in range(len(nb_filters)):
            nIn = n_in_channel if i == 0 else nb_filters[i - 1]
            nOut = nb_filters[i]

            block = nn.Sequential()
            block.add_module(
                f"conv{i}",
                nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]),
            )

            if normalization == "batch":
                block.add_module(f"batchnorm{i}", nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99))
            elif normalization == "layer":
                block.add_module(f"layernorm{i}", nn.GroupNorm(1, nOut))

            if activation.lower() == "leakyrelu":
                block.add_module(f"relu{i}", nn.LeakyReLU(0.2))
            elif activation.lower() == "relu":
                block.add_module(f"relu{i}", nn.ReLU())

            if conv_dropout > 0:
                block.add_module(f"dropout{i}", nn.Dropout(conv_dropout))

            block.add_module(f"pooling{i}", nn.AvgPool2d(pooling[i]))

            self.conv_blocks.append(block)

        # --- MixStyleレイヤーを定義 ---
        # Calculate n_freq for each position based on pooling
        n_freq_pre = self.n_freq_bins  # Before any pooling
        n_freq_post1 = n_freq_pre // pooling[0][1]  # After first conv+pooling
        n_freq_post2 = n_freq_post1 // pooling[1][1]  # After second conv+pooling

        # 1層目の前
        self.attn_mixstyle_pre = self._create_mixstyle_layer(
            channels=n_in_channel,
            n_freq=n_freq_pre,
            **kwargs,
        )
        # 1層目の後
        self.attn_mixstyle_post1 = self._create_mixstyle_layer(
            channels=nb_filters[0],
            n_freq=n_freq_post1,
            **kwargs,
        )
        # 2層目の後
        self.attn_mixstyle_post2 = self._create_mixstyle_layer(
            channels=nb_filters[1],
            n_freq=n_freq_post2,
            **kwargs,
        )

    def _create_mixstyle_layer(self, channels, n_freq, **kwargs):
        """Factory method to create appropriate MixStyle layer based on type.

        Args:
            channels: Number of input channels
            n_freq: Frequency dimension size (for Transformer/CrossAttention)
            **kwargs: Configuration options including mixstyle_type

        Returns:
            Appropriate MixStyle layer instance
        """
        mixstyle_type = kwargs.get("mixstyle_type", "disabled")

        if mixstyle_type == "disabled":
            # No MixStyle augmentation
            return nn.Identity()

        if mixstyle_type == "resMix":
            # B0: Pure MixStyle without attention (baseline)
            return BasicMixStyleWrapper(channels=channels, **kwargs)

        if mixstyle_type == "freqAttn":
            # P1: CNN-based frequency attention
            return FrequencyAttentionMixStyle(channels=channels, **kwargs)

        if mixstyle_type == "freqTransformer":
            # P2-1: Transformer-based frequency attention
            return FrequencyTransformerMixStyle(channels=channels, n_freq=n_freq, **kwargs)

        if mixstyle_type == "crossAttn":
            # P2-2: Cross-attention based
            return CrossAttentionMixStyle(channels=channels, n_freq=n_freq, **kwargs)

        # Unknown type - raise error
        msg = (
            f"Unknown mixstyle_type: {mixstyle_type}. "
            f"Choose from ['disabled', 'resMix', 'freqAttn', 'freqTransformer', 'crossAttn']"
        )
        raise ValueError(msg)

    def forward(self, x):
        """Forward step of the CNN module.

        Args:
            x (Tensor): input batch of size (batch_size, n_channels, n_frames, n_freq)

        Returns:
            Tensor: batch embedded

        """
        # conv features
        # x = self.cnn(x)

        if self.training and self.mixstyle_type != "disabled":
            # 1. 1層目の前に適用
            x = self.attn_mixstyle_pre(x)

            # 2. 1層目の畳み込みブロックを適用
            x = self.conv_blocks[0](x)

            # 3. 1層目の後に適用
            x = self.attn_mixstyle_post1(x)

            # 4. 2層目の畳み込みブロックを適用
            x = self.conv_blocks[1](x)

            # 5. 2層目の後に適用
            x = self.attn_mixstyle_post2(x)

            # 6. 3層目以降の畳み込みブロックを適用
            for i in range(2, len(self.conv_blocks)):
                x = self.conv_blocks[i](x)
        else:
            for i in range(len(self.conv_blocks)):
                x = self.conv_blocks[i](x)
        return x
