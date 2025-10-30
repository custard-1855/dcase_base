import torch
import torch.nn as nn
from .mixstyle import FrequencyAttentionMixStyle


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
        **transformer_kwargs
    ):
        """
            Initialization of CNN network s

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
        # cnn = nn.Sequential()

        # def conv(i, normalization="batch", dropout=None, activ="relu"):
        #     nIn = n_in_channel if i == 0 else nb_filters[i - 1]
        #     nOut = nb_filters[i]
        #     cnn.add_module(
        #         "conv{0}".format(i),
        #         nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]),
        #     )
        #     if normalization == "batch":
        #         cnn.add_module(
        #             "batchnorm{0}".format(i),
        #             nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99),
        #         )
        #     elif normalization == "layer":
        #         cnn.add_module("layernorm{0}".format(i), nn.GroupNorm(1, nOut))

        #     if activ.lower() == "leakyrelu":
        #         cnn.add_module("relu{0}".format(i), nn.LeakyReLU(0.2))
        #     elif activ.lower() == "relu":
        #         cnn.add_module("relu{0}".format(i), nn.ReLU())
        #     elif activ.lower() == "glu":
        #         cnn.add_module("glu{0}".format(i), GLU(nOut))
        #     elif activ.lower() == "cg":
        #         cnn.add_module("cg{0}".format(i), ContextGating(nOut))

        #     if dropout is not None:
        #         cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        # --- 畳み込みブロックを個別に定義 ---
        # 単一のnn.Sequentialではなく、各ブロックをリストで管理
        self.conv_blocks = nn.ModuleList()
        for i in range(len(nb_filters)):
            nIn = n_in_channel if i == 0 else nb_filters[i - 1]
            nOut = nb_filters[i]
            
            block = nn.Sequential()
            block.add_module(f"conv{i}", nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]))
            
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

        # --- FrequencyAttentionMixStyleレイヤーを定義 ---
        # 1層目の前
        self.attn_mixstyle_pre = FrequencyAttentionMixStyle(channels=n_in_channel)
        # 1層目の後
        self.attn_mixstyle_post1 = FrequencyAttentionMixStyle(channels=nb_filters[0])
        # 2層目の後
        self.attn_mixstyle_post2 = FrequencyAttentionMixStyle(channels=nb_filters[1])

        self.attn_mixstyle_post3 = FrequencyAttentionMixStyle(channels=nb_filters[2])
        self.attn_mixstyle_post4 = FrequencyAttentionMixStyle(channels=nb_filters[3])


        # # 128x862x64
        # for i in range(len(nb_filters)):
        #     conv(i, normalization=normalization, dropout=conv_dropout, activ=activation)
        #     cnn.add_module(
        #         "pooling{0}".format(i), nn.AvgPool2d(pooling[i])
        #     )  # bs x tframe x mels

        # self.cnn = cnn

    def forward(self, x):
        """
        Forward step of the CNN module

        Args:
            x (Tensor): input batch of size (batch_size, n_channels, n_frames, n_freq)

        Returns:
            Tensor: batch embedded
        """
        # conv features
        # x = self.cnn(x)

        if self.training:
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


            x = self.conv_blocks[2](x)
            x = self.attn_mixstyle_post3(x)

            x = self.conv_blocks[3](x)
            x = self.attn_mixstyle_post4(x)

            # 6. 3層目以降の畳み込みブロックを適用
            for i in range(4, len(self.conv_blocks)):
                x = self.conv_blocks[i](x)
        else: 
            for i in range(0, len(self.conv_blocks)):
                x = self.conv_blocks[i](x)
        return x
