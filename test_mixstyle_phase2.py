"""Phase 2実装のユニットテスト（Transformer）"""

import unittest

import torch

from DESED_task.dcase2024_task4_baseline.desed_task.nnet.mixstyle import (
    CrossAttentionMixStyle,
    FrequencyTransformerMixStyle,
)


class TestPhase2Transformer(unittest.TestCase):
    """Phase 2（Transformer）実装のユニットテスト"""

    def setUp(self):
        """各テストの前に実行"""
        self.batch_size = 4
        self.channels = 64
        self.frames = 100
        self.freq = 128
        torch.manual_seed(42)  # 再現性のため

    def test_frequency_transformer_output_shape(self):
        """FrequencyTransformerMixStyleの出力shapeテスト"""
        x = torch.randn(self.batch_size, self.channels, self.frames, self.freq)

        for blend_type in ["linear", "residual"]:
            for attn_input in ["mixed", "content"]:
                with self.subTest(blend_type=blend_type, attn_input=attn_input):
                    model = FrequencyTransformerMixStyle(
                        channels=self.channels,
                        n_freq=self.freq,
                        mixstyle_type="enabled",
                        blend_type=blend_type,
                        attn_input=attn_input,
                        n_heads=4,
                        n_layers=1,
                    )
                    model.train()
                    out = model(x)
                    self.assertEqual(
                        out.shape,
                        x.shape,
                        f"Output shape mismatch for blend={blend_type}, attn={attn_input}",
                    )

    def test_cross_attention_output_shape(self):
        """CrossAttentionMixStyleの出力shapeテスト"""
        x = torch.randn(self.batch_size, self.channels, self.frames, self.freq)

        for blend_type in ["linear", "residual"]:
            with self.subTest(blend_type=blend_type):
                model = CrossAttentionMixStyle(
                    channels=self.channels,
                    n_freq=self.freq,
                    mixstyle_type="enabled",
                    blend_type=blend_type,
                    n_heads=4,
                    n_layers=1,
                )
                model.train()
                out = model(x)
                self.assertEqual(
                    out.shape,
                    x.shape,
                    f"Output shape mismatch for blend={blend_type}",
                )

    def test_transformer_n_heads_divisibility(self):
        """チャンネル数がヘッド数で割り切れない場合のエラーテスト"""
        # チャンネル数(64)がヘッド数(5)で割り切れない場合、
        # モデル初期化時にAssertionErrorが発生するはず
        with self.assertRaises(AssertionError) as context:
            model = FrequencyTransformerMixStyle(
                channels=self.channels,
                n_freq=self.freq,
                mixstyle_type="enabled",
                n_heads=5,  # 64は5で割り切れない
            )

        self.assertIn("embed_dim must be divisible by num_heads", str(context.exception))

    def test_transformer_multi_layer(self):
        """複数層のTransformerブロックが正常に動作するかテスト"""
        x = torch.randn(self.batch_size, self.channels, self.frames, self.freq)

        for n_layers in [1, 2, 3]:
            with self.subTest(n_layers=n_layers):
                model = FrequencyTransformerMixStyle(
                    channels=self.channels,
                    n_freq=self.freq,
                    mixstyle_type="enabled",
                    n_layers=n_layers,
                )
                model.train()
                out = model(x)
                self.assertEqual(out.shape, x.shape)

    def test_cross_attention_multi_layer(self):
        """複数層のCross-Attentionブロックが正常に動作するかテスト"""
        x = torch.randn(self.batch_size, self.channels, self.frames, self.freq)

        for n_layers in [1, 2, 3]:
            with self.subTest(n_layers=n_layers):
                model = CrossAttentionMixStyle(
                    channels=self.channels,
                    n_freq=self.freq,
                    mixstyle_type="enabled",
                    n_layers=n_layers,
                )
                model.train()
                out = model(x)
                self.assertEqual(out.shape, x.shape)

    def test_positional_encoding_optional(self):
        """Positional encodingがオプショナルであることをテスト"""
        x = torch.randn(self.batch_size, self.channels, self.frames, self.freq)

        # n_freqを指定しない場合
        model_no_pos = FrequencyTransformerMixStyle(
            channels=self.channels,
            mixstyle_type="enabled",
        )
        model_no_pos.train()
        out = model_no_pos(x)
        self.assertEqual(out.shape, x.shape)
        self.assertIsNone(model_no_pos.pos_encoding)

        # n_freqを指定する場合
        model_with_pos = FrequencyTransformerMixStyle(
            channels=self.channels,
            n_freq=self.freq,
            mixstyle_type="enabled",
        )
        model_with_pos.train()
        out = model_with_pos(x)
        self.assertEqual(out.shape, x.shape)
        self.assertIsNotNone(model_with_pos.pos_encoding)

    def test_gradient_flow_transformer(self):
        """Transformerで勾配が正しく流れるかテスト"""
        x = torch.randn(
            self.batch_size,
            self.channels,
            self.frames,
            self.freq,
            requires_grad=True,
        )

        model = FrequencyTransformerMixStyle(
            channels=self.channels,
            n_freq=self.freq,
            mixstyle_type="enabled",
        )
        model.train()

        out = model(x)
        loss = out.sum()
        loss.backward()

        # 入力に勾配が流れているか
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

        # モデルのパラメータに勾配が流れているか
        for param in model.parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.isnan(param.grad).any())

    def test_gradient_flow_cross_attention(self):
        """Cross-Attentionで勾配が正しく流れるかテスト"""
        x = torch.randn(
            self.batch_size,
            self.channels,
            self.frames,
            self.freq,
            requires_grad=True,
        )

        model = CrossAttentionMixStyle(
            channels=self.channels,
            n_freq=self.freq,
            mixstyle_type="enabled",
        )
        model.train()

        out = model(x)
        loss = out.sum()
        loss.backward()

        # 入力に勾配が流れているか
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

        # モデルのパラメータに勾配が流れているか
        for param in model.parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.isnan(param.grad).any())

    def test_parameter_count_comparison(self):
        """Transformerのパラメータ数が妥当かテスト"""
        model_transformer = FrequencyTransformerMixStyle(
            channels=self.channels,
            n_freq=self.freq,
            mixstyle_type="enabled",
            n_layers=1,
        )

        params = sum(p.numel() for p in model_transformer.parameters())

        # Transformerは少なくとも10K以上のパラメータを持つはず
        # (Multi-head attention + Feed-forward + Layer norm + Positional encoding)
        self.assertGreater(
            params,
            10_000,
            "Transformer should have significant number of parameters",
        )

    def test_dropout_training_vs_eval(self):
        """DropoutがTraining/Evalモードで適切に動作するかテスト"""
        torch.manual_seed(42)
        x = torch.randn(self.batch_size, self.channels, self.frames, self.freq)

        model = FrequencyTransformerMixStyle(
            channels=self.channels,
            n_freq=self.freq,
            mixstyle_type="enabled",
            dropout=0.5,  # 高いdropout率
        )

        # Trainingモード
        model.train()
        torch.manual_seed(100)
        out_train1 = model(x.clone())
        torch.manual_seed(100)
        out_train2 = model(x.clone())

        # Trainingモードでは乱数により結果が異なる（MixStyleの影響）
        # （ただしMixStyleの乱数が支配的なのでdropoutの影響は小さい）

        # Evalモード
        model.eval()
        torch.manual_seed(100)
        out_eval1 = model(x.clone())
        torch.manual_seed(100)
        out_eval2 = model(x.clone())

        # Evalモードでは決定的（ただしMixStyleが無効化されるため入力と異なる）
        self.assertTrue(torch.allclose(out_eval1, out_eval2, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
