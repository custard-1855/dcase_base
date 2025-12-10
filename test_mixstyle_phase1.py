"""Phase 1実装のユニットテスト"""

import unittest

import torch
from torch import nn

from DESED_task.dcase2024_task4_baseline.desed_task.nnet.mixstyle import (
    FrequencyAttentionMixStyle,
    mix_style,
)


class TestFrequencyAttentionMixStylePhase1(unittest.TestCase):
    """Phase 1実装のユニットテスト"""

    def setUp(self):
        """各テストの前に実行"""
        self.batch_size = 4
        self.channels = 64
        self.frames = 100
        self.freq = 128
        torch.manual_seed(42)  # 再現性のため

    def test_output_shape(self):
        """出力のshapeが正しいかテスト"""
        x = torch.randn(self.batch_size, self.channels, self.frames, self.freq)

        for blend_type in ["linear", "residual"]:
            for attn_input in ["mixed", "content", "dual_stream"]:
                with self.subTest(blend_type=blend_type, attn_input=attn_input):
                    model = FrequencyAttentionMixStyle(
                        channels=self.channels,
                        mixstyle_type="enabled",
                        blend_type=blend_type,
                        attn_input=attn_input,
                    )
                    model.train()
                    out = model(x)
                    self.assertEqual(
                        out.shape,
                        x.shape,
                        f"Output shape mismatch for blend={blend_type}, attn={attn_input}",
                    )

    def test_blend_type_mathematical_equivalence(self):
        """blend_type='linear'と'residual'が数学的に等価かテスト"""
        torch.manual_seed(42)
        x = torch.randn(self.batch_size, self.channels, self.frames, self.freq)

        # 同じattention networkを使うため、同じseedで初期化
        torch.manual_seed(100)
        model_linear = FrequencyAttentionMixStyle(
            channels=self.channels,
            mixstyle_type="enabled",
            blend_type="linear",
            attn_input="mixed",
        )

        torch.manual_seed(100)
        model_residual = FrequencyAttentionMixStyle(
            channels=self.channels,
            mixstyle_type="enabled",
            blend_type="residual",
            attn_input="mixed",
        )

        # 同じパラメータを持つことを確認
        for p1, p2 in zip(model_linear.parameters(), model_residual.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

        # evalモードで同じ乱数シードを使用
        model_linear.eval()
        model_residual.eval()

        # MixStyleの乱数を固定
        torch.manual_seed(200)
        out_linear = model_linear(x.clone())

        torch.manual_seed(200)
        out_residual = model_residual(x.clone())

        # 数学的に等価な結果になるはず
        self.assertTrue(
            torch.allclose(out_linear, out_residual, atol=1e-6),
            f"Linear and Residual formulations should be mathematically equivalent. "
            f"Max diff: {(out_linear - out_residual).abs().max().item()}",
        )

    def test_attn_input_content_uses_original_features(self):
        """attn_input='content'が元の特徴量を使っているかテスト"""

        class InstrumentedFrequencyAttentionMixStyle(FrequencyAttentionMixStyle):
            """内部状態を記録できるようにしたクラス"""

            def forward(self, x_content):
                self.x_content = x_content.clone()
                self.x_mixed = mix_style(x_content)
                return super().forward(x_content)

        x = torch.randn(self.batch_size, self.channels, self.frames, self.freq)

        # attn_input="content"の場合
        model_content = InstrumentedFrequencyAttentionMixStyle(
            channels=self.channels,
            mixstyle_type="enabled",
            attn_input="content",
        )
        model_content.train()

        # Attention計算をフックで監視
        attn_input_captured = []

        def hook(module, input, output):
            attn_input_captured.append(input[0].clone())

        handle = model_content.attention_network.register_forward_hook(hook)

        out = model_content(x)

        # Attention networkへの入力が x_content の時間平均であることを確認
        expected_input = model_content.x_content.mean(dim=2)
        actual_input = attn_input_captured[0]

        self.assertTrue(
            torch.allclose(expected_input, actual_input, atol=1e-6),
            "attn_input='content' should use x_content features",
        )

        handle.remove()

    def test_attn_input_mixed_uses_mixed_features(self):
        """attn_input='mixed'が混合特徴量を使っているかテスト"""
        x = torch.randn(self.batch_size, self.channels, self.frames, self.freq)

        # attn_input="mixed"の場合
        model_mixed = FrequencyAttentionMixStyle(
            channels=self.channels,
            mixstyle_type="enabled",
            attn_input="mixed",
        )
        model_mixed.train()

        # Attention計算をフックで監視
        attn_input_captured = []
        x_mixed_captured = []

        def hook_attn(module, input, output):
            attn_input_captured.append(input[0].clone())

        # mix_styleの出力を捕捉
        original_mix_style = mix_style

        def instrumented_mix_style(x):
            result = original_mix_style(x)
            x_mixed_captured.append(result.clone())
            return result

        # mix_styleを一時的に置き換え
        import DESED_task.dcase2024_task4_baseline.desed_task.nnet.mixstyle as mixstyle_module

        original_fn = mixstyle_module.mix_style
        mixstyle_module.mix_style = instrumented_mix_style

        handle = model_mixed.attention_network.register_forward_hook(hook_attn)

        try:
            torch.manual_seed(200)
            out = model_mixed(x)

            # Attention networkへの入力が x_mixed の時間平均であることを確認
            expected_input = x_mixed_captured[0].mean(dim=2)
            actual_input = attn_input_captured[0]

            self.assertTrue(
                torch.allclose(expected_input, actual_input, atol=1e-6),
                "attn_input='mixed' should use x_mixed features",
            )
        finally:
            handle.remove()
            mixstyle_module.mix_style = original_fn

    def test_dual_stream_has_two_networks(self):
        """dual_streamが2つのネットワークを持つかテスト"""
        model_single = FrequencyAttentionMixStyle(
            channels=self.channels,
            mixstyle_type="enabled",
            attn_input="mixed",
        )

        model_dual = FrequencyAttentionMixStyle(
            channels=self.channels,
            mixstyle_type="enabled",
            attn_input="dual_stream",
        )

        # dual_streamは追加のネットワークを持つ
        self.assertTrue(hasattr(model_dual, "attention_network_content"))
        self.assertFalse(hasattr(model_single, "attention_network_content"))

        # パラメータ数がほぼ2倍になっているはず
        params_single = sum(p.numel() for p in model_single.parameters())
        params_dual = sum(p.numel() for p in model_dual.parameters())

        self.assertAlmostEqual(
            params_dual / params_single,
            2.0,
            delta=0.1,
            msg="Dual-stream should have roughly 2x parameters",
        )

    def test_invalid_blend_type_raises_error(self):
        """無効なblend_typeでエラーが出るかテスト"""
        x = torch.randn(self.batch_size, self.channels, self.frames, self.freq)

        model = FrequencyAttentionMixStyle(
            channels=self.channels,
            mixstyle_type="enabled",
            blend_type="invalid_type",  # 無効な値
        )
        model.train()

        with self.assertRaises(ValueError) as context:
            model(x)

        self.assertIn("Unknown blend_type", str(context.exception))

    def test_invalid_attn_input_raises_error(self):
        """無効なattn_inputでエラーが出るかテスト"""
        x = torch.randn(self.batch_size, self.channels, self.frames, self.freq)

        model = FrequencyAttentionMixStyle(
            channels=self.channels,
            mixstyle_type="enabled",
            attn_input="invalid_input",  # 無効な値
        )
        model.train()

        with self.assertRaises(ValueError) as context:
            model(x)

        self.assertIn("Unknown attn_input", str(context.exception))

    def test_output_dtype_matches_input(self):
        """出力のdtypeが入力と一致するかテスト（float32のみ）"""
        # Note: float16はMixStyleの実装でサポートされていないため、float32のみテスト
        x = torch.randn(
            self.batch_size,
            self.channels,
            self.frames,
            self.freq,
            dtype=torch.float32,
        )

        model = FrequencyAttentionMixStyle(
            channels=self.channels,
            mixstyle_type="enabled",
        )

        model.train()
        out = model(x)

        self.assertEqual(out.dtype, x.dtype)

    def test_gradient_flow(self):
        """勾配が正しく流れるかテスト"""
        x = torch.randn(
            self.batch_size,
            self.channels,
            self.frames,
            self.freq,
            requires_grad=True,
        )

        for blend_type in ["linear", "residual"]:
            for attn_input in ["mixed", "content", "dual_stream"]:
                with self.subTest(blend_type=blend_type, attn_input=attn_input):
                    model = FrequencyAttentionMixStyle(
                        channels=self.channels,
                        mixstyle_type="enabled",
                        blend_type=blend_type,
                        attn_input=attn_input,
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

                    # 次のテストのため勾配をクリア
                    x.grad = None
                    model.zero_grad()


if __name__ == "__main__":
    unittest.main()
