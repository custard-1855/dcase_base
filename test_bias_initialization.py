"""Test script to verify attention network bias initialization.

This validates that the final layer bias is initialized to 1.0
to avoid sigmoid saturation at 0.5.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "DESED_task/dcase2024_task4_baseline"))

from desed_task.nnet.mixstyle import FrequencyAttentionMixStyle


def test_bias_initialization():
    """Test that attention network bias is initialized to 1.0."""
    print("=" * 80)
    print("Testing Attention Network Bias Initialization")
    print("=" * 80)

    # Test different attention types
    attention_types = ["default", "residual_deep", "multiscale"]

    for attn_type in attention_types:
        print(f"\nTesting attn_type='{attn_type}':")

        layer = FrequencyAttentionMixStyle(
            channels=64,
            mixstyle_type="freqAttn",
            attn_type=attn_type,
            attn_deepen=2,
        )

        # Get final layer bias
        final_layer_bias = layer.attention_network[-1].bias.data
        print(f"  Final layer bias shape: {final_layer_bias.shape}")
        print(f"  Final layer bias mean: {final_layer_bias.mean().item():.4f}")
        print(f"  Final layer bias std: {final_layer_bias.std().item():.4f}")

        # Check if initialized to 1.0
        expected_value = 1.0
        tolerance = 0.001
        assert torch.allclose(
            final_layer_bias,
            torch.ones_like(final_layer_bias) * expected_value,
            atol=tolerance,
        ), f"Expected bias={expected_value}, got mean={final_layer_bias.mean().item():.4f}"

        print(f"  ✓ Bias correctly initialized to {expected_value}")

    # Test initial attention weights
    print("\n" + "=" * 80)
    print("Testing Initial Attention Weights")
    print("=" * 80)

    layer = FrequencyAttentionMixStyle(
        channels=64,
        mixstyle_type="freqAttn",
        attn_type="default",
    )
    layer.eval()

    # Create dummy input
    torch.manual_seed(42)
    x = torch.randn(4, 64, 100, 32)

    with torch.no_grad():
        # Compute attention weights
        x_avg = x.mean(dim=2)  # (B, C, F)
        attn_logits = layer.attention_network(x_avg)
        attn_weights = torch.sigmoid(attn_logits)

        mean_weight = attn_weights.mean().item()
        std_weight = attn_weights.std().item()
        min_weight = attn_weights.min().item()
        max_weight = attn_weights.max().item()

        print(f"\nInitial attention weights:")
        print(f"  mean={mean_weight:.4f}, std={std_weight:.4f}")
        print(f"  min={min_weight:.4f}, max={max_weight:.4f}")

        # Check if weights are NOT stuck at 0.5
        assert abs(mean_weight - 0.5) > 0.1, (
            f"Attention weights still stuck near 0.5! mean={mean_weight:.4f}"
        )

        print(f"  ✓ Attention weights are NOT stuck at 0.5 (mean={mean_weight:.4f})")

        # Expected behavior: sigmoid(1.0) = 0.73
        expected_mean = 0.73
        if abs(mean_weight - expected_mean) < 0.15:
            print(f"  ✓ Weights close to expected sigmoid(1.0)={expected_mean:.2f}")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("""
Summary:
- Final layer bias correctly initialized to 1.0 for all attention types
- Initial attention weights are NOT stuck at 0.5
- Expected behavior: sigmoid(1.0) ≈ 0.73

This initialization avoids sigmoid saturation and allows the attention
mechanism to learn from the beginning of training.
""")


if __name__ == "__main__":
    try:
        test_bias_initialization()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
