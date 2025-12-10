"""Debug script to identify why P1 experiments show no variation.

This script checks:
1. Whether B0 (baseline) uses FrequencyAttentionMixStyle or pure MixStyle
2. Whether attention weights are being learned or staying constant
3. Whether different P1 configurations produce different attention patterns
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add DESED task path
sys.path.insert(0, str(Path(__file__).parent / "DESED_task/dcase2024_task4_baseline"))

from desed_task.nnet.mixstyle import (
    FrequencyAttentionMixStyle,
    mix_style,
)


def test_baseline_vs_p1():
    """Test if B0 (baseline) and P1-1 use different implementations."""
    print("=" * 80)
    print("Test 1: B0 vs P1-1 Implementation")
    print("=" * 80)

    # B0 configuration (run_tier1_ablation.sh:71)
    b0_kwargs = {
        "mixstyle_type": "resMix",
        "attn_type": "default",
    }

    # P1-1 configuration (run_tier1_ablation.sh:85-89)
    p1_1_kwargs = {
        "mixstyle_type": "freqAttn",
        "attn_type": "default",
        "attn_deepen": 2,
        "blend_type": "linear",
        "attn_input": "mixed",
    }

    # Create instances
    try:
        b0_layer = FrequencyAttentionMixStyle(channels=64, **b0_kwargs)
        print(f"✓ B0 layer created: {type(b0_layer).__name__}")
        print(f"  - mixstyle_type: {b0_layer.mixstyle_type}")
        print(f"  - blend_type: {b0_layer.blend_type}")
        print(f"  - attn_input: {b0_layer.attn_input}")
    except Exception as e:
        print(f"✗ B0 layer creation failed: {e}")

    try:
        p1_1_layer = FrequencyAttentionMixStyle(channels=64, **p1_1_kwargs)
        print(f"\n✓ P1-1 layer created: {type(p1_1_layer).__name__}")
        print(f"  - mixstyle_type: {p1_1_layer.mixstyle_type}")
        print(f"  - blend_type: {p1_1_layer.blend_type}")
        print(f"  - attn_input: {p1_1_layer.attn_input}")
    except Exception as e:
        print(f"✗ P1-1 layer creation failed: {e}")

    print("\n⚠️  CRITICAL FINDING:")
    print("Both B0 and P1-1 use FrequencyAttentionMixStyle!")
    print("The only difference is the mixstyle_type parameter, but it's NOT used in forward().")


def test_attention_weights_behavior():
    """Test if attention weights are actually learning or staying constant."""
    print("\n" + "=" * 80)
    print("Test 2: Attention Weights Behavior")
    print("=" * 80)

    # Create layer
    layer = FrequencyAttentionMixStyle(
        channels=64,
        mixstyle_type="freqAttn",
        attn_type="default",
        blend_type="linear",
        attn_input="mixed",
    )
    layer.train()

    # Create dummy input: (batch=8, channels=64, frames=100, freq=32)
    torch.manual_seed(42)
    x = torch.randn(8, 64, 100, 32)

    # Forward pass
    output = layer(x)

    # Check attention network parameters
    print("\nAttention network parameters:")
    for name, param in layer.attention_network.named_parameters():
        print(f"  {name}: shape={param.shape}, mean={param.mean().item():.4f}, std={param.std().item():.4f}")

    # Multiple forward passes to check attention weight variation
    print("\nAttention weights over multiple forward passes:")
    layer.eval()  # Disable training mode to remove randomness from MixStyle

    # Hook to capture attention weights
    attn_weights_list = []

    def hook_fn(module, input, output):
        # This hooks the sigmoid in forward
        pass

    # Instead, let's manually compute attention weights
    for i in range(5):
        torch.manual_seed(i)
        x_test = torch.randn(8, 64, 100, 32)

        with torch.no_grad():
            # Simulate forward pass
            x_avg = x_test.mean(dim=2)  # (B, C, F)
            attn_logits = layer.attention_network(x_avg)
            attn_weights = torch.sigmoid(attn_logits)

            mean_weight = attn_weights.mean().item()
            std_weight = attn_weights.std().item()
            min_weight = attn_weights.min().item()
            max_weight = attn_weights.max().item()

            print(f"  Pass {i+1}: mean={mean_weight:.4f}, std={std_weight:.4f}, "
                  f"min={min_weight:.4f}, max={max_weight:.4f}")
            attn_weights_list.append(attn_weights)

    # Check if attention weights are similar across channels
    print("\nAttention weights per channel (Pass 1, Batch 0):")
    attn_sample = attn_weights_list[0][0]  # (C, F)
    print(f"  Shape: {attn_sample.shape}")
    print(f"  Channel 0: mean={attn_sample[0].mean():.4f}, std={attn_sample[0].std():.4f}")
    print(f"  Channel 31: mean={attn_sample[31].mean():.4f}, std={attn_sample[31].std():.4f}")
    print(f"  Channel 63: mean={attn_sample[63].mean():.4f}, std={attn_sample[63].std():.4f}")

    # Check if weights are close to 0.5 (indicating no learning)
    avg_distance_from_half = abs(attn_weights_list[0].mean().item() - 0.5)
    if avg_distance_from_half < 0.1:
        print("\n⚠️  WARNING: Attention weights are close to 0.5!")
        print("This suggests the attention network may not be learning effectively.")


def test_p1_variants():
    """Test if different P1 configurations produce different behaviors."""
    print("\n" + "=" * 80)
    print("Test 3: P1 Variants Comparison")
    print("=" * 80)

    variants = {
        "P1-1 (linear, mixed)": {
            "mixstyle_type": "freqAttn",
            "attn_type": "default",
            "blend_type": "linear",
            "attn_input": "mixed",
        },
        "P1-2 (residual, mixed)": {
            "mixstyle_type": "freqAttn",
            "attn_type": "default",
            "blend_type": "residual",
            "attn_input": "mixed",
        },
        "P1-3 (linear, content)": {
            "mixstyle_type": "freqAttn",
            "attn_type": "default",
            "blend_type": "linear",
            "attn_input": "content",
        },
        "P1-4 (linear, dual_stream)": {
            "mixstyle_type": "freqAttn",
            "attn_type": "default",
            "blend_type": "linear",
            "attn_input": "dual_stream",
        },
    }

    torch.manual_seed(42)
    x_content = torch.randn(4, 64, 100, 32)

    outputs = {}
    for name, kwargs in variants.items():
        layer = FrequencyAttentionMixStyle(channels=64, **kwargs)
        layer.eval()

        with torch.no_grad():
            output = layer(x_content)
            outputs[name] = output

            diff = (output - x_content).abs().mean().item()
            print(f"{name}:")
            print(f"  Output diff from input: {diff:.6f}")

    # Compare outputs
    print("\nPairwise output differences:")
    names = list(outputs.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            diff = (outputs[names[i]] - outputs[names[j]]).abs().mean().item()
            print(f"  {names[i]} vs {names[j]}: {diff:.6f}")


def test_mixstyle_effect():
    """Test if MixStyle is actually having an effect."""
    print("\n" + "=" * 80)
    print("Test 4: MixStyle Effect")
    print("=" * 80)

    torch.manual_seed(42)
    x_content = torch.randn(8, 64, 100, 32)

    # Test mix_style function directly
    print("\nTesting mix_style function:")

    # Multiple runs to account for 50% probability
    effects = []
    for i in range(10):
        torch.manual_seed(i)
        x_mixed = mix_style(x_content)
        diff = (x_mixed - x_content).abs().mean().item()
        effects.append(diff)
        print(f"  Run {i+1}: diff={diff:.6f}")

    avg_effect = sum(effects) / len(effects)
    print(f"\nAverage effect: {avg_effect:.6f}")

    if avg_effect < 0.01:
        print("⚠️  WARNING: MixStyle has very small effect!")


if __name__ == "__main__":
    test_baseline_vs_p1()
    test_attention_weights_behavior()
    test_p1_variants()
    test_mixstyle_effect()

    print("\n" + "=" * 80)
    print("Summary of Findings")
    print("=" * 80)
    print("""
Expected issues:
1. B0 and P1 variants all use FrequencyAttentionMixStyle
2. Attention weights may not be learning (staying near 0.5)
3. P1 variants may not show significant differences
4. MixStyle effect may be too small

Recommended fixes:
1. Make B0 use pure mix_style (no attention)
2. Improve attention network initialization
3. Increase MixStyle effect (change beta parameter)
4. Add gradient monitoring during training
""")
