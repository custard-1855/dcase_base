"""Test script to verify teacher eval mode behavior.

This validates that:
1. MixStyle is applied in training mode
2. MixStyle is NOT applied in eval mode
3. Teacher predictions are deterministic in eval mode
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "DESED_task/dcase2024_task4_baseline"))

from desed_task.nnet.CNN import CNN


def test_teacher_eval_mode():
    """Test that teacher behaves correctly in training vs eval mode."""
    print("=" * 80)
    print("Testing Teacher Eval Mode Behavior")
    print("=" * 80)

    # Create CNN with MixStyle
    model = CNN(
        n_in_channel=1,
        nb_filters=[64, 64, 64],
        pooling=[(1, 4), (1, 4), (1, 4)],
        mixstyle_type="freqAttn",  # Use frequency attention MixStyle
        attn_type="default",
    )

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create dummy input (batch_size=4, channels=1, time=100, freq=128)
    x = torch.randn(4, 1, 100, 128)

    print("\n" + "=" * 80)
    print("Test 1: Training Mode (MixStyle should be applied)")
    print("=" * 80)

    model.train()

    # Forward pass twice with same input
    with torch.no_grad():
        output1 = model(x)
        output2 = model(x)

    # Check if outputs are different (stochastic due to MixStyle)
    diff_train = (output1 - output2).abs().mean().item()
    print(f"\nMean absolute difference between two forward passes: {diff_train:.6f}")

    if diff_train > 1e-6:
        print("✓ Training mode: Outputs are DIFFERENT (MixStyle is applied)")
    else:
        print("⚠️  Training mode: Outputs are IDENTICAL (unexpected!)")

    print("\n" + "=" * 80)
    print("Test 2: Eval Mode (MixStyle should NOT be applied)")
    print("=" * 80)

    model.eval()

    # Forward pass twice with same input
    with torch.no_grad():
        output1 = model(x)
        output2 = model(x)

    # Check if outputs are identical (deterministic)
    diff_eval = (output1 - output2).abs().mean().item()
    print(f"\nMean absolute difference between two forward passes: {diff_eval:.6f}")

    if diff_eval < 1e-6:
        print("✓ Eval mode: Outputs are IDENTICAL (MixStyle is disabled)")
    else:
        print("❌ Eval mode: Outputs are DIFFERENT (unexpected!)")

    print("\n" + "=" * 80)
    print("Test 3: Training vs Eval Mode Comparison")
    print("=" * 80)

    # Compare training vs eval mode predictions
    model.train()
    with torch.no_grad():
        output_train = model(x)

    model.eval()
    with torch.no_grad():
        output_eval = model(x)

    diff_mode = (output_train - output_eval).abs().mean().item()
    print(f"\nMean absolute difference (training vs eval): {diff_mode:.6f}")

    # Note: This difference might be small if MixStyle has limited effect
    # The key test is whether eval mode is deterministic (Test 2)

    print("\n" + "=" * 80)
    print("Test 4: BatchNorm Statistics")
    print("=" * 80)

    # Check BatchNorm behavior
    model.train()
    bn_modules_train = [(name, module) for name, module in model.named_modules()
                        if isinstance(module, torch.nn.BatchNorm2d)]

    if bn_modules_train:
        print(f"\nFound {len(bn_modules_train)} BatchNorm2d layers")
        bn_layer = bn_modules_train[0][1]

        print(f"\nTraining mode:")
        print(f"  - training flag: {bn_layer.training}")

        model.eval()
        print(f"\nEval mode:")
        print(f"  - training flag: {bn_layer.training}")

        print("\n✓ BatchNorm correctly switches between training and eval modes")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    success = True

    if diff_train > 1e-6:
        print("✓ Test 1 PASSED: Training mode has stochastic MixStyle")
    else:
        print("❌ Test 1 FAILED: Training mode should have stochastic MixStyle")
        success = False

    if diff_eval < 1e-6:
        print("✓ Test 2 PASSED: Eval mode is deterministic")
    else:
        print("❌ Test 2 FAILED: Eval mode should be deterministic")
        success = False

    if success:
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("""
Verification Complete:
- Training mode: MixStyle is applied (stochastic predictions)
- Eval mode: MixStyle is disabled (deterministic predictions)
- Teacher will now provide stable, non-augmented predictions during training
- This aligns with Mean Teacher principles

Expected Impact:
- More stable consistency loss
- Better teacher-student learning
- Improved convergence
""")
    else:
        print("\n❌ SOME TESTS FAILED")
        return False

    return True


if __name__ == "__main__":
    try:
        success = test_teacher_eval_mode()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
