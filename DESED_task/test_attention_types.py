"""
Attention Networkの各パターンをテストするスクリプト
"""
import torch
import sys
sys.path.append('dcase2024_task4_baseline')

from desed_task.nnet.mixstyle import FrequencyAttentionMixStyle

def test_attention_pattern(attn_type, deepen=2):
    """各attention typeの動作をテスト"""
    print(f"\n{'='*60}")
    print(f"Testing: {attn_type} (depth={deepen})")
    print(f"{'='*60}")

    # テスト用のパラメータ
    batch_size = 4
    channels = 64
    frames = 100
    freq = 128

    # モジュール作成
    try:
        module = FrequencyAttentionMixStyle(
            channels=channels,
            attn_type=attn_type,
            attn_deepen=deepen,
            mixstyle_type="resMix"
        )

        # パラメータ数を計算
        total_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)

        print(f"✓ Module created successfully")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")

        # フォワードパス
        module.eval()
        with torch.no_grad():
            x = torch.randn(batch_size, channels, frames, freq)
            output = module(x)

            print(f"✓ Forward pass successful")
            print(f"  - Input shape:  {tuple(x.shape)}")
            print(f"  - Output shape: {tuple(output.shape)}")

            # 出力の形状チェック
            assert output.shape == x.shape, "Output shape mismatch!"
            print(f"✓ Output shape matches input")

            # NaNチェック
            assert not torch.isnan(output).any(), "NaN detected in output!"
            print(f"✓ No NaN values in output")

        return True

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """全パターンをテスト"""
    print("\n" + "="*60)
    print("Testing Multiple Attention Network Patterns")
    print("="*60)

    # テストするパターン
    test_cases = [
        ("default", 2),
        ("residual_deep", 2),
        ("residual_deep", 4),
        ("multiscale", 2),
        ("se_deep", 2),
        ("dilated_deep", 3),
    ]

    results = {}

    for attn_type, depth in test_cases:
        success = test_attention_pattern(attn_type, depth)
        results[f"{attn_type}_d{depth}"] = success

    # 結果サマリー
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)

    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("="*60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
