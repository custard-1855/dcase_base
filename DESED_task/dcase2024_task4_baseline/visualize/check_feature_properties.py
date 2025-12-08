#!/usr/bin/env python3
"""抽出された特徴量の性質を確認し、UMAP可視化に適しているかを検証"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def analyze_feature_distribution(features, name):
    """特徴量の分布を分析"""
    print(f"\n{'=' * 60}")
    print(f"{name} の分析")
    print(f"{'=' * 60}")

    print(f"形状: {features.shape}")
    print(f"次元数: {features.shape[1]}")
    print(f"サンプル数: {features.shape[0]}")

    # 統計量
    print("\n統計量:")
    print(f"  平均: {features.mean():.4f}")
    print(f"  標準偏差: {features.std():.4f}")
    print(f"  最小値: {features.min():.4f}")
    print(f"  最大値: {features.max():.4f}")

    # 次元ごとの分散
    dim_variance = features.var(axis=0)
    print("\n次元ごとの分散:")
    print(f"  平均分散: {dim_variance.mean():.4f}")
    print(f"  分散の標準偏差: {dim_variance.std():.4f}")
    print(f"  最小分散: {dim_variance.min():.6f}")
    print(f"  最大分散: {dim_variance.max():.4f}")

    # ゼロに近い次元の確認
    low_variance_dims = (dim_variance < 0.001).sum()
    print(f"  低分散次元（< 0.001）: {low_variance_dims} / {features.shape[1]}")

    # スパース性の確認
    zero_ratio = (np.abs(features) < 1e-6).sum() / features.size
    print("\nスパース性:")
    print(f"  ほぼゼロの要素比率: {zero_ratio:.4f}")

    # サンプル間の距離分布
    # ランダムに100サンプル選んで距離を計算（計算量削減）
    if features.shape[0] > 100:
        sample_idx = np.random.choice(features.shape[0], 100, replace=False)
        sample_features = features[sample_idx]
    else:
        sample_features = features

    from scipy.spatial.distance import pdist

    distances = pdist(sample_features, metric="euclidean")

    print("\nサンプル間距離（ユークリッド）:")
    print(f"  平均距離: {distances.mean():.4f}")
    print(f"  標準偏差: {distances.std():.4f}")
    print(f"  最小距離: {distances.min():.4f}")
    print(f"  最大距離: {distances.max():.4f}")

    return {
        "shape": features.shape,
        "mean": features.mean(),
        "std": features.std(),
        "dim_variance_mean": dim_variance.mean(),
        "low_variance_dims": low_variance_dims,
        "zero_ratio": zero_ratio,
        "distance_mean": distances.mean(),
        "distance_std": distances.std(),
    }


def compare_features_vs_probs(data_path):
    """featuresとprobsの性質を比較"""
    print(f"\nデータ読み込み: {data_path}")
    data = np.load(data_path)

    # 利用可能なキーを表示
    print(f"\n利用可能なキー: {list(data.keys())}")

    # features（学生モデル）
    features_student = data["features_student"]
    stats_features = analyze_feature_distribution(features_student, "Features (Student)")

    # probs（学生モデル）
    probs_student = data["probs_student"]
    stats_probs = analyze_feature_distribution(probs_student, "Weak Probs (Student)")

    # 比較表
    print(f"\n{'=' * 60}")
    print("Features vs Probs の比較")
    print(f"{'=' * 60}")
    print(f"{'指標':<30} | {'Features':<15} | {'Probs':<15}")
    print(f"{'-' * 30}-+-{'-' * 15}-+-{'-' * 15}")
    print(f"{'次元数':<30} | {stats_features['shape'][1]:<15} | {stats_probs['shape'][1]:<15}")
    print(
        f"{'平均分散':<30} | {stats_features['dim_variance_mean']:<15.4f} | {stats_probs['dim_variance_mean']:<15.4f}"
    )
    print(
        f"{'低分散次元数':<30} | {stats_features['low_variance_dims']:<15} | {stats_probs['low_variance_dims']:<15}"
    )
    print(
        f"{'スパース性（ゼロ比率）':<30} | {stats_features['zero_ratio']:<15.4f} | {stats_probs['zero_ratio']:<15.4f}"
    )
    print(
        f"{'平均サンプル間距離':<30} | {stats_features['distance_mean']:<15.4f} | {stats_probs['distance_mean']:<15.4f}"
    )

    # UMAP適性の判定
    print(f"\n{'=' * 60}")
    print("UMAP可視化への適性")
    print(f"{'=' * 60}")

    def evaluate_suitability(stats, name):
        score = 0
        reasons = []

        # 次元数（高次元が望ましい）
        if stats["shape"][1] >= 128:
            score += 2
            reasons.append(f"✓ 十分な次元数 ({stats['shape'][1]}次元)")
        elif stats["shape"][1] >= 64:
            score += 1
            reasons.append(f"△ やや低次元 ({stats['shape'][1]}次元)")
        else:
            reasons.append(f"✗ 低次元すぎる ({stats['shape'][1]}次元)")

        # 分散の分布（各次元が情報を持っている）
        if stats["low_variance_dims"] / stats["shape"][1] < 0.1:
            score += 2
            reasons.append(f"✓ ほとんどの次元が有用 (低分散次元: {stats['low_variance_dims']})")
        elif stats["low_variance_dims"] / stats["shape"][1] < 0.3:
            score += 1
            reasons.append(f"△ 一部の次元が低分散 (低分散次元: {stats['low_variance_dims']})")
        else:
            reasons.append(f"✗ 多くの次元が低分散 (低分散次元: {stats['low_variance_dims']})")

        # スパース性（密な特徴が望ましい）
        if stats["zero_ratio"] < 0.1:
            score += 2
            reasons.append(f"✓ 密な特徴量 (ゼロ比率: {stats['zero_ratio']:.2%})")
        elif stats["zero_ratio"] < 0.3:
            score += 1
            reasons.append(f"△ やや疎 (ゼロ比率: {stats['zero_ratio']:.2%})")
        else:
            reasons.append(f"✗ 疎すぎる (ゼロ比率: {stats['zero_ratio']:.2%})")

        # サンプル間距離の分布
        if stats["distance_std"] / stats["distance_mean"] > 0.2:
            score += 2
            reasons.append(
                f"✓ 距離の分布が広い（変動係数: {stats['distance_std'] / stats['distance_mean']:.2f}）"
            )
        else:
            score += 1
            reasons.append(
                f"△ 距離の分布がやや狭い（変動係数: {stats['distance_std'] / stats['distance_mean']:.2f}）"
            )

        print(f"\n[{name}]")
        print(f"スコア: {score}/8")
        for reason in reasons:
            print(f"  {reason}")

        if score >= 7:
            print("  → 判定: 非常に適している ⭐⭐⭐")
        elif score >= 5:
            print("  → 判定: 適している ⭐⭐")
        elif score >= 3:
            print("  → 判定: やや適している ⭐")
        else:
            print("  → 判定: 適していない")

        return score

    score_features = evaluate_suitability(stats_features, "Features (384次元)")
    score_probs = evaluate_suitability(stats_probs, "Weak Probs (27次元)")

    # 推奨
    print(f"\n{'=' * 60}")
    print("推奨")
    print(f"{'=' * 60}")
    if score_features > score_probs:
        print("✅ Featuresの使用を強く推奨します")
        print("   理由: 高次元で情報量が多く、UMAPによる次元削減の効果が高い")
    elif score_features == score_probs:
        print("⚠️  FeaturesとProbsは同程度の適性です")
        print("   推奨: より高次元のFeaturesを使用")
    else:
        print("⚠️  Probsの方が適している可能性があります（要確認）")


def main():
    # データパスの設定
    base_dir = Path(
        "/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/visualize/get_features/inference_outputs"
    )

    # 利用可能なモデルディレクトリを確認
    model_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])

    if not model_dirs:
        print("エラー: 特徴量が見つかりません")
        return

    print("利用可能なモデルディレクトリ:")
    for i, model_dir in enumerate(model_dirs):
        print(f"  {i}: {model_dir.name}")

    # 最初のモデルディレクトリを使用
    model_dir = model_dirs[0]
    print(f"\n使用するモデル: {model_dir.name}")

    # データセットを選択（DESED validationを使用）
    data_path = model_dir / "desed_validation.npz"

    if not data_path.exists():
        print(f"エラー: {data_path} が見つかりません")
        return

    # 分析実行
    compare_features_vs_probs(data_path)


if __name__ == "__main__":
    main()
