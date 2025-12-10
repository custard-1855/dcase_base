"""Analyze Tier1 experiment results to identify potential issues."""

import json
import os
import re
from pathlib import Path

# Experiment directory
exp_dir = Path(
    "/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline"
    "/experiments/experiments/train/ablation/mixstyle_tier1_50"
)

# Expected experiments
experiments = {
    "baseline": {"type": "resMix", "desc": "B0: Pure MixStyle"},
    "linear_mixed_CNN": {"type": "freqAttn", "desc": "P1-1: linear, mixed, CNN"},
    "residual_mixed_CNN": {"type": "freqAttn", "desc": "P1-2: residual, mixed, CNN"},
    "linear_content_CNN": {"type": "freqAttn", "desc": "P1-3: linear, content, CNN"},
    "linear_dual_stream_CNN": {
        "type": "freqAttn",
        "desc": "P1-4: linear, dual_stream, CNN",
    },
    "linear_mixed_transformer": {
        "type": "freqTransformer",
        "desc": "P2-1: linear, mixed, Transformer",
    },
    "linear_cross-attn": {"type": "crossAttn", "desc": "P2-2: Cross-Attention"},
}


def extract_best_metric(log_file):
    """Extract best validation metric from output log."""
    if not log_file.exists():
        return None

    best_metric = None
    with open(log_file) as f:
        for line in f:
            # Look for lines like: "Metric val/obj_metric improved ... New best score: 1.572"
            match = re.search(r"New best score:\s*([\d.]+)", line)
            if match:
                best_metric = float(match.group(1))

    return best_metric


def check_completion(log_file):
    """Check if training completed successfully."""
    if not log_file.exists():
        return False

    with open(log_file) as f:
        content = f.read()
        return "`Trainer.fit` stopped: `max_epochs=50` reached" in content


def extract_model_info(log_file):
    """Extract model architecture information from log."""
    if not log_file.exists():
        return {}

    info = {}
    with open(log_file) as f:
        for line in f:
            # Look for model parameters
            if "Total params" in line:
                match = re.search(r"([\d.]+\s*[MK])\s+Total params", line)
                if match:
                    info["total_params"] = match.group(1)

    return info


def analyze_experiments():
    """Analyze all experiments."""
    print("=" * 80)
    print("Tier1 Experiment Results Analysis")
    print("=" * 80)
    print()

    results = []

    for exp_name, exp_info in experiments.items():
        exp_path = exp_dir / exp_name
        log_file = exp_path / "wandb" / "latest-run" / "files" / "output.log"

        if not exp_path.exists():
            print(f"❌ {exp_name}: NOT FOUND")
            continue

        # Extract metrics
        best_metric = extract_best_metric(log_file)
        completed = check_completion(log_file)
        model_info = extract_model_info(log_file)

        status = "✓ COMPLETED" if completed else "⚠️  INCOMPLETE"

        result = {
            "name": exp_name,
            "desc": exp_info["desc"],
            "type": exp_info["type"],
            "best_metric": best_metric,
            "completed": completed,
            "status": status,
        }
        results.append(result)

        print(f"{exp_name}:")
        print(f"  Type: {exp_info['type']}")
        print(f"  Description: {exp_info['desc']}")
        print(f"  Best val/obj_metric: {best_metric}")
        print(f"  Status: {status}")
        print()

    # Analysis
    print("=" * 80)
    print("Comparison Table")
    print("=" * 80)
    print()
    print(f"{'Experiment':<25} {'Type':<15} {'Best Metric':<15} {'Status':<15}")
    print("-" * 80)
    for r in results:
        metric_str = f"{r['best_metric']:.3f}" if r['best_metric'] else "N/A"
        print(
            f"{r['name']:<25} {r['type']:<15} {metric_str:<15} {r['status']:<15}",
        )

    print()
    print("=" * 80)
    print("Issue Detection")
    print("=" * 80)
    print()

    # Check if all completed
    incomplete = [r for r in results if not r["completed"]]
    if incomplete:
        print(f"⚠️  WARNING: {len(incomplete)} experiments did not complete:")
        for r in incomplete:
            print(f"  - {r['name']}")
        print()

    # Check P1 variants (should have different results if fix worked)
    p1_results = [r for r in results if r["name"].startswith(("linear_", "residual_"))]
    if len(p1_results) >= 2:
        p1_metrics = [r["best_metric"] for r in p1_results if r["best_metric"]]
        if len(set(p1_metrics)) == 1:
            print("❌ CRITICAL: P1 variants have IDENTICAL metrics!")
            print("   This suggests the fix did NOT work.")
            print(f"   All P1 metrics: {p1_metrics}")
        elif len(set(p1_metrics)) < len(p1_metrics):
            print("⚠️  WARNING: Some P1 variants have identical metrics:")
            for r in p1_results:
                print(f"   {r['name']}: {r['best_metric']}")
        else:
            print("✓ GOOD: P1 variants have different metrics")
            for r in p1_results:
                print(f"   {r['name']}: {r['best_metric']}")
        print()

    # Check if baseline differs from P1
    baseline = next((r for r in results if r["name"] == "baseline"), None)
    if baseline and baseline["best_metric"]:
        p1_avg = (
            sum(r["best_metric"] for r in p1_results if r["best_metric"])
            / len(p1_results)
            if p1_results
            else None
        )
        if p1_avg:
            diff = abs(baseline["best_metric"] - p1_avg)
            if diff < 0.01:
                print(
                    f"⚠️  WARNING: Baseline and P1 average are very close (diff={diff:.4f})",
                )
                print("   This might indicate insufficient differentiation")
            else:
                print(f"✓ GOOD: Baseline differs from P1 average (diff={diff:.4f})")
        print()

    # Check P2 variants
    p2_results = [r for r in results if r["name"].startswith("linear_mixed_t") or r["name"].startswith("linear_cross")]
    if p2_results:
        print("P2 (Transformer/CrossAttention) variants:")
        for r in p2_results:
            print(f"   {r['name']}: {r['best_metric']}")
        print()

    return results


if __name__ == "__main__":
    results = analyze_experiments()
