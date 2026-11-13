"""Summarize Reviewer P0 Issue 1 Causal Within-Subject Training Ablation across 3 seeds.

Compares:
Condition A (Leaked Train): runs/ablation_naive/yolo11n_e100_seed{s} (train with 31 partners)
Condition B (Purged Train): runs/ablation_causal_purged/yolo11n_purged_e100_seed{s} (train without 31 partners)

Evaluated on the EXACT SAME:
1. Leaked Test Subset (N=25) -> Pure Causal Effect of Leakage
2. Clean Test Subset (N=155) -> Negative Control
3. Overall Test Set (N=180)
"""
from pathlib import Path
import json, numpy as np

bench_dir = Path(__file__).resolve().parent
naive_dir = bench_dir / 'runs/ablation_naive'
purged_dir = bench_dir / 'runs/ablation_causal_purged'
seeds = [0, 1, 2]

naive_summaries = {}
purged_summaries = {}
for s in seeds:
    pn = naive_dir / f'yolo11n_e100_seed{s}/ablation_summary.json'
    pp = purged_dir / f'yolo11n_purged_e100_seed{s}/ablation_summary.json'
    if pn.exists(): naive_summaries[s] = json.loads(pn.read_text(encoding='utf-8'))
    if pp.exists(): purged_summaries[s] = json.loads(pp.read_text(encoding='utf-8'))

print(f"Loaded Naive runs: {list(naive_summaries.keys())}")
print(f"Loaded Purged runs: {list(purged_summaries.keys())}")

common_seeds = sorted(set(naive_summaries.keys()) & set(purged_summaries.keys()))

def stats(vals):
    arr = np.array(vals)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        'min': float(np.min(arr)),
        'max': float(np.max(arr))
    }

if common_seeds:
    print(f"\nPaired Causal Comparison across {len(common_seeds)} seeds ({common_seeds}):")
    # Metric on the 25 leaked test images
    leaked_naive_50 = [naive_summaries[s]['test_leaked_subset']['map50'] * 100 for s in common_seeds]
    leaked_purged_50 = [purged_summaries[s]['test_leaked_subset']['map50'] * 100 for s in common_seeds]
    delta_leaked_50 = [n - p for n, p in zip(leaked_naive_50, leaked_purged_50)]

    leaked_naive_50_95 = [naive_summaries[s]['test_leaked_subset']['map50_95'] * 100 for s in common_seeds]
    leaked_purged_50_95 = [purged_summaries[s]['test_leaked_subset']['map50_95'] * 100 for s in common_seeds]
    delta_leaked_50_95 = [n - p for n, p in zip(leaked_naive_50_95, leaked_purged_50_95)]

    # Clean test images (N=155, negative control)
    clean_naive_50 = [naive_summaries[s]['test_clean_subset']['map50'] * 100 for s in common_seeds]
    clean_purged_50 = [purged_summaries[s]['test_clean_subset']['map50'] * 100 for s in common_seeds]
    delta_clean_50 = [n - p for n, p in zip(clean_naive_50, clean_purged_50)]

    clean_naive_50_95 = [naive_summaries[s]['test_clean_subset']['map50_95'] * 100 for s in common_seeds]
    clean_purged_50_95 = [purged_summaries[s]['test_clean_subset']['map50_95'] * 100 for s in common_seeds]
    delta_clean_50_95 = [n - p for n, p in zip(clean_naive_50_95, clean_purged_50_95)]

    rollup = {
        'common_seeds': common_seeds,
        'leaked_test_subset_N25': {
            'naive_map50': stats(leaked_naive_50),
            'purged_map50': stats(leaked_purged_50),
            'causal_gain_map50': stats(delta_leaked_50),
            'naive_map50_95': stats(leaked_naive_50_95),
            'purged_map50_95': stats(leaked_purged_50_95),
            'causal_gain_map50_95': stats(delta_leaked_50_95)
        },
        'clean_test_subset_N155': {
            'naive_map50': stats(clean_naive_50),
            'purged_map50': stats(clean_purged_50),
            'causal_gain_map50': stats(delta_clean_50),
            'naive_map50_95': stats(clean_naive_50_95),
            'purged_map50_95': stats(clean_purged_50_95),
            'causal_gain_map50_95': stats(delta_clean_50_95)
        }
    }
    out_f = purged_dir / 'causal_ablation_rollup.json'
    out_f.parent.mkdir(parents=True, exist_ok=True)
    out_f.write_text(json.dumps(rollup, indent=2), encoding='utf-8')
    print(f"Saved causal rollup to {out_f}")

    print("\n--- RESULTS PREVIEW ---")
    print(f"Leaked Test Subset (N=25):")
    print(f"  mAP@0.5: Naive={np.mean(leaked_naive_50):.2f}% vs. Purged={np.mean(leaked_purged_50):.2f}% | Causal Delta = {np.mean(delta_leaked_50):+.2f} pp")
    print(f"  mAP@[0.5:0.95]: Naive={np.mean(leaked_naive_50_95):.2f}% vs. Purged={np.mean(leaked_purged_50_95):.2f}% | Causal Delta = {np.mean(delta_leaked_50_95):+.2f} pp")
    print(f"Clean Test Subset (N=155, Control):")
    print(f"  mAP@0.5: Naive={np.mean(clean_naive_50):.2f}% vs. Purged={np.mean(clean_purged_50):.2f}% | Control Delta = {np.mean(delta_clean_50):+.2f} pp")
    print(f"  mAP@[0.5:0.95]: Naive={np.mean(clean_naive_50_95):.2f}% vs. Purged={np.mean(clean_purged_50_95):.2f}% | Control Delta = {np.mean(delta_clean_50_95):+.2f} pp")
else:
    print("No matching seeds completed in purged directory yet.")
