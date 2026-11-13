"""Summarize Reviewer D2 Naive Stratified Random Split Leakage Ablation across 3 seeds.

Aggregates seed0, seed1, seed2 from runs/ablation_naive/yolo11n_e100_seed{seed}/ablation_summary.json,
computes mean, standard deviation, min, max, and delta (leaked vs clean),
and formats a clean LaTeX table comparing with neu_v5_frozen.
"""
from pathlib import Path
import json, numpy as np

bench_dir = Path(__file__).resolve().parent
ablation_dir = bench_dir / 'runs/ablation_naive'
seeds = [0, 1, 2]

summaries = {}
for s in seeds:
    p = ablation_dir / f'yolo11n_e100_seed{s}/ablation_summary.json'
    if p.exists():
        summaries[s] = json.loads(p.read_text(encoding='utf-8'))
    else:
        print(f"Seed {s} not found yet ({p})")

print(f"Loaded {len(summaries)}/3 completed seeds: {list(summaries.keys())}")

def stats(vals):
    arr = np.array(vals)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'range': float(np.max(arr) - np.min(arr))
    }

results = {'seeds_available': list(summaries.keys())}

metrics = [
    ('val_map50', lambda d: d['val']['map50'] * 100),
    ('val_map50_95', lambda d: d['val']['map50_95'] * 100),
    ('test_all_map50', lambda d: d['test_all']['map50'] * 100),
    ('test_all_map50_95', lambda d: d['test_all']['map50_95'] * 100),
    ('test_leaked_map50', lambda d: d['test_leaked_subset']['map50'] * 100),
    ('test_leaked_map50_95', lambda d: d['test_leaked_subset']['map50_95'] * 100),
    ('test_clean_map50', lambda d: d['test_clean_subset']['map50'] * 100),
    ('test_clean_map50_95', lambda d: d['test_clean_subset']['map50_95'] * 100),
    ('leakage_delta_pp_map50', lambda d: d['leakage_delta_pp_map50']),
    ('leakage_delta_pp_map50_95', lambda d: d['leakage_delta_pp_map50_95']),
]

for name, getter in metrics:
    vals = [getter(summaries[s]) for s in summaries]
    results[name] = {
        'per_seed': {s: getter(summaries[s]) for s in summaries},
        'stats': stats(vals)
    }

out_json = ablation_dir / 'ablation_rollup.json'
out_json.write_text(json.dumps(results, indent=2), encoding='utf-8')
print(f"Saved rollup to {out_json}")

# Print markdown preview
print("\n=== ABLATION RESULTS SUMMARY ===")
print(f"{'Metric':<30} | " + " | ".join(f"Seed {s}" for s in summaries) + " | Mean ± SD | Range")
print("-" * 75)
for name, getter in metrics:
    st = results[name]['stats']
    seed_strs = " | ".join(f"{getter(summaries[s]):.2f}" for s in summaries)
    print(f"{name:<30} | {seed_strs} | {st['mean']:.2f} ± {st['std']:.2f} | {st['range']:.2f}")

