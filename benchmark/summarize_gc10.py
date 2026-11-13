"""Aggregate and summarize 3-seed benchmark results for GC10-DET (YOLO11n)."""
import json
from pathlib import Path
import numpy as np

base = Path(__file__).resolve().parent
runs_dir = base / "runs/baseline_gc10"
seeds = [0, 1, 2]
CLASSES = [
    "punching_hole", "welding_line", "crescent_gap", "water_spot",
    "oil_spot", "silk_spot", "inclusion", "rolled_pit", "crease", "waist_folding"
]

results = {}
for seed in seeds:
    summary_file = runs_dir / f"gc10_yolo11n_e100_seed{seed}/gc10_summary.json"
    if not summary_file.exists():
        print(f"Warning: {summary_file} not found yet.")
        continue
    results[seed] = json.loads(summary_file.read_text(encoding="utf-8"))

if len(results) < len(seeds):
    print(f"Only found {len(results)}/{len(seeds)} runs.")

rollup = {"dataset": "GC10-DET", "architecture": "YOLO11n", "seeds": list(results.keys()), "metrics": {}}

for split in ["val", "test"]:
    map50_vals = [results[s][split]["map50"] * 100 for s in results]
    map50_95_vals = [results[s][split]["map50_95"] * 100 for s in results]
    rollup["metrics"][f"{split}_map50"] = {
        "per_seed": map50_vals,
        "mean": float(np.mean(map50_vals)) if map50_vals else None,
        "std": float(np.std(map50_vals, ddof=1)) if len(map50_vals) > 1 else None,
        "range": float(np.max(map50_vals) - np.min(map50_vals)) if map50_vals else None
    }
    rollup["metrics"][f"{split}_map50_95"] = {
        "per_seed": map50_95_vals,
        "mean": float(np.mean(map50_95_vals)) if map50_95_vals else None,
        "std": float(np.std(map50_95_vals, ddof=1)) if len(map50_95_vals) > 1 else None,
        "range": float(np.max(map50_95_vals) - np.min(map50_95_vals)) if map50_95_vals else None
    }

# Per-class AP50 on test set
rollup["per_class_test_ap50"] = {}
for c in CLASSES:
    c_vals = [results[s]["test"]["per_class_ap50"].get(c, 0.0) * 100 for s in results]
    rollup["per_class_test_ap50"][c] = {
        "per_seed": c_vals,
        "mean": float(np.mean(c_vals)) if c_vals else None,
        "std": float(np.std(c_vals, ddof=1)) if len(c_vals) > 1 else None,
        "range": float(np.max(c_vals) - np.min(c_vals)) if c_vals else None
    }

# Latency
latency_infer = [results[s]["test"]["speed_ms_per_image"]["inference"] for s in results]
rollup["latency_ms"] = {
    "inference_mean": float(np.mean(latency_infer)) if latency_infer else None,
    "inference_std": float(np.std(latency_infer, ddof=1)) if len(latency_infer) > 1 else None
}

out_file = runs_dir / "gc10_rollup.json"
out_file.write_text(json.dumps(rollup, indent=2), encoding="utf-8")
print(f"Rollup written to {out_file}")
print(json.dumps(rollup, indent=2))
