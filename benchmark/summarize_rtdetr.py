"""Aggregate and summarize 3-seed cross-architecture benchmark (YOLO11n vs. RT-DETR-L on NEU-DET)."""
import json
from pathlib import Path
import numpy as np

base = Path(__file__).resolve().parent
rtdetr_dir = base / "runs/baseline_rtdetr"
yolo_dir = base / "runs/baseline_v5"
seeds = [0, 1, 2]
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

# Load RT-DETR results
rtdetr_results = {}
for seed in seeds:
    sf = rtdetr_dir / f"rtdetr_l_e100_seed{seed}/rtdetr_summary.json"
    if sf.exists():
        rtdetr_results[seed] = json.loads(sf.read_text(encoding="utf-8"))

# Load YOLO11n results
yolo_results = {}
for seed in seeds:
    sf = yolo_dir / f"yolo11n_e100_seed{seed}/baseline_summary.json"
    if sf.exists():
        yolo_results[seed] = json.loads(sf.read_text(encoding="utf-8"))

rollup = {
    "dataset": "NEU-DET (neu_v5_frozen)",
    "architectures": ["YOLO11n", "RT-DETR-L"],
    "seeds": seeds,
    "comparison": {}
}

for model_name, res in [("YOLO11n", yolo_results), ("RT-DETR-L", rtdetr_results)]:
    if not res:
        continue
    m_data = {"test_map50": {}, "test_map50_95": {}, "per_class_test_ap50": {}, "latency_ms": {}}
    t50 = [res[s]["test"]["map50"] * 100 for s in res]
    t50_95 = [res[s]["test"]["map50_95"] * 100 for s in res]
    m_data["test_map50"] = {
        "per_seed": t50,
        "mean": float(np.mean(t50)),
        "std": float(np.std(t50, ddof=1)) if len(t50) > 1 else 0.0,
        "range": float(np.max(t50) - np.min(t50))
    }
    m_data["test_map50_95"] = {
        "per_seed": t50_95,
        "mean": float(np.mean(t50_95)),
        "std": float(np.std(t50_95, ddof=1)) if len(t50_95) > 1 else 0.0,
        "range": float(np.max(t50_95) - np.min(t50_95))
    }
    for c in CLASSES:
        c_vals = [res[s]["test"]["per_class_ap50"].get(c, 0.0) * 100 for s in res]
        m_data["per_class_test_ap50"][c] = {
            "mean": float(np.mean(c_vals)),
            "std": float(np.std(c_vals, ddof=1)) if len(c_vals) > 1 else 0.0,
            "range": float(np.max(c_vals) - np.min(c_vals))
        }
    lat = [res[s]["test"]["speed_ms_per_image"]["inference"] for s in res]
    m_data["latency_ms"]["inference_mean"] = float(np.mean(lat))
    rollup["comparison"][model_name] = m_data

out_file = rtdetr_dir / "rtdetr_cross_architecture_rollup.json"
out_file.write_text(json.dumps(rollup, indent=2), encoding="utf-8")
print(f"Cross-architecture rollup written to {out_file}")
print(json.dumps(rollup, indent=2))
