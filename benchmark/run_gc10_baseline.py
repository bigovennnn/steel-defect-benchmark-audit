"""Paper 2 Multi-Dataset Benchmark: YOLO11n on GC10-DET across seeds 0, 1, 2.

Protocol: official COCO-pretrained yolo11n.pt with Ultralytics head adaptation to 10 classes,
640 px, 100 epochs, batch 8, SGD defaults, seeds 0/1/2.
Checkpoint is chosen by validation fitness (best.pt); the test split is evaluated
exactly once per seed after training, reporting standard COCO metrics and per-class AP.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time
import torch
from ultralytics import YOLO

base = Path(__file__).resolve().parent
data_dir = base / "datasets/gc10_review_v1"
weight = base / "weights/yolo11n_coco_official.pt"
expected_weight = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
CLASSES = [
    "punching_hole", "welding_line", "crescent_gap", "water_spot",
    "oil_spot", "silk_spot", "inclusion", "rolled_pit", "crease", "waist_folding"
]

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--fraction", type=float, default=1.0)
ap.add_argument("--device", default="0")
ap.add_argument("--project", default="runs/baseline_gc10")
ap.add_argument("--smoke", action="store_true", help="pipeline check only; result is not a paper result")
ap.add_argument("--resume-eval", action="store_true", help="skip training if best.pt exists and only re-evaluate")
args = ap.parse_args()

if hashlib.sha256(weight.read_bytes()).hexdigest() != expected_weight:
    raise SystemExit("Pretrained weight hash mismatch")

name = f"{'smoke_' if args.smoke else ''}gc10_yolo11n_e{args.epochs}_seed{args.seed}"
project = base / args.project
run = project / name
device = args.device if args.device == "cpu" else int(args.device)

best_pt = run / "weights/best.pt"
train_seconds = 0.0

if args.resume_eval and best_pt.exists():
    print(f"[{name}] Found existing weights at {best_pt}, skipping training and evaluating...", flush=True)
else:
    if run.exists() and not args.resume_eval:
        raise SystemExit(f"Run directory exists; refusing overwrite: {run}")
    start = time.time()
    model = YOLO(str(weight))
    model.train(
        data=str(data_dir / "gc10.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=0,
        seed=args.seed,
        deterministic=True,
        fraction=args.fraction,
        project=str(project),
        name=name,
        exist_ok=True,
        plots=False,
        verbose=False,
        amp=False,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        patience=0,
        val=True
    )
    save = Path(model.trainer.save_dir)
    if save.resolve() != run.resolve():
        raise SystemExit(f"Unexpected run directory: {save} vs {run}")
    with (save / "results.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if [int(x["epoch"]) for x in rows] != list(range(1, args.epochs + 1)):
        raise SystemExit("Unexpected epoch history")
    train_seconds = time.time() - start

print(f"[{name}] Evaluating validation and test splits with best checkpoint...", flush=True)
best = YOLO(str(best_pt))

def evaluate(part):
    m = best.val(
        data=str(data_dir / "gc10.yaml"),
        split=part,
        imgsz=args.imgsz,
        batch=1,
        device=device,
        workers=0,
        plots=False,
        verbose=False,
        project=str(run),
        name=f"eval_{part}",
        exist_ok=True
    )
    return {
        "map50": float(m.box.map50),
        "map50_95": float(m.box.map),
        "per_class_ap50": {CLASSES[int(c)]: float(a) for c, a in zip(m.box.ap_class_index, m.box.ap50)},
        "speed_ms_per_image": {k: float(v) for k, v in m.speed.items()}
    }

result = {
    "purpose": "smoke check, not a paper result" if args.smoke else "paper 2 multi-dataset benchmark (GC10-DET on YOLO11n)",
    "dataset": "GC10-DET",
    "architecture": "YOLO11n",
    "seed": args.seed,
    "epochs": args.epochs,
    "imgsz": args.imgsz,
    "batch": args.batch,
    "fraction": args.fraction,
    "checkpoint_selection": "best.pt by validation fitness",
    "val": evaluate("val"),
    "test": evaluate("test"),
    "train_seconds": round(train_seconds, 1),
    "weight_sha256": expected_weight,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() and device != "cpu" else "cpu"
}

summary_file = run / "gc10_summary.json"
summary_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(f"[{name}] Summary written to {summary_file}", flush=True)
print(json.dumps(result, indent=2), flush=True)
