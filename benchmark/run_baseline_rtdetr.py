"""Paper 2 Cross-Architecture Baseline: RT-DETR-L on the frozen NEU v5 split.

Protocol: official COCO-pretrained rtdetr-l.pt with Ultralytics head adaptation,
640 px, 100 epochs, batch 4, deterministic mode, seeds 0/1/2.
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
from ultralytics import RTDETR

base = Path(__file__).resolve().parent
data_dir = base / "datasets/neu_v5_frozen"
expected_design = "372b6745f651e7d9723b1151d307305dc0c35407338a2e8d6ad8a0ca04e33c57"
weight = base / "weights/rtdetr-l.pt"
expected_weight = "6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f"
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--fraction", type=float, default=1.0)
ap.add_argument("--device", default="0")
ap.add_argument("--project", default="runs/baseline_rtdetr")
ap.add_argument("--smoke", action="store_true", help="pipeline check only; result is not a paper result")
ap.add_argument("--resume-eval", action="store_true", help="skip training if best.pt exists and only re-evaluate")
args = ap.parse_args()

split = json.loads((data_dir / "split.json").read_text(encoding="utf-8"))
if split["provenance"]["design_sha256"] != expected_design:
    raise SystemExit("Unexpected split design")
if hashlib.sha256(weight.read_bytes()).hexdigest() != expected_weight:
    raise SystemExit("Pretrained weight hash mismatch")

name = f"{'smoke_' if args.smoke else ''}rtdetr_l_e{args.epochs}_seed{args.seed}"
project = base / args.project
run = project / name
device = args.device if args.device == "cpu" else int(args.device)

best_pt = run / "weights/best.pt"
last_pt = run / "weights/last.pt"
results_csv = run / "results.csv"
summary_file = run / "rtdetr_summary.json"
train_seconds = 0.0

def check_complete_epochs(csv_path, target_epochs):
    if not csv_path.exists():
        return False
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = []
        for r in csv.DictReader(f):
            ep_val = (r.get("epoch") or r.get("             epoch") or "").strip()
            if ep_val.isdigit():
                rows.append(int(ep_val))
    return len(rows) >= target_epochs and set(range(1, target_epochs + 1)).issubset(set(rows)) and rows[-1] == target_epochs

is_complete = check_complete_epochs(results_csv, args.epochs)

if args.resume_eval and is_complete and summary_file.exists():
    print(f"[{name}] Already completed all {args.epochs} epochs and summary exists at {summary_file}, skipping.", flush=True)
    raise SystemExit(0)
elif args.resume_eval and is_complete and best_pt.exists():
    print(f"[{name}] Found complete {args.epochs}-epoch weights at {best_pt}, skipping training and evaluating...", flush=True)
elif args.resume_eval and not is_complete and last_pt.exists():
    print(f"[{name}] Found incomplete run with checkpoint at {last_pt}; resuming training to {args.epochs} epochs...", flush=True)
    start = time.time()
    model = RTDETR(str(last_pt))
    model.train(resume=True)
    if not check_complete_epochs(results_csv, args.epochs):
        raise SystemExit("Unexpected epoch history after resume")
    train_seconds = time.time() - start
else:
    if run.exists() and not args.resume_eval:
        raise SystemExit(f"Run directory exists; refusing overwrite: {run}")
    start = time.time()
    model = RTDETR(str(weight))
    model.train(
        data=str(data_dir / "neu.yaml"),
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
        patience=0,
        val=True
    )
    save = Path(model.trainer.save_dir)
    if save.resolve() != run.resolve():
        raise SystemExit(f"Unexpected run directory: {save} vs {run}")
    if not check_complete_epochs(results_csv, args.epochs):
        raise SystemExit("Unexpected epoch history")
    train_seconds = time.time() - start

print(f"[{name}] Evaluating validation and test splits with best checkpoint...", flush=True)
best = RTDETR(str(best_pt))

def evaluate(part):
    m = best.val(
        data=str(data_dir / "neu.yaml"),
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
    "purpose": "smoke check, not a paper result" if args.smoke else "paper 2 cross-architecture benchmark (RT-DETR-L on frozen v5)",
    "architecture": "RT-DETR-L",
    "seed": args.seed,
    "epochs": args.epochs,
    "imgsz": args.imgsz,
    "batch": args.batch,
    "fraction": args.fraction,
    "checkpoint_selection": "best.pt by validation fitness",
    "val": evaluate("val"),
    "test": evaluate("test"),
    "train_seconds": round(train_seconds, 1),
    "split_design_sha256": expected_design,
    "weight_sha256": expected_weight,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() and device != "cpu" else "cpu"
}

summary_file = run / "rtdetr_summary.json"
summary_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(f"[{name}] Summary written to {summary_file}", flush=True)
print(json.dumps(result, indent=2), flush=True)
