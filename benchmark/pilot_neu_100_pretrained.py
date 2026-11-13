"""One-hundred-epoch exploratory GPU run; results are not research comparisons."""
from pathlib import Path
import json, time, torch
from ultralytics import YOLO
base = Path(__file__).resolve().parent
start = time.time()
model = YOLO(str(base / "weights/yolo11n_coco_official.pt"))
model.train(data=str(base / "datasets/neu_dedup_v1/neu.yaml"), epochs=100, imgsz=640,
    batch=8, device=0, workers=0, pretrained=True, seed=0, deterministic=True,
    project=str(base / "runs/pilot"), name="neu_100epoch_pretrained_exploratory", exist_ok=False,
    plots=False, verbose=False, amp=False, save=True, optimizer="SGD", lr0=0.01, momentum=0.937, weight_decay=0.0005, patience=0)
summary = {"purpose": "exploratory run only; not final benchmark", "seconds": round(time.time()-start, 2),
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0), "run_dir": str(model.trainer.save_dir)}
(base / "runs/pilot/neu_100epoch_pretrained_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
