"""Three-epoch GPU pipeline smoke test; results are not research comparisons."""
from pathlib import Path
import json, time, torch
from ultralytics import YOLO
base = Path(__file__).resolve().parent
start = time.time()
model = YOLO("yolo11n.yaml")
model.train(data=str(base / "datasets/neu_dedup_v1/neu.yaml"), epochs=3, imgsz=640,
    batch=8, device=0, workers=0, pretrained=False, seed=0, deterministic=True,
    project=str(base / "runs/pilot"), name="neu_640_smoke", exist_ok=False,
    plots=False, verbose=False, amp=False, save=True)
summary = {"purpose": "pipeline smoke test only", "seconds": round(time.time()-start, 2),
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0), "run_dir": str(model.trainer.save_dir)}
(base / "runs/pilot/smoke_640_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
