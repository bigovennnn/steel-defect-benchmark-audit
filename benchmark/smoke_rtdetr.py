"""RT-DETR interface/memory smoke test. Not a benchmark; never uses test set."""
from pathlib import Path
import json,time,torch
from ultralytics import RTDETR
base=Path(__file__).resolve().parent
start=time.time()
model=RTDETR("rtdetr-l.yaml")
model.train(data=str(base/"datasets/neu_dedup_v1/neu.yaml"),epochs=1,imgsz=320,batch=2,
 device=0,workers=0,pretrained=False,seed=0,deterministic=True,fraction=0.02,
 project=str(base/"runs/pilot"),name="rtdetr_smoke",exist_ok=False,plots=False,
 verbose=False,amp=False,save=True,optimizer="AdamW",lr0=0.0001)
result={"purpose":"RT-DETR smoke test only", "seconds":round(time.time()-start,2), "gpu":torch.cuda.get_device_name(0), "run_dir":str(model.trainer.save_dir)}
(base/"runs/pilot/rtdetr_smoke_summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
