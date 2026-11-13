"""Resume the interrupted exploratory training; test set remains unused."""
from pathlib import Path
import hashlib,json,time,torch
from ultralytics import YOLO
base=Path(__file__).resolve().parent
ckpt=base/'runs/pilot/neu_100epoch_exploratory/weights/last.pt'
summary=base/'runs/pilot/neu_100epoch_summary.json'
if summary.exists(): raise SystemExit('Existing summary: inspect before restarting')
model=YOLO(str(ckpt));epoch=model.ckpt.get('epoch')
if model.ckpt.get('optimizer') is None or model.ckpt.get('train_args',{}).get('epochs')!=100: raise SystemExit('Invalid resume checkpoint')
print(f'Resuming from completed epoch {epoch+1}',flush=True)
start=time.time();model.train(resume=True,device=0,workers=0)
result={'purpose':'exploratory run only; not final benchmark','resumed_from_completed_epoch':epoch+1,'resume_session_seconds':round(time.time()-start,2),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'run_dir':str(model.trainer.save_dir)}
summary.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)
