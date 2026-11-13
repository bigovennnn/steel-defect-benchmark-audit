"""Paper 2 reference baseline: standard Ultralytics YOLO11n fine-tune on the frozen NEU v5 split.

Protocol (frozen 2026-11-01, see 基准协议_v5.md): official COCO-pretrained yolo11n.pt with the
framework's default head adaptation, 640 px, 100 epochs, batch 8, SGD defaults below, seeds 0/1/2.
The checkpoint is chosen by validation fitness (best.pt); the test split is evaluated exactly once
per seed after training, and validation metrics of the same checkpoint are reported alongside.
"""
from pathlib import Path
import argparse, csv, hashlib, json, time
import torch
from ultralytics import YOLO

base = Path(__file__).resolve().parent
data_dir = base / 'datasets/neu_v5_frozen'
expected_design = '372b6745f651e7d9723b1151d307305dc0c35407338a2e8d6ad8a0ca04e33c57'
weight = base / 'weights/yolo11n_coco_official.pt'
expected_weight = '0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--epochs', type=int, default=100)
ap.add_argument('--imgsz', type=int, default=640)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--fraction', type=float, default=1.0)
ap.add_argument('--device', default='0')
ap.add_argument('--project', default='runs/baseline_v5')
ap.add_argument('--smoke', action='store_true', help='pipeline check only; result is not a paper result')
args = ap.parse_args()

split = json.loads((data_dir / 'split.json').read_text(encoding='utf-8'))
if split['provenance']['design_sha256'] != expected_design: raise SystemExit('Unexpected split design')
if hashlib.sha256(weight.read_bytes()).hexdigest() != expected_weight: raise SystemExit('Pretrained weight hash mismatch')
name = f"{'smoke_' if args.smoke else ''}yolo11n_e{args.epochs}_seed{args.seed}"
project = base / args.project; run = project / name
if run.exists(): raise SystemExit('Run directory exists; refusing overwrite: ' + str(run))

device = args.device if args.device == 'cpu' else int(args.device)
start = time.time()
model = YOLO(str(weight))
model.train(data=str(data_dir / 'neu.yaml'), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device,
            workers=0, seed=args.seed, deterministic=True, fraction=args.fraction, project=str(project), name=name,
            exist_ok=False, plots=False, verbose=False, amp=False, optimizer='SGD', lr0=0.01, momentum=0.937,
            weight_decay=0.0005, patience=0, val=True)
save = Path(model.trainer.save_dir)
if save.resolve() != run.resolve(): raise SystemExit('Unexpected run directory')
with (save / 'results.csv').open(encoding='utf-8-sig', newline='') as f: rows = list(csv.DictReader(f))
if [int(x['epoch']) for x in rows] != list(range(1, args.epochs + 1)): raise SystemExit('Unexpected epoch history')
train_seconds = time.time() - start

best = YOLO(str(save / 'weights/best.pt'))
def evaluate(part):
    m = best.val(data=str(data_dir / 'neu.yaml'), split=part, imgsz=args.imgsz, batch=1, device=device, workers=0,
                 plots=False, verbose=False, project=str(save), name=f'eval_{part}', exist_ok=False)
    return {'map50': float(m.box.map50), 'map50_95': float(m.box.map),
            'per_class_ap50': {CLASSES[int(c)]: float(a) for c, a in zip(m.box.ap_class_index, m.box.ap50)},
            'speed_ms_per_image': {k: float(v) for k, v in m.speed.items()}}
result = {'purpose': 'smoke check, not a paper result' if args.smoke else 'paper 2 reference baseline (frozen protocol v5)',
          'seed': args.seed, 'epochs': args.epochs, 'imgsz': args.imgsz, 'batch': args.batch, 'fraction': args.fraction,
          'checkpoint_selection': 'best.pt by Ultralytics 8.4.189 validation fitness (mAP50-95)',
          'val': evaluate('val'), 'test': evaluate('test'), 'train_seconds': round(train_seconds, 1),
          'split_design_sha256': expected_design, 'weight_sha256': expected_weight,
          'torch': torch.__version__, 'cuda': torch.version.cuda,
          'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() and device != 'cpu' else 'cpu'}
(save / 'baseline_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2), flush=True)
