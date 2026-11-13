"""Paper 2 Reviewer P0 Issue 1: Within-Subject Causal Ablation of Leakage on NEU-DET.

Evaluates standard Ultralytics YOLO11n on the purged naive split (neu_naive_purged),
which removes the 31 training images that are near-duplicate partners to the 25 test images.

Evaluates the EXACT same test sets (all 180, clean 155, leaked 25).
Comparing performance on the 25 test images when trained with partners vs. without partners
isolates the pure causal effect of leakage, controlling for sample difficulty and class bias.
"""
from pathlib import Path
import argparse, csv, hashlib, json, time
import torch
from ultralytics import YOLO

base = Path(__file__).resolve().parent
data_dir = base / 'datasets/neu_naive_purged'
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
ap.add_argument('--project', default='runs/ablation_causal_purged')
ap.add_argument('--resume-eval', action='store_true', help='skip training if run directory has best.pt and results.csv, only run evaluations')
ap.add_argument('--smoke', action='store_true', help='pipeline check only; result is not a paper result')
args = ap.parse_args()

if hashlib.sha256(weight.read_bytes()).hexdigest() != expected_weight:
    raise SystemExit('Pretrained weight hash mismatch')

split = json.loads((data_dir / 'split.json').read_text(encoding='utf-8'))
name = f"{'smoke_' if args.smoke else ''}yolo11n_purged_e{args.epochs}_seed{args.seed}"
project = base / args.project
run = project / name

device = args.device if args.device == 'cpu' else int(args.device)

if run.exists():
    best_pt = run / 'weights/best.pt'
    results_csv = run / 'results.csv'
    if args.resume_eval and best_pt.exists() and results_csv.exists():
        with results_csv.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        if len(rows) == args.epochs:
            print(f"Resuming evaluation for existing run: {run}")
            save = run
            train_seconds = results_csv.stat().st_mtime - (run / 'args.yaml').stat().st_mtime
        else:
            raise SystemExit(f"Run exists with incomplete epochs ({len(rows)}/{args.epochs}): {run}")
    else:
        raise SystemExit('Run directory exists; refusing overwrite: ' + str(run))
else:
    start = time.time()
    model = YOLO(str(weight))
    model.train(data=str(data_dir / 'neu.yaml'), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device,
                workers=0, seed=args.seed, deterministic=True, fraction=args.fraction, project=str(project), name=name,
                exist_ok=False, plots=False, verbose=False, amp=False, optimizer='SGD', lr0=0.01, momentum=0.937,
                weight_decay=0.0005, patience=0, val=True)
    save = Path(model.trainer.save_dir)
    if save.resolve() != run.resolve():
        raise SystemExit('Unexpected run directory')
    with (save / 'results.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if [int(x['epoch']) for x in rows] != list(range(1, args.epochs + 1)):
        raise SystemExit('Unexpected epoch history')
    train_seconds = time.time() - start

best = YOLO(str(save / 'weights/best.pt'))

def evaluate(yaml_name, part, eval_name):
    m = best.val(data=str(data_dir / yaml_name), split=part, imgsz=args.imgsz, batch=1, device=device, workers=0,
                 plots=False, verbose=False, project=str(save), name=eval_name, exist_ok=True)
    return {
        'map50': float(m.box.map50),
        'map50_95': float(m.box.map),
        'per_class_ap50': {CLASSES[int(c)]: float(a) for c, a in zip(m.box.ap_class_index, m.box.ap50)},
        'speed_ms_per_image': {k: float(v) for k, v in m.speed.items()}
    }

val_metrics = evaluate('neu.yaml', 'val', 'eval_val')
test_all = evaluate('neu.yaml', 'test', 'eval_test')
test_leaked = evaluate('neu_leaked.yaml', 'test', 'eval_test_leaked')
test_clean = evaluate('neu_clean.yaml', 'test', 'eval_test_clean')

result = {
    'purpose': 'smoke check' if args.smoke else 'Reviewer P0 Issue 1: within-subject causal training ablation',
    'seed': args.seed,
    'epochs': args.epochs,
    'imgsz': args.imgsz,
    'batch': args.batch,
    'fraction': args.fraction,
    'split_strategy': 'purged naive stratified random (31 near-duplicate partner frames excised from train)',
    'train_images_count': len(split['train']),
    'purged_partners_count': len(split['purged_partners']),
    'checkpoint_selection': 'best.pt by Ultralytics validation fitness (mAP50-95)',
    'val': val_metrics,
    'test_all': test_all,
    'test_leaked_subset': test_leaked,
    'test_clean_subset': test_clean,
    'leakage_delta_pp_map50': round((test_leaked['map50'] - test_clean['map50']) * 100, 2),
    'leakage_delta_pp_map50_95': round((test_leaked['map50_95'] - test_clean['map50_95']) * 100, 2),
    'train_seconds': round(train_seconds, 1),
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
}

out_json = save / 'ablation_summary.json'
out_json.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(f"Saved run result to {out_json}")
