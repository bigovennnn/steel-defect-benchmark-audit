"""Paper 2 Phase 3: Matched-Size Counterfactual Partner-Replacement Ablation Runner.

Trains and evaluates YOLO11n on matched-size counterfactual splits:
- Condition L (neu_matched_m{m}_L): D_base + P_leak (1,406 train images, retaining 31 train partners of the 25 leaked test images)
- Condition C (neu_matched_m{m}_C): D_base + R_holdout^(m) (1,406 train images, replacing the 31 partners with class-matched clean train images)

Both conditions use the exact same decontaminated validation set (N=143) for best.pt selection
and evaluate on the exact same test sets:
- test_all (N=180)
- test_leaked (N=25)
- test_clean (N=155)

Also records per-image detection results on test_leaked and test_clean to enable
component-clustered bootstrap confidence intervals on the 25 leaked test images.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
import numpy as np
import torch
from ultralytics import YOLO

base = Path(__file__).resolve().parent
root = base.parent
weight = base / 'weights/yolo11n_coco_official.pt'
expected_weight = '0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']


def compute_box_iou(box1, box2):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def collect_per_image_records(model, img_dir, lbl_dir, imgsz, device):
    """Run inference on each image in img_dir and store raw predictions + GT boxes for cluster bootstrap."""
    records = {}
    img_paths = sorted(img_dir.glob('*.jpg'))
    for img_path in img_paths:
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"
        gt_boxes = []
        # Read image size from prediction result
        preds = model.predict(source=str(img_path), imgsz=imgsz, device=device, conf=0.001, iou=0.7,
                              max_det=300, verbose=False, augment=False)
        res = preds[0]
        h, w = res.orig_shape
        if lbl_path.exists():
            for line in lbl_path.read_text(encoding='utf-8').splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                    x1 = (cx - bw / 2.0) * w
                    y1 = (cy - bh / 2.0) * h
                    x2 = (cx + bw / 2.0) * w
                    y2 = (cy + bh / 2.0) * h
                    gt_boxes.append({'cls': cls_id, 'xyxy': [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]})
        pred_boxes = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy()
            for b, c, k in zip(xyxy, confs, clss):
                pred_boxes.append({
                    'cls': int(k),
                    'conf': round(float(c), 6),
                    'xyxy': [round(float(x), 3) for x in b]
                })
        records[stem] = {
            'gt': gt_boxes,
            'pred': pred_boxes
        }
    return records


def run_single(draw_m, seed_s, cond, args, device):
    data_dir = base / f"datasets/neu_matched_m{draw_m}_{cond}"
    if not data_dir.exists():
        raise SystemExit(f"Dataset directory not found: {data_dir}")

    project = base / args.project
    name = f"m{draw_m}_s{seed_s}_{cond}"
    run = project / name
    eval_json = run / "eval_summary.json"

    if eval_json.exists() and not args.force_eval:
        print(f"[SKIP] {name} already completed ({eval_json})", flush=True)
        return json.loads(eval_json.read_text(encoding='utf-8'))

    best_pt = run / 'weights/best.pt'
    results_csv = run / 'results.csv'

    need_train = True
    train_seconds = 0.0
    if run.exists() and best_pt.exists() and results_csv.exists():
        with results_csv.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        if len(rows) == args.epochs:
            print(f"[RESUME-EVAL] Found completed 100-epoch checkpoint for {name}", flush=True)
            need_train = False
            train_seconds = results_csv.stat().st_mtime - (run / 'args.yaml').stat().st_mtime

    if need_train:
        print(f"\n==========================================", flush=True)
        print(f"[TRAIN START] {name} (draw={draw_m}, seed={seed_s}, cond={cond})", flush=True)
        print(f"==========================================", flush=True)
        start = time.time()
        model = YOLO(str(weight))
        model.train(
            data=str(data_dir / 'neu.yaml'),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=0,
            seed=seed_s,
            deterministic=True,
            fraction=args.fraction,
            project=str(project),
            name=name,
            exist_ok=True,
            plots=False,
            verbose=False,
            amp=False,
            optimizer='SGD',
            lr0=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            patience=0,
            val=True
        )
        train_seconds = time.time() - start
        print(f"[TRAIN DONE] {name} in {train_seconds:.1f}s", flush=True)

    best = YOLO(str(best_pt))

    def evaluate(yaml_name, part, eval_name):
        m = best.val(
            data=str(data_dir / yaml_name),
            split=part,
            imgsz=args.imgsz,
            batch=1,
            device=device,
            workers=0,
            plots=False,
            verbose=False,
            project=str(run),
            name=eval_name,
            exist_ok=True
        )
        return {
            'precision': float(m.box.mp),
            'recall': float(m.box.mr),
            'map50': float(m.box.map50),
            'map50_95': float(m.box.map),
            'per_class_ap50': {CLASSES[int(c)]: float(a) for c, a in zip(m.box.ap_class_index, m.box.ap50)},
            'per_class_ap50_95': {CLASSES[int(c)]: float(a) for c, a in zip(m.box.ap_class_index, m.box.ap)},
            'speed_ms_per_image': {k: float(v) for k, v in m.speed.items()}
        }

    val_metrics = evaluate('neu.yaml', 'val', 'eval_val')
    test_all = evaluate('neu.yaml', 'test', 'eval_test')
    test_leaked = evaluate('neu_leaked.yaml', 'test', 'eval_test_leaked')
    test_clean = evaluate('neu_clean.yaml', 'test', 'eval_test_clean')

    # Save per-image predictions on test_leaked (N=25) and test_clean (N=155) for cluster bootstrap
    leaked_records = collect_per_image_records(
        best,
        data_dir / 'images/test_leaked',
        data_dir / 'labels/test_leaked',
        args.imgsz,
        device
    )
    (run / 'per_image_test_leaked.json').write_text(json.dumps(leaked_records, indent=2), encoding='utf-8')

    result = {
        'purpose': 'Phase 3: Matched-Size Counterfactual Partner-Replacement Ablation',
        'draw': draw_m,
        'seed': seed_s,
        'condition': cond,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'val': val_metrics,
        'test_all': test_all,
        'test_leaked': test_leaked,
        'test_clean': test_clean,
        'leakage_delta_pp_map50': round((test_leaked['map50'] - test_clean['map50']) * 100, 4),
        'leakage_delta_pp_map50_95': round((test_leaked['map50_95'] - test_clean['map50_95']) * 100, 4),
        'train_seconds': round(train_seconds, 1),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() and device != 'cpu' else 'cpu'
    }
    eval_json.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(
        f"[EVAL DONE] {name}: test_leaked mAP50={test_leaked['map50']*100:.2f}%, "
        f"mAP50-95={test_leaked['map50_95']*100:.2f}%, "
        f"test_clean mAP50={test_clean['map50']*100:.2f}%",
        flush=True
    )
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--draws', type=int, nargs='+', default=[0, 1, 2], help='Draw indices m to run')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help='Seed indices s to run')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--fraction', type=float, default=1.0)
    ap.add_argument('--device', default='0')
    ap.add_argument('--project', default='runs/counterfactual_matched')
    ap.add_argument('--force-eval', action='store_true')
    args = ap.parse_args()

    if hashlib.sha256(weight.read_bytes()).hexdigest() != expected_weight:
        raise SystemExit('Pretrained weight hash mismatch')

    device = args.device if args.device == 'cpu' else int(args.device)

    # Order by seed first, then draw, running L and C back-to-back so every 2 runs forms a complete matched pair!
    for s in args.seeds:
        for m in args.draws:
            run_single(m, s, 'L', args, device)
            run_single(m, s, 'C', args, device)
            # Update summary JSON after every completed pair
            summarize_script = root / 'scripts/step3_summarize_counterfactual.py'
            if summarize_script.exists():
                subprocess.run([sys.executable, str(summarize_script)], check=False)


if __name__ == '__main__':
    main()
