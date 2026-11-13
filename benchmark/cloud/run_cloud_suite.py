#!/usr/bin/env python3
"""
Unified Benchmark & Ablation Runner (Local RTX 4060 & Cloud RTX 4090 compatible).

Supports five experiment suites on the clean grouped and counterfactual splits:
  1. `baseline_neu`      : Multi-architecture 10-seed baseline on `neu_grouped_v6`
                           (yolo11n, yolo11s, yolov8n, yolov8s, rtdetr-l)
  2. `e7_modules`        : E7 / Mod-10S 10-seed paired module replication on `neu_grouped_v6`
                           (yolo11n-base, yolo11n-cbam, yolo11n-ema, yolo11n-bifpn)
                           with verified 493/499 COCO pretrained tensor transfer across all 4 variants.
  3. `split_interaction` : Split x Model interaction (G0..G9 x [yolo11n-base, yolo11n-bifpn] x 3 seeds = 60 runs)
                           using the exact same verified 493/499 COCO loader across all 10 zero-leakage splits.
  4. `baseline_gc10`     : Multi-seed baseline on `gc10_grouped_v1` (yolo11n, yolov8n, yolo11s)
  5. `counterfactual_k10`: K=10 naive splits x S=5 seeds x 2 conditions (L, C) = 100 runs on
                           `neu_k00..k09_{L,C}`

Protocol rules:
  - `--smoke` runs 1 epoch on 5% data and evaluates ONLY `val` (NEVER `test`).
  - Every completed formal run writes `eval_summary.json` and per-image test predictions
    (`per_image_test.json`) to support image-level recall, micro-AP, and hierarchical bootstrap.
  - Resumable: any run whose `eval_summary.json` already exists is automatically skipped.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
import torch
from ultralytics import RTDETR, YOLO

CLOUD_DIR = Path(__file__).resolve().parent
BENCH_DIR = CLOUD_DIR.parent
ROOT_DIR = BENCH_DIR.parent

sys.path.insert(0, str(CLOUD_DIR))
from custom_modules import load_coco_weights_exact_493, register_custom_modules

register_custom_modules()

NEU_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
GC10_CLASSES = [
    "punching_hole",
    "welding_line",
    "crescent_gap",
    "water_spot",
    "oil_spot",
    "silk_spot",
    "inclusion",
    "rolled_pit",
    "crease",
    "waist_folding",
]


def collect_per_image_records(model, img_dir: Path, lbl_dir: Path, imgsz: int, device):
    records = {}
    img_paths = sorted(img_dir.glob("*.jpg"))
    for img_path in img_paths:
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"
        preds = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            device=device,
            conf=0.001,
            iou=0.7,
            max_det=300,
            verbose=False,
            augment=False,
        )
        res = preds[0]
        h, w = res.orig_shape
        gt_boxes = []
        if lbl_path.exists():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                    x1 = (cx - bw / 2.0) * w
                    y1 = (cy - bh / 2.0) * h
                    x2 = (cx + bw / 2.0) * w
                    y2 = (cy + bh / 2.0) * h
                    gt_boxes.append(
                        {"cls": cls_id, "xyxy": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]}
                    )
        pred_boxes = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy()
            for b, c, k in zip(xyxy, confs, clss):
                pred_boxes.append(
                    {"cls": int(k), "conf": round(float(c), 6), "xyxy": [round(float(x), 3) for x in b]}
                )
        records[stem] = {"gt": gt_boxes, "pred": pred_boxes}
    return records


def evaluate_split(best_model, yaml_path: Path, split_name: str, eval_tag: str, run_dir: Path, imgsz: int, device, class_names: list[str]):
    m = best_model.val(
        data=str(yaml_path),
        split=split_name,
        imgsz=imgsz,
        batch=1,
        device=device,
        workers=0,
        plots=False,
        verbose=False,
        project=str(run_dir),
        name=eval_tag,
        exist_ok=True,
    )
    return {
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "map50": float(m.box.map50),
        "map50_95": float(m.box.map),
        "per_class_ap50": {class_names[int(c)]: float(a) for c, a in zip(m.box.ap_class_index, m.box.ap50)},
        "per_class_ap50_95": {class_names[int(c)]: float(a) for c, a in zip(m.box.ap_class_index, m.box.ap)},
        "speed_ms_per_image": {k: float(v) for k, v in m.speed.items()},
    }


CUSTOM_MODULE_ARCHS = {"yolo11n-base", "yolo11n-cbam", "yolo11n-ema", "yolo11n-bifpn"}


def resolve_weight_path(arch: str) -> Path:
    weights_dir = BENCH_DIR / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    if arch in {"yolo11n"} | CUSTOM_MODULE_ARCHS:
        for cand in [
            weights_dir / "yolo11n_coco_official.pt",
            weights_dir / "yolo11n.pt",
            ROOT_DIR / "yolo11n.pt",
            Path("yolo11n.pt"),
        ]:
            if cand.exists():
                return cand
        # Trigger single official download via Ultralytics if no local file exists
        dl = YOLO("yolo11n.pt")
        ckpt_p = Path(dl.ckpt_path) if getattr(dl, "ckpt_path", None) else Path("yolo11n.pt")
        if ckpt_p.exists():
            target = weights_dir / "yolo11n_coco_official.pt"
            if not target.exists():
                shutil.copy2(ckpt_p, target)
            return target
        return Path("yolo11n.pt")
    return weights_dir / f"{arch}.pt" if (weights_dir / f"{arch}.pt").exists() else Path(f"{arch}.pt")


def build_detector(arch: str, seed: int | None = None):
    w_path = resolve_weight_path(arch)
    if arch in CUSTOM_MODULE_ARCHS:
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        yaml_p = CLOUD_DIR / "models" / f"{arch}.yaml"
        model = YOLO(str(yaml_p))
        n_trans = load_coco_weights_exact_493(model, str(w_path))
        if not getattr(model, "ckpt", None):
            raise RuntimeError(f"model.ckpt missing after load_coco_weights_exact_493 for {arch}")
        if n_trans != 493:
            raise RuntimeError(f"Expected 493 effective COCO tensors for {arch}, got {n_trans}")
        return model, str(w_path), n_trans
    if arch.startswith("rtdetr"):
        return RTDETR(str(w_path)), str(w_path), -1
    return YOLO(str(w_path)), str(w_path), -1


def is_stale_custom_run(run_dir: Path, arch: str) -> tuple[bool, str]:
    """Detect prior random-init runs produced under the old nc=6 / pretrained=False bug."""
    if arch not in CUSTOM_MODULE_ARCHS or not run_dir.exists():
        return False, ""
    eval_json = run_dir / "eval_summary.json"
    if eval_json.exists():
        try:
            es = json.loads(eval_json.read_text(encoding="utf-8"))
        except Exception:
            return True, "corrupt eval_summary.json"
        if es.get("pretrained_coco_verified") is not True or es.get("pretrained_tensors_transferred") != 493:
            return True, f"old unverified transfer ({es.get('pretrained_tensors_transferred')})"
    args_yaml = run_dir / "args.yaml"
    if args_yaml.exists():
        txt = args_yaml.read_text(encoding="utf-8", errors="ignore")
        for line in txt.splitlines():
            if line.strip().lower() == "pretrained: false":
                return True, "args.yaml had pretrained: false"
    return False, ""


def archive_stale_run(run_dir: Path, suite_name: str, reason: str):
    archive_root = BENCH_DIR / "runs_v6" / "_archived_stale_random_init" / suite_name
    archive_root.mkdir(parents=True, exist_ok=True)
    dst = archive_root / f"{run_dir.name}_{int(time.time())}"
    print(f"[ARCHIVE-STALE] Moving {run_dir.name} -> {dst} ({reason})", flush=True)
    shutil.move(str(run_dir), str(dst))


def run_standard_job(
    suite_name: str,
    arch: str,
    seed: int,
    data_dir: Path,
    yaml_name: str,
    class_names: list[str],
    args,
    device,
    run_name_prefix: str = "",
):
    project_dir = BENCH_DIR / "runs_v6" / suite_name
    run_name = f"{'smoke_' if args.smoke else ''}{run_name_prefix}{arch}_e{args.epochs}_seed{seed}"
    run_dir = project_dir / run_name
    eval_json = run_dir / "eval_summary.json"

    if not args.smoke:
        stale, reason = is_stale_custom_run(run_dir, arch)
        if stale:
            archive_stale_run(run_dir, suite_name, reason)

    if eval_json.exists() and not args.force_eval:
        print(f"[SKIP] {suite_name}/{run_name} already completed.", flush=True)
        return

    best_pt = run_dir / "weights" / "best.pt"
    results_csv = run_dir / "results.csv"
    epochs = 1 if args.smoke else args.epochs
    fraction = 0.05 if args.smoke else args.fraction

    need_train = True
    train_seconds = 0.0
    n_transferred = -1
    coco_verified = arch not in CUSTOM_MODULE_ARCHS
    conv0_diff = -1.0

    if run_dir.exists() and best_pt.exists() and results_csv.exists() and not args.smoke:
        with results_csv.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) == epochs:
            print(f"[RESUME-EVAL] Found completed {epochs}-epoch checkpoint for {run_name}", flush=True)
            need_train = False
            if arch in CUSTOM_MODULE_ARCHS:
                if eval_json.exists():
                    prev = json.loads(eval_json.read_text(encoding="utf-8"))
                    n_transferred = prev.get("pretrained_tensors_transferred", 493)
                    coco_verified = bool(prev.get("pretrained_coco_verified", False))
                    conv0_diff = prev.get("pretrained_conv0_max_diff", -1.0)
                else:
                    args_y = run_dir / "args.yaml"
                    has_true = args_y.exists() and any(
                        ln.strip().lower() == "pretrained: true"
                        for ln in args_y.read_text(encoding="utf-8", errors="ignore").splitlines()
                    )
                    if has_true:
                        n_transferred = 493
                        coco_verified = True
                    else:
                        need_train = True

    if need_train:
        print(f"\n=== [TRAIN START] {suite_name} | {run_name} ===", flush=True)
        start = time.time()
        model, w_used, n_transferred = build_detector(arch, seed=seed)
        train_kwargs = dict(
            data=str(data_dir / yaml_name),
            epochs=epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=args.workers,
            seed=seed,
            deterministic=True,
            fraction=fraction,
            project=str(project_dir),
            name=run_name,
            exist_ok=True,
            plots=False,
            verbose=False,
            patience=0,
            val=True,
            pretrained=True,
        )
        if arch.startswith("rtdetr"):
            train_kwargs.update(dict(amp=True))
        else:
            train_kwargs.update(dict(amp=False, optimizer="SGD", lr0=0.01, momentum=0.937, weight_decay=0.0005))

        cb_state = {}
        if arch in CUSTOM_MODULE_ARCHS:
            coco_ref_conv0 = model.model.model[0].conv.weight.detach().clone()

            def verify_coco_on_train_start(trainer):
                if trainer.args.pretrained is False:
                    raise RuntimeError(f"Aborting {run_name}: trainer.args.pretrained became False!")
                diff = float((trainer.model.model[0].conv.weight.detach().cpu() - coco_ref_conv0).abs().max().item())
                cb_state["conv0_diff"] = diff
                if diff > 1e-6:
                    raise RuntimeError(
                        f"Aborting {run_name}: COCO pretrained conv0 mismatch in trainer.model (diff={diff})"
                    )
                cb_state["verified"] = True
                print(
                    f"[PRETRAINED VERIFIED] {run_name}: n_coco={n_transferred}/499, conv0_max_diff={diff:.2e}",
                    flush=True,
                )

            model.add_callback("on_train_start", verify_coco_on_train_start)

        model.train(**train_kwargs)
        if arch in CUSTOM_MODULE_ARCHS:
            if not cb_state.get("verified"):
                raise RuntimeError(f"on_train_start verification did not fire for {run_name}")
            coco_verified = True
            conv0_diff = cb_state["conv0_diff"]
        train_seconds = time.time() - start
        print(f"=== [TRAIN DONE] {run_name} in {train_seconds:.1f}s ===", flush=True)

    best_model = RTDETR(str(best_pt)) if arch.startswith("rtdetr") else YOLO(str(best_pt))
    val_metrics = evaluate_split(best_model, data_dir / yaml_name, "val", "eval_val", run_dir, args.imgsz, device, class_names)

    if args.smoke:
        summary = {
            "purpose": "SMOKE CHECK ONLY (val evaluated; test split strictly untouched)",
            "suite": suite_name,
            "arch": arch,
            "seed": seed,
            "pretrained_tensors_transferred": n_transferred,
            "pretrained_coco_verified": coco_verified,
            "pretrained_conv0_max_diff": conv0_diff,
            "val": val_metrics,
        }
        (run_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[SMOKE PASS] {run_name} val mAP50={val_metrics['map50']*100:.2f}% (test untouched)", flush=True)
        return

    test_metrics = evaluate_split(best_model, data_dir / yaml_name, "test", "eval_test", run_dir, args.imgsz, device, class_names)
    per_img = collect_per_image_records(best_model, data_dir / "images/test", data_dir / "labels/test", args.imgsz, device)
    (run_dir / "per_image_test.json").write_text(json.dumps(per_img, indent=2), encoding="utf-8")

    # Also save per-image val records for E6 inflation ladder
    per_img_val = collect_per_image_records(best_model, data_dir / "images/val", data_dir / "labels/val", args.imgsz, device)
    (run_dir / "per_image_val.json").write_text(json.dumps(per_img_val, indent=2), encoding="utf-8")

    split_raw = json.loads((data_dir / "split.json").read_text(encoding="utf-8"))
    split_meta = split_raw.get("provenance", {})
    summary = {
        "suite": suite_name,
        "arch": arch,
        "seed": seed,
        "epochs": epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "pretrained_tensors_transferred": n_transferred,
        "pretrained_coco_verified": coco_verified,
        "pretrained_conv0_max_diff": conv0_diff,
        "split_name": split_meta.get("name", split_raw.get("split_name", data_dir.name)),
        "split_design_sha256": split_meta.get("design_sha256", ""),
        "val": val_metrics,
        "test": test_metrics,
        "train_seconds": round(train_seconds, 1),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() and device != "cpu" else "cpu",
    }
    eval_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[EVAL DONE] {run_name}: val mAP50={val_metrics['map50']*100:.2f}%, test mAP50={test_metrics['map50']*100:.2f}%, "
        f"test mAP50-95={test_metrics['map50_95']*100:.2f}%",
        flush=True,
    )


def run_counterfactual_k10_job(k_idx: int, seed: int, cond: str, args, device):
    data_dir = BENCH_DIR / "datasets" / f"neu_k{k_idx:02d}_{cond}"
    if not data_dir.exists():
        raise SystemExit(f"Dataset not found: {data_dir}. Run scripts/step2_generate_new_splits_v6.py first.")

    project_dir = BENCH_DIR / "runs_v6" / "counterfactual_k10"
    run_name = f"{'smoke_' if args.smoke else ''}k{k_idx:02d}_s{seed}_{cond}"
    run_dir = project_dir / run_name
    eval_json = run_dir / "eval_summary.json"

    if eval_json.exists() and not args.force_eval:
        print(f"[SKIP] counterfactual_k10/{run_name} already completed.", flush=True)
        return

    best_pt = run_dir / "weights" / "best.pt"
    results_csv = run_dir / "results.csv"
    epochs = 1 if args.smoke else args.epochs
    fraction = 0.05 if args.smoke else args.fraction

    need_train = True
    train_seconds = 0.0
    if run_dir.exists() and best_pt.exists() and results_csv.exists() and not args.smoke:
        with results_csv.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) == epochs:
            need_train = False

    if need_train:
        print(f"\n=== [TRAIN START] counterfactual_k10 | {run_name} ===", flush=True)
        start = time.time()
        model, _, _ = build_detector("yolo11n")
        model.train(
            data=str(data_dir / "neu.yaml"),
            epochs=epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=args.workers,
            seed=seed,
            deterministic=True,
            fraction=fraction,
            project=str(project_dir),
            name=run_name,
            exist_ok=True,
            plots=False,
            verbose=False,
            amp=False,
            optimizer="SGD",
            lr0=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            patience=0,
            val=True,
        )
        train_seconds = time.time() - start

    best_model = YOLO(str(best_pt))
    val_metrics = evaluate_split(best_model, data_dir / "neu.yaml", "val", "eval_val", run_dir, args.imgsz, device, NEU_CLASSES)
    if args.smoke:
        print(f"[SMOKE PASS] {run_name} val mAP50={val_metrics['map50']*100:.2f}% (test untouched)", flush=True)
        return

    test_all = evaluate_split(best_model, data_dir / "neu.yaml", "test", "eval_test", run_dir, args.imgsz, device, NEU_CLASSES)
    test_leaked = evaluate_split(best_model, data_dir / "neu_leaked.yaml", "test", "eval_test_leaked", run_dir, args.imgsz, device, NEU_CLASSES)
    test_clean = evaluate_split(best_model, data_dir / "neu_clean.yaml", "test", "eval_test_clean", run_dir, args.imgsz, device, NEU_CLASSES)

    rec_leaked = collect_per_image_records(best_model, data_dir / "images/test_leaked", data_dir / "labels/test_leaked", args.imgsz, device)
    rec_clean = collect_per_image_records(best_model, data_dir / "images/test_clean", data_dir / "labels/test_clean", args.imgsz, device)
    (run_dir / "per_image_test_leaked.json").write_text(json.dumps(rec_leaked, indent=2), encoding="utf-8")
    (run_dir / "per_image_test_clean.json").write_text(json.dumps(rec_clean, indent=2), encoding="utf-8")

    summary = {
        "suite": "counterfactual_k10",
        "split_k": k_idx,
        "seed": seed,
        "condition": cond,
        "epochs": epochs,
        "val": val_metrics,
        "test_all": test_all,
        "test_leaked": test_leaked,
        "test_clean": test_clean,
        "train_seconds": round(train_seconds, 1),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() and device != "cpu" else "cpu",
    }
    eval_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[EVAL DONE] {run_name}: test_all mAP50={test_all['map50']*100:.2f}%, "
        f"test_leaked mAP50={test_leaked['map50']*100:.2f}%, test_clean mAP50={test_clean['map50']*100:.2f}%",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--suite",
        required=True,
        choices=["baseline_neu", "e7_modules", "split_interaction", "baseline_gc10", "counterfactual_k10"],
    )
    ap.add_argument("--archs", nargs="+", default=None, help="Override architectures for the chosen suite")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="Seed indices to run")
    ap.add_argument("--splits-g", type=int, nargs="+", default=list(range(10)), help="For split_interaction: G0..G9 indices 0..9")
    ap.add_argument("--splits-k", type=int, nargs="+", default=list(range(10)), help="For counterfactual_k10: split indices 0..9")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--smoke", action="store_true", help="1-epoch 5% val-only smoke check (never touches test)")
    ap.add_argument("--force-eval", action="store_true")
    args = ap.parse_args()

    device = args.device if args.device == "cpu" else int(args.device)

    if args.suite == "baseline_neu":
        archs = args.archs or ["yolo11n", "yolov8n", "yolo11s", "yolov8s", "rtdetr-l"]
        seeds = args.seeds if args.seeds is not None else list(range(10))
        data_dir = BENCH_DIR / "datasets" / "neu_grouped_v6"
        # Interleave seeds first so all architectures get 3, 5, 10 seeds balanced
        for s in seeds:
            for arch in archs:
                run_standard_job("baseline_neu", arch, s, data_dir, "neu.yaml", NEU_CLASSES, args, device)

    elif args.suite == "e7_modules":
        archs = args.archs or ["yolo11n-base", "yolo11n-cbam", "yolo11n-ema", "yolo11n-bifpn"]
        seeds = args.seeds if args.seeds is not None else list(range(10))
        data_dir = BENCH_DIR / "datasets" / "neu_grouped_v6"
        for s in seeds:
            for arch in archs:
                run_standard_job("e7_modules", arch, s, data_dir, "neu.yaml", NEU_CLASSES, args, device)

    elif args.suite == "split_interaction":
        archs = args.archs or ["yolo11n-base", "yolo11n-bifpn"]
        seeds = args.seeds if args.seeds is not None else [0, 1, 2]
        for g_idx in args.splits_g:
            data_dir = BENCH_DIR / "datasets" / ("neu_grouped_v6" if g_idx == 0 else f"neu_grouped_g{g_idx:02d}")
            if not (data_dir / "neu.yaml").exists():
                raise SystemExit(
                    f"Dataset {data_dir} not materialized. Run benchmark/cloud/run_3090_fix_100runs.py or unpack_on_cloud.py first."
                )
            for s in seeds:
                for arch in archs:
                    run_standard_job(
                        "split_interaction",
                        arch,
                        s,
                        data_dir,
                        "neu.yaml",
                        NEU_CLASSES,
                        args,
                        device,
                        run_name_prefix=f"g{g_idx:02d}_",
                    )

    elif args.suite == "baseline_gc10":
        archs = args.archs or ["yolo11n"]
        seeds = args.seeds if args.seeds is not None else list(range(10))
        data_dir = BENCH_DIR / "datasets" / "gc10_grouped_v1"
        for s in seeds:
            for arch in archs:
                run_standard_job("baseline_gc10", arch, s, data_dir, "gc10.yaml", GC10_CLASSES, args, device)

    elif args.suite == "counterfactual_k10":
        seeds = args.seeds if args.seeds is not None else list(range(5))
        # Run split k outer or seed s outer; pair L and C back-to-back!
        for s in seeds:
            for k in args.splits_k:
                run_counterfactual_k10_job(k, s, "L", args, device)
                run_counterfactual_k10_job(k, s, "C", args, device)


if __name__ == "__main__":
    main()
