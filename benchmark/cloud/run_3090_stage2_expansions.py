#!/usr/bin/env python3
"""
Stage-2 Expansion Suite for RTX 3090 (`run_3090_stage2_expansions.py`).

Designed to be chained directly after the initial 220-run bundle finishes on the 3090
so you never have to release and re-queue for a GPU.

It executes three high-impact additions from the JIM revision plan:
  1. [GC10 30->50 Completion] (20 runs, ~4.0 hrs on 3090):
     Runs `yolov8s` and `rtdetr-l` (seeds 0..9) on `gc10_grouped_v1` so GC10-DET has
     the exact same 5-detector x 10-seed (50/50) matrix as NEU-DET.
  2. [E2 Variance Decomposition Across 10 Grouped Splits] (27 runs, ~3.5 hrs on 3090):
     Generates 9 additional independent zero-leakage grouped partitions (`neu_grouped_g01`..`g09`,
     0 cross-split edges each, complementing `neu_grouped_v6` as `g00`) and trains `yolo11n`
     across 3 seeds (0, 1, 2) per split to decompose Split Variance vs. Seed Variance vs. Bootstrap Variance.
  3. [E5 Leakage Dose-Response Curve (0%, 25%, 50%, 75%, 100%)] (27 runs, ~3.5 hrs on 3090):
     For splits k=0, 1, 2 of `counterfactual_k10` (where 0% = Clean C and 100% = Leaked L are already run),
     constructs matched training sets with 25%, 50%, and 75% of the leaked training partners retained
     (and the rest replaced by clean holdout images of the exact same class) across seeds 0, 1, 2.

Every completed run is immediately checkpointed and packed into `paper2_cloud_results.zip`.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

CLOUD_DIR = Path(__file__).resolve().parent
BENCH_DIR = CLOUD_DIR.parent
ROOT_DIR = BENCH_DIR.parent
DATASETS_DIR = BENCH_DIR / "datasets"
RUNS_DIR = BENCH_DIR / "runs_v6"

NEU_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def find_neu_image_and_label(stem: str) -> tuple[Path, Path]:
    pool_img = ROOT_DIR / "raw_pool" / "neu" / "images" / f"{stem}.jpg"
    pool_lbl = ROOT_DIR / "raw_pool" / "neu" / "labels" / f"{stem}.txt"
    if pool_img.exists() and pool_lbl.exists():
        return pool_img, pool_lbl
    v6_dir = DATASETS_DIR / "neu_grouped_v6"
    for sub in ["train", "val", "test"]:
        p_img = v6_dir / "images" / sub / f"{stem}.jpg"
        p_lbl = v6_dir / "labels" / sub / f"{stem}.txt"
        if p_img.exists() and p_lbl.exists():
            return p_img, p_lbl
    raise FileNotFoundError(f"Cannot locate NEU image/label for {stem}")


def write_yolo_split(target_dir: Path, splits_map: dict[str, list[str]], yaml_specs: dict[str, dict]):
    target_dir.mkdir(parents=True, exist_ok=True)
    for sub_name, stems in splits_map.items():
        img_sub = target_dir / "images" / sub_name
        lbl_sub = target_dir / "labels" / sub_name
        img_sub.mkdir(parents=True, exist_ok=True)
        lbl_sub.mkdir(parents=True, exist_ok=True)
        for st in stems:
            src_img, src_lbl = find_neu_image_and_label(st)
            link_or_copy(src_img, img_sub / f"{st}.jpg")
            link_or_copy(src_lbl, lbl_sub / f"{st}.txt")

    names_block = "\n".join([f"  {i}: {c}" for i, c in enumerate(NEU_CLASSES)])
    path_str = str(target_dir.resolve()).replace("\\", "/")
    for y_name, spec in yaml_specs.items():
        content = (
            f"path: {path_str}\n"
            f"train: {spec['train']}\n"
            f"val: {spec['val']}\n"
            f"test: {spec['test']}\n"
            f"names:\n{names_block}\n"
        )
        (target_dir / y_name).write_text(content, encoding="utf-8")


def materialize_e2_grouped_splits():
    """
    Constructs 9 additional zero-leakage grouped splits (`neu_grouped_g01`..`g09`)
    preserving exact 147 connected components + 1,475 singletons (0 cross-split edges)
    and class balance (30 val / 30 test per class, 1,437 train / 180 val / 180 test).
    """
    manifest = json.loads((ROOT_DIR / "results" / "step2_v6_splits_manifest.json").read_text(encoding="utf-8"))
    v6_sp = json.loads((DATASETS_DIR / "neu_grouped_v6" / "split.json").read_text(encoding="utf-8"))
    all_stems = sorted(set(v6_sp["train"] + v6_sp["val"] + v6_sp["test"]))

    # Reconstruct connected components from step2_v6_splits_manifest or candidate pairs
    # Notice that in neu_grouped_v6, any two stems in the same component never cross train/val/test.
    # Even simpler and 100% leak-free: we keep all multi-image connected components tracked via
    # the counterfactual manifest's union of leaked/partner stems or candidate pairs if present,
    # OR we use the component grouping from neu_grouped_v6:
    cand_path = BENCH_DIR / "near_duplicate_20261101" / "candidate_pairs_hs_v1.json"
    edges_csv_path = BENCH_DIR / "audits" / "neu_candidate_edges_223_20261101.csv"
    parent = {s: s for s in all_stems}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    found_edges = False
    if cand_path.exists():
        cands = json.loads(cand_path.read_text(encoding="utf-8"))
        pairs = cands.get("candidates", cands.get("edges", cands)) if isinstance(cands, dict) else cands
        for item in pairs:
            u = Path(item.get("img_a", item.get("a", ""))).stem
            v = Path(item.get("img_b", item.get("b", ""))).stem
            if u in parent and v in parent:
                union(u, v)
        found_edges = True
    elif edges_csv_path.exists():
        import pandas as pd
        df = pd.read_csv(edges_csv_path)
        for _, r in df.iterrows():
            u = Path(str(r["a_file"])).stem
            v = Path(str(r["b_file"])).stem
            if u in parent and v in parent:
                union(u, v)
        found_edges = True

    if not found_edges:
        raise FileNotFoundError(
            f"CRITICAL ERROR: Neither {cand_path} nor {edges_csv_path} was found on cloud!\n"
            "Cannot construct zero-leakage partitions without exact candidate registration edges."
        )

    comps_by_root = defaultdict(list)
    for s in all_stems:
        comps_by_root[find(s)].append(s)

    # Group components by primary defect class
    comps_by_class = defaultdict(list)
    for root_s, members in comps_by_root.items():
        cls_counts = Counter(m.rsplit("_", 1)[0] for m in members)
        dom_cls = cls_counts.most_common(1)[0][0]
        comps_by_class[dom_cls].append(sorted(members))

    for g_idx in range(1, 10):
        target_dir = DATASETS_DIR / f"neu_grouped_g{g_idx:02d}"
        if (target_dir / "neu.yaml").exists():
            continue
        rng = np.random.default_rng(20261100 + g_idx * 97)
        train_stems, val_stems, test_stems = [], [], []
        for c in NEU_CLASSES:
            c_comps = list(comps_by_class[c])
            # Separate singletons and multi-image components
            singletons = [comp[0] for comp in c_comps if len(comp) == 1]
            multis = [comp for comp in c_comps if len(comp) > 1]
            rng.shuffle(singletons)
            rng.shuffle(multis)

            c_test, c_val, c_train = [], [], []
            # Sample ~4 multi-image components into test/val and fill exact quota=30 with singletons
            for comp in multis:
                if len(comp) <= 4 and len(c_test) + len(comp) <= 12 and rng.random() < 0.25:
                    c_test.extend(comp)
                elif len(comp) <= 4 and len(c_val) + len(comp) <= 12 and rng.random() < 0.25:
                    c_val.extend(comp)
                else:
                    c_train.extend(comp)
            need_test = 30 - len(c_test)
            need_val = 30 - len(c_val)
            c_test.extend(singletons[:need_test])
            c_val.extend(singletons[need_test : need_test + need_val])
            c_train.extend(singletons[need_test + need_val :])

            test_stems.extend(sorted(c_test))
            val_stems.extend(sorted(c_val))
            train_stems.extend(sorted(c_train))

        split_payload = {
            "split_name": f"neu_grouped_g{g_idx:02d}",
            "train": sorted(train_stems),
            "val": sorted(val_stems),
            "test": sorted(test_stems),
        }
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "split.json").write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
        write_yolo_split(
            target_dir,
            {"train": split_payload["train"], "val": split_payload["val"], "test": split_payload["test"]},
            {"neu.yaml": {"train": "images/train", "val": "images/val", "test": "images/test"}},
        )
    print("[OK] Materialized E2 9 additional grouped splits: neu_grouped_g01..g09")


def materialize_e5_dose_response_splits():
    """
    Constructs E5 dose-response datasets for k=0, 1, 2 at leakage doses d in {25, 50, 75}%:
      `benchmark/datasets/neu_k0{k}_dose{25|50|75}`
    """
    manifest = json.loads((ROOT_DIR / "results" / "step2_v6_splits_manifest.json").read_text(encoding="utf-8"))
    v6_sp = json.loads((DATASETS_DIR / "neu_grouped_v6" / "split.json").read_text(encoding="utf-8"))
    all_neu_stems = sorted(set(v6_sp["train"] + v6_sp["val"] + v6_sp["test"]))
    by_class = {c: sorted([s for s in all_neu_stems if s.rsplit("_", 1)[0] == c]) for c in NEU_CLASSES}

    yaml_specs = {
        "neu.yaml": {"train": "images/train", "val": "images/val", "test": "images/test"},
        "neu_leaked.yaml": {"train": "images/train", "val": "images/val", "test": "images/test_leaked"},
        "neu_clean.yaml": {"train": "images/train", "val": "images/val", "test": "images/test_clean"},
    }

    for sp_info in manifest["multi_split_counterfactual_k10"]["splits"][:3]:
        k = sp_info["split_index_k"]
        seed_k = sp_info["split_seed"]
        rng = np.random.default_rng(seed_k)
        train_s, val_s, test_s = [], [], []
        for c in NEU_CLASSES:
            c_stems = list(by_class[c])
            perm = rng.permutation(len(c_stems))
            c_stems = [c_stems[i] for i in perm]
            test_s.extend(c_stems[:30])
            val_s.extend(c_stems[30:60])
            train_s.extend(c_stems[60:])
        train_s, test_s = sorted(train_s), sorted(test_s)

        p_leak = sorted(sp_info["p_leak_stems"])
        r_holdout = sorted(sp_info["r_holdout_stems"])
        val_clean = sp_info["val_clean_stems"]
        test_leaked = sp_info["test_leaked_stems"]
        test_clean = sorted([u for u in test_s if u not in set(test_leaked)])
        d_core = [u for u in train_s if u not in set(p_leak) and u not in set(r_holdout)]

        # Group p_leak and r_holdout by class so every dose preserves exact class counts
        p_by_cls = defaultdict(list)
        r_by_cls = defaultdict(list)
        for u in p_leak:
            p_by_cls[u.rsplit("_", 1)[0]].append(u)
        for u in r_holdout:
            r_by_cls[u.rsplit("_", 1)[0]].append(u)

        for dose_pct in [25, 50, 75]:
            target_dir = DATASETS_DIR / f"neu_k{k:02d}_dose{dose_pct}"
            if (target_dir / "neu.yaml").exists():
                continue
            frac = dose_pct / 100.0
            chosen_add = []
            for c in NEU_CLASSES:
                p_list = p_by_cls[c]
                r_list = r_by_cls[c]
                n_total = len(p_list)
                n_keep_leak = int(round(n_total * frac))
                n_keep_clean = n_total - n_keep_leak
                chosen_add.extend(p_list[:n_keep_leak])
                chosen_add.extend(r_list[:n_keep_clean])
            train_dose = sorted(d_core + chosen_add)
            write_yolo_split(
                target_dir,
                {
                    "train": train_dose,
                    "val": val_clean,
                    "test": test_s,
                    "test_leaked": test_leaked,
                    "test_clean": test_clean,
                },
                yaml_specs,
            )
    print("[OK] Materialized E5 dose-response datasets: neu_k00..02_dose{25,50,75}")


def run_custom_train_and_eval(suite_name: str, run_name: str, data_yaml: Path, seed: int, workers: int, extra_evals: dict[str, Path] | None = None):
    out_dir = RUNS_DIR / suite_name / run_name
    summary_file = out_dir / "eval_summary.json"
    if summary_file.exists():
        print(f"[SKIP] {suite_name}/{run_name} already completed.")
        return

    from ultralytics import YOLO
    from custom_modules import register_custom_modules

    register_custom_modules()
    w_path = BENCH_DIR / "weights" / "yolo11n_coco_official.pt"
    model = YOLO(str(w_path) if w_path.exists() else "yolo11n.pt")

    t0 = time.time()
    model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=workers,
        seed=seed,
        deterministic=True,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        close_mosaic=10,
        val=True,
        save=True,
        plots=False,
        amp=False,
        patience=0,
        project=str(RUNS_DIR / suite_name),
        name=run_name,
        exist_ok=True,
        verbose=False,
    )
    train_sec = round(time.time() - t0, 1)
    best_pt = out_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = out_dir / "weights" / "last.pt"

    def eval_split(y_path: Path, sp: str, tag: str):
        ev = YOLO(str(best_pt))
        m = ev.val(
            data=str(y_path),
            split=sp,
            imgsz=640,
            batch=8,
            device=0,
            workers=workers,
            conf=0.001,
            iou=0.7,
            save_json=False,
            plots=False,
            verbose=False,
            project=str(out_dir),
            name=f"eval_{tag}",
            exist_ok=True,
        )
        return {
            "precision": float(m.box.mp),
            "recall": float(m.box.mr),
            "map50": float(m.box.map50),
            "map50_95": float(m.box.map),
        }

    res = {
        "suite": suite_name,
        "run_name": run_name,
        "seed": seed,
        "train_seconds": train_sec,
        "val": eval_split(data_yaml, "val", "val"),
        "test": eval_split(data_yaml, "test", "test"),
    }
    if extra_evals:
        for k_tag, y_p in extra_evals.items():
            res[k_tag] = eval_split(y_p, "test", k_tag)

    summary_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[DONE] {suite_name}/{run_name} | test_mAP50={res['test']['map50']*100:.2f}% ({train_sec:.0f}s)")


def pack_checkpoint():
    pack_script = CLOUD_DIR / "pack_cloud_results.py"
    if pack_script.exists():
        subprocess.run([sys.executable, str(pack_script)], cwd=ROOT_DIR, check=False)


def main():
    parser = argparse.ArgumentParser(description="Stage-2 RTX 3090 Expansion Queue")
    parser.add_argument("--workers", type=int, default=4 if os.name != "nt" else 2)
    parser.add_argument("--skip-gc10-extra", action="store_true")
    parser.add_argument("--skip-e2-splits", action="store_true")
    parser.add_argument("--skip-e5-dose", action="store_true")
    parser.add_argument("--shutdown", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(CLOUD_DIR))

    # 1. Complete GC10-DET 50/50 (add yolov8s & rtdetr-l x 10 seeds = 20 runs)
    if not args.skip_gc10_extra:
        print("\n=== [Stage 2A] Completing GC10-DET 50/50 (yolov8s & rtdetr-l x 10 seeds = 20 runs) ===")
        subprocess.run(
            [
                sys.executable,
                str(CLOUD_DIR / "run_cloud_suite.py"),
                "--suite",
                "baseline_gc10",
                "--archs",
                "yolov8s",
                "rtdetr-l",
                "--seeds",
                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                "--workers",
                str(args.workers),
            ],
            cwd=ROOT_DIR,
            check=True,
        )
        pack_checkpoint()

    # 2. E2 Multi-Split Grouped Variance Decomposition (9 additional grouped splits x 3 seeds = 27 runs)
    if not args.skip_e2_splits:
        print("\n=== [Stage 2B] E2 Variance Decomposition (9 grouped splits g01..g09 x 3 seeds = 27 runs) ===")
        materialize_e2_grouped_splits()
        for g_idx in range(1, 10):
            g_dir = DATASETS_DIR / f"neu_grouped_g{g_idx:02d}"
            for seed in [0, 1, 2]:
                run_custom_train_and_eval(
                    suite_name="e2_grouped_splits",
                    run_name=f"g{g_idx:02d}_yolo11n_e100_seed{seed}",
                    data_yaml=g_dir / "neu.yaml",
                    seed=seed,
                    workers=args.workers,
                )
            pack_checkpoint()

    # 3. E5 Dose-Response Curve (k=0,1,2 x dose {25,50,75}% x 3 seeds = 27 runs)
    if not args.skip_e5_dose:
        print("\n=== [Stage 2C] E5 Leakage Dose-Response (k=0..2 x dose 25/50/75% x 3 seeds = 27 runs) ===")
        materialize_e5_dose_response_splits()
        for k in [0, 1, 2]:
            for dose_pct in [25, 50, 75]:
                d_dir = DATASETS_DIR / f"neu_k{k:02d}_dose{dose_pct}"
                for seed in [0, 1, 2]:
                    run_custom_train_and_eval(
                        suite_name="e5_dose_response",
                        run_name=f"k{k:02d}_dose{dose_pct}_yolo11n_e100_seed{seed}",
                        data_yaml=d_dir / "neu.yaml",
                        seed=seed,
                        workers=args.workers,
                        extra_evals={
                            "test_leaked": d_dir / "neu_leaked.yaml",
                            "test_clean": d_dir / "neu_clean.yaml",
                        },
                    )
                pack_checkpoint()

    print("\n=== ALL STAGE-2 EXPANSIONS COMPLETED AND PACKED INTO paper2_cloud_results.zip ===")
    if args.shutdown and os.name != "nt":
        print("Shutting down in 30 seconds...")
        time.sleep(30)
        subprocess.run(["shutdown", "-h", "now"], check=False)


if __name__ == "__main__":
    main()
