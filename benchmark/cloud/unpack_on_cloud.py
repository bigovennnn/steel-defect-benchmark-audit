#!/usr/bin/env python3
"""
Run this script once after unzipping `paper2_cloud_bundle.zip` on the cloud GPU instance:
  python benchmark/cloud/unpack_on_cloud.py

It materializes `benchmark/datasets/neu_grouped_v6`, `benchmark/datasets/gc10_grouped_v1`,
and all 20 counterfactual datasets `benchmark/datasets/neu_k00..09_{L,C}` using hardlinks,
and writes all dataset `.yaml` files with the cloud server's exact absolute paths.
"""

import json
import os
import shutil
from collections import Counter
from pathlib import Path

CLOUD_DIR = Path(__file__).resolve().parent
BENCH_DIR = CLOUD_DIR.parent
ROOT_DIR = BENCH_DIR.parent
DATASETS_DIR = BENCH_DIR / "datasets"

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


def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def write_yolo_dataset(target_dir: Path, splits_map: dict[str, list[str]], pool_dir: Path, classes: list[str], yaml_specs: dict):
    target_dir.mkdir(parents=True, exist_ok=True)
    for sub_name, stems in splits_map.items():
        img_sub = target_dir / "images" / sub_name
        lbl_sub = target_dir / "labels" / sub_name
        img_sub.mkdir(parents=True, exist_ok=True)
        lbl_sub.mkdir(parents=True, exist_ok=True)
        for st in stems:
            link_or_copy(pool_dir / "images" / f"{st}.jpg", img_sub / f"{st}.jpg")
            link_or_copy(pool_dir / "labels" / f"{st}.txt", lbl_sub / f"{st}.txt")

    names_block = "\n".join([f"  {i}: {c}" for i, c in enumerate(classes)])
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


def main():
    neu_pool = ROOT_DIR / "raw_pool" / "neu"
    gc10_pool = ROOT_DIR / "raw_pool" / "gc10"

    # 1. Materialize neu_grouped_v6
    v6_dir = DATASETS_DIR / "neu_grouped_v6"
    v6_sp = json.loads((v6_dir / "split.json").read_text(encoding="utf-8"))
    write_yolo_dataset(
        v6_dir,
        {"train": v6_sp["train"], "val": v6_sp["val"], "test": v6_sp["test"]},
        neu_pool,
        NEU_CLASSES,
        {"neu.yaml": {"train": "images/train", "val": "images/val", "test": "images/test"}},
    )
    print("[OK] Materialized neu_grouped_v6")

    # 2. Materialize gc10_grouped_v1
    gc10_dir = DATASETS_DIR / "gc10_grouped_v1"
    gc10_sp = json.loads((gc10_dir / "split.json").read_text(encoding="utf-8"))
    write_yolo_dataset(
        gc10_dir,
        {"train": gc10_sp["train"], "val": gc10_sp["val"], "test": gc10_sp["test"]},
        gc10_pool,
        GC10_CLASSES,
        {"gc10.yaml": {"train": "images/train", "val": "images/val", "test": "images/test"}},
    )
    print("[OK] Materialized gc10_grouped_v1")

    # 3. Materialize K=10 counterfactual pairs from step2_v6_splits_manifest.json
    manifest = json.loads((ROOT_DIR / "results" / "step2_v6_splits_manifest.json").read_text(encoding="utf-8"))
    all_neu_stems = sorted(set(v6_sp["train"] + v6_sp["val"] + v6_sp["test"]))
    by_class = {}
    for c in NEU_CLASSES:
        by_class[c] = sorted([s for s in all_neu_stems if s.rsplit("_", 1)[0] == c])

    import numpy as np

    yaml_specs = {
        "neu.yaml": {"train": "images/train", "val": "images/val", "test": "images/test"},
        "neu_leaked.yaml": {"train": "images/train", "val": "images/val", "test": "images/test_leaked"},
        "neu_clean.yaml": {"train": "images/train", "val": "images/val", "test": "images/test_clean"},
    }
    for sp_info in manifest["multi_split_counterfactual_k10"]["splits"]:
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
        p_leak = set(sp_info["p_leak_stems"])
        r_holdout = set(sp_info["r_holdout_stems"])
        val_clean = sp_info["val_clean_stems"]
        test_leaked = sp_info["test_leaked_stems"]
        test_clean = sorted([u for u in test_s if u not in set(test_leaked)])
        d_core = [u for u in train_s if u not in p_leak and u not in r_holdout]
        train_L = sorted(d_core + list(p_leak))
        train_C = sorted(d_core + list(r_holdout))

        write_yolo_dataset(
            DATASETS_DIR / f"neu_k{k:02d}_L",
            {"train": train_L, "val": val_clean, "test": test_s, "test_leaked": test_leaked, "test_clean": test_clean},
            neu_pool,
            NEU_CLASSES,
            yaml_specs,
        )
        write_yolo_dataset(
            DATASETS_DIR / f"neu_k{k:02d}_C",
            {"train": train_C, "val": val_clean, "test": test_s, "test_leaked": test_leaked, "test_clean": test_clean},
            neu_pool,
            NEU_CLASSES,
            yaml_specs,
        )
    print("[OK] Materialized all 20 counterfactual datasets neu_k00..09_{L,C}")


if __name__ == "__main__":
    main()
