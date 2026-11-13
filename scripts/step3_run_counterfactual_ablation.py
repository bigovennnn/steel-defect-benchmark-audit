#!/usr/bin/env python3
"""
Step 3.1: Matched-Size Counterfactual Partner-Replacement Experiment Pipeline & Split Generator
Part of AGENT EXECUTION PLAYBOOK (Phase 3).
"""

import os
import sys
import glob
import json
import shutil
import random
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCHMARK_DIR = os.path.join(ROOT_DIR, "benchmark")
DATASETS_DIR = os.path.join(BENCHMARK_DIR, "datasets")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
NAIVE_SPLIT_PATH = os.path.join(DATASETS_DIR, "neu_naive_random", "split.json")
EDGES_CSV_PATH = os.path.join(BENCHMARK_DIR, "audits", "neu_candidate_edges_223_20261101.csv")

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

def build_counterfactual_splits(M=3, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    print("=== Step 3.1: Preparing Matched-Size Counterfactual Partner-Replacement Splits ===")
    
    with open(NAIVE_SPLIT_PATH, "r", encoding="utf-8") as f:
        naive_data = json.load(f)

    train_raw = naive_data["train"]
    val_raw = naive_data["val"]
    test_raw = naive_data["test"]
    
    # Leaked test subset (Tl, N=25) and Clean test subset (Tc, N=155)
    leaked_test_stems = set(naive_data["candidate_edges_cross_split_audit"]["leaked_test_images"])
    clean_test_stems = set(test_raw) - leaked_test_stems
    print(f"Test split: |T| = {len(test_raw)}, |T_l| (leaked) = {len(leaked_test_stems)}, |T_c| (clean) = {len(clean_test_stems)}")

    # Load edges to find unique training partners P_leak
    import pandas as pd
    df_edges = pd.read_csv(EDGES_CSV_PATH)
    train_partners = set()
    for _, row in df_edges.iterrows():
        a = os.path.splitext(row["a_file"])[0]
        b = os.path.splitext(row["b_file"])[0]
        if a in leaked_test_stems and b in train_raw:
            train_partners.add(b)
        elif b in leaked_test_stems and a in train_raw:
            train_partners.add(a)

    k = len(train_partners)
    print(f"Unique training partners |P_leak|: k = {k}")
    
    # Class breakdown of P_leak
    p_leak_classes = Counter([s.rsplit("_", 1)[0] for s in train_partners])
    print(f"P_leak class distribution: {dict(p_leak_classes)}")

    # Validation decontaminating
    # Check if any validation images link to train_raw or T_l
    val_contaminated = set()
    for _, row in df_edges.iterrows():
        a = os.path.splitext(row["a_file"])[0]
        b = os.path.splitext(row["b_file"])[0]
        if a in val_raw and (b in train_raw or b in test_raw):
            val_contaminated.add(a)
        elif b in val_raw and (a in train_raw or a in test_raw):
            val_contaminated.add(b)
    
    clean_val = [s for s in val_raw if s not in val_contaminated]
    print(f"Validation set: raw = {len(val_raw)}, contaminated = {len(val_contaminated)}, decontaminated V_clean = {len(clean_val)}")

    # Clean pool of training images
    clean_train_pool = [s for s in train_raw if s not in train_partners]
    print(f"Clean training pool: {len(clean_train_pool)} (1437 - {k})")

    # Group clean pool by class
    clean_by_class = defaultdict(list)
    for s in clean_train_pool:
        c = s.rsplit("_", 1)[0]
        clean_by_class[c].append(s)

    draws_manifest = []

    # Generate M matched draws
    for m in range(M):
        draw_seed = seed + m * 100
        random.seed(draw_seed)

        # Draw R_holdout matching exactly the class distribution of P_leak
        r_holdout = []
        for c, count in p_leak_classes.items():
            pool_c = list(clean_by_class[c])
            random.shuffle(pool_c)
            sampled = pool_c[:count]
            r_holdout.extend(sampled)

        assert len(r_holdout) == k, f"Expected {k} replacement images, got {len(r_holdout)}"
        r_set = set(r_holdout)

        # Common base set
        d_base = [s for s in clean_train_pool if s not in r_set]
        assert len(d_base) == len(train_raw) - 2 * k

        # Condition L: D_base + P_leak (size 1437 - k)
        d_L = sorted(d_base + list(train_partners))
        # Condition C: D_base + R_holdout (size 1437 - k)
        d_C = sorted(d_base + r_holdout)

        assert len(d_L) == len(d_C) == len(train_raw) - k
        # Verify class distribution identity
        c_L = Counter([s.rsplit("_", 1)[0] for s in d_L])
        c_C = Counter([s.rsplit("_", 1)[0] for s in d_C])
        assert c_L == c_C, f"Class distributions must match! L: {c_L}, C: {c_C}"

        draw_info = {
            "draw_index": m,
            "seed": draw_seed,
            "size_L": len(d_L),
            "size_C": len(d_C),
            "base_size": len(d_base),
            "replacement_k": k,
            "p_leak_stems": sorted(list(train_partners)),
            "r_holdout_stems": sorted(r_holdout),
            "d_L_stems": d_L,
            "d_C_stems": d_C
        }
        draws_manifest.append(draw_info)

        # Write datasets on disk
        for stems, cond, name in [(d_L, "L", f"neu_matched_m{m}_L"), (d_C, "C", f"neu_matched_m{m}_C")]:
            cond_dir = os.path.join(DATASETS_DIR, name)
            os.makedirs(os.path.join(cond_dir, "images", "train"), exist_ok=True)
            os.makedirs(os.path.join(cond_dir, "images", "val"), exist_ok=True)
            os.makedirs(os.path.join(cond_dir, "images", "test"), exist_ok=True)
            os.makedirs(os.path.join(cond_dir, "images", "test_leaked"), exist_ok=True)
            os.makedirs(os.path.join(cond_dir, "images", "test_clean"), exist_ok=True)
            os.makedirs(os.path.join(cond_dir, "labels", "train"), exist_ok=True)
            os.makedirs(os.path.join(cond_dir, "labels", "val"), exist_ok=True)
            os.makedirs(os.path.join(cond_dir, "labels", "test"), exist_ok=True)
            os.makedirs(os.path.join(cond_dir, "labels", "test_leaked"), exist_ok=True)
            os.makedirs(os.path.join(cond_dir, "labels", "test_clean"), exist_ok=True)

            naive_dir = os.path.join(DATASETS_DIR, "neu_naive_random")
            
            # Copy val (using decontaminated clean val)
            for s in clean_val:
                src_img = os.path.join(naive_dir, "images", "val", f"{s}.jpg")
                src_lbl = os.path.join(naive_dir, "labels", "val", f"{s}.txt")
                if os.path.exists(src_img): shutil.copy2(src_img, os.path.join(cond_dir, "images", "val", f"{s}.jpg"))
                if os.path.exists(src_lbl): shutil.copy2(src_lbl, os.path.join(cond_dir, "labels", "val", f"{s}.txt"))

            # Copy test subsets
            for s in leaked_test_stems:
                src_img = os.path.join(naive_dir, "images", "test", f"{s}.jpg")
                src_lbl = os.path.join(naive_dir, "labels", "test", f"{s}.txt")
                if os.path.exists(src_img):
                    shutil.copy2(src_img, os.path.join(cond_dir, "images", "test", f"{s}.jpg"))
                    shutil.copy2(src_img, os.path.join(cond_dir, "images", "test_leaked", f"{s}.jpg"))
                if os.path.exists(src_lbl):
                    shutil.copy2(src_lbl, os.path.join(cond_dir, "labels", "test", f"{s}.txt"))
                    shutil.copy2(src_lbl, os.path.join(cond_dir, "labels", "test_leaked", f"{s}.txt"))

            for s in clean_test_stems:
                src_img = os.path.join(naive_dir, "images", "test", f"{s}.jpg")
                src_lbl = os.path.join(naive_dir, "labels", "test", f"{s}.txt")
                if os.path.exists(src_img):
                    shutil.copy2(src_img, os.path.join(cond_dir, "images", "test", f"{s}.jpg"))
                    shutil.copy2(src_img, os.path.join(cond_dir, "images", "test_clean", f"{s}.jpg"))
                if os.path.exists(src_lbl):
                    shutil.copy2(src_lbl, os.path.join(cond_dir, "labels", "test", f"{s}.txt"))
                    shutil.copy2(src_lbl, os.path.join(cond_dir, "labels", "test_clean", f"{s}.txt"))

            # Copy training images
            for s in stems:
                src_img = os.path.join(naive_dir, "images", "train", f"{s}.jpg")
                src_lbl = os.path.join(naive_dir, "labels", "train", f"{s}.txt")
                if os.path.exists(src_img): shutil.copy2(src_img, os.path.join(cond_dir, "images", "train", f"{s}.jpg"))
                if os.path.exists(src_lbl): shutil.copy2(src_lbl, os.path.join(cond_dir, "labels", "train", f"{s}.txt"))

            # Write YAML
            yaml_content = f"""path: {Path(cond_dir).as_posix()}
train: images/train
val: images/val
test: images/test
names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
"""
            with open(os.path.join(cond_dir, "neu.yaml"), "w", encoding="utf-8") as yf:
                yf.write(yaml_content)
            with open(os.path.join(cond_dir, "neu_leaked.yaml"), "w", encoding="utf-8") as yf:
                yf.write(yaml_content.replace("test: images/test", "test: images/test_leaked"))
            with open(os.path.join(cond_dir, "neu_clean.yaml"), "w", encoding="utf-8") as yf:
                yf.write(yaml_content.replace("test: images/test", "test: images/test_clean"))

    # Save manifest
    manifest_out = os.path.join(RESULTS_DIR, "step3_counterfactual_splits_manifest.json")
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump({
            "total_draws": M,
            "train_size_matched": len(train_raw) - k,
            "p_leak_count_k": k,
            "leaked_test_count": len(leaked_test_stems),
            "clean_test_count": len(clean_test_stems),
            "decontaminated_val_count": len(clean_val),
            "draws": draws_manifest
        }, f, indent=2)
    print(f"Generated {M} matched-size draws and saved manifest to {manifest_out}")

if __name__ == "__main__":
    build_counterfactual_splits(M=3, seed=42)
