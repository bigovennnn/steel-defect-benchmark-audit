#!/usr/bin/env python3
"""
Step 2 (v6 Re-split per Decision D7):
Generates:
1. `benchmark/datasets/neu_grouped_v6`: A brand-new MIP-grouped partition of the 1,797 NEU-DET images
   (1,437 train / 180 val / 180 test, 0 cross-split edges across all 223 registration-verified pairs,
   exact class balance, deterministic tie-breaking seed=20261111 so its test/val sets are fresh and
   never touched by any prior smoke test).
2. `benchmark/datasets/neu_k{00..09}_{L,C}`: K=10 independent naive stratified random splits
   (seeds 100..109), each paired with a matched-size counterfactual partner-replacement dataset:
   - Condition L: D_base^(k) + P_leak^(k) (retains training near-duplicate partners of T_leak^(k))
   - Condition C: D_base^(k) + R_holdout^(k) (replaces P_leak^(k) with an equal number of class-matched
     clean training images)
   - Both conditions use the exact same decontaminated validation set V_clean^(k) and evaluate on
     identical test_all (180), test_leaked (|T_leak^(k)|), and test_clean (180 - |T_leak^(k)|).
"""

import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT_DIR / "benchmark"
DATASETS_DIR = BENCH_DIR / "datasets"
SRC_V5_DIR = DATASETS_DIR / "neu_v5_frozen"
EDGES_CSV = BENCH_DIR / "audits" / "neu_candidate_edges_223_20261101.csv"
RESULTS_DIR = ROOT_DIR / "results"

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def stem_class(stem: str) -> str:
    return stem.rsplit("_", 1)[0]


def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def build_stem_lookup(v5_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    img_map, lbl_map = {}, {}
    for part in ["train", "val", "test"]:
        for p in (v5_dir / "images" / part).glob("*.jpg"):
            img_map[p.stem] = p
        for p in (v5_dir / "labels" / part).glob("*.txt"):
            lbl_map[p.stem] = p
    return img_map, lbl_map


def find_connected_components(stems: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    adj = defaultdict(set)
    stem_set = set(stems)
    for u, v in edges:
        if u in stem_set and v in stem_set:
            adj[u].add(v)
            adj[v].add(u)
    visited = set()
    comps = []
    for s in sorted(stems):
        if s not in visited:
            comp = []
            q = [s]
            visited.add(s)
            while q:
                curr = q.pop(0)
                comp.append(curr)
                for nb in sorted(adj[curr]):
                    if nb not in visited:
                        visited.add(nb)
                        q.append(nb)
            comps.append(sorted(comp))
    return comps


def solve_fresh_mip_grouped_split(stems: list[str], edges: list[tuple[str, str]], old_split: dict, seed: int = 20261111) -> dict:
    comps = find_connected_components(stems, edges)
    K = len(comps)
    m_k = np.array([len(c) for c in comps], dtype=np.float64)
    n_img = np.zeros((K, 6), dtype=np.float64)
    for k, comp in enumerate(comps):
        for st in comp:
            c_idx = CLASSES.index(stem_class(st))
            n_img[k, c_idx] += 1.0

    # Target exact class counts per split matching standard stratified 1437 / 180 / 180:
    # Classes with 300 images -> 240 train, 30 val, 30 test
    # Classes with 299 images -> 239 train, 30 val, 30 test
    total_cls = n_img.sum(axis=0)  # [300, 300, 299, 299, 300, 299]
    target_cls = np.zeros((6, 3), dtype=np.float64)
    for c in range(6):
        target_cls[c, 1] = 30.0
        target_cls[c, 2] = 30.0
        target_cls[c, 0] = total_cls[c] - 60.0

    # Decision variables: x_{k,s} in {0,1} for k=0..K-1, s=0..2
    # Plus random perturbation in objective AND a discouragement weight on overlapping with old_split test/val
    rng = np.random.default_rng(seed)
    old_val_set = set(old_split["val"])
    old_test_set = set(old_split["test"])

    n_x = 3 * K
    c_obj = rng.uniform(-1e-3, 1e-3, size=n_x)
    for k, comp in enumerate(comps):
        # Penalize putting old val/test components into the same val/test slot so v6 is genuinely fresh
        if any(st in old_val_set for st in comp):
            c_obj[k * 3 + 1] += 1.0
        if any(st in old_test_set for st in comp):
            c_obj[k * 3 + 2] += 1.0
            c_obj[k * 3 + 1] += 0.5

    rows, cols, vals, lhs, rhs = [], [], [], [], []
    r_idx = 0

    # 1. Each component assigned to exactly one split
    for k in range(K):
        for s in range(3):
            rows.append(r_idx)
            cols.append(k * 3 + s)
            vals.append(1.0)
        lhs.append(1.0)
        rhs.append(1.0)
        r_idx += 1

    # 2. Exact class count constraints per split: sum_k n_img[k, c] * x_{k, s} == target_cls[c, s]
    for c in range(6):
        for s in range(3):
            for k in range(K):
                if n_img[k, c] > 0:
                    rows.append(r_idx)
                    cols.append(k * 3 + s)
                    vals.append(n_img[k, c])
            lhs.append(target_cls[c, s])
            rhs.append(target_cls[c, s])
            r_idx += 1

    A_mat = csc_matrix((vals, (rows, cols)), shape=(r_idx, n_x))
    res = milp(
        c=c_obj,
        integrality=np.ones(n_x, dtype=np.int32),
        bounds=Bounds(np.zeros(n_x), np.ones(n_x)),
        constraints=LinearConstraint(A_mat, lhs, rhs),
    )
    if not res.success:
        raise RuntimeError(f"MIP failed: {res.message}")

    x_sol = np.round(res.x).astype(int).reshape((K, 3))
    subsets = {0: "train", 1: "val", 2: "test"}
    new_split = {"train": [], "val": [], "test": []}
    for k in range(K):
        s_idx = int(np.argmax(x_sol[k]))
        new_split[subsets[s_idx]].extend(comps[k])
    for s in new_split:
        new_split[s] = sorted(new_split[s])

    # Verify 0 cross-split edges and exact counts
    stem_to_s = {st: s for s, lst in new_split.items() for st in lst}
    cross = sum(1 for u, v in edges if stem_to_s[u] != stem_to_s[v])
    assert cross == 0, f"Expected 0 cross edges, got {cross}"
    assert len(new_split["train"]) == 1437 and len(new_split["val"]) == 180 and len(new_split["test"]) == 180
    return new_split


def write_yolo_dataset(
    target_dir: Path,
    splits_map: dict[str, list[str]],
    img_map: dict[str, Path],
    lbl_map: dict[str, Path],
    yaml_files: dict[str, dict[str, str]],
):
    target_dir.mkdir(parents=True, exist_ok=True)
    for sub_name, stems in splits_map.items():
        img_sub = target_dir / "images" / sub_name
        lbl_sub = target_dir / "labels" / sub_name
        if img_sub.exists():
            shutil.rmtree(img_sub)
        if lbl_sub.exists():
            shutil.rmtree(lbl_sub)
        img_sub.mkdir(parents=True, exist_ok=True)
        lbl_sub.mkdir(parents=True, exist_ok=True)
        for st in stems:
            link_or_copy(img_map[st], img_sub / f"{st}.jpg")
            link_or_copy(lbl_map[st], lbl_sub / f"{st}.txt")

    names_block = "\n".join([f"  {i}: {c}" for i, c in enumerate(CLASSES)])
    path_str = str(target_dir.resolve()).replace("\\", "/")
    for y_name, spec in yaml_files.items():
        content = (
            f"path: {path_str}\n"
            f"train: {spec['train']}\n"
            f"val: {spec['val']}\n"
            f"test: {spec['test']}\n"
            f"names:\n{names_block}\n"
        )
        (target_dir / y_name).write_text(content, encoding="utf-8")


def split_sha256(split_obj: dict) -> str:
    canonical = json.dumps(
        {k: sorted(split_obj[k]) for k in ["train", "val", "test"]}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generate_k_naive_counterfactual_splits(
    stems: list[str],
    edges: list[tuple[str, str]],
    img_map: dict[str, Path],
    lbl_map: dict[str, Path],
    k_count: int = 10,
    base_seed: int = 100,
) -> list[dict]:
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    comps = find_connected_components(stems, edges)
    stem_to_comp = {}
    for cid, comp in enumerate(comps):
        for st in comp:
            stem_to_comp[st] = cid

    by_class = defaultdict(list)
    for st in sorted(stems):
        by_class[stem_class(st)].append(st)

    manifests = []
    for k in range(k_count):
        seed_k = base_seed + k
        rng = np.random.default_rng(seed_k)
        train_s, val_s, test_s = [], [], []
        for c in CLASSES:
            c_stems = list(by_class[c])
            perm = rng.permutation(len(c_stems))
            c_stems = [c_stems[i] for i in perm]
            test_s.extend(c_stems[:30])
            val_s.extend(c_stems[30:60])
            train_s.extend(c_stems[60:])
        train_s, val_s, test_s = sorted(train_s), sorted(val_s), sorted(test_s)
        train_set, val_set, test_set = set(train_s), set(val_s), set(test_s)

        # Identify leaked test images (have >=1 candidate partner in train_set)
        test_leaked = sorted([u for u in test_s if any(v in train_set for v in adj[u])])
        test_clean = sorted([u for u in test_s if u not in set(test_leaked)])

        # Identify training partners of test_leaked
        p_leak = sorted([v for v in train_s if any(u in set(test_leaked) for u in adj[v])])
        p_leak_set = set(p_leak)

        # Also identify any training images connected to validation images, and decontaminate validation set
        val_clean = sorted([v for v in val_s if not any((nb in train_set or nb in test_set) for nb in adj[v])])

        # Base training set excludes P_leak
        d_base = sorted([u for u in train_s if u not in p_leak_set])

        # Eligible clean pool in d_base for replacement R_holdout^(k): no edges to test_s or val_clean
        eligible_by_cls = defaultdict(list)
        for u in d_base:
            if not any((nb in test_set or nb in set(val_clean) or nb in p_leak_set) for nb in adj[u]):
                eligible_by_cls[stem_class(u)].append(u)

        p_leak_cls_counts = Counter(stem_class(u) for u in p_leak)
        r_holdout = []
        for c, cnt in sorted(p_leak_cls_counts.items()):
            pool_c = sorted(eligible_by_cls[c])
            if len(pool_c) < cnt:
                raise RuntimeError(f"Split k={k}: insufficient eligible clean pool for {c}")
            chosen_idx = rng.choice(len(pool_c), size=cnt, replace=False)
            r_holdout.extend([pool_c[i] for i in sorted(chosen_idx)])
        r_holdout = sorted(r_holdout)
        r_holdout_set = set(r_holdout)

        # Wait: In matched-size counterfactual design:
        # From the remaining clean training images (d_base), we hold out r_holdout so that:
        # D_core = d_base \ r_holdout
        # Condition L training set = D_core + P_leak  (size = |d_base| = 1437 - |P_leak|)
        # Condition C training set = D_core + R_holdout (= d_base, exact same size and exact same class counts!)
        d_core = [u for u in d_base if u not in r_holdout_set]
        train_L = sorted(d_core + p_leak)
        train_C = sorted(d_core + r_holdout)

        assert len(train_L) == len(train_C), "Train L and C size mismatch!"
        assert Counter(stem_class(u) for u in train_L) == Counter(stem_class(u) for u in train_C), "Class mismatch!"
        assert set(train_L).isdisjoint(test_set) and set(train_C).isdisjoint(test_set)

        # Materialize datasets neu_k{k:02d}_L and neu_k{k:02d}_C
        yaml_specs = {
            "neu.yaml": {"train": "images/train", "val": "images/val", "test": "images/test"},
            "neu_leaked.yaml": {"train": "images/train", "val": "images/val", "test": "images/test_leaked"},
            "neu_clean.yaml": {"train": "images/train", "val": "images/val", "test": "images/test_clean"},
        }
        dir_L = DATASETS_DIR / f"neu_k{k:02d}_L"
        dir_C = DATASETS_DIR / f"neu_k{k:02d}_C"

        write_yolo_dataset(
            dir_L,
            {
                "train": train_L,
                "val": val_clean,
                "test": test_s,
                "test_leaked": test_leaked,
                "test_clean": test_clean,
            },
            img_map,
            lbl_map,
            yaml_specs,
        )
        write_yolo_dataset(
            dir_C,
            {
                "train": train_C,
                "val": val_clean,
                "test": test_s,
                "test_leaked": test_leaked,
                "test_clean": test_clean,
            },
            img_map,
            lbl_map,
            yaml_specs,
        )

        leaked_clusters = len(set(stem_to_comp[u] for u in test_leaked))
        manifests.append(
            {
                "split_index_k": k,
                "split_seed": seed_k,
                "naive_sha256": split_sha256({"train": train_s, "val": val_s, "test": test_s}),
                "n_train_matched": len(train_L),
                "n_val_clean": len(val_clean),
                "n_test_all": len(test_s),
                "n_test_leaked": len(test_leaked),
                "n_test_leaked_clusters": leaked_clusters,
                "n_test_clean": len(test_clean),
                "n_train_partners_P_leak": len(p_leak),
                "p_leak_by_class": dict(p_leak_cls_counts),
                "test_leaked_by_class": dict(Counter(stem_class(u) for u in test_leaked)),
                "test_leaked_stems": test_leaked,
                "p_leak_stems": p_leak,
                "r_holdout_stems": r_holdout,
                "val_clean_stems": val_clean,
            }
        )
    return manifests


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    old_split = json.loads((SRC_V5_DIR / "split.json").read_text(encoding="utf-8"))
    stems = sorted(set(old_split["train"] + old_split["val"] + old_split["test"]))
    df_edges = pd.read_csv(EDGES_CSV)
    edges = [
        ( os.path.splitext(r["a_file"])[0], os.path.splitext(r["b_file"])[0] )
        for _, r in df_edges.iterrows()
    ]
    img_map, lbl_map = build_stem_lookup(SRC_V5_DIR)

    print("1. Solving fresh MIP grouped split `neu_grouped_v6` (seed=20261111)...")
    v6_split = solve_fresh_mip_grouped_split(stems, edges, old_split, seed=20261111)
    v6_sha = split_sha256(v6_split)
    overlap_test = len(set(v6_split["test"]) & set(old_split["test"]))
    overlap_val = len(set(v6_split["val"]) & set(old_split["val"]))
    print(f"   neu_grouped_v6 sha256={v6_sha}")
    print(f"   Overlap with old neu_v5_frozen: test={overlap_test}/180, val={overlap_val}/180")

    v6_dir = DATASETS_DIR / "neu_grouped_v6"
    write_yolo_dataset(
        v6_dir,
        v6_split,
        img_map,
        lbl_map,
        {"neu.yaml": {"train": "images/train", "val": "images/val", "test": "images/test"}},
    )
    v6_split_json = {
        "train": v6_split["train"],
        "val": v6_split["val"],
        "test": v6_split["test"],
        "provenance": {
            "name": "neu_grouped_v6",
            "created_date": "2026-11-11",
            "mip_seed": 20261111,
            "design_sha256": v6_sha,
            "cross_split_edges": 0,
            "smoke_test_policy": "Test set strictly untouched before and during freeze; smoke checks restricted to val.",
        },
    }
    (v6_dir / "split.json").write_text(json.dumps(v6_split_json, indent=2), encoding="utf-8")

    print("2. Generating K=10 independent naive splits & counterfactual pairs `neu_k00..k09_{L,C}`...")
    k_manifests = generate_k_naive_counterfactual_splits(stems, edges, img_map, lbl_map, k_count=10, base_seed=100)

    total_leaked_test_instances = sum(m["n_test_leaked"] for m in k_manifests)
    unique_leaked_test_images = len(set(st for m in k_manifests for st in m["test_leaked_stems"]))
    print(
        f"   Across K=10 splits: total leaked test image-evaluations = {total_leaked_test_instances} "
        f"(mean {total_leaked_test_instances/10:.1f}/split), unique images = {unique_leaked_test_images}"
    )

    out_manifest = {
        "generated_at": "2026-11-11",
        "neu_grouped_v6": {
            "design_sha256": v6_sha,
            "counts": {k: len(v6_split[k]) for k in ["train", "val", "test"]},
            "class_counts": {
                k: dict(Counter(stem_class(u) for u in v6_split[k])) for k in ["train", "val", "test"]
            },
            "overlap_with_legacy_v5": {"test_overlap": overlap_test, "val_overlap": overlap_val},
            "cross_split_edges": 0,
        },
        "multi_split_counterfactual_k10": {
            "k_count": 10,
            "base_seed": 100,
            "total_leaked_test_image_evaluations": total_leaked_test_instances,
            "unique_leaked_test_images_across_k10": unique_leaked_test_images,
            "splits": k_manifests,
        },
    }
    out_path = RESULTS_DIR / "step2_v6_splits_manifest.json"
    out_path.write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")
    print(f"Saved complete manifest to: {out_path}")


if __name__ == "__main__":
    main()
