#!/usr/bin/env python3
"""
Step 2 (GC10-DET per Decision D5):
Generates `benchmark/datasets/gc10_grouped_v1` (1,831 train / 230 val / 230 test across 2,291 valid GC10-DET images)
using Mixed-Integer Programming (MIP) to assign each of the 467 coil/sequence groups (`img_XX_<coil_id>_<frame>`)
entirely to a single partition (0 cross-split coil edges) while balancing the 10 defect classes across splits.
"""

import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT_DIR / "benchmark" / "datasets"
SRC_GC10_DIR = DATASETS_DIR / "gc10_review_v1"
DST_GC10_DIR = DATASETS_DIR / "gc10_grouped_v1"
RESULTS_DIR = ROOT_DIR / "results"

CLASSES = [
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


def main():
    old_sp = json.loads((SRC_GC10_DIR / "split.json").read_text(encoding="utf-8"))
    img_map, lbl_map = {}, {}
    for part in ["train", "val", "test"]:
        for p in (SRC_GC10_DIR / "images" / part).glob("*.jpg"):
            img_map[p.stem] = p
        for p in (SRC_GC10_DIR / "labels" / part).glob("*.txt"):
            lbl_map[p.stem] = p

    stems = sorted(img_map.keys())
    N_total = len(stems)
    N_target = np.array([1831.0, 230.0, 230.0])
    ratios = N_target / N_total

    # Group by coil ID (3rd token of img_XX_<coil>_<seq>)
    coil_groups = defaultdict(list)
    for s in stems:
        parts = s.split("_")
        cid = parts[2] if len(parts) >= 4 else s
        coil_groups[cid].append(s)

    comps = [sorted(v) for _, v in sorted(coil_groups.items())]
    K = len(comps)
    m_k = np.array([len(c) for c in comps], dtype=np.float64)

    # Count bounding-box classes per image and per component
    n_cls = np.zeros((K, 10), dtype=np.float64)
    for k, comp in enumerate(comps):
        for st in comp:
            lines = lbl_map[st].read_text(encoding="utf-8").splitlines()
            present_cls = set(int(line.split()[0]) for line in lines if line.strip())
            for c_id in present_cls:
                if 0 <= c_id < 10:
                    n_cls[k, c_id] += 1.0

    T_cls = np.outer(n_cls.sum(axis=0), ratios)  # (10, 3)

    # MIP variables: x_{k, s} binary (3*K), e_s continuous (3), d_{c, s} continuous (30)
    n_x = 3 * K
    n_e = 3
    n_d = 30
    n_vars = n_x + n_e + n_d

    rng = np.random.default_rng(20261111)
    c_obj = np.zeros(n_vars, dtype=np.float64)
    c_obj[:n_x] = rng.uniform(-1e-4, 1e-4, size=n_x)
    for s in range(3):
        c_obj[n_x + s] = 20.0 / N_target[s]
    for c in range(10):
        for s in range(3):
            c_obj[n_x + n_e + c * 3 + s] = 1.0 / max(1.0, T_cls[c, s])

    integrality = np.zeros(n_vars, dtype=np.int32)
    integrality[:n_x] = 1
    lb = np.zeros(n_vars, dtype=np.float64)
    ub = np.full(n_vars, np.inf, dtype=np.float64)
    ub[:n_x] = 1.0

    rows, cols, vals, lhs, rhs = [], [], [], [], []
    r_idx = 0

    for k in range(K):
        for s in range(3):
            rows.append(r_idx)
            cols.append(k * 3 + s)
            vals.append(1.0)
        lhs.append(1.0)
        rhs.append(1.0)
        r_idx += 1

    # Exact total subset sizes: 1831, 230, 230
    for s in range(3):
        for k in range(K):
            rows.append(r_idx)
            cols.append(k * 3 + s)
            vals.append(m_k[k])
        lhs.append(N_target[s])
        rhs.append(N_target[s])
        r_idx += 1

    # Soft class deviations d_{c, s}
    for c in range(10):
        for s in range(3):
            d_idx = n_x + n_e + c * 3 + s
            for k in range(K):
                if n_cls[k, c] > 0:
                    rows.append(r_idx)
                    cols.append(k * 3 + s)
                    vals.append(n_cls[k, c])
            rows.append(r_idx)
            cols.append(d_idx)
            vals.append(-1.0)
            lhs.append(-np.inf)
            rhs.append(T_cls[c, s])
            r_idx += 1

            for k in range(K):
                if n_cls[k, c] > 0:
                    rows.append(r_idx)
                    cols.append(k * 3 + s)
                    vals.append(n_cls[k, c])
            rows.append(r_idx)
            cols.append(d_idx)
            vals.append(1.0)
            lhs.append(T_cls[c, s])
            rhs.append(np.inf)
            r_idx += 1

    A_mat = csc_matrix((vals, (rows, cols)), shape=(r_idx, n_vars))
    res = milp(
        c=c_obj,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(A_mat, lhs, rhs),
    )
    if not res.success:
        raise RuntimeError(f"GC10 MIP failed: {res.message}")

    x_sol = np.round(res.x[:n_x]).astype(int).reshape((K, 3))
    subsets = {0: "train", 1: "val", 2: "test"}
    new_split = {"train": [], "val": [], "test": []}
    for k in range(K):
        s_idx = int(np.argmax(x_sol[k]))
        new_split[subsets[s_idx]].extend(comps[k])
    for s in new_split:
        new_split[s] = sorted(new_split[s])

    # Materialize dataset
    DST_GC10_DIR.mkdir(parents=True, exist_ok=True)
    for part, part_stems in new_split.items():
        img_sub = DST_GC10_DIR / "images" / part
        lbl_sub = DST_GC10_DIR / "labels" / part
        if img_sub.exists():
            shutil.rmtree(img_sub)
        if lbl_sub.exists():
            shutil.rmtree(lbl_sub)
        img_sub.mkdir(parents=True, exist_ok=True)
        lbl_sub.mkdir(parents=True, exist_ok=True)
        for st in part_stems:
            link_or_copy(img_map[st], img_sub / f"{st}.jpg")
            link_or_copy(lbl_map[st], lbl_sub / f"{st}.txt")

    names_block = "\n".join([f"  {i}: {c}" for i, c in enumerate(CLASSES)])
    path_str = str(DST_GC10_DIR.resolve()).replace("\\", "/")
    yaml_content = f"path: {path_str}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names_block}\n"
    (DST_GC10_DIR / "gc10.yaml").write_text(yaml_content, encoding="utf-8")

    canonical = json.dumps({k: sorted(new_split[k]) for k in ["train", "val", "test"]}, sort_keys=True, separators=(",", ":"))
    sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    split_json = {
        "train": new_split["train"],
        "val": new_split["val"],
        "test": new_split["test"],
        "provenance": {
            "name": "gc10_grouped_v1",
            "created_date": "2026-11-11",
            "grouping_criterion": "MIP coil-disjoint grouping (467 coil groups from filename token 3, 0 cross-split coil edges)",
            "mip_seed": 20261111,
            "design_sha256": sha,
            "counts": {k: len(new_split[k]) for k in ["train", "val", "test"]},
            "excluded_from_raw_2294": old_sp.get("provenance", {}).get("excluded", {}),
        },
    }
    (DST_GC10_DIR / "split.json").write_text(json.dumps(split_json, indent=2), encoding="utf-8")
    print(f"Generated gc10_grouped_v1 (sha256={sha}):", split_json["provenance"]["counts"])


if __name__ == "__main__":
    main()
