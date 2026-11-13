#!/usr/bin/env python3
"""
Step 2.3: Solve Mixed-Integer Programming (MIP) Partition Model with Slack Variables
Part of AGENT EXECUTION PLAYBOOK (Phase 2).
"""

import os
import sys
import glob
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AUDITS_DIR = os.path.join(ROOT_DIR, "benchmark", "audits")
SPLIT_PATH = os.path.join(ROOT_DIR, "benchmark", "datasets", "neu_v5_frozen", "split.json")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

def parse_xml_class_counts(raw_xml_dirs):
    xml_map = {}
    for d in raw_xml_dirs:
        for f in glob.glob(os.path.join(d, "*.xml")):
            stem = os.path.splitext(os.path.basename(f))[0]
            xml_map[stem] = f

    stem_box_counts = {}
    for stem, path in xml_map.items():
        tree = ET.parse(path)
        root = tree.getroot()
        counts = Counter()
        for obj in root.findall("object"):
            c = obj.find("name").text.strip().replace(" ", "_")
            if c in CLASSES:
                counts[c] += 1
        stem_box_counts[stem] = counts
    return stem_box_counts

def find_connected_components(stems, edges):
    stem_to_idx = {s: i for i, s in enumerate(stems)}
    adj = defaultdict(set)
    for u, v in edges:
        if u in stem_to_idx and v in stem_to_idx:
            adj[u].add(v)
            adj[v].add(u)

    visited = set()
    components = []
    for s in stems:
        if s not in visited:
            comp = []
            queue = [s]
            visited.add(s)
            while queue:
                curr = queue.pop()
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(comp)
    return components

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== Step 2.3: Formulating & Solving MIP Partition Model ===")

    # 1. Load frozen stems and edges
    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    frozen_stems = sorted(list(set(split_data["train"] + split_data["val"] + split_data["test"])))
    print(f"Total frozen stems: {len(frozen_stems)}")

    edges_csv = os.path.join(AUDITS_DIR, "neu_candidate_edges_223_20261101.csv")
    df_edges = pd.read_csv(edges_csv)
    edges = []
    for _, row in df_edges.iterrows():
        a = os.path.splitext(row["a_file"])[0]
        b = os.path.splitext(row["b_file"])[0]
        edges.append((a, b))
    print(f"Loaded {len(edges)} verified candidate edges")

    # 2. Extract Connected Components
    components = find_connected_components(frozen_stems, edges)
    K = len(components)
    comp_sizes = [len(c) for c in components]
    max_comp_size = max(comp_sizes)
    multi_comps = [c for c in components if len(c) > 1]
    print(f"Total connected components K = {K}")
    print(f"Multi-image components: {len(multi_comps)}, Max component size: {max_comp_size}")

    # Load XML annotations for box counts
    raw_xml_dirs = [
        os.path.join(ROOT_DIR, "benchmark", "paper-try", "datasets", "raw", "NEU-DET", "ANNOTATIONS"),
        os.path.join(ROOT_DIR, "benchmark", "paper-try", "datasets", "raw", "NEU-DET", "Validation_Annotations"),
    ]
    stem_box_counts = parse_xml_class_counts(raw_xml_dirs)

    # Component attributes
    # m_k: size of comp k
    # n_img[k, c]: number of images in comp k with class c prefix
    # n_box[k, c]: number of boxes in comp k with class c
    m_k = np.array(comp_sizes, dtype=np.float64)
    n_img = np.zeros((K, 6), dtype=np.float64)
    n_box = np.zeros((K, 6), dtype=np.float64)

    for k, comp in enumerate(components):
        for stem in comp:
            cls_prefix = stem.rsplit("_", 1)[0]
            if cls_prefix in CLASSES:
                c_idx = CLASSES.index(cls_prefix)
                n_img[k, c_idx] += 1
            for c_name, b_cnt in stem_box_counts.get(stem, {}).items():
                c_idx = CLASSES.index(c_name)
                n_box[k, c_idx] += b_cnt

    # Target capacities and class targets
    # N = [1437, 180, 180]
    N_s = np.array([1437.0, 180.0, 180.0])
    N_total = 1797.0
    split_ratios = N_s / N_total

    # Class target by image prefix:
    total_class_imgs = n_img.sum(axis=0)  # e.g. [300, 300, 299, 299, 300, 299]
    T_img = np.outer(total_class_imgs, split_ratios)  # shape (6, 3)

    # 3. Formulate MIP Problem
    # Decision variables:
    # x_{k, s} for k in 0..K-1, s in 0..2 (3*K binary variables)
    # e_s for s in 0..2 (3 continuous variables)
    # d_{c, s} for c in 0..5, s in 0..2 (18 continuous variables)
    # Total variables = 3*K + 3 + 18
    n_x = 3 * K
    n_e = 3
    n_d = 18
    n_vars = n_x + n_e + n_d

    # Variable types: 1 = integer (binary), 0 = continuous
    integrality = np.zeros(n_vars, dtype=np.int32)
    integrality[:n_x] = 1

    # Variable bounds
    lb = np.zeros(n_vars, dtype=np.float64)
    ub = np.full(n_vars, np.inf, dtype=np.float64)
    ub[:n_x] = 1.0  # binary x_{k, s} in [0, 1]

    # Objective function: min lambda * sum(e_s / N_s) + sum(d_{c,s} / T_{c,s})
    lam = 10.0  # prioritize capacity matching while balancing classes
    c_obj = np.zeros(n_vars, dtype=np.float64)
    
    # e_s coefficients
    for s in range(3):
        c_obj[n_x + s] = lam / N_s[s]

    # d_{c, s} coefficients
    for c in range(6):
        for s in range(3):
            c_obj[n_x + n_e + c * 3 + s] = 1.0 / max(1.0, T_img[c, s])

    # Constraints:
    rows = []
    cols = []
    vals = []
    lhs = []
    rhs = []
    row_idx = 0

    # Constraint 1: Exact partition for each component: sum_{s=0}^2 x_{k, s} = 1
    for k in range(K):
        for s in range(3):
            rows.append(row_idx)
            cols.append(k * 3 + s)
            vals.append(1.0)
        lhs.append(1.0)
        rhs.append(1.0)
        row_idx += 1

    # Constraint 2: Capacity soft bounds:
    # sum_k m_k * x_{k, s} - e_s <= N_s  =>  sum_k m_k * x_{k,s} - e_s in (-inf, N_s]
    # sum_k m_k * x_{k, s} + e_s >= N_s  =>  sum_k m_k * x_{k,s} + e_s in [N_s, +inf)
    for s in range(3):
        # Upper bound
        for k in range(K):
            rows.append(row_idx)
            cols.append(k * 3 + s)
            vals.append(m_k[k])
        rows.append(row_idx)
        cols.append(n_x + s)
        vals.append(-1.0)
        lhs.append(-np.inf)
        rhs.append(N_s[s])
        row_idx += 1

        # Lower bound
        for k in range(K):
            rows.append(row_idx)
            cols.append(k * 3 + s)
            vals.append(m_k[k])
        rows.append(row_idx)
        cols.append(n_x + s)
        vals.append(1.0)
        lhs.append(N_s[s])
        rhs.append(np.inf)
        row_idx += 1

    # Constraint 3: Class balance soft bounds:
    # sum_k n_img[k, c] * x_{k, s} - d_{c, s} <= T_img[c, s]
    # sum_k n_img[k, c] * x_{k, s} + d_{c, s} >= T_img[c, s]
    for c in range(6):
        for s in range(3):
            d_idx = n_x + n_e + c * 3 + s
            # Upper bound
            for k in range(K):
                if n_img[k, c] > 0:
                    rows.append(row_idx)
                    cols.append(k * 3 + s)
                    vals.append(n_img[k, c])
            rows.append(row_idx)
            cols.append(d_idx)
            vals.append(-1.0)
            lhs.append(-np.inf)
            rhs.append(T_img[c, s])
            row_idx += 1

            # Lower bound
            for k in range(K):
                if n_img[k, c] > 0:
                    rows.append(row_idx)
                    cols.append(k * 3 + s)
                    vals.append(n_img[k, c])
            rows.append(row_idx)
            cols.append(d_idx)
            vals.append(1.0)
            lhs.append(T_img[c, s])
            rhs.append(np.inf)
            row_idx += 1

    A_mat = csc_matrix((vals, (rows, cols)), shape=(row_idx, n_vars))
    constraints = LinearConstraint(A_mat, lhs, rhs)
    bounds = Bounds(lb, ub)

    print(f"Solving MIP with {n_vars} variables ({n_x} binary) and {row_idx} linear constraints using HiGHS...")
    res = milp(c=c_obj, integrality=integrality, bounds=bounds, constraints=constraints)

    print(f"MIP Status: {res.status} ({res.message})")
    print(f"Success: {res.success}")
    print(f"Objective value: {res.fun:.4f}")

    # Extract solution
    x_sol = res.x[:n_x].reshape((K, 3))
    comp_assignment = np.argmax(x_sol, axis=1)

    e_sol = res.x[n_x:n_x+n_e]
    d_sol = res.x[n_x+n_e:n_x+n_e+n_d].reshape((6, 3))

    subsets = {0: "train", 1: "val", 2: "test"}
    opt_split = {"train": [], "val": [], "test": []}
    for k, s_idx in enumerate(comp_assignment):
        opt_split[subsets[s_idx]].extend(components[k])

    opt_counts = {k: len(v) for k, v in opt_split.items()}
    print(f"Optimized subset sizes: {opt_counts} (Target: 1437, 180, 180)")
    print(f"Capacity slack e*: train={e_sol[0]:.2f}, val={e_sol[1]:.2f}, test={e_sol[2]:.2f}")

    # Verify cross-split edges in optimized split
    stem_to_split = {}
    for s, stems in opt_split.items():
        for st in stems:
            stem_to_split[st] = s

    leaked_edges = []
    for u, v in edges:
        if stem_to_split.get(u) != stem_to_split.get(v):
            leaked_edges.append((u, v))
    print(f"Cross-split edges in MIP solution: {len(leaked_edges)} (Expected 0)")

    # Also verify frozen split
    frozen_stem_to_split = {}
    for s in ["train", "val", "test"]:
        for st in split_data[s]:
            frozen_stem_to_split[st] = s
    frozen_leaked_edges = []
    for u, v in edges:
        if frozen_stem_to_split.get(u) != frozen_stem_to_split.get(v):
            frozen_leaked_edges.append((u, v))
    print(f"Cross-split edges in neu_v5_frozen: {len(frozen_leaked_edges)} (Expected 0)")

    # Test set cross-fold edges in Li (2026 v2)
    comp_folds_path = os.path.join(ROOT_DIR, "benchmark", "competitor_reference_li2026_v2", "splits", "neu", "folds.json")
    li_cross_fold_edges = 0
    if os.path.exists(comp_folds_path):
        with open(comp_folds_path, "r", encoding="utf-8") as f:
            li_folds = json.load(f)["fold_of"]
        for u, v in edges:
            if u in li_folds and v in li_folds:
                if li_folds[u] != li_folds[v]:
                    li_cross_fold_edges += 1
        print(f"Cross-fold leaked edges in Li (2026) 5-fold CV: {li_cross_fold_edges} / {len(edges)} ({li_cross_fold_edges/len(edges)*100:.1f}%)")

    # Output solution report
    mip_report = {
        "solver": "HiGHS (via scipy.optimize.milp)",
        "scipy_version": scipy.__version__ if "scipy" in sys.modules else "1.10+",
        "success": bool(res.success),
        "status_code": int(res.status),
        "status_message": str(res.message),
        "components_count_K": K,
        "max_component_size": max_comp_size,
        "multi_image_components_count": len(multi_comps),
        "optimal_objective_value": round(float(res.fun), 4),
        "capacity_slacks": {
            "e_train": round(float(e_sol[0]), 2),
            "e_val": round(float(e_sol[1]), 2),
            "e_test": round(float(e_sol[2]), 2)
        },
        "optimized_subset_sizes": opt_counts,
        "category_deviations_matrix_d": [[round(float(val), 2) for val in row] for row in d_sol],
        "cross_split_edges_mip": len(leaked_edges),
        "cross_split_edges_neu_v5_frozen": len(frozen_leaked_edges),
        "competitor_li2026_cross_fold_leaked_edges": {
            "leaked_edges_count": li_cross_fold_edges,
            "total_shared_edges": len(edges),
            "leakage_percentage": round(li_cross_fold_edges / len(edges) * 100.0, 2)
        }
    }

    out_json = os.path.join(RESULTS_DIR, "step2_mip_solution_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(mip_report, f, indent=2)
    print(f"Saved MIP solution report to: {out_json}")

if __name__ == "__main__":
    import scipy
    main()
