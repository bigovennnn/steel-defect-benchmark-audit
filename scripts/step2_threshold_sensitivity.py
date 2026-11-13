#!/usr/bin/env python3
"""
Step 2B: Near-Duplicate Registration Threshold Sensitivity Analysis
Analyzes how coarse screening thresholds (0.80, 0.85, 0.90) and native-resolution
verification thresholds (pixel_ncc, gradient_ncc) affect candidate edge counts,
connected component topology, and naive cross-split leakage counts.
"""
import csv
import json
from pathlib import Path
from collections import defaultdict
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AUDITS = ROOT / "benchmark" / "audits"
RESULTS = ROOT / "results"


def analyze_graph(edges):
    G = nx.Graph()
    for a, b in edges:
        G.add_edge(a, b)
    n_edges = G.number_of_edges()
    n_nodes = G.number_of_nodes()
    comps = list(nx.connected_components(G))
    n_comps = len(comps)
    max_comp = max((len(c) for c in comps), default=0)
    return {
        "edges": n_edges,
        "unique_images": n_nodes,
        "connected_components": n_comps,
        "max_component_size": max_comp
    }


def main():
    edges_223_path = AUDITS / "neu_candidate_edges_223_20261101.csv"
    edges_219_path = AUDITS / "neu_candidate_edges_219_20261101.csv"
    band_path = AUDITS / "neu_clean_v2_rigid_band_080_085_20261101" / "band_audit.json"
    cand_path = AUDITS / "neu_clean_v2_rigid_20261031" / "candidate_audit.json"
    supp_path = AUDITS / "neu_clean_v2_rigid_supplement_v2_20261101" / "supplemental_audit.json"
    naive_split_path = ROOT / "benchmark" / "datasets" / "neu_naive_random" / "split.json"

    rows_223 = list(csv.DictReader(edges_223_path.open(encoding="utf-8")))
    rows_219 = list(csv.DictReader(edges_219_path.open(encoding="utf-8")))
    band_data = json.loads(band_path.read_text(encoding="utf-8"))
    cand_data = json.loads(cand_path.read_text(encoding="utf-8"))
    supp_data = json.loads(supp_path.read_text(encoding="utf-8"))
    naive_split = json.loads(naive_split_path.read_text(encoding="utf-8"))

    naive_map = {}
    for s in naive_split["train"]:
        naive_map[f"{s}.jpg"] = "train"
    for s in naive_split["val"]:
        naive_map[f"{s}.jpg"] = "val"
    for s in naive_split["test"]:
        naive_map[f"{s}.jpg"] = "test"

    # Build lookup of coarse_ncc for each of the 223 edges
    coarse_lookup = {}
    for item in cand_data["review_candidates"]:
        pair = tuple(sorted([item["a"]["file"], item["b"]["file"]]))
        coarse_lookup[pair] = float(item["coarse_ncc"])
    for item in supp_data["review_candidates"]:
        pair = tuple(sorted([item["a"]["file"], item["b"]["file"]]))
        coarse_lookup[pair] = float(item["coarse_ncc"])
    for item in band_data["review_candidates"]:
        pair = tuple(sorted([item["a"]["file"], item["b"]["file"]]))
        coarse_lookup[pair] = float(item["coarse_ncc"])

    enriched_edges = []
    for r in rows_223:
        pair = tuple(sorted([r["a_file"], r["b_file"]]))
        c_ncc = coarse_lookup.get(pair, None)
        p_ncc = float(r["pixel_ncc"])
        g_ncc = float(r["gradient_ncc"])
        ov = float(r["overlap_fraction"])
        sa = naive_map[r["a_file"]]
        sb = naive_map[r["b_file"]]
        enriched_edges.append({
            "pair": pair,
            "coarse_ncc": c_ncc,
            "pixel_ncc": p_ncc,
            "gradient_ncc": g_ncc,
            "overlap_fraction": ov,
            "naive_cross_split": sa != sb,
            "naive_train_test": (sa == "train" and sb == "test") or (sa == "test" and sb == "train")
        })

    # 1. Coarse screening threshold sensitivity (holding native verification fixed at pixel>=0.95 & grad>=0.90)
    coarse_thresholds = [0.80, 0.85, 0.90, 0.95]
    coarse_sensitivity = []
    for tau_c in coarse_thresholds:
        sub = [e for e in enriched_edges if e["coarse_ncc"] is not None and e["coarse_ncc"] >= tau_c]
        topo = analyze_graph([e["pair"] for e in sub])
        topo["coarse_threshold"] = tau_c
        topo["naive_cross_split_edges"] = sum(1 for e in sub if e["naive_cross_split"])
        topo["naive_train_test_edges"] = sum(1 for e in sub if e["naive_train_test"])
        coarse_sensitivity.append(topo)

    # 2. Native-resolution dual NCC threshold sensitivity (across the 223 verified pool)
    native_thresholds = [
        {"label": "pixel>=0.95 & grad>=0.90 (Frozen v5 baseline, coarse>=0.80)", "p_th": 0.95, "g_th": 0.90, "c_th": 0.80},
        {"label": "pixel>=0.95 & grad>=0.90 (Primary coarse>=0.85 only)", "p_th": 0.95, "g_th": 0.90, "c_th": 0.85},
        {"label": "pixel>=0.98 & grad>=0.95 (Strict native)", "p_th": 0.98, "g_th": 0.95, "c_th": 0.80},
        {"label": "pixel>=0.99 & grad>=0.98 (Ultra-strict native)", "p_th": 0.99, "g_th": 0.98, "c_th": 0.80},
    ]
    native_sensitivity = []
    for cfg in native_thresholds:
        sub = [
            e for e in enriched_edges
            if e["coarse_ncc"] >= cfg["c_th"]
            and e["pixel_ncc"] >= cfg["p_th"]
            and e["gradient_ncc"] >= cfg["g_th"]
        ]
        topo = analyze_graph([e["pair"] for e in sub])
        topo.update(cfg)
        topo["naive_cross_split_edges"] = sum(1 for e in sub if e["naive_cross_split"])
        topo["naive_train_test_edges"] = sum(1 for e in sub if e["naive_train_test"])
        native_sensitivity.append(topo)

    p_vals = [e["pixel_ncc"] for e in enriched_edges]
    g_vals = [e["gradient_ncc"] for e in enriched_edges]
    c_vals = [e["coarse_ncc"] for e in enriched_edges]
    ov_vals = [e["overlap_fraction"] for e in enriched_edges]

    report = {
        "pipeline_exact_parameters": {
            "total_valid_images": 1797,
            "total_unordered_pairs_screened": 1613706,
            "stage1_coarse_16x16": {
                "primary_threshold": 0.85,
                "marginal_band_audited": [0.80, 0.85],
                "min_overlap_fraction": 0.60,
                "max_shift_fraction": 0.25,
                "pairs_passing_085": 122706,
                "pairs_in_band_080_085": 96572
            },
            "stage2_intermediate_64x64": {
                "threshold": 0.85,
                "passed_count": 44915,
                "rejected_but_supplementally_audited_at_200x200": 77791
            },
            "stage3_native_200x200_verification": {
                "pixel_ncc_threshold": 0.95,
                "gradient_ncc_threshold": 0.90,
                "min_overlap_fraction": 0.60,
                "from_stage2_pass": 213,
                "from_stage2_supplement": 6,
                "subtotal_coarse_ge_085": 219,
                "from_coarse_band_080_085": 4,
                "total_merged_candidate_edges": 223
            }
        },
        "distribution_across_223_edges": {
            "coarse_ncc_min": round(min(c_vals), 4),
            "coarse_ncc_max": round(max(c_vals), 4),
            "pixel_ncc_min": round(min(p_vals), 4),
            "pixel_ncc_median": round(float(np.median(p_vals)), 4),
            "pixel_ncc_max": round(max(p_vals), 4),
            "gradient_ncc_min": round(min(g_vals), 4),
            "gradient_ncc_median": round(float(np.median(g_vals)), 4),
            "gradient_ncc_max": round(max(g_vals), 4),
            "overlap_min": round(min(ov_vals), 4),
            "overlap_max": round(max(ov_vals), 4)
        },
        "borderline_4_pairs_in_080_085": [
            {
                "a_file": item["a"]["file"],
                "b_file": item["b"]["file"],
                "coarse_ncc": round(float(item["coarse_ncc"]), 4),
                "pixel_ncc": round(float(item["pixel_ncc"]), 4),
                "gradient_ncc": round(float(item["gradient_ncc"]), 4),
                "overlap_fraction": round(float(item["overlap_fraction"]), 4)
            }
            for item in band_data["review_candidates"]
        ],
        "coarse_threshold_sensitivity": coarse_sensitivity,
        "native_threshold_sensitivity": native_sensitivity
    }

    out_path = RESULTS / "step2_threshold_sensitivity.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
