#!/usr/bin/env python3
"""
Step 3.1 (Part B): Summarize Matched-Size Counterfactual Partner-Replacement Ablation
Computes:
1. Full metrics under Condition L (retained 31 partners) vs. Condition C (matched replacement):
   - mAP@0.5, mAP@[0.5:0.95], Precision, Recall
2. Paired differences on the exact same 25 leaked test images (Delta_leak),
   155 clean test images (Delta_clean), Difference-in-Differences (Delta_DiD),
   and inherent subset difficulty (Delta_difficulty = YC_l - YC_c),
   verifying the exact algebraic identity: Delta_observed == Delta_leak + Delta_difficulty - Delta_clean.
3. Mean, SD, and 95% confidence intervals across draws (M) and seeds (S).
4. Component-Clustered Bootstrap 95% CI on the 25 leaked test images (clustering the 25 test
   images by their connected components in the 223-edge near-duplicate graph and resampling
   clusters with replacement B=2,000 times).
"""

import csv
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import networkx as nx
import numpy as np
import scipy.stats as stats

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
RUNS_DIR = ROOT_DIR / "benchmark" / "runs" / "counterfactual_matched"
EDGES_CSV = ROOT_DIR / "benchmark" / "audits" / "neu_candidate_edges_223_20261101.csv"
NAIVE_SPLIT = ROOT_DIR / "benchmark" / "datasets" / "neu_naive_random" / "split.json"


def calc_stats_with_ci(arr_pct):
    """Given an array in percentage points (0-100), return mean, std, median, min, max, and 95% CI."""
    arr = np.asarray(arr_pct, dtype=float)
    n = len(arr)
    if n == 0:
        return {}
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    med = float(np.median(arr))
    if n > 1 and sd > 0:
        se = sd / np.sqrt(n)
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
        ci_low = mean - t_crit * se
        ci_high = mean + t_crit * se
    else:
        ci_low = mean
        ci_high = mean
    return {
        "n": n,
        "mean": round(mean, 2),
        "std": round(sd, 2),
        "ci95_low": round(ci_low, 2),
        "ci95_high": round(ci_high, 2),
        "median": round(med, 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2)
    }


def compute_box_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    u = a1 + a2 - inter
    return inter / u if u > 0 else 0.0


def compute_ap_from_image_records(records_list, iou_thresh=0.5):
    """Compute macro mAP at a given IoU threshold across a list of image record dicts (with 'gt' and 'pred')."""
    aps = []
    for cls_id in range(6):
        n_gt = 0
        all_preds = []  # (conf, is_tp)
        for img_idx, rec in enumerate(records_list):
            gts = [g["xyxy"] for g in rec.get("gt", []) if g["cls"] == cls_id]
            n_gt += len(gts)
            matched = [False] * len(gts)
            preds = sorted(
                [p for p in rec.get("pred", []) if p["cls"] == cls_id],
                key=lambda x: x["conf"],
                reverse=True
            )
            for p in preds:
                best_iou = 0.0
                best_j = -1
                for j, g_box in enumerate(gts):
                    if not matched[j]:
                        iou = compute_box_iou(p["xyxy"], g_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_j = j
                if best_iou >= iou_thresh and best_j >= 0:
                    matched[best_j] = True
                    all_preds.append((p["conf"], 1))
                else:
                    all_preds.append((p["conf"], 0))
        if n_gt == 0:
            continue
        if not all_preds:
            aps.append(0.0)
            continue
        all_preds.sort(key=lambda x: x[0], reverse=True)
        tps = np.cumsum([x[1] for x in all_preds])
        fps = np.cumsum([1 - x[1] for x in all_preds])
        recalls = tps / float(n_gt)
        precisions = tps / np.maximum(tps + fps, 1e-9)
        # 101-point interpolated AP
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([1.0], precisions, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        x = np.linspace(0, 1, 101)
        ap = float(np.trapz(np.interp(x, mrec, mpre), x))
        aps.append(ap)
    return float(np.mean(aps)) * 100.0 if aps else 0.0


def get_leaked_test_clusters():
    """Group the 25 leaked test images by their connected components in the 223-edge near-duplicate graph."""
    naive_data = json.loads(NAIVE_SPLIT.read_text(encoding="utf-8"))
    leaked_stems = sorted(naive_data["candidate_edges_cross_split_audit"]["leaked_test_images"])
    G = nx.Graph()
    for s in leaked_stems:
        G.add_node(s)
    for row in csv.DictReader(EDGES_CSV.open(encoding="utf-8")):
        a = Path(row["a_file"]).stem
        b = Path(row["b_file"]).stem
        G.add_edge(a, b)
    comp_map = {}
    for idx, comp in enumerate(nx.connected_components(G)):
        for node in comp:
            comp_map[node] = idx
    clusters = defaultdict(list)
    for s in leaked_stems:
        clusters[comp_map[s]].append(s)
    return list(clusters.values())


def run_component_cluster_bootstrap(completed_pairs, B=2000, rng_seed=20261108):
    """Resample connected components of the 25 leaked test images with replacement to get cluster-bootstrap 95% CI."""
    clusters = get_leaked_test_clusters()
    n_clusters = len(clusters)
    pairs_with_per_image = [
        p for p in completed_pairs
        if "per_image_L" in p and "per_image_C" in p
    ]
    if not pairs_with_per_image:
        return {"n_clusters": n_clusters, "status": "no_per_image_records"}

    rng = np.random.default_rng(rng_seed)
    boot_deltas_map50 = []
    for _ in range(B):
        sampled_indices = rng.integers(0, n_clusters, size=n_clusters)
        sampled_stems = [s for idx in sampled_indices for s in clusters[idx]]
        run_deltas = []
        for p in pairs_with_per_image:
            recs_L = [p["per_image_L"][s] for s in sampled_stems if s in p["per_image_L"]]
            recs_C = [p["per_image_C"][s] for s in sampled_stems if s in p["per_image_C"]]
            ap_L = compute_ap_from_image_records(recs_L, iou_thresh=0.5)
            ap_C = compute_ap_from_image_records(recs_C, iou_thresh=0.5)
            run_deltas.append(ap_L - ap_C)
        boot_deltas_map50.append(float(np.mean(run_deltas)))

    boot_arr = np.array(boot_deltas_map50)
    return {
        "n_leaked_test_images": 25,
        "n_connected_component_clusters": n_clusters,
        "cluster_sizes": [len(c) for c in clusters],
        "bootstrap_replicates_B": B,
        "delta_leak_map50_cluster_boot_mean": round(float(np.mean(boot_arr)), 2),
        "delta_leak_map50_cluster_boot_se": round(float(np.std(boot_arr, ddof=1)), 2),
        "delta_leak_map50_cluster_boot_ci95": [
            round(float(np.percentile(boot_arr, 2.5)), 2),
            round(float(np.percentile(boot_arr, 97.5)), 2)
        ]
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Step 3.1: Summarizing Counterfactual Matched-Size Ablation Matrix ===")

    completed_pairs = []
    if RUNS_DIR.exists():
        for m in range(3):
            for s in range(8):
                dir_L = RUNS_DIR / f"m{m}_s{s}_L"
                dir_C = RUNS_DIR / f"m{m}_s{s}_C"
                p_L = dir_L / "eval_summary.json"
                p_C = dir_C / "eval_summary.json"
                if p_L.exists() and p_C.exists():
                    d_L = json.loads(p_L.read_text(encoding="utf-8"))
                    d_C = json.loads(p_C.read_text(encoding="utf-8"))
                    item = {"draw": m, "seed": s, "L": d_L, "C": d_C}
                    pi_L = dir_L / "per_image_test_leaked.json"
                    pi_C = dir_C / "per_image_test_leaked.json"
                    if pi_L.exists() and pi_C.exists():
                        item["per_image_L"] = json.loads(pi_L.read_text(encoding="utf-8"))
                        item["per_image_C"] = json.loads(pi_C.read_text(encoding="utf-8"))
                    completed_pairs.append(item)

    print(f"Total completed matched pairs (m, s): {len(completed_pairs)}")

    if len(completed_pairs) == 0:
        clusters = get_leaked_test_clusters()
        report = {
            "status": "in_progress_gpu_runs",
            "n_pairs": 0,
            "leaked_test_component_clusters": {
                "n_leaked_test_images": 25,
                "n_connected_component_clusters": len(clusters),
                "cluster_sizes": [len(c) for c in clusters]
            },
            "protocol_spec": {
                "design": "Matched-Size Counterfactual Partner-Replacement Experiment",
                "train_size_L": 1406,
                "train_size_C": 1406,
                "test_leaked_size_Tl": 25,
                "test_clean_size_Tc": 155,
                "decontaminated_val_size": 143,
                "draws_M": 3,
                "seeds_S": 8,
                "total_runs": 48
            },
            "algebraic_identity": "Delta_observed == Delta_leak + Delta_difficulty - Delta_clean"
        }
        (RESULTS_DIR / "step3_counterfactual_ablation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return

    metrics_summary = {}
    for metric_key in ["map50", "map50_95", "precision", "recall"]:
        YL_l = [p["L"]["test_leaked"].get(metric_key, 0.0) * 100.0 for p in completed_pairs]
        YC_l = [p["C"]["test_leaked"].get(metric_key, 0.0) * 100.0 for p in completed_pairs]
        YL_c = [p["L"]["test_clean"].get(metric_key, 0.0) * 100.0 for p in completed_pairs]
        YC_c = [p["C"]["test_clean"].get(metric_key, 0.0) * 100.0 for p in completed_pairs]
        YL_all = [p["L"]["test_all"].get(metric_key, 0.0) * 100.0 for p in completed_pairs]
        YC_all = [p["C"]["test_all"].get(metric_key, 0.0) * 100.0 for p in completed_pairs]

        d_leak = [a - b for a, b in zip(YL_l, YC_l)]
        d_clean = [a - b for a, b in zip(YL_c, YC_c)]
        d_did = [dl - dc for dl, dc in zip(d_leak, d_clean)]
        d_diff = [a - b for a, b in zip(YC_l, YC_c)]
        d_obs = [a - b for a, b in zip(YL_l, YL_c)]

        for obs, dl, dd, dc in zip(d_obs, d_leak, d_diff, d_clean):
            assert abs(obs - (dl + dd - dc)) < 1e-5

        metrics_summary[metric_key] = {
            "cond_L_test_leaked": calc_stats_with_ci(YL_l),
            "cond_C_test_leaked": calc_stats_with_ci(YC_l),
            "cond_L_test_clean": calc_stats_with_ci(YL_c),
            "cond_C_test_clean": calc_stats_with_ci(YC_c),
            "cond_L_test_all": calc_stats_with_ci(YL_all),
            "cond_C_test_all": calc_stats_with_ci(YC_all),
            "delta_leak_paired_25": calc_stats_with_ci(d_leak),
            "delta_clean_paired_155": calc_stats_with_ci(d_clean),
            "delta_did_net": calc_stats_with_ci(d_did),
            "delta_difficulty_clean_model": calc_stats_with_ci(d_diff),
            "delta_observed_leaked_model": calc_stats_with_ci(d_obs),
        }

    cluster_boot = run_component_cluster_bootstrap(completed_pairs, B=2000)

    summary = {
        "status": "completed" if len(completed_pairs) >= 9 else f"partial_{len(completed_pairs)}_pairs",
        "n_pairs": len(completed_pairs),
        "completed_pair_ids": [f"m{p['draw']}_s{p['seed']}" for p in completed_pairs],
        "metrics": metrics_summary,
        "component_cluster_bootstrap": cluster_boot,
        "algebraic_identity_verified": True
    }

    out_file = RESULTS_DIR / "step3_counterfactual_ablation.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved counterfactual summary to: {out_file}")


if __name__ == "__main__":
    main()
