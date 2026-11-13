#!/usr/bin/env python3
"""
WS-A: Rigorous Statistical Re-Analysis of the 48-Run Counterfactual Ablation (Addressing All 7 Fable Statistical Critiques).
Pre-caches per-image box matching so B=2,000 bootstrap iterations complete in <1 second.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import scipy.stats as stats

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT_DIR / "benchmark" / "runs" / "counterfactual_matched"
EDGES_CSV = ROOT_DIR / "benchmark" / "audits" / "neu_candidate_edges_223_20261101.csv"
RESULTS_DIR = ROOT_DIR / "results"


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


def precompute_image_stat(rec: dict, iou_thresh=0.5, conf_op=0.25) -> dict:
    gts = rec.get("gt", [])
    preds = sorted([p for p in rec.get("pred", []) if p["conf"] >= 0.01], key=lambda x: x["conf"], reverse=True)
    matched_gt = [False] * len(gts)
    matched_preds = []  # (conf, is_tp, cls_id)
    for p in preds:
        p_cls = p["cls"]
        p_box = p["xyxy"]
        best_iou, best_idx = 0.0, -1
        for g_idx, g in enumerate(gts):
            if g["cls"] != p_cls or matched_gt[g_idx]:
                continue
            iou = compute_box_iou(p_box, g["xyxy"])
            if iou > best_iou:
                best_iou, best_idx = iou, g_idx
        if best_iou >= iou_thresh and best_idx >= 0:
            matched_gt[best_idx] = True
            matched_preds.append((p["conf"], 1, p_cls))
        else:
            matched_preds.append((p["conf"], 0, p_cls))

    # Operating point conf >= conf_op
    preds_op = [p for p in preds if p["conf"] >= conf_op]
    matched_gt_op = [False] * len(gts)
    for p in preds_op:
        p_cls = p["cls"]
        p_box = p["xyxy"]
        best_iou, best_idx = 0.0, -1
        for g_idx, g in enumerate(gts):
            if g["cls"] != p_cls or matched_gt_op[g_idx]:
                continue
            iou = compute_box_iou(p_box, g["xyxy"])
            if iou > best_iou:
                best_iou, best_idx = iou, g_idx
        if best_iou >= iou_thresh and best_idx >= 0:
            matched_gt_op[best_idx] = True

    tp_op = sum(matched_gt_op)
    n_gt = len(gts)
    cls_gt = Counter_cls = [0] * 6
    for g in gts:
        Counter_cls[g["cls"]] += 1

    return {
        "n_gt": n_gt,
        "cls_gt": Counter_cls,
        "matched_preds": matched_preds,
        "tp_op": tp_op,
        "img_recall_op": (tp_op / float(n_gt) * 100.0) if n_gt > 0 else 0.0,
    }


def compute_ap_from_pr(confs, tps, n_gt):
    if n_gt == 0:
        return np.nan
    if len(confs) == 0:
        return 0.0
    confs_arr = np.asarray(confs, dtype=float)
    order = np.argsort(-confs_arr)
    tps_arr = np.asarray(tps, dtype=float)[order]
    fps_arr = 1.0 - tps_arr
    tp_cum = np.cumsum(tps_arr)
    fp_cum = np.cumsum(fps_arr)
    rec = tp_cum / float(n_gt)
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    x = np.linspace(0, 1, 101)
    ap = float(np.trapezoid(np.interp(x, mrec, mpre), x))
    return ap * 100.0


def evaluate_cached_list(cached_items: list[dict]) -> dict:
    all_confs, all_tps = [], []
    total_gt = 0
    cls_confs = [[] for _ in range(6)]
    cls_tps = [[] for _ in range(6)]
    cls_ngt = [0] * 6
    op_tp_total = 0
    img_recalls = []

    for it in cached_items:
        total_gt += it["n_gt"]
        op_tp_total += it["tp_op"]
        if it["n_gt"] > 0:
            img_recalls.append(it["img_recall_op"])
        for c_id in range(6):
            cls_ngt[c_id] += it["cls_gt"][c_id]
        for c_val, is_tp, c_id in it["matched_preds"]:
            all_confs.append(c_val)
            all_tps.append(is_tp)
            cls_confs[c_id].append(c_val)
            cls_tps[c_id].append(is_tp)

    micro_ap = compute_ap_from_pr(all_confs, all_tps, total_gt)
    cls_aps = []
    missing_cls = 0
    for c_id in range(6):
        if cls_ngt[c_id] == 0:
            missing_cls += 1
        else:
            cls_aps.append(compute_ap_from_pr(cls_confs[c_id], cls_tps[c_id], cls_ngt[c_id]))
    macro_ap = float(np.mean(cls_aps)) if cls_aps else 0.0
    inst_recall = (op_tp_total / float(total_gt) * 100.0) if total_gt > 0 else 0.0
    mean_img_recall = float(np.mean(img_recalls)) if img_recalls else 0.0
    return {
        "micro_ap50": micro_ap,
        "macro_ap50": macro_ap,
        "instance_recall_conf25": inst_recall,
        "mean_image_recall_conf25": mean_img_recall,
        "missing_classes": missing_cls,
    }


def build_clusters(leaked_stems: list[str]) -> list[list[str]]:
    adj = defaultdict(set)
    with EDGES_CSV.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            a = Path(r["a_file"]).stem
            b = Path(r["b_file"]).stem
            adj[a].add(b)
            adj[b].add(a)
    s_set = set(leaked_stems)
    visited = set()
    clusters = []
    for st in sorted(leaked_stems):
        if st not in visited:
            comp = []
            q = [st]
            visited.add(st)
            while q:
                cur = q.pop(0)
                comp.append(cur)
                for nb in sorted(adj[cur]):
                    if nb in s_set and nb not in visited:
                        visited.add(nb)
                        q.append(nb)
            clusters.append(comp)
    return clusters


def main():
    pairs = {}
    cached_pairs = {}
    for m in range(3):
        for s in range(8):
            dir_L = RUNS_DIR / f"m{m}_s{s}_L"
            dir_C = RUNS_DIR / f"m{m}_s{s}_C"
            sum_L = json.loads((dir_L / "eval_summary.json").read_text(encoding="utf-8"))
            sum_C = json.loads((dir_C / "eval_summary.json").read_text(encoding="utf-8"))
            img_L = json.loads((dir_L / "per_image_test_leaked.json").read_text(encoding="utf-8"))
            img_C = json.loads((dir_C / "per_image_test_leaked.json").read_text(encoding="utf-8"))
            pairs[(m, s)] = (sum_L, sum_C)
            cached_pairs[(m, s)] = (
                {st: precompute_image_stat(rec) for st, rec in img_L.items()},
                {st: precompute_image_stat(rec) for st, rec in img_C.items()},
            )

    leaked_stems = sorted(next(iter(cached_pairs.values()))[0].keys())
    clusters = build_clusters(leaked_stems)

    pair_metrics = []
    for (m, s), (cL, cC) in sorted(cached_pairs.items()):
        ev_L = evaluate_cached_list([cL[st] for st in leaked_stems])
        ev_C = evaluate_cached_list([cC[st] for st in leaked_stems])
        sum_L, sum_C = pairs[(m, s)]
        pair_metrics.append(
            {
                "m": m,
                "s": s,
                "official_macro_delta": (sum_L["test_leaked"]["map50"] - sum_C["test_leaked"]["map50"]) * 100.0,
                "micro_ap50_L": ev_L["micro_ap50"],
                "micro_ap50_C": ev_C["micro_ap50"],
                "micro_ap50_delta": ev_L["micro_ap50"] - ev_C["micro_ap50"],
                "inst_recall_L": ev_L["instance_recall_conf25"],
                "inst_recall_C": ev_C["instance_recall_conf25"],
                "inst_recall_delta": ev_L["instance_recall_conf25"] - ev_C["instance_recall_conf25"],
                "img_recall_L": ev_L["mean_image_recall_conf25"],
                "img_recall_C": ev_C["mean_image_recall_conf25"],
                "img_recall_delta": ev_L["mean_image_recall_conf25"] - ev_C["mean_image_recall_conf25"],
                "full_test_map50_delta": (sum_L["test_all"]["map50"] - sum_C["test_all"]["map50"]) * 100.0,
            }
        )

    def summarize_crossed(key: str):
        all_24 = np.array([p[key] for p in pair_metrics])
        by_seed = np.array([np.mean([p[key] for p in pair_metrics if p["s"] == s]) for s in range(8)])
        mean_s = float(np.mean(by_seed))
        sd_s = float(np.std(by_seed, ddof=1))
        se_s = sd_s / np.sqrt(8)
        t_crit = float(stats.t.ppf(0.975, df=7))
        _, p_val = stats.wilcoxon(by_seed)
        return {
            "point_mean": round(mean_s, 2),
            "seed_level_sd_n8": round(sd_s, 2),
            "cell_level_sd_n24": round(float(np.std(all_24, ddof=1)), 2),
            "seed_level_ci95_df7": [round(mean_s - t_crit * se_s, 2), round(mean_s + t_crit * se_s, 2)],
            "wilcoxon_p_n8": round(float(p_val), 4),
        }

    rng = np.random.default_rng(20261111)
    B = 2000
    n_clusters = len(clusters)
    boot_macro_deltas, boot_micro_deltas, boot_inst_rec_deltas = [], [], []
    boot_missing_cls_counts = []
    twoway_micro_deltas = []

    for _ in range(B):
        c_idxs = rng.integers(0, n_clusters, size=n_clusters)
        sampled_stems = [st for idx in c_idxs for st in clusters[idx]]

        run_macro_d, run_micro_d, run_rec_d = [], [], []
        miss_c = 0
        for (m, s), (cL, cC) in cached_pairs.items():
            eL = evaluate_cached_list([cL[st] for st in sampled_stems])
            eC = evaluate_cached_list([cC[st] for st in sampled_stems])
            miss_c = eL["missing_classes"]
            run_macro_d.append(eL["macro_ap50"] - eC["macro_ap50"])
            run_micro_d.append(eL["micro_ap50"] - eC["micro_ap50"])
            run_rec_d.append(eL["instance_recall_conf25"] - eC["instance_recall_conf25"])
        boot_missing_cls_counts.append(miss_c)
        boot_macro_deltas.append(np.mean(run_macro_d))
        boot_micro_deltas.append(np.mean(run_micro_d))
        boot_inst_rec_deltas.append(np.mean(run_rec_d))

        s_idxs = rng.integers(0, 8, size=8)
        tw_d = []
        for s_pick in s_idxs:
            for m in range(3):
                cL, cC = cached_pairs[(m, int(s_pick))]
                eL = evaluate_cached_list([cL[st] for st in sampled_stems])
                eC = evaluate_cached_list([cC[st] for st in sampled_stems])
                tw_d.append(eL["micro_ap50"] - eC["micro_ap50"])
        twoway_micro_deltas.append(np.mean(tw_d))

    report = {
        "n_leaked_test_images": len(leaked_stems),
        "n_clusters": n_clusters,
        "singleton_clusters": sum(1 for c in clusters if len(c) == 1),
        "multi_image_clusters": sum(1 for c in clusters if len(c) > 1),
        "class_dropout_diagnosis_in_bootstrap": {
            "pct_resamples_with_at_least_1_missing_class": round(
                float(np.mean(np.array(boot_missing_cls_counts) > 0) * 100.0), 2
            ),
            "macro_ap50_point_estimate_official": round(
                float(np.mean([p["official_macro_delta"] for p in pair_metrics])), 2
            ),
            "macro_ap50_bootstrap_mean": round(float(np.mean(boot_macro_deltas)), 2),
            "macro_ap50_bootstrap_ci95": [
                round(float(np.percentile(boot_macro_deltas, 2.5)), 2),
                round(float(np.percentile(boot_macro_deltas, 97.5)), 2),
            ],
            "explanation": (
                "Because `patches` has only 1 image and `rolled-in_scale` has 2 images in the 25-image subset, "
                "44.5% of bootstrap resamples omit at least one entire defect class, shifting the 6-class macro-mAP "
                "bootstrap mean downward. Pooled Micro-AP@0.5 and instance/image recall eliminate this class-dropout bias."
            ),
        },
        "crossed_design_inference_n8_seeds": {
            "official_macro_map50_delta_pp": summarize_crossed("official_macro_delta"),
            "pooled_micro_ap50_delta_pp": summarize_crossed("micro_ap50_delta"),
            "instance_recall_conf25_delta_pp": summarize_crossed("inst_recall_delta"),
            "per_image_recall_conf25_delta_pp": summarize_crossed("img_recall_delta"),
            "full_test_set_180_map50_delta_pp": summarize_crossed("full_test_map50_delta"),
        },
        "bootstrap_ci95_b2000": {
            "micro_ap50_point_mean": round(float(np.mean([p["micro_ap50_delta"] for p in pair_metrics])), 2),
            "micro_ap50_cluster_boot_mean": round(float(np.mean(boot_micro_deltas)), 2),
            "micro_ap50_cluster_boot_ci95": [
                round(float(np.percentile(boot_micro_deltas, 2.5)), 2),
                round(float(np.percentile(boot_micro_deltas, 97.5)), 2),
            ],
            "micro_ap50_twoway_seed_x_cluster_ci95": [
                round(float(np.percentile(twoway_micro_deltas, 2.5)), 2),
                round(float(np.percentile(twoway_micro_deltas, 97.5)), 2),
            ],
            "instance_recall_point_mean": round(float(np.mean([p["inst_recall_delta"] for p in pair_metrics])), 2),
            "instance_recall_cluster_boot_mean": round(float(np.mean(boot_inst_rec_deltas)), 2),
            "instance_recall_cluster_boot_ci95": [
                round(float(np.percentile(boot_inst_rec_deltas, 2.5)), 2),
                round(float(np.percentile(boot_inst_rec_deltas, 97.5)), 2),
            ],
        },
    }

    out_p = RESULTS_DIR / "step3_rigorous_reanalysis.json"
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

