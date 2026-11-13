#!/usr/bin/env python3
"""Comprehensive Statistical Aggregation and Audit of All 297 Cloud GPU Runs (2026-11-13).

Enforces all platform-comparability rules from README.md:
- Excludes _excluded_contaminated_stage1/ (no runs in cloud_results_20261113/runs).
- Never compares wall-clock train_minutes or speed_ms_per_image across concurrent cloud jobs.
- Computes paired E7 differences within the same seed and platform (RTX 3090 seeds 0-4, A100 seeds 5-9).
- Evaluates E5 dose-response across matched k in {0,1,2}, s in {0,1,2} on RTX 3090.
- Computes native Ultralytics metrics plus Pooled Micro-AP@0.5 and Instance Recall@0.5 (conf >= 0.25)
  from per_image_test_{clean,leaked}.json for both counterfactual_k10 (100 runs) and e5_dose (45 runs).
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
CLOUD_DIR = ROOT_DIR / "benchmark" / "cloud_results_20261113"
RUNS_DIR = CLOUD_DIR / "runs"
MANIFEST_CSV = CLOUD_DIR / "manifest.csv"
OUT_JSON = ROOT_DIR / "results" / "cloud_297_comprehensive_stats.json"


def compute_box_iou(b1, b2) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    u = a1 + a2 - inter
    return inter / u if u > 0 else 0.0


def precompute_image_stat(rec: dict, iou_thresh: float = 0.5, conf_op: float = 0.25) -> dict:
    gts = rec.get("gt", [])
    preds = sorted([p for p in rec.get("pred", []) if p["conf"] >= 0.01], key=lambda x: x["conf"], reverse=True)
    matched_gt = [False] * len(gts)
    matched_preds = []
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
    return {
        "n_gt": n_gt,
        "matched_preds": matched_preds,
        "tp_op": tp_op,
    }


def compute_ap_from_pr(confs, tps, n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
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
    return float(np.trapezoid(np.interp(x, mrec, mpre), x)) * 100.0


def evaluate_per_image_json(json_path: Path, return_stats: bool = False) -> dict:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    all_confs, all_tps = [], []
    total_gt, op_tp_total = 0, 0
    per_img_stats = []
    for rec in data.values():
        st = precompute_image_stat(rec)
        total_gt += st["n_gt"]
        op_tp_total += st["tp_op"]
        if return_stats:
            per_img_stats.append({"n_gt": st["n_gt"], "preds": st["matched_preds"]})
        for c_val, is_tp, _ in st["matched_preds"]:
            all_confs.append(c_val)
            all_tps.append(is_tp)
    out = {
        "n_images": len(data),
        "n_gt": total_gt,
        "micro_ap50": compute_ap_from_pr(all_confs, all_tps, total_gt),
        "instance_recall_conf25": (op_tp_total / float(total_gt) * 100.0) if total_gt > 0 else 0.0,
    }
    if return_stats:
        out["per_img_stats"] = per_img_stats
    return out


def eval_at_fixed_fppi(imgs_leak: list, imgs_clean: list, target_fppi: float) -> dict:
    """Calibrate confidence threshold tau on combined test set (imgs_leak + imgs_clean)
    to achieve target_fppi false positives per image, then compute instance miss rate (%)
    and image escape rate (% defective images with 0 TP detections) on each subset."""
    imgs_all = imgs_leak + imgs_clean
    fp_confs = sorted(
        [c for im in imgs_all for (c, tp, _) in im["preds"] if tp == 0],
        reverse=True,
    )
    k = int(round(target_fppi * len(imgs_all)))
    if k <= 0:
        tau = 1.01
    elif len(fp_confs) >= k:
        tau = 0.5 * (fp_confs[k - 1] + fp_confs[k]) if len(fp_confs) > k and fp_confs[k] < fp_confs[k - 1] else fp_confs[k - 1]
    else:
        tau = 0.01

    fp_all = sum(1 for im in imgs_all for (c, tp, _) in im["preds"] if tp == 0 and c >= tau)

    def _subset_rates(imgs: list) -> tuple[float, float]:
        tot_gt = sum(im["n_gt"] for im in imgs)
        tot_tp = sum(sum(1 for (c, tp, _) in im["preds"] if tp == 1 and c >= tau) for im in imgs)
        defective = [im for im in imgs if im["n_gt"] > 0]
        escaped = sum(
            1 for im in defective if sum(1 for (c, tp, _) in im["preds"] if tp == 1 and c >= tau) == 0
        )
        inst_miss = 100.0 * (1.0 - tot_tp / float(tot_gt)) if tot_gt > 0 else 0.0
        img_escape = 100.0 * (escaped / float(len(defective))) if defective else 0.0
        return inst_miss, img_escape

    leak_miss, leak_esc = _subset_rates(imgs_leak)
    clean_miss, clean_esc = _subset_rates(imgs_clean)
    return {
        "tau": float(tau),
        "achieved_fppi": float(fp_all / float(len(imgs_all))),
        "leak_inst_miss": leak_miss,
        "leak_img_escape": leak_esc,
        "clean_inst_miss": clean_miss,
        "clean_img_escape": clean_esc,
    }


def summ_1d(vals) -> dict:
    arr = np.asarray(vals, dtype=float)
    n = int(arr.size)
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    ci = float(stats.t.ppf(0.975, n - 1) * se) if n > 1 else 0.0
    mean = float(arr.mean())
    return {
        "n": n,
        "mean": round(mean, 4),
        "sd": round(sd, 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "range": round(float(arr.max() - arr.min()), 4),
        "ci95_lo": round(mean - ci, 4),
        "ci95_hi": round(mean + ci, 4),
        "values": [round(float(x), 4) for x in arr],
    }


def paired_test(diffs) -> dict:
    arr = np.asarray(diffs, dtype=float)
    s = summ_1d(arr)
    t_res = stats.ttest_1samp(arr, 0.0)
    try:
        w_res = stats.wilcoxon(arr)
        w_p = float(w_res.pvalue)
    except Exception:
        w_p = 1.0
    s["p_ttest"] = float(t_res.pvalue)
    s["p_wilcoxon"] = w_p
    s["n_pos"] = int(np.sum(arr > 0))
    return s


def analyze_baseline(df: pd.DataFrame, suite: str) -> dict:
    sub = df[df["suite"] == suite]
    arches = ["yolo11n", "yolov8n", "yolo11s", "yolov8s", "rtdetr-l"]
    out = {"n_runs": int(len(sub)), "architectures": {}}
    # Get class list from first eval_summary
    first_run = sub.iloc[0]["run"]
    first_es = json.loads((RUNS_DIR / suite / first_run / "eval_summary.json").read_text(encoding="utf-8"))
    classes = list(first_es["test"]["per_class_ap50"].keys())

    for arch in arches:
        adf = sub[sub["arch"] == arch].sort_values("seed")
        per_class = {c: [] for c in classes}
        pt_tensors = []
        gpus = set()
        for rname in adf["run"]:
            es = json.loads((RUNS_DIR / suite / rname / "eval_summary.json").read_text(encoding="utf-8"))
            pt_tensors.append(es.get("pretrained_tensors_transferred"))
            gpus.add(es.get("gpu"))
            for c in classes:
                per_class[c].append(es["test"]["per_class_ap50"][c] * 100.0)

        out["architectures"][arch] = {
            "gpus": sorted(list(gpus)),
            "pretrained_tensors_transferred": sorted(list(set(pt_tensors))),
            "val_map50": summ_1d(adf["val_map50"] * 100.0),
            "val_map50_95": summ_1d(adf["val_map50_95"] * 100.0),
            "test_map50": summ_1d(adf["test_map50"] * 100.0),
            "test_map50_95": summ_1d(adf["test_map50_95"] * 100.0),
            "test_precision": summ_1d(adf["test_precision"] * 100.0),
            "test_recall": summ_1d(adf["test_recall"] * 100.0),
            "val_minus_test_map50": summ_1d((adf["val_map50"] - adf["test_map50"]) * 100.0),
            "per_class_test_ap50": {c: summ_1d(per_class[c]) for c in classes},
        }
    return out


def analyze_e2(df: pd.DataFrame) -> dict:
    sub = df[df["suite"].str.startswith("e2_g")].copy()
    splits = sorted(sub["suite"].unique())
    per_split = {}
    split_means = {m: [] for m in ["val_map50", "val_map50_95", "test_map50", "test_map50_95"]}
    all_vals = {m: [] for m in ["val_map50", "val_map50_95", "test_map50", "test_map50_95"]}

    for sp in splits:
        sdf = sub[sub["suite"] == sp].sort_values("seed")
        per_split[sp] = {
            "seeds": sdf["seed"].tolist(),
            "val_map50": summ_1d(sdf["val_map50"] * 100.0),
            "val_map50_95": summ_1d(sdf["val_map50_95"] * 100.0),
            "test_map50": summ_1d(sdf["test_map50"] * 100.0),
            "test_map50_95": summ_1d(sdf["test_map50_95"] * 100.0),
        }
        for m in split_means:
            vals = (sdf[m] * 100.0).tolist()
            split_means[m].append(float(np.mean(vals)))
            all_vals[m].append(vals)

    anova = {}
    G, S = len(splits), 3
    for m, mat_list in all_vals.items():
        mat = np.array(mat_list, dtype=float)  # (G, S)
        grand = mat.mean()
        ss_between = S * np.sum((mat.mean(axis=1) - grand) ** 2)
        ss_within = np.sum((mat - mat.mean(axis=1, keepdims=True)) ** 2)
        ms_between = ss_between / (G - 1)
        ms_within = ss_within / (G * (S - 1))
        var_split = max(0.0, (ms_between - ms_within) / S)
        var_seed = ms_within
        var_tot = var_split + var_seed
        f_stat = ms_between / ms_within if ms_within > 0 else float("inf")
        p_val = float(stats.f.sf(f_stat, G - 1, G * (S - 1)))
        anova[m] = {
            "overall_summary": summ_1d(mat.flatten()),
            "split_means_summary": summ_1d(mat.mean(axis=1)),
            "ms_between": round(float(ms_between), 4),
            "ms_within": round(float(ms_within), 4),
            "var_split": round(float(var_split), 4),
            "var_seed": round(float(var_seed), 4),
            "sd_split": round(float(np.sqrt(var_split)), 4),
            "sd_seed": round(float(np.sqrt(var_seed)), 4),
            "pct_split": round(float(var_split / var_tot * 100.0), 2),
            "pct_seed": round(float(var_seed / var_tot * 100.0), 2),
            "f_stat": round(float(f_stat), 4),
            "p_value": p_val,
        }

    return {"n_runs": int(len(sub)), "n_splits": G, "per_split": per_split, "variance_decomposition": anova}


def analyze_counterfactual_k10(df: pd.DataFrame) -> dict:
    cf = df[(df["suite"] == "counterfactual_k10") & (df["condition"].isin(["C", "L"]))].copy()
    classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

    # Cache per-image micro-AP and instance recall for all 100 counterfactual runs
    per_run_micro = {}
    for _, row in cf.iterrows():
        rname = row["run"]
        rdir = RUNS_DIR / "counterfactual_k10" / rname
        es = json.loads((rdir / "eval_summary.json").read_text(encoding="utf-8"))
        pi_leak = evaluate_per_image_json(rdir / "per_image_test_leaked.json", return_stats=True)
        pi_clean = evaluate_per_image_json(rdir / "per_image_test_clean.json", return_stats=True)
        op_rates = {
            f"fppi_{fppi:.2f}": eval_at_fixed_fppi(pi_leak["per_img_stats"], pi_clean["per_img_stats"], fppi)
            for fppi in (0.15, 0.25)
        }
        per_run_micro[rname] = {
            "es": es,
            "leak_micro": pi_leak["micro_ap50"],
            "leak_inst_rec": pi_leak["instance_recall_conf25"],
            "clean_micro": pi_clean["micro_ap50"],
            "clean_inst_rec": pi_clean["instance_recall_conf25"],
            "n_leak_img": pi_leak["n_images"],
            "n_clean_img": pi_clean["n_images"],
            "op_rates": op_rates,
        }

    per_split = {}
    split_series = {
        m: {"C_clean": [], "L_clean": [], "C_leak": [], "L_leak": [], "C_all": [], "L_all": []}
        for m in ["map50", "map50_95", "precision", "recall", "micro_ap50", "inst_rec25"]
    }
    cls_series = {c: {"C_leak": [], "L_leak": [], "C_clean": [], "L_clean": []} for c in classes}

    for k in range(10):
        rc = cf[(cf["split_k"] == k) & (cf["condition"] == "C")].sort_values("seed")
        rl = cf[(cf["split_k"] == k) & (cf["condition"] == "L")].sort_values("seed")
        per_split[f"k{k:02d}"] = {
            "n_leak_images": per_run_micro[rc.iloc[0]["run"]]["n_leak_img"],
            "n_clean_images": per_run_micro[rc.iloc[0]["run"]]["n_clean_img"],
            "C_leak_map50": round(float(rc["test_leaked_map50"].mean() * 100.0), 4),
            "L_leak_map50": round(float(rl["test_leaked_map50"].mean() * 100.0), 4),
            "delta_leak_map50": round(float((rl["test_leaked_map50"].values - rc["test_leaked_map50"].values).mean() * 100.0), 4),
            "C_leak_map50_95": round(float(rc["test_leaked_map50_95"].mean() * 100.0), 4),
            "L_leak_map50_95": round(float(rl["test_leaked_map50_95"].mean() * 100.0), 4),
            "delta_leak_map50_95": round(float((rl["test_leaked_map50_95"].values - rc["test_leaked_map50_95"].values).mean() * 100.0), 4),
            "delta_clean_map50": round(float((rl["test_clean_map50"].values - rc["test_clean_map50"].values).mean() * 100.0), 4),
        }
        for m, col_c, col_l, col_a in [
            ("map50", "test_clean_map50", "test_leaked_map50", "test_all_map50"),
            ("map50_95", "test_clean_map50_95", "test_leaked_map50_95", "test_all_map50_95"),
            ("precision", "test_clean_precision", "test_leaked_precision", "test_all_precision"),
            ("recall", "test_clean_recall", "test_leaked_recall", "test_all_recall"),
        ]:
            split_series[m]["C_clean"].append(float(rc[col_c].mean() * 100.0))
            split_series[m]["L_clean"].append(float(rl[col_c].mean() * 100.0))
            split_series[m]["C_leak"].append(float(rc[col_l].mean() * 100.0))
            split_series[m]["L_leak"].append(float(rl[col_l].mean() * 100.0))
            split_series[m]["C_all"].append(float(rc[col_a].mean() * 100.0))
            split_series[m]["L_all"].append(float(rl[col_a].mean() * 100.0))

        # Per-image micro AP and instance recall
        split_series["micro_ap50"]["C_clean"].append(float(np.mean([per_run_micro[r]["clean_micro"] for r in rc["run"]])))
        split_series["micro_ap50"]["L_clean"].append(float(np.mean([per_run_micro[r]["clean_micro"] for r in rl["run"]])))
        split_series["micro_ap50"]["C_leak"].append(float(np.mean([per_run_micro[r]["leak_micro"] for r in rc["run"]])))
        split_series["micro_ap50"]["L_leak"].append(float(np.mean([per_run_micro[r]["leak_micro"] for r in rl["run"]])))
        split_series["inst_rec25"]["C_clean"].append(float(np.mean([per_run_micro[r]["clean_inst_rec"] for r in rc["run"]])))
        split_series["inst_rec25"]["L_clean"].append(float(np.mean([per_run_micro[r]["clean_inst_rec"] for r in rl["run"]])))
        split_series["inst_rec25"]["C_leak"].append(float(np.mean([per_run_micro[r]["leak_inst_rec"] for r in rc["run"]])))
        split_series["inst_rec25"]["L_leak"].append(float(np.mean([per_run_micro[r]["leak_inst_rec"] for r in rl["run"]])))

        for cls in classes:
            cls_series[cls]["C_leak"].append(float(np.mean([per_run_micro[r]["es"]["test_leaked"]["per_class_ap50"][cls] * 100.0 for r in rc["run"]])))
            cls_series[cls]["L_leak"].append(float(np.mean([per_run_micro[r]["es"]["test_leaked"]["per_class_ap50"][cls] * 100.0 for r in rl["run"]])))
            cls_series[cls]["C_clean"].append(float(np.mean([per_run_micro[r]["es"]["test_clean"]["per_class_ap50"][cls] * 100.0 for r in rc["run"]])))
            cls_series[cls]["L_clean"].append(float(np.mean([per_run_micro[r]["es"]["test_clean"]["per_class_ap50"][cls] * 100.0 for r in rl["run"]])))

    split_level = {}
    for m, d in split_series.items():
        cc = np.array(d["C_clean"])
        lc = np.array(d["L_clean"])
        cl = np.array(d["C_leak"])
        ll = np.array(d["L_leak"])
        entry = {
            "C_clean": summ_1d(cc),
            "L_clean": summ_1d(lc),
            "C_leak": summ_1d(cl),
            "L_leak": summ_1d(ll),
            "delta_obs": paired_test(ll - cc),
            "delta_leak": paired_test(ll - cl),
            "delta_diff": paired_test(cl - cc),
            "delta_clean": paired_test(lc - cc),
        }
        if d["C_all"]:
            ca = np.array(d["C_all"])
            la = np.array(d["L_all"])
            entry["C_all"] = summ_1d(ca)
            entry["L_all"] = summ_1d(la)
            entry["delta_all"] = paired_test(la - ca)
        split_level[m] = entry

    per_class = {}
    for cls, d in cls_series.items():
        cl = np.array(d["C_leak"])
        ll = np.array(d["L_leak"])
        cc = np.array(d["C_clean"])
        lc = np.array(d["L_clean"])
        per_class[cls] = {
            "C_leak": summ_1d(cl),
            "L_leak": summ_1d(ll),
            "delta_leak": paired_test(ll - cl),
            "C_clean": summ_1d(cc),
            "L_clean": summ_1d(lc),
            "delta_clean": paired_test(lc - cc),
        }

    op_escape = {}
    for fkey in ("fppi_0.15", "fppi_0.25"):
        op_entry = {}
        for metric in ("leak_inst_miss", "leak_img_escape", "clean_inst_miss", "clean_img_escape"):
            c_vals = [
                float(np.mean([
                    per_run_micro[r]["op_rates"][fkey][metric]
                    for r in cf[(cf["split_k"] == k) & (cf["condition"] == "C")]["run"]
                ]))
                for k in range(10)
            ]
            l_vals = [
                float(np.mean([
                    per_run_micro[r]["op_rates"][fkey][metric]
                    for r in cf[(cf["split_k"] == k) & (cf["condition"] == "L")]["run"]
                ]))
                for k in range(10)
            ]
            ca, la = np.array(c_vals), np.array(l_vals)
            op_entry[metric] = {
                "C": summ_1d(ca),
                "L": summ_1d(la),
                "delta_L_minus_C": paired_test(la - ca),
            }
        op_entry["achieved_fppi_C"] = summ_1d([
            float(np.mean([
                per_run_micro[r]["op_rates"][fkey]["achieved_fppi"]
                for r in cf[(cf["split_k"] == k) & (cf["condition"] == "C")]["run"]
            ]))
            for k in range(10)
        ])
        op_entry["achieved_fppi_L"] = summ_1d([
            float(np.mean([
                per_run_micro[r]["op_rates"][fkey]["achieved_fppi"]
                for r in cf[(cf["split_k"] == k) & (cf["condition"] == "L")]["run"]
            ]))
            for k in range(10)
        ])
        op_escape[fkey] = op_entry

    return {
        "n_runs": int(len(cf)),
        "total_leaked_image_evals": int(sum(per_split[k]["n_leak_images"] for k in per_split)),
        "per_split": per_split,
        "split_level_k10": split_level,
        "per_class_ap50": per_class,
        "operating_point_escape_rates": op_escape,
    }


def analyze_e5_dose(df: pd.DataFrame) -> dict:
    sub = df[(df["suite"] == "counterfactual_k10") & (df["split_k"].isin([0, 1, 2])) & (df["seed"].isin([0, 1, 2]))].copy()
    doses = [("C", 0), ("D25", 25), ("D50", 50), ("D75", 75), ("L", 100)]
    per_run = {}
    for _, row in sub.iterrows():
        rname = row["run"]
        rdir = RUNS_DIR / "counterfactual_k10" / rname
        pi_leak = evaluate_per_image_json(rdir / "per_image_test_leaked.json")
        pi_clean = evaluate_per_image_json(rdir / "per_image_test_clean.json")
        per_run[(int(row["split_k"]), int(row["seed"]), row["condition"])] = {
            "leak_map50": float(row["test_leaked_map50"] * 100.0),
            "leak_map50_95": float(row["test_leaked_map50_95"] * 100.0),
            "leak_micro": pi_leak["micro_ap50"],
            "leak_inst_rec": pi_leak["instance_recall_conf25"],
            "clean_map50": float(row["test_clean_map50"] * 100.0),
            "clean_map50_95": float(row["test_clean_map50_95"] * 100.0),
            "clean_micro": pi_clean["micro_ap50"],
            "clean_inst_rec": pi_clean["instance_recall_conf25"],
            "all_map50": float(row["test_all_map50"] * 100.0),
            "all_map50_95": float(row["test_all_map50_95"] * 100.0),
        }

    pairs = [(k, s) for k in range(3) for s in range(3)]
    metrics = [
        "leak_map50", "leak_map50_95", "leak_micro", "leak_inst_rec",
        "clean_map50", "clean_map50_95", "clean_micro", "clean_inst_rec",
        "all_map50", "all_map50_95",
    ]
    by_dose = {}
    for cond, pct in doses:
        d_entry = {"dose_pct": pct}
        for m in metrics:
            vals = np.array([per_run[(k, s, cond)][m] for (k, s) in pairs])
            base_vals = np.array([per_run[(k, s, "C")][m] for (k, s) in pairs])
            d_entry[m] = summ_1d(vals)
            if cond != "C":
                d_entry[f"delta_vs_C_{m}"] = paired_test(vals - base_vals)
        by_dose[cond] = d_entry

    return {"n_runs": int(len(sub)), "n_matched_series": len(pairs), "by_dose": by_dose}


def analyze_e7_modules(df: pd.DataFrame) -> dict:
    sub = df[df["suite"] == "e7_modules"].copy()
    variants = ["yolo11n-base", "yolo11n-cbam", "yolo11n-ema", "yolo11n-bifpn"]
    out = {"n_runs": int(len(sub)), "variants": {}, "paired_vs_base": {}, "cherry_pick_inflation": {}}

    seed_data = {v: {} for v in variants}
    for _, row in sub.iterrows():
        rdir = RUNS_DIR / "e7_modules" / row["run"]
        es = json.loads((rdir / "eval_summary.json").read_text(encoding="utf-8"))
        seed_data[row["arch"]][int(row["seed"])] = {
            "val_map50": float(row["val_map50"] * 100.0),
            "test_map50": float(row["test_map50"] * 100.0),
            "test_map50_95": float(row["test_map50_95"] * 100.0),
            "tensors": es.get("pretrained_tensors_transferred"),
            "machine": row["machine"],
            "gpu": row["gpu"],
        }

    for v in variants:
        seeds = sorted(seed_data[v].keys())
        out["variants"][v] = {
            "pretrained_tensors_transferred": sorted(list(set(seed_data[v][s]["tensors"] for s in seeds))),
            "all_10_seeds": {
                "val_map50": summ_1d([seed_data[v][s]["val_map50"] for s in seeds]),
                "test_map50": summ_1d([seed_data[v][s]["test_map50"] for s in seeds]),
                "test_map50_95": summ_1d([seed_data[v][s]["test_map50_95"] for s in seeds]),
            },
            "rtx3090_s0_4": {
                "val_map50": summ_1d([seed_data[v][s]["val_map50"] for s in range(5)]),
                "test_map50": summ_1d([seed_data[v][s]["test_map50"] for s in range(5)]),
                "test_map50_95": summ_1d([seed_data[v][s]["test_map50_95"] for s in range(5)]),
            },
            "a100_s5_9": {
                "val_map50": summ_1d([seed_data[v][s]["val_map50"] for s in range(5, 10)]),
                "test_map50": summ_1d([seed_data[v][s]["test_map50"] for s in range(5, 10)]),
                "test_map50_95": summ_1d([seed_data[v][s]["test_map50_95"] for s in range(5, 10)]),
            },
        }

    for v in variants[1:]:
        d50_all = [seed_data[v][s]["test_map50"] - seed_data["yolo11n-base"][s]["test_map50"] for s in range(10)]
        d95_all = [seed_data[v][s]["test_map50_95"] - seed_data["yolo11n-base"][s]["test_map50_95"] for s in range(10)]
        dv50_all = [seed_data[v][s]["val_map50"] - seed_data["yolo11n-base"][s]["val_map50"] for s in range(10)]
        out["paired_vs_base"][v] = {
            "all_10_test_map50": paired_test(d50_all),
            "all_10_test_map50_95": paired_test(d95_all),
            "all_10_val_map50": paired_test(dv50_all),
            "rtx3090_s0_4_test_map50": paired_test(d50_all[:5]),
            "a100_s5_9_test_map50": paired_test(d50_all[5:]),
        }

        # Within-platform cherry picking: best seed of variant vs worst seed of base
        for plat_name, s_range in [("rtx3090_s0_4", range(5)), ("a100_s5_9", range(5, 10))]:
            base_vals = [seed_data["yolo11n-base"][s]["test_map50"] for s in s_range]
            var_vals = [seed_data[v][s]["test_map50"] for s in s_range]
            var_vals_val = [seed_data[v][s]["val_map50"] for s in s_range]
            base_vals_val = [seed_data["yolo11n-base"][s]["val_map50"] for s in s_range]
            out["cherry_pick_inflation"][f"{v}_{plat_name}"] = {
                "best_var_test_minus_worst_base_test": round(max(var_vals) - min(base_vals), 4),
                "best_var_val_minus_worst_base_test": round(max(var_vals_val) - min(base_vals), 4),
                "best_var_val_minus_mean_base_test": round(max(var_vals_val) - float(np.mean(base_vals)), 4),
                "best_var_test_minus_mean_base_test": round(max(var_vals) - float(np.mean(base_vals)), 4),
            }

    return out


def main():
    df = pd.read_csv(MANIFEST_CSV)
    print(f"Loaded manifest.csv: {len(df)} runs across suites {df['suite'].value_counts().to_dict()}")

    results = {
        "total_runs": int(len(df)),
        "baseline_neu": analyze_baseline(df, "baseline_neu"),
        "baseline_gc10": analyze_baseline(df, "baseline_gc10"),
        "e2_variance_decomposition": analyze_e2(df),
        "counterfactual_k10": analyze_counterfactual_k10(df),
        "e5_dose_response": analyze_e5_dose(df),
        "e7_modules": analyze_e7_modules(df),
    }

    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved comprehensive stats to {OUT_JSON}")

    # Concise console verification summary
    e2_50 = results["e2_variance_decomposition"]["variance_decomposition"]["test_map50"]
    print(f"\n[E2 ANOVA Test mAP@0.5] split_means range={e2_50['split_means_summary']['range']:.2f} pp "
          f"({e2_50['split_means_summary']['min']:.2f}..{e2_50['split_means_summary']['max']:.2f}), "
          f"sd_split={e2_50['sd_split']:.2f} ({e2_50['pct_split']}%), sd_seed={e2_50['sd_seed']:.2f} ({e2_50['pct_seed']}%)")

    cf_50 = results["counterfactual_k10"]["split_level_k10"]["map50"]["delta_leak"]
    cf_mic = results["counterfactual_k10"]["split_level_k10"]["micro_ap50"]["delta_leak"]
    cf_rec = results["counterfactual_k10"]["split_level_k10"]["inst_rec25"]["delta_leak"]
    cf_cln = results["counterfactual_k10"]["split_level_k10"]["map50"]["delta_clean"]
    print(f"[K=10 CF] delta_leak map50={cf_50['mean']:+.2f}+-{cf_50['sd']:.2f} (p_w={cf_50['p_wilcoxon']:.4f}, pos={cf_50['n_pos']}/10) | "
          f"micro_ap50={cf_mic['mean']:+.2f}+-{cf_mic['sd']:.2f} (p_w={cf_mic['p_wilcoxon']:.4f}) | "
          f"inst_rec={cf_rec['mean']:+.2f}+-{cf_rec['sd']:.2f} (p_w={cf_rec['p_wilcoxon']:.4f}) | "
          f"delta_clean map50={cf_cln['mean']:+.2f}+-{cf_cln['sd']:.2f} (p_w={cf_cln['p_wilcoxon']:.4f})")

    print("[E5 Dose-Response on test_leaked]:")
    for cond in ["C", "D25", "D50", "D75", "L"]:
        d = results["e5_dose_response"]["by_dose"][cond]
        print(f"  {cond:4s} ({d['dose_pct']:3d}%): map50={d['leak_micro']['mean']:.2f} (micro), "
              f"macro={d['leak_map50']['mean']:.2f}, inst_rec={d['leak_inst_rec']['mean']:.2f} | "
              f"clean_macro={d['clean_map50']['mean']:.2f}")

    print("[E7 Modules 10-seed paired vs yolo11n-base]:")
    for v in ["yolo11n-cbam", "yolo11n-ema", "yolo11n-bifpn"]:
        p = results["e7_modules"]["paired_vs_base"][v]["all_10_test_map50"]
        print(f"  {v:14s}: {p['mean']:+.2f}+-{p['sd']:.2f} (95% CI [{p['ci95_lo']:+.2f}, {p['ci95_hi']:+.2f}], p_w={p['p_wilcoxon']:.4f})")


if __name__ == "__main__":
    main()
